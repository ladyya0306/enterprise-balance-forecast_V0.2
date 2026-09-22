"""SYN-B1-I4A-2: turn reconciled monthly cash into dated transaction candidates.

This module is deliberately a *planner*, not a ledger.  It keeps the observed
account's routed monthly amounts unchanged, splits ordinary items according to
user-declared frequency/concentration/amount-variation settings, and retains
tax and funding dates supplied by their existing modules.  It then asks the
shared stop controller whether the resulting ordered plan is payable.

No CSV, SQLite, transaction-after-balance, automatic funding, or legacy SG2/
SG3 code belongs here.  I4B owns actual posting and balances.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from enum import Enum
from hashlib import sha256
from heapq import heappop, heappush
from random import Random
from typing import Mapping, Sequence

from syn_b1.generation_stop import (
    CashExecutionPreviewResult,
    MovementDirection,
    PlannedCashMovement,
    preview_cash_sufficiency,
)
from syn_b1.funding_contracts import FundingEventType, FundingKind, LoanContract
from syn_b1.funding_timeline import FundingTimelineState, ShareholderCapitalPlan, TailStatus
from syn_b1.tax_routing import CashCategory, CashDirection, TaxRoutingResult


class TransactionPlanningError(ValueError):
    """A candidate-plan input conflicts with the frozen I4A-2 contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MonthConcentration(str, Enum):
    """A simple, visible choice for the dates of ordinary monthly cash items."""

    UNIFORM = "uniform"
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


@dataclass(frozen=True)
class CategoryTransactionPolicy:
    """User-controlled split settings for one ordinary cash category.

    ``transaction_count`` is an exact count for each non-zero monthly item in
    the category.  ``amount_variation_bp`` is a permitted amount spread around
    an equal split; it never changes the monthly total.  This deliberately
    exposes only the three controls needed by this stage, rather than adding a
    large set of overlapping amount-distribution knobs.
    """

    transaction_count: int = 1
    month_concentration: MonthConcentration = MonthConcentration.UNIFORM
    amount_variation_bp: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.transaction_count, int)
            or isinstance(self.transaction_count, bool)
            or self.transaction_count <= 0
        ):
            raise TransactionPlanningError("INVALID_TRANSACTION_COUNT", "每月拆分笔数必须是正整数。")
        if not isinstance(self.month_concentration, MonthConcentration):
            raise TransactionPlanningError("INVALID_MONTH_CONCENTRATION", "月份集中方式必须为均匀、月初、月中或月末。")
        if (
            not isinstance(self.amount_variation_bp, int)
            or isinstance(self.amount_variation_bp, bool)
            or not 0 <= self.amount_variation_bp <= 9_999
        ):
            raise TransactionPlanningError("INVALID_AMOUNT_VARIATION", "金额波动幅度必须是0至9999个基点。")


@dataclass(frozen=True)
class DatedSourceEvent:
    """An immutable date supplied by an earlier contract/tax module.

    ``source_item_id`` points to the full enterprise monthly source item.  A
    partially routed observed amount is allocated proportionally over these
    dated events, with integer-cent rounding closed inside the group.  The
    planner therefore never guesses or moves a funding or tax date.
    """

    event_id: str
    source_item_id: str
    booking_datetime: datetime
    source_amount_fen: int
    source_frozen_order: int = 0
    funding_tail_item_id: str | None = None
    # 新订单来源只用于内部追踪，不包含未来模型标签。
    business_order_id: str | None = None
    # 公开备注仅描述本笔已经发生的业务。
    historical_note_cn: str | None = None
    # 原合同到期时刻永久保留，实际支付可能在已协商宽限内不同。
    original_due_booking_datetime: datetime | None = None
    # 宽限和回款依赖都必须来自预算中的已协商条款。
    negotiated_grace_days: int = 0
    dependent_receipt_event_id: str | None = None
    payment_deferral_reason_cn: str | None = None
    receipt_dependency_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.event_id or not self.source_item_id:
            raise TransactionPlanningError("EMPTY_DATED_EVENT_ID", "定日事项及其来源编号不能为空。")
        if not isinstance(self.booking_datetime, datetime):
            raise TransactionPlanningError("INVALID_DATED_EVENT_TIME", "定日事项必须提供日期时间。")
        if (
            not isinstance(self.source_amount_fen, int)
            or isinstance(self.source_amount_fen, bool)
            or self.source_amount_fen <= 0
        ):
            raise TransactionPlanningError("INVALID_DATED_EVENT_AMOUNT", "定日事项金额必须为正整数分。")
        if (
            not isinstance(self.source_frozen_order, int)
            or isinstance(self.source_frozen_order, bool)
            or self.source_frozen_order < 0
        ):
            raise TransactionPlanningError("INVALID_DATED_EVENT_ORDER", "定日事项冻结顺序必须为非负整数。")
        if self.funding_tail_item_id is not None and not self.funding_tail_item_id:
            raise TransactionPlanningError("INVALID_FUNDING_TAIL_REFERENCE", "融资尾巴引用不能为空。")


@dataclass(frozen=True)
class TransactionPlanningRequest:
    """Input boundary for the dated-candidate planner.

    Financing dates come from I3B/I3C as ``dated_source_events``.  Taxes due
    in the current horizon are filled from I4A-1 automatically.  Ordinary
    operating/investing items use the default policy or a category override.
    """

    tax_routing_result: TaxRoutingResult
    random_seed: int
    default_policy: CategoryTransactionPolicy = CategoryTransactionPolicy()
    category_policies: Mapping[CashCategory, CategoryTransactionPolicy] | None = None
    dated_source_events: tuple[DatedSourceEvent, ...] = ()
    version: str = "syn_b1_i4a2_transaction_candidate_1_0"
    # 与融资税费分开输入，防止普通外部日期越权改写来源。
    business_order_events: tuple[DatedSourceEvent, ...] = ()
    # 完整合同节点覆盖普通经营及设备，不接管税费融资。
    contract_source_events: tuple[DatedSourceEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TransactionPlanningError("INVALID_RANDOM_SEED", "随机种子必须是整数。")
        if not isinstance(self.default_policy, CategoryTransactionPolicy):
            raise TransactionPlanningError("INVALID_DEFAULT_POLICY", "默认拆分规则无效。")
        if not self.version:
            raise TransactionPlanningError("EMPTY_TRANSACTION_PLANNING_VERSION", "逐笔候选版本不能为空。")
        for category, policy in (self.category_policies or {}).items():
            if not isinstance(category, CashCategory) or not isinstance(policy, CategoryTransactionPolicy):
                raise TransactionPlanningError("INVALID_CATEGORY_POLICY", "类别规则必须使用现金类别和拆分规则。")
        event_ids = [item.event_id for item in self.dated_source_events]
        if len(event_ids) != len(set(event_ids)):
            raise TransactionPlanningError("DUPLICATE_DATED_EVENT_ID", "定日事项编号不能重复。")
        # 所有事件共享唯一编号空间。
        all_ids = event_ids + [event.event_id for event in (*self.business_order_events, *self.contract_source_events)]
        # 一张订单的同一收付动作不能记两次。
        if len(all_ids) != len(set(all_ids)) or any(not e.business_order_id or not e.historical_note_cn for e in (*self.business_order_events, *self.contract_source_events)):
            # 明确拒绝重复事件或缺失来源。
            raise TransactionPlanningError("INVALID_BUSINESS_ORDER_EVENT", "新增订单现金编号重复或缺少业务来源。")


@dataclass(frozen=True)
class TransactionCandidate:
    """One observed-account candidate before I4B posts it to the ledger."""

    candidate_id: str
    source_item_id: str
    category: CashCategory
    direction: CashDirection
    booking_datetime: datetime
    sequence_no: int
    amount_fen: int
    date_basis: str
    source_frozen_order: int = 0
    funding_tail_item_id: str | None = None
    # 来源贯穿逐笔候选、实际流水对应和受限核对清单。
    business_order_id: str | None = None
    # 中文备注不包含未发生的后续动作。
    historical_note_cn: str | None = None
    # 下列字段形成逐笔可审计的原到期、实际支付和协商依据链。
    original_due_booking_datetime: datetime | None = None
    negotiated_grace_days: int = 0
    dependent_receipt_event_id: str | None = None
    payment_deferral_reason_cn: str | None = None
    # 计划级开关必须贯穿到候选，默认关闭以兼容旧画像。
    receipt_dependency_enabled: bool = False
    receipt_dependency_applied: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.source_item_id:
            raise TransactionPlanningError("EMPTY_CANDIDATE_ID", "逐笔候选及其来源编号不能为空。")
        if not isinstance(self.category, CashCategory) or not isinstance(self.direction, CashDirection):
            raise TransactionPlanningError("INVALID_CANDIDATE_DIRECTION", "逐笔候选必须提供已知类别和收付方向。")
        if not isinstance(self.booking_datetime, datetime):
            raise TransactionPlanningError("INVALID_CANDIDATE_TIME", "逐笔候选必须提供日期时间。")
        if not isinstance(self.sequence_no, int) or isinstance(self.sequence_no, bool) or self.sequence_no < 0:
            raise TransactionPlanningError("INVALID_CANDIDATE_SEQUENCE", "逐笔候选顺序必须为非负整数。")
        if not isinstance(self.amount_fen, int) or isinstance(self.amount_fen, bool) or self.amount_fen <= 0:
            raise TransactionPlanningError("INVALID_CANDIDATE_AMOUNT", "逐笔候选金额必须为正整数分。")
        # 新增订单允许使用独立的冻结业务日期标识。
        if self.date_basis not in {"policy_split", "frozen_tax_date", "frozen_funding_date", "frozen_order_date", "receipt_dependent_negotiated_payment"}:
            raise TransactionPlanningError("INVALID_DATE_BASIS", "逐笔候选日期依据无效。")
        if (
            not isinstance(self.source_frozen_order, int)
            or isinstance(self.source_frozen_order, bool)
            or self.source_frozen_order < 0
        ):
            raise TransactionPlanningError("INVALID_CANDIDATE_SOURCE_ORDER", "逐笔候选来源顺序必须为非负整数。")
        if self.funding_tail_item_id is not None:
            if self.date_basis != "frozen_funding_date" or not self.funding_tail_item_id:
                raise TransactionPlanningError("INVALID_CANDIDATE_FUNDING_REFERENCE", "融资候选必须保留有效融资尾巴引用。")

    def as_cash_movement(self) -> PlannedCashMovement:
        return PlannedCashMovement(
            movement_id=self.candidate_id,
            booking_datetime=self.booking_datetime,
            sequence_no=self.sequence_no,
            direction=(MovementDirection.INFLOW if self.direction is CashDirection.INFLOW else MovementDirection.OUTFLOW),
            amount_fen=self.amount_fen,
        )


@dataclass(frozen=True)
class CandidateAmountReconciliationRow:
    source_item_id: str
    month: str
    category: CashCategory
    direction: CashDirection
    routed_observed_amount_fen: int
    candidate_amount_fen: int

    @property
    def difference_fen(self) -> int:
        return self.candidate_amount_fen - self.routed_observed_amount_fen

    @property
    def is_reconciled(self) -> bool:
        return self.difference_fen == 0


@dataclass(frozen=True)
class TransactionPlanningResult:
    request: TransactionPlanningRequest
    candidates: tuple[TransactionCandidate, ...]
    reconciliation_rows: tuple[CandidateAmountReconciliationRow, ...]
    cash_preview: CashExecutionPreviewResult

    @property
    def is_amount_reconciled(self) -> bool:
        return all(row.is_reconciled for row in self.reconciliation_rows)


_FUNDING_CATEGORIES = frozenset({
    CashCategory.BANK_LOAN_DRAWDOWN,
    CashCategory.BANK_LOAN_REPAYMENT,
    CashCategory.BANK_INTEREST_PAYMENT,
    CashCategory.SHAREHOLDER_FUNDING,
})
_TAX_CATEGORIES = frozenset({
    CashCategory.CORPORATE_INCOME_TAX_PAYMENT,
    CashCategory.SIMULATED_VAT_PAYMENT,
    CashCategory.SURCHARGE_PAYMENT,
})


def build_transaction_candidates(request: TransactionPlanningRequest) -> TransactionPlanningResult:
    """Split all non-zero observed-account monthly sources without changing totals.

    The returned plan intentionally includes candidates after an insufficiency;
    ``cash_preview.completed_movements`` marks the payable prefix.  I4B must
    only post that prefix and preserve the fact sheet when the preview stops.
    """

    routed = tuple(item for item in request.tax_routing_result.routed_cash_items if item.observed_amount_fen)
    tax_events = _tax_dated_events(request.tax_routing_result)
    supplied = _group_dated_events((*tax_events, *request.dated_source_events))
    # 新订单按原月度现金来源分组，合计不会再加一次。
    order_groups = _group_dated_events(request.business_order_events)
    # 完整合同模式必须独占对应月度来源。
    contract_groups = _group_dated_events(request.contract_source_events)
    # 两种合同模式不得同时消费同一来源。
    if set(contract_groups) & set(order_groups):
        # 防止重复记账。
        raise TransactionPlanningError("CONTRACT_SOURCE_CONFLICT", "合同节点与补充订单来源重叠")
    candidates: list[TransactionCandidate] = []
    rows: list[CandidateAmountReconciliationRow] = []

    for route in sorted(routed, key=lambda item: item.source_item.item_id):
        source = route.source_item
        fixed_events = supplied.get(source.item_id, ())
        if source.category in _TAX_CATEGORIES:
            if not fixed_events:
                raise TransactionPlanningError("MISSING_TAX_DATE", "本期应从I4A-1取得税费扣款日期，不能随机安排。")
            basis = "frozen_tax_date"
        elif source.category in _FUNDING_CATEGORIES:
            if not fixed_events:
                raise TransactionPlanningError(
                    "MISSING_FUNDING_DATE",
                    "资金合同事项缺少I3B/I3C提供的明确日期，不能由逐笔模块猜测。",
                )
            basis = "frozen_funding_date"
        elif fixed_events:
            raise TransactionPlanningError(
                "UNEXPECTED_FIXED_DATE_SOURCE",
                "普通经营或投资事项不能混入外部定日清单；请通过类别拆分规则安排日期。",
            )
        else:
            basis = "policy_split"

        # 新订单可以只占月度来源一部分，其余仍按原拆分规则安排。
        if source.item_id in contract_groups:
            # 新模式只接管明确普通业务来源，不改变税费和融资。
            if fixed_events or source.category not in {CashCategory.SALES_COLLECTION, CashCategory.SUPPLIER_PAYMENT, CashCategory.PAYROLL, CashCategory.RENT_AND_UTILITIES, CashCategory.OTHER_DECLARED_EVENT, CashCategory.FIXED_ASSET_PAYMENT}:
                # 非经营定日必须仍由原合同或税费模块提供。
                raise TransactionPlanningError("CONTRACT_SOURCE_CONFLICT", "合同节点不可接管税费或融资")
            # 要求完整金额，不能用随机余款掩盖漏项。
            if sum(e.source_amount_fen for e in contract_groups[source.item_id]) != source.amount_fen:
                # 所有合同都需与中央来源一分不差。
                raise TransactionPlanningError("DATED_SOURCE_TOTAL_MISMATCH", "合同节点分期与月度来源不一致")
            # 复用原定日候选、金额分配和现金不足停止逻辑。
            item_candidates = _mixed_order_candidates(request, route, contract_groups[source.item_id])
        # 补充订单继续按原允许的类别处理。
        elif source.item_id in order_groups:
            # 同一来源不能同时是经营订单和融资税费。
            if fixed_events or source.category not in {CashCategory.SALES_COLLECTION, CashCategory.SUPPLIER_PAYMENT}:
                # 业务类型冲突必须在记账前阻断。
                raise TransactionPlanningError("ORDER_SOURCE_CONFLICT", "新增订单只能对应销售回款或供应商付款。")
            # 冻结订单与普通经营余款在这里共同对平。
            item_candidates = _mixed_order_candidates(request, route, order_groups[source.item_id])
        # 非订单的冻结来源继续使用原融资与税费处理。
        elif fixed_events:
            if any(
                f"{item.booking_datetime.year:04d}-{item.booking_datetime.month:02d}" != source.month
                for item in fixed_events
            ):
                raise TransactionPlanningError(
                    "DATED_EVENT_MONTH_MISMATCH",
                    "定日事项必须落在其已对平月度来源所属月份，不能跨月挪动现金。",
                )
            source_events_total = sum(item.source_amount_fen for item in fixed_events)
            if source_events_total != source.amount_fen:
                raise TransactionPlanningError(
                    "DATED_SOURCE_TOTAL_MISMATCH",
                    "定日事项合计必须等于对应企业月度来源金额，不能改变合同或税费金额。",
                )
            amounts = _allocate_exact(route.observed_amount_fen, [item.source_amount_fen for item in fixed_events])
            item_candidates = [
                TransactionCandidate(
                    candidate_id=f"candidate:{source.item_id}:{event.event_id}",
                    source_item_id=source.item_id,
                    category=source.category,
                    direction=source.direction,
                    booking_datetime=event.booking_datetime,
                    sequence_no=0,
                    amount_fen=amount,
                    date_basis=basis,
                    source_frozen_order=event.source_frozen_order,
                    funding_tail_item_id=event.funding_tail_item_id,
                )
                for event, amount in zip(fixed_events, amounts)
                if amount
            ]
        else:
            policy = (request.category_policies or {}).get(source.category, request.default_policy)
            if policy.transaction_count > route.observed_amount_fen:
                raise TransactionPlanningError(
                    "TRANSACTION_COUNT_EXCEEDS_AMOUNT",
                    "拆分笔数超过可观察金额的分数，无法生成每笔至少一分的真实候选。",
                )
            rng = Random(_derived_seed(request.random_seed, source.item_id))
            dates = _policy_dates(source.month, policy.month_concentration, policy.transaction_count, rng)
            amounts = _split_ordinary_amount(route.observed_amount_fen, policy.transaction_count, policy.amount_variation_bp, rng)
            item_candidates = [
                TransactionCandidate(
                    candidate_id=f"candidate:{source.item_id}:{position:03d}",
                    source_item_id=source.item_id,
                    category=source.category,
                    direction=source.direction,
                    booking_datetime=value,
                    sequence_no=0,
                    amount_fen=amount,
                    date_basis=basis,
                )
                for position, (value, amount) in enumerate(zip(dates, amounts), start=1)
            ]
        candidates.extend(item_candidates)
        rows.append(CandidateAmountReconciliationRow(
            source_item_id=source.item_id,
            month=source.month,
            category=source.category,
            direction=source.direction,
            routed_observed_amount_fen=route.observed_amount_fen,
            candidate_amount_fen=sum(item.amount_fen for item in item_candidates),
        ))

    all_source_ids = {item.source_item.item_id for item in request.tax_routing_result.routed_cash_items}
    # 未消费的订单来源也视为错误，防止清单被静默遗漏。
    unexpected_sources = (set(supplied) | set(order_groups) | set(contract_groups)) - all_source_ids
    if unexpected_sources:
        raise TransactionPlanningError("UNUSED_DATED_SOURCE", "定日清单包含本期未路由到观察账户的来源事项。")
    # 先统一同日收入优先级；随后才按余额决定是否使用已声明宽限。
    ordered = tuple(
        replace(item, sequence_no=sequence)
        for sequence, item in enumerate(
            sorted(candidates, key=_candidate_order_key)
        )
    )
    reconciliation = tuple(rows)
    if not all(row.is_reconciled for row in reconciliation):
        raise TransactionPlanningError("CANDIDATE_AMOUNT_NOT_RECONCILED", "逐笔候选合计与观察账户月度路由金额不一致。")
    # 只对显式启用且资金确实不足的供应商合同做一次有界重排。
    deferred = _apply_receipt_dependent_supplier_payments(
        request.tax_routing_result.request.observed_account.opening_balance_fen, ordered,
    )
    ordered = tuple(replace(item, sequence_no=sequence) for sequence, item in enumerate(sorted(deferred, key=_candidate_order_key)))
    preview = preview_cash_sufficiency(
        request.tax_routing_result.request.observed_account.opening_balance_fen,
        tuple(item.as_cash_movement() for item in ordered),
    )
    return TransactionPlanningResult(request, ordered, reconciliation, preview)


# 部分冻结的经营来源仍复用原拆分金额和日期函数。
def _mixed_order_candidates(request, route, events):
    # 取原经营月度来源作为总金额所有者。
    source = route.source_item
    # 合同日期必须落在原合计所属月份。
    if any(e.booking_datetime.strftime("%Y-%m") != source.month for e in events):
        # 不准为对平而挪动现金月份。
        raise TransactionPlanningError("DATED_EVENT_MONTH_MISMATCH", "订单日期与现金来源月份不一致。")
    # 普通业务余款由合计减订单得出，订单不能重复加进总额。
    remainder = source.amount_fen - sum(e.source_amount_fen for e in events)
    # 订单不得超过中央经营计算的月度来源。
    if remainder < 0:
        # 超额意味着业务驱动或来源连接错误。
        raise TransactionPlanningError("DATED_SOURCE_TOTAL_MISMATCH", "订单合计超过月度经营现金。")
    # 完整主账户直接引用冻结金额，不用拆分函数再分配一分钱。
    weights = [remainder, *[e.source_amount_fen for e in events]]
    # 全额路由保持每笔原金额，部分路由复用既有规则并跳过零余款。
    positive = iter(_allocate_exact(route.observed_amount_fen, [v for v in weights if v]) if route.observed_amount_fen != source.amount_fen else [v for v in weights if v])
    # 第一项即使没有普通业务，也明确保留零以对齐订单索引。
    allocations = [next(positive) if v else 0 for v in weights]
    # 订单各自保留日期和来源，不再随机拆分。
    result = [TransactionCandidate(f"candidate:{source.item_id}:{event.event_id}", source.item_id, source.category, source.direction, event.booking_datetime, 0, amount, "frozen_order_date", business_order_id=event.business_order_id, historical_note_cn=event.historical_note_cn, original_due_booking_datetime=event.original_due_booking_datetime, negotiated_grace_days=event.negotiated_grace_days, dependent_receipt_event_id=event.dependent_receipt_event_id, payment_deferral_reason_cn=event.payment_deferral_reason_cn, receipt_dependency_enabled=event.receipt_dependency_enabled) for event, amount in zip(events, allocations[1:], strict=True) if amount]
    # 只有非零普通业务余款才按原规则生成。
    if allocations[0]:
        # 读取原月度类别笔数、集中度和幅度。
        policy = (request.category_policies or {}).get(source.category, request.default_policy)
        # 保留原先每笔至少一分的保护。
        if policy.transaction_count > allocations[0]:
            # 不通过压低笔数偷偷修复非法参数。
            raise TransactionPlanningError("TRANSACTION_COUNT_EXCEEDS_AMOUNT", "普通业务余款不足以按原笔数拆分。")
        # 原业务随机编号与没有新订单时完全相同。
        rng = Random(_derived_seed(request.random_seed, source.item_id))
        # 日期仍使用原集中安排。
        dates = _policy_dates(source.month, policy.month_concentration, policy.transaction_count, rng)
        # 金额仍使用原精确拆分函数。
        amounts = _split_ordinary_amount(allocations[0], policy.transaction_count, policy.amount_variation_bp, rng)
        # 原业务编号不因新增订单而重新编号。
        result.extend(TransactionCandidate(f"candidate:{source.item_id}:{index:03d}", source.item_id, source.category, source.direction, value, 0, amount, "policy_split") for index, (value, amount) in enumerate(zip(dates, amounts, strict=True), start=1))
    # 所有候选回到原排序、现金不足保护和账本。
    return result


def _tax_dated_events(result: TaxRoutingResult) -> tuple[DatedSourceEvent, ...]:
    return tuple(
        DatedSourceEvent(
            event_id=f"tax-date:{candidate.candidate_id}",
            source_item_id=f"cash:{candidate.candidate_id}",
            booking_datetime=datetime.combine(candidate.scheduled_debit_date, time(9, 0)),
            source_amount_fen=candidate.amount_fen,
        )
        for candidate in result.due_tax_payment_candidates
    )


def build_funding_dated_source_events(
    tax_routing_result: TaxRoutingResult,
    timeline_state: FundingTimelineState,
) -> tuple[DatedSourceEvent, ...]:
    """Bridge I3B's immutable contract tail into I4A-2's dated-source input.

    This is an adapter, not a second funding engine: it only names the same
    month/category buckets already used by I3C, retains each due date and
    frozen order, and ignores events outside the current routed horizon.
    """

    known_sources = {item.source_item.item_id for item in tax_routing_result.routed_cash_items}
    source_by_contract = {record.source_key: record.contract for record in timeline_state.contract_book}
    events: list[DatedSourceEvent] = []
    for tail in timeline_state.pending_tail:
        if tail.status is TailStatus.SUPERSEDED_BEFORE_FIRST_EVENT:
            continue
        if tail.status is not TailStatus.PENDING:
            raise TransactionPlanningError("INVALID_FUNDING_TAIL", "被阻断的资金尾巴必须由用户处理，不能进入逐笔候选。")
        event = tail.event
        source = source_by_contract.get((event.loan_contract_id, event.contract_version))
        if source is None:
            raise TransactionPlanningError("MISSING_FUNDING_CONTRACT", "资金尾巴没有对应的合同记录。")
        source_item_id = _funding_source_item_id(event, source)
        if source_item_id not in known_sources:
            continue
        events.append(DatedSourceEvent(
            event_id=f"funding-date:{event.tail_item_id}",
            source_item_id=source_item_id,
            booking_datetime=datetime.combine(event.due_date, time(9, 0)),
            source_amount_fen=event.amount_fen,
            source_frozen_order=tail.contract_frozen_order * 1_000_000 + event.frozen_sequence,
            funding_tail_item_id=event.tail_item_id,
        ))
    return tuple(sorted(events, key=lambda item: (item.booking_datetime, item.source_frozen_order, item.event_id)))


def _funding_source_item_id(event, source: LoanContract | ShareholderCapitalPlan) -> str:
    month = f"{event.due_date.year:04d}-{event.due_date.month:02d}"
    if event.event_type is FundingEventType.SHAREHOLDER_CAPITAL_INJECTION:
        if not isinstance(source, ShareholderCapitalPlan):
            raise TransactionPlanningError("FUNDING_SOURCE_MISMATCH", "股东注资尾巴没有对应的股东投入计划。")
        suffix = "shareholder_capital"
    else:
        if not isinstance(source, LoanContract):
            raise TransactionPlanningError("FUNDING_SOURCE_MISMATCH", "贷款尾巴没有对应的贷款合同。")
        if event.event_type is FundingEventType.INTEREST_PAYMENT:
            suffix = "interest"
        elif event.event_type is FundingEventType.LOAN_DRAWDOWN:
            suffix = "bank_draw" if source.funding_kind is FundingKind.BANK_LOAN else "shareholder_loan_draw"
        elif event.event_type is FundingEventType.PRINCIPAL_REPAYMENT:
            suffix = "bank_repay" if source.funding_kind is FundingKind.BANK_LOAN else "shareholder_repay"
        else:
            raise TransactionPlanningError("UNKNOWN_FUNDING_EVENT", "出现未冻结的资金事项类型，不能猜测逐笔来源。")
    return f"i3c:{month}:{suffix}"


def _group_dated_events(events: Sequence[DatedSourceEvent]) -> Mapping[str, tuple[DatedSourceEvent, ...]]:
    result: dict[str, list[DatedSourceEvent]] = {}
    for item in events:
        result.setdefault(item.source_item_id, []).append(item)
    return {
        source_id: tuple(sorted(values, key=lambda item: (item.booking_datetime, item.event_id)))
        for source_id, values in result.items()
    }


def _candidate_order_key(value: TransactionCandidate) -> tuple[datetime, int, int, str]:
    """Keep declared funding before ordinary cash only when timestamps tie."""
    funding_priority = 0 if value.category in _FUNDING_CATEGORIES else 1
    return value.booking_datetime, funding_priority, value.source_frozen_order, value.candidate_id


def _apply_receipt_dependent_supplier_payments(opening_balance_fen: int, candidates: Sequence[TransactionCandidate]) -> tuple[TransactionCandidate, ...]:
    """Only defer an unaffordable supplier item to its declared receipt within grace."""
    by_event = {item.business_order_id: item for item in candidates if item.direction is CashDirection.INFLOW and item.business_order_id}
    # 时间队列会把延期付款在实际时刻重新入队，不能只改日期而漏扣余额。
    queue = []
    for index, item in enumerate(candidates):
        heappush(queue, (_candidate_order_key(item), index, item))
    balance = opening_balance_fen
    changed: list[TransactionCandidate] = []
    deferred_ids: set[str] = set()
    next_index = len(candidates)
    while queue:
        # 取当前最早的实际现金事项。
        _, _, item = heappop(queue)
        # 回款在其真实到账时刻才增加可用余额。
        if item.direction is CashDirection.INFLOW:
            balance += item.amount_fen
            changed.append(item)
            continue
        # 余额足够即按当前（原到期或已延期）时刻付款并实际扣款。
        if item.amount_fen <= balance:
            balance -= item.amount_fen
            changed.append(item)
            continue
        # 仅第一次资金不足时，供应商可按已协商条款等指定且尚未来到的回款。
        if item.receipt_dependency_enabled and item.category is CashCategory.SUPPLIER_PAYMENT and item.negotiated_grace_days and item.dependent_receipt_event_id and item.amount_fen > balance:
            receipt = by_event.get(item.dependent_receipt_event_id)
            original = item.original_due_booking_datetime or item.booking_datetime
            if receipt and item.candidate_id not in deferred_ids and original.date() <= receipt.booking_datetime.date() <= original.date().fromordinal(original.date().toordinal() + item.negotiated_grace_days) and receipt.booking_datetime >= item.booking_datetime:
                # 回款后一秒重新尝试同一付款，金额和原到期均不改变。
                actual = receipt.booking_datetime + timedelta(seconds=1)
                deferred = replace(item, booking_datetime=actual, date_basis="receipt_dependent_negotiated_payment", receipt_dependency_applied=True)
                deferred_ids.add(item.candidate_id)
                heappush(queue, (_candidate_order_key(deferred), next_index, deferred))
                next_index += 1
                continue
        # 不可延期或宽限失败时保留原后缀，由统一停止器如实停止。
        changed.append(item)
        changed.extend(queued_item for _, _, queued_item in sorted(queue))
        break
    return tuple(sorted(changed, key=_candidate_order_key))


def _policy_dates(month: str, concentration: MonthConcentration, count: int, rng: Random) -> tuple[datetime, ...]:
    try:
        year, month_number = (int(month[:4]), int(month[5:]))
        if len(month) != 7 or month[4] != "-" or not 1 <= month_number <= 12:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise TransactionPlanningError("INVALID_SOURCE_MONTH", "来源事项月份必须使用YYYY-MM。") from error
    days = list(range(1, monthrange(year, month_number)[1] + 1))
    weights = _day_weights(days, concentration)
    return tuple(
        datetime.combine(date(year, month_number, rng.choices(days, weights=weights, k=1)[0]), _random_time(rng))
        for _ in range(count)
    )


def _day_weights(days: Sequence[int], concentration: MonthConcentration) -> tuple[int, ...]:
    def band(day: int) -> str:
        return "early" if day <= 10 else "middle" if day <= 20 else "late"

    preferred = {
        MonthConcentration.UNIFORM: None,
        MonthConcentration.EARLY: "early",
        MonthConcentration.MIDDLE: "middle",
        MonthConcentration.LATE: "late",
    }[concentration]
    return tuple(1 if preferred is None or band(day) != preferred else 4 for day in days)


def _random_time(rng: Random) -> time:
    return time(9 + rng.randrange(9), rng.randrange(60), rng.randrange(60))


def _split_ordinary_amount(total_fen: int, count: int, variation_bp: int, rng: Random) -> tuple[int, ...]:
    if variation_bp == 0:
        weights = [1] * count
    else:
        weights = [10_000 + rng.randint(-variation_bp, variation_bp) for _ in range(count)]
    return _allocate_exact(total_fen, weights)


def _allocate_exact(total_fen: int, weights: Sequence[int]) -> tuple[int, ...]:
    """Allocate positive integer cents without changing the group total."""
    if total_fen < len(weights) or not weights or any(value <= 0 for value in weights):
        raise TransactionPlanningError("INVALID_AMOUNT_ALLOCATION", "金额不能按所设笔数和权重拆成正数分。")
    remaining = total_fen - len(weights)
    total_weight = sum(weights)
    extra = [remaining * value // total_weight for value in weights]
    remainder = remaining - sum(extra)
    ranked = sorted(range(len(weights)), key=lambda index: (-weights[index], index))
    for index in ranked[:remainder]:
        extra[index] += 1
    return tuple(1 + value for value in extra)


def _derived_seed(root_seed: int, source_item_id: str) -> int:
    digest = sha256(f"{root_seed}:{source_item_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
