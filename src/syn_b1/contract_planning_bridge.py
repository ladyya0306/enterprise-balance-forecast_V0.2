"""SYN-B1-I3C: reconnect declared contracts to the monthly planning chain.

This module is deliberately a bridge.  It reads an already reconciled I1/I1B
pre-financing plan and an I3B contract book, then asks I1 for one final named
cash plan and I1B for post-contract statements.  It never chooses financing,
adds capital, changes a date, executes a daily event, writes a ledger, or
creates CSV/SQLite/model inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from enum import Enum
import hashlib
import json

from syn_b1.cash_plan import (
    BalanceRange,
    Direction,
    FinalNamedCashPlanResult,
    NamedMonthlyCashComponent,
    PlanningMode,
    build_final_named_cash_plan,
)
from syn_b1.contract_statement_rollforward import (
    ContractFinancedStatementResult,
    ContractFinancingMonthlyAdjustment,
    build_contract_financed_statement_rollforward,
)
from syn_b1.funding_contracts import CashDirection, FundingEventType, FundingKind, LoanContract
from syn_b1.funding_timeline import (
    FundingTimelineState,
    HistoryStatus,
    ShareholderCapitalPlan,
    TailStatus,
)
from syn_b1.statement_rollforward import StatementRollforwardResult


class ContractPlanningBridgeError(ValueError):
    """A declared contract cannot safely be connected to a monthly plan."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ContractBridgeDiagnostic:
    severity: str
    code: str
    message_cn: str


@dataclass(frozen=True)
class ContractPlanningBridgeRequest:
    """Inputs already produced by I1/I1B plus an immutable I3B state."""

    pre_financing_statements: StatementRollforwardResult
    timeline_state: FundingTimelineState
    planning_mode: PlanningMode
    minimum_liquidity_fen: int
    ending_balance_target_range: BalanceRange | None = None
    fixed_total_inflow_fen: int | None = None
    fixed_total_outflow_fen: int | None = None
    allow_monthly_cash_shortage_for_daily_stop: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.planning_mode, PlanningMode):
            raise ContractPlanningBridgeError("INVALID_PLANNING_MODE", "三种规划方式必须使用冻结的方式枚举。")
        if not isinstance(self.minimum_liquidity_fen, int) or isinstance(self.minimum_liquidity_fen, bool) or self.minimum_liquidity_fen < 0:
            raise ContractPlanningBridgeError("INVALID_MINIMUM_LIQUIDITY", "最低可用余额必须为非负整数分。")
        if self.planning_mode is PlanningMode.FIXED_BANK_TOTALS:
            if self.fixed_total_inflow_fen is None or self.fixed_total_outflow_fen is None:
                raise ContractPlanningBridgeError("MISSING_FIXED_TOTAL", "固定总额方式必须填写期间总流入和总流出。")
            if (
                not isinstance(self.fixed_total_inflow_fen, int)
                or isinstance(self.fixed_total_inflow_fen, bool)
                or self.fixed_total_inflow_fen < 0
                or not isinstance(self.fixed_total_outflow_fen, int)
                or isinstance(self.fixed_total_outflow_fen, bool)
                or self.fixed_total_outflow_fen < 0
            ):
                raise ContractPlanningBridgeError("INVALID_FIXED_TOTAL", "固定期间总流入和总流出必须为非负整数分。")
            if self.ending_balance_target_range is not None:
                raise ContractPlanningBridgeError("UNUSED_TARGET_RANGE", "固定总额方式不能同时填写独立期末余额目标。")
        else:
            if self.ending_balance_target_range is None:
                raise ContractPlanningBridgeError("MISSING_TARGET_RANGE", "目标余额或受约束自动方式必须填写期末余额目标。")
            if self.fixed_total_inflow_fen is not None or self.fixed_total_outflow_fen is not None:
                raise ContractPlanningBridgeError("UNUSED_FIXED_TOTAL", "非固定总额方式不能填写固定总收付。")
        if not isinstance(self.allow_monthly_cash_shortage_for_daily_stop, bool):
            raise ContractPlanningBridgeError("INVALID_DAILY_STOP_POLICY", "月度资金不足处理方式无效。")


@dataclass(frozen=True)
class ContractPlanningBridgeFailure:
    planning_mode: PlanningMode
    first_conflict_month_or_date: str
    conflict_code: str
    required_amount_fen: int
    available_or_reserved_amount_fen: int
    affected_named_components: tuple[str, ...]
    allowed_user_adjustments_cn: tuple[str, ...]


@dataclass(frozen=True)
class ContractPlanningBridgeResult:
    """Monthly projection only; it is not proof that daily payments can clear."""

    request: ContractPlanningBridgeRequest
    is_feasible: bool
    final_named_plan: FinalNamedCashPlanResult | None
    post_contract_statements: ContractFinancedStatementResult | None
    restricted_contract_component_ids: tuple[str, ...]
    diagnostics: tuple[ContractBridgeDiagnostic, ...]
    enterprise_profile_id: str
    plan_snapshot_fingerprint: str
    failure: ContractPlanningBridgeFailure | None = None


def build_contract_planning_bridge(request: ContractPlanningBridgeRequest) -> ContractPlanningBridgeResult:
    """Make one fully named final plan from contracts the user has already set.

    I3B remains the only component that can execute a dated payment.  Since
    the pre-financing operating plan is monthly, this function does not infer
    an intra-month receipt/payment order and cannot silently certify it.
    """

    statements = request.pre_financing_statements
    if not statements.is_fully_reconciled:
        raise ContractPlanningBridgeError("PRE_FINANCING_STATEMENTS_NOT_RECONCILED", "融资前三张表未对平，不能接入资金合同。")
    pre_plan = statements.request.cash_plan
    start_date = pre_plan.request.start_date
    end_date = pre_plan.request.end_date
    enterprise_profile_id, snapshot_fingerprint = _plan_identities(request)
    _validate_contract_state_for_projection(request.timeline_state, start_date)
    base_components = _named_pre_financing_components(statements)
    contract_components, adjustments = _scheduled_contract_components(
        request.timeline_state,
        start_date,
        end_date,
        tuple(row.month for row in pre_plan.monthly_rows),
    )
    final_plan = build_final_named_cash_plan(
        start_date=start_date,
        end_date=end_date,
        opening_balance_fen=pre_plan.request.opening_balance_fen,
        components=(*base_components, *contract_components),
    )
    failure = _validate_final_plan(request, final_plan, contract_components)
    if failure is not None:
        return ContractPlanningBridgeResult(
            request=request,
            is_feasible=False,
            final_named_plan=None,
            post_contract_statements=None,
            restricted_contract_component_ids=tuple(item.component_id for item in contract_components),
            diagnostics=(),
            enterprise_profile_id=enterprise_profile_id,
            plan_snapshot_fingerprint=snapshot_fingerprint,
            failure=failure,
        )
    post_statements = build_contract_financed_statement_rollforward(
        pre_financing_statements=statements,
        final_named_plan=final_plan,
        adjustments=adjustments,
    )
    diagnostics = _soft_diagnostics(request, final_plan)
    return ContractPlanningBridgeResult(
        request=request,
        is_feasible=True,
        final_named_plan=final_plan,
        post_contract_statements=post_statements,
        restricted_contract_component_ids=tuple(item.component_id for item in contract_components),
        diagnostics=diagnostics,
        enterprise_profile_id=enterprise_profile_id,
        plan_snapshot_fingerprint=snapshot_fingerprint,
    )


def _validate_contract_state_for_projection(state: FundingTimelineState, start_date: date) -> None:
    if any(item.status is TailStatus.BLOCKED for item in state.pending_tail) or any(item.status is HistoryStatus.BLOCKED for item in state.event_history):
        raise ContractPlanningBridgeError("BLOCKED_TAIL_REQUIRES_RESOLUTION", "存在被阻断的资金事项，必须由用户处理，不能用于新的月度投影。")
    if state.processed_through_date is not None and state.processed_through_date != start_date - timedelta(days=1):
        raise ContractPlanningBridgeError("NON_CONTIGUOUS_PROJECTION", "月度投影必须紧接已处理日期，不能跳过或重算中间日期。")
    overdue = [item for item in state.pending_tail if item.status is TailStatus.PENDING and item.event.due_date < start_date]
    if overdue:
        raise ContractPlanningBridgeError("OVERDUE_PENDING_EVENT", "存在早于本次开始日的未执行资金事项，不能静默延期。")


def _named_pre_financing_components(statements: StatementRollforwardResult) -> tuple[NamedMonthlyCashComponent, ...]:
    components: list[NamedMonthlyCashComponent] = []
    for row in statements.request.cash_plan.monthly_rows:
        if row.unallocated_inflow_fen or row.unallocated_outflow_fen:
            raise ContractPlanningBridgeError("UNALLOCATED_CASH_BUDGET", f"{row.month}仍有未命名预算，不能形成最终计划。")
        for suffix, cash_class, direction, amount in (
            ("operating_inflow", "operating", Direction.INFLOW, row.operating_inflow_fen),
            ("operating_outflow", "operating", Direction.OUTFLOW, row.operating_outflow_fen),
            ("investing_inflow", "investing", Direction.INFLOW, row.investing_inflow_fen),
            ("investing_outflow", "investing", Direction.OUTFLOW, row.investing_outflow_fen),
        ):
            if amount:
                components.append(NamedMonthlyCashComponent(
                    component_id=f"pre_financing:{row.month}:{suffix}", month=row.month,
                    cash_flow_class=cash_class, direction=direction, amount_fen=amount,
                    source_stage="I1A", truth_visibility="public_amount_only",
                ))
    return tuple(components)


def _scheduled_contract_components(
    state: FundingTimelineState,
    start_date: date,
    end_date: date,
    months: tuple[str, ...],
) -> tuple[tuple[NamedMonthlyCashComponent, ...], tuple[ContractFinancingMonthlyAdjustment, ...]]:
    record_by_key = {record.source_key: record.contract for record in state.contract_book}
    grouped = {
        month: {
            "bank_draw": 0,
            "shareholder_draw": 0,
            "capital": 0,
            "bank_repay": 0,
            "shareholder_repay": 0,
            "interest": 0,
        }
        for month in months
    }
    components: list[NamedMonthlyCashComponent] = []
    for item in state.pending_tail:
        if item.status is TailStatus.SUPERSEDED_BEFORE_FIRST_EVENT or not start_date <= item.event.due_date <= end_date:
            continue
        if item.status is not TailStatus.PENDING:
            raise ContractPlanningBridgeError("INVALID_TAIL_FOR_PROJECTION", "资金事项状态不能用于新的月度投影。")
        event = item.event
        source = record_by_key[(event.loan_contract_id, event.contract_version)]
        month = f"{event.due_date.year:04d}-{event.due_date.month:02d}"
        key = _adjustment_key(source, event.event_type)
        grouped[month][key] += event.amount_fen
        components.append(NamedMonthlyCashComponent(
            component_id=f"contract:{event.tail_item_id}", month=month, cash_flow_class="financing",
            direction=Direction.INFLOW if event.cash_direction is CashDirection.INFLOW else Direction.OUTFLOW,
            amount_fen=event.amount_fen, source_stage="owner_declared_other",
            truth_visibility="restricted_reason_and_schedule",
        ))
    adjustments = tuple(
        ContractFinancingMonthlyAdjustment(
            month=month,
            bank_loan_drawdown_fen=values["bank_draw"],
            shareholder_loan_drawdown_fen=values["shareholder_draw"],
            shareholder_capital_injection_fen=values["capital"],
            bank_principal_repayment_fen=values["bank_repay"],
            shareholder_principal_repayment_fen=values["shareholder_repay"],
            interest_cash_payment_fen=values["interest"],
        )
        for month, values in grouped.items()
    )
    return tuple(components), adjustments


def _adjustment_key(source: LoanContract | ShareholderCapitalPlan, event_type: FundingEventType) -> str:
    if event_type is FundingEventType.SHAREHOLDER_CAPITAL_INJECTION:
        if not isinstance(source, ShareholderCapitalPlan):
            raise ContractPlanningBridgeError("CAPITAL_SOURCE_MISMATCH", "股东投入事项没有对应的用户声明计划。")
        return "capital"
    if not isinstance(source, LoanContract):
        raise ContractPlanningBridgeError("LOAN_SOURCE_MISMATCH", "贷款事项没有对应的用户贷款合同。")
    if event_type is FundingEventType.INTEREST_PAYMENT:
        return "interest"
    if event_type is FundingEventType.LOAN_DRAWDOWN:
        return "bank_draw" if source.funding_kind is FundingKind.BANK_LOAN else "shareholder_draw"
    if event_type is FundingEventType.PRINCIPAL_REPAYMENT:
        return "bank_repay" if source.funding_kind is FundingKind.BANK_LOAN else "shareholder_repay"
    raise ContractPlanningBridgeError("UNKNOWN_FUNDING_EVENT_TYPE", "出现未冻结的资金事项类型，不能猜测其账务含义。")


def _validate_final_plan(
    request: ContractPlanningBridgeRequest,
    final_plan: FinalNamedCashPlanResult,
    contract_components: tuple[NamedMonthlyCashComponent, ...],
) -> ContractPlanningBridgeFailure | None:
    if request.planning_mode is PlanningMode.FIXED_BANK_TOTALS:
        assert request.fixed_total_inflow_fen is not None and request.fixed_total_outflow_fen is not None
        inflow_failure = _fixed_total_failure(
            request,
            final_plan.monthly_rows[-1].month,
            "inflow",
            final_plan.total_inflow_fen,
            request.fixed_total_inflow_fen,
            contract_components,
        )
        if inflow_failure is not None:
            return inflow_failure
        outflow_failure = _fixed_total_failure(
            request,
            final_plan.monthly_rows[-1].month,
            "outflow",
            final_plan.total_outflow_fen,
            request.fixed_total_outflow_fen,
            contract_components,
        )
        if outflow_failure is not None:
            return outflow_failure
    negative = next((row for row in final_plan.monthly_rows if row.ending_balance_fen < 0), None)
    if request.allow_monthly_cash_shortage_for_daily_stop:
        # Formal I7 must allow I4B to find the exact first unpayable outflow.
        # This never approves a negative ledger balance or invents financing;
        # it changes only the bridge from a premature monthly rejection into a
        # diagnostic for the already frozen daily-stop controller.
        return None
    if negative is not None:
        return _failure(request, negative.month, "MONTH_END_CASH_NEGATIVE", -negative.ending_balance_fen, 0, contract_components)
    return None


def _soft_diagnostics(request: ContractPlanningBridgeRequest, final_plan: FinalNamedCashPlanResult) -> tuple[ContractBridgeDiagnostic, ...]:
    result: list[ContractBridgeDiagnostic] = [ContractBridgeDiagnostic("warning", "MONTHLY_PROJECTION_NOT_DAILY_EXECUTION", "本结果只按月展示已声明合同影响；逐日到账与付款顺序仍须由I3B按实际日期执行。")]
    if any(row.ending_balance_fen < request.minimum_liquidity_fen for row in final_plan.monthly_rows):
        result.append(ContractBridgeDiagnostic("warning", "MONTH_END_BELOW_MINIMUM_LIQUIDITY", "合同后月末余额低于最低可用余额；系统不会自行新增融资。"))
    if request.allow_monthly_cash_shortage_for_daily_stop and any(row.ending_balance_fen < 0 for row in final_plan.monthly_rows):
        result.append(ContractBridgeDiagnostic("warning", "MONTHLY_SHORTAGE_DEFERRED_TO_DAILY_STOP", "月度计划显示资金缺口；逐笔账本将在首笔无法支付的流出前停止并出具事实单。"))
    if request.ending_balance_target_range is not None:
        ending = final_plan.ending_balance_fen
        target = request.ending_balance_target_range
        if ending < target.minimum_fen:
            result.append(ContractBridgeDiagnostic("warning", "ENDING_BALANCE_BELOW_TARGET", "已声明资金合同后期末余额仍低于目标下限；请由用户修改经营或资金合同。"))
        elif ending > target.maximum_fen:
            result.append(ContractBridgeDiagnostic("warning", "ENDING_BALANCE_ABOVE_TARGET", "已声明资金合同后期末余额高于目标上限；系统不会虚构支出压低余额。"))
    return tuple(result)


def _failure(request: ContractPlanningBridgeRequest, month: str, code: str, required: int, available: int, components: tuple[NamedMonthlyCashComponent, ...]) -> ContractPlanningBridgeFailure:
    adjustments = ["修改用户设定的经营收付或资金合同后重新生成计划。"]
    if request.planning_mode is PlanningMode.FIXED_BANK_TOTALS:
        adjustments.append("修改固定期间总收付，使它与全部有名称项目一致。")
    return ContractPlanningBridgeFailure(
        planning_mode=request.planning_mode, first_conflict_month_or_date=month, conflict_code=code,
        required_amount_fen=required, available_or_reserved_amount_fen=available,
        affected_named_components=tuple(item.component_id for item in components),
        allowed_user_adjustments_cn=tuple(adjustments),
    )


def _fixed_total_failure(
    request: ContractPlanningBridgeRequest,
    month: str,
    direction: str,
    actual_fen: int,
    fixed_fen: int,
    components: tuple[NamedMonthlyCashComponent, ...],
) -> ContractPlanningBridgeFailure | None:
    if actual_fen == fixed_fen:
        return None
    if actual_fen > fixed_fen:
        code = f"FIXED_FINANCING_{direction.upper()}_RESERVE_EXCEEDED"
        return _failure(request, month, code, actual_fen - fixed_fen, fixed_fen, components)
    return _failure(
        request,
        month,
        "FIXED_UNEXPLAINED_REMAINDER",
        fixed_fen - actual_fen,
        actual_fen,
        components,
    )


def _plan_identities(request: ContractPlanningBridgeRequest) -> tuple[str, str]:
    account_id = request.pre_financing_statements.request.operating_cycle.request.synthetic_account_id
    profile_payload = {
        "identity_version": "syn_b1_enterprise_profile_v1",
        "synthetic_account_id": account_id,
    }
    snapshot_payload = {
        "snapshot_version": "syn_b1_i3c_plan_snapshot_v1",
        "request": asdict(request),
    }
    return _sha256_json(profile_payload), _sha256_json(snapshot_payload)


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_value,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _json_value(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    raise TypeError(f"不能为{type(value).__name__}生成计划快照。")
