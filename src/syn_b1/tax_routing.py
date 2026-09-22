"""SYN-B1-I4A-1: simplified tax schedule and one observed-account routing.

This module is intentionally a planning adapter.  It reads the existing
monthly operating plan and the I3C post-financing income statement, then
creates restricted tax schedules and splits already named enterprise cash
items between one observed account and peripheral cash.  It never creates
ordinary transactions, calculates transaction balances, writes files, or
reimplements profit, income-tax, financing, or calendar rules.

All monetary values are fictional integer fen.  No function in this module
reads real data or returns model-input rows.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
# 仅切换日期检查方式来识别期外待确认日，不改税额。
from dataclasses import replace
from datetime import date, timedelta
from enum import Enum
from typing import Callable, Mapping, Sequence

from syn_b1.cash_plan import Direction, FinalNamedCashPlanResult, NamedMonthlyCashComponent, build_final_named_cash_plan
from syn_b1.contract_statement_rollforward import ContractFinancedStatementResult
from syn_b1.operating_cycle import OperatingCycleResult
from syn_b1.tax_projection import build_tax_replaced_contract_statements


class TaxRoutingError(ValueError):
    """A frozen F5 tax or routing rule cannot form a safe plan."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AccountRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class CashCategory(str, Enum):
    SALES_COLLECTION = "sales_collection"
    SUPPLIER_PAYMENT = "supplier_payment"
    PAYROLL = "payroll"
    RENT_AND_UTILITIES = "rent_and_utilities"
    BANK_LOAN_DRAWDOWN = "bank_loan_drawdown"
    BANK_LOAN_REPAYMENT = "bank_loan_repayment"
    BANK_INTEREST_PAYMENT = "bank_interest_payment"
    SHAREHOLDER_FUNDING = "shareholder_funding"
    FIXED_ASSET_PAYMENT = "fixed_asset_payment"
    CORPORATE_INCOME_TAX_PAYMENT = "corporate_income_tax_payment"
    SIMULATED_VAT_PAYMENT = "simulated_vat_payment"
    SURCHARGE_PAYMENT = "surcharge_payment"
    OTHER_DECLARED_EVENT = "other_declared_event"


class CashDirection(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"


class TaxType(str, Enum):
    CORPORATE_INCOME_TAX = "corporate_income_tax"
    SIMULATED_VAT = "simulated_vat"
    SURCHARGE = "surcharge"


class TaxFrequency(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class TaxScheduleMode(str, Enum):
    RECURRING_DAY = "recurring_day"
    EXPLICIT_PERIOD_DATES = "explicit_period_dates"


class TaxQuarterOpeningContext(str, Enum):
    NEW_ENTERPRISE_NO_PRIOR_TAX = "new_enterprise_no_prior_tax"
    CONTINUING_WITH_CARRY = "continuing_with_carry"


class NonBankDayPolicy(str, Enum):
    NEXT_BANK_WORKDAY = "next_bank_workday"
    KEEP_CALENDAR_DATE = "keep_calendar_date"


class RouteDestination(str, Enum):
    OBSERVED_ACCOUNT = "observed_account"
    PERIPHERAL_CASH = "peripheral_cash"


@dataclass(frozen=True)
class ObservedAccount:
    """The one fictional RMB account represented by this generation run."""

    synthetic_account_id: str
    account_role: AccountRole
    opening_balance_fen: int

    def __post_init__(self) -> None:
        if not self.synthetic_account_id:
            raise TaxRoutingError("EMPTY_SYNTHETIC_ACCOUNT_ID", "虚构观察账户编号不能为空。")
        _require_nonnegative_fen("opening_balance_fen", self.opening_balance_fen)


@dataclass(frozen=True)
class RoutingRule:
    """User-confirmed share of one enterprise cash category in the account."""

    enabled: bool
    routed_share_bp: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TaxRoutingError("INVALID_ROUTE_ENABLED", "账户路由开关必须为布尔值。")
        _require_rate("routed_share_bp", self.routed_share_bp)
        if not self.enabled and self.routed_share_bp != 0:
            raise TaxRoutingError("DISABLED_ROUTE_HAS_SHARE", "未启用的账户路由比例必须为0。")


@dataclass(frozen=True)
class EnterpriseCashItem:
    """One already-named enterprise monthly cash item, not a transaction."""

    item_id: str
    month: str
    category: CashCategory
    direction: CashDirection
    amount_fen: int

    def __post_init__(self) -> None:
        if not self.item_id:
            raise TaxRoutingError("EMPTY_CASH_ITEM_ID", "企业现金事项编号不能为空。")
        _validate_month_key(self.month)
        _require_nonnegative_fen("amount_fen", self.amount_fen)


@dataclass(frozen=True)
class RoutedCashItem:
    """A conservation-preserving split of an enterprise item."""

    source_item: EnterpriseCashItem
    observed_amount_fen: int
    peripheral_amount_fen: int

    def __post_init__(self) -> None:
        _require_nonnegative_fen("observed_amount_fen", self.observed_amount_fen)
        _require_nonnegative_fen("peripheral_amount_fen", self.peripheral_amount_fen)
        if self.observed_amount_fen + self.peripheral_amount_fen != self.source_item.amount_fen:
            raise TaxRoutingError("ROUTE_NOT_CONSERVED", "观察账户金额与外围金额必须刚好等于企业事项总额。")


@dataclass(frozen=True)
class TaxPaymentRule:
    """One user-confirmed rule for a simplified tax type."""

    enabled: bool
    frequency: TaxFrequency
    schedule_mode: TaxScheduleMode
    payer_account_role: AccountRole
    non_bank_day_policy: NonBankDayPolicy = NonBankDayPolicy.NEXT_BANK_WORKDAY
    recurring_debit_day: int | None = None
    explicit_period_dates: Mapping[str, date] | None = None
    effective_vat_burden_rate_bp: int | None = None
    surcharge_rate_on_simulated_vat_bp: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TaxRoutingError("INVALID_TAX_ENABLED", "税种启用状态必须为布尔值。")
        if not isinstance(self.frequency, TaxFrequency):
            raise TaxRoutingError("INVALID_TAX_FREQUENCY", "税费周期必须为月度或季度。")
        if not isinstance(self.schedule_mode, TaxScheduleMode):
            raise TaxRoutingError("INVALID_TAX_SCHEDULE_MODE", "税费日期方式无效。")
        if not isinstance(self.payer_account_role, AccountRole):
            raise TaxRoutingError("INVALID_TAX_PAYER_ROLE", "税费扣款账户必须为主账户或次要账户。")
        if not isinstance(self.non_bank_day_policy, NonBankDayPolicy):
            raise TaxRoutingError("INVALID_NON_BANK_DAY_POLICY", "非银行工作日处理方式无效。")
        if self.schedule_mode is TaxScheduleMode.RECURRING_DAY:
            if self.explicit_period_dates is not None:
                raise TaxRoutingError("MIXED_TAX_DATE_INPUT", "固定扣款日不能同时填写逐期明确日期。")
            if not isinstance(self.recurring_debit_day, int) or isinstance(self.recurring_debit_day, bool) or not 1 <= self.recurring_debit_day <= 31:
                raise TaxRoutingError("INVALID_RECURRING_DEBIT_DAY", "固定扣款日必须为1至31。")
        else:
            if self.recurring_debit_day is not None:
                raise TaxRoutingError("MIXED_TAX_DATE_INPUT", "逐期明确日期不能同时填写固定扣款日。")
            if self.explicit_period_dates is None:
                raise TaxRoutingError("MISSING_EXPLICIT_TAX_DATES", "逐期明确日期方式必须提供日期表。")
            for period, debit_date in self.explicit_period_dates.items():
                _validate_assessment_period(period)
                if not isinstance(debit_date, date):
                    raise TaxRoutingError("INVALID_EXPLICIT_TAX_DATE", "逐期税费扣款日期必须为date。")
        for name, rate in (
            ("effective_vat_burden_rate_bp", self.effective_vat_burden_rate_bp),
            ("surcharge_rate_on_simulated_vat_bp", self.surcharge_rate_on_simulated_vat_bp),
        ):
            if rate is not None:
                _require_rate(name, rate)


@dataclass(frozen=True)
class TaxPaymentCandidate:
    """A tax payment planned for a date; I4B later decides actual execution."""

    candidate_id: str
    tax_type: TaxType
    assessment_period: str
    amount_fen: int
    scheduled_debit_date: date
    payer_account_role: AccountRole
    route_destination: RouteDestination
    # 期外日历尚未公布时仅保留合同日期，不冒充已核定工作日。
    calendar_confirmed: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise TaxRoutingError("EMPTY_TAX_CANDIDATE_ID", "税费候选事项编号不能为空。")
        _validate_assessment_period(self.assessment_period)
        _require_positive_fen("tax_payment_amount_fen", self.amount_fen)
        if not isinstance(self.scheduled_debit_date, date):
            raise TaxRoutingError("INVALID_TAX_DEBIT_DATE", "税费扣款日期必须为date。")


@dataclass(frozen=True)
class TaxPaymentHistoryEntry:
    """An append-only result supplied by the later final ledger."""

    candidate_id: str
    actual_debit_date: date
    amount_fen: int

    def __post_init__(self) -> None:
        if not self.candidate_id or not isinstance(self.actual_debit_date, date):
            raise TaxRoutingError("INVALID_TAX_PAYMENT_HISTORY", "税费支付历史必须包含候选编号和实际日期。")
        _require_positive_fen("tax_payment_history_amount_fen", self.amount_fen)


@dataclass(frozen=True)
class TaxCarryForwardState:
    """Restricted tax memory; it is never a model input or public export."""

    processed_through_month: str | None = None
    payable_by_tax_type: Mapping[TaxType, int] | None = None
    unassessed_quarterly_fen: Mapping[tuple[TaxType, str], int] | None = None
    pending_payment_candidates: tuple[TaxPaymentCandidate, ...] = ()
    payment_history: tuple[TaxPaymentHistoryEntry, ...] = ()

    def __post_init__(self) -> None:
        if self.processed_through_month is not None:
            _validate_month_key(self.processed_through_month)
        for tax_type, amount in (self.payable_by_tax_type or {}).items():
            if not isinstance(tax_type, TaxType):
                raise TaxRoutingError("INVALID_TAX_TYPE", "税费余额必须按冻结税种保存。")
            _require_nonnegative_fen("opening_tax_payable_fen", amount)
        for (tax_type, period), amount in (self.unassessed_quarterly_fen or {}).items():
            if not isinstance(tax_type, TaxType) or not _is_quarter_period(period):
                raise TaxRoutingError("INVALID_UNASSESSED_QUARTER", "季度待计税金额必须使用税种和YYYY-QN。")
            _require_nonnegative_fen("unassessed_quarterly_fen", amount)
        candidate_ids = [item.candidate_id for item in self.pending_payment_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise TaxRoutingError("DUPLICATE_TAX_TAIL", "待处理税费尾巴编号不能重复。")
        history_ids = [item.candidate_id for item in self.payment_history]
        if len(history_ids) != len(set(history_ids)):
            raise TaxRoutingError("DUPLICATE_TAX_HISTORY", "税费支付历史编号不能重复。")
        if set(candidate_ids) & set(history_ids):
            raise TaxRoutingError("TAX_TAIL_ALREADY_PAID", "已支付税费不能同时保留在待处理尾巴中。")
        pending_by_type = {tax_type: 0 for tax_type in TaxType}
        for candidate in self.pending_payment_candidates:
            pending_by_type[candidate.tax_type] += candidate.amount_fen
        payable = self.payable_by_tax_type or {}
        for tax_type in TaxType:
            if payable.get(tax_type, 0) != pending_by_type[tax_type]:
                raise TaxRoutingError(
                    "TAX_PAYABLE_WITHOUT_MATCHING_TAIL",
                    f"{tax_type.value}应付余额必须由同额待扣款尾巴完整解释。",
                )


@dataclass(frozen=True)
class TaxLedgerRow:
    """One tax type and one month/quarter assessment period."""

    assessment_period: str
    tax_type: TaxType
    opening_payable_fen: int
    current_accrual_fen: int
    scheduled_payment_fen: int
    actual_payment_fen: int
    ending_payable_fen: int
    scheduled_debit_date: date | None
    payer_account_role: AccountRole | None
    route_destination: RouteDestination | None
    status: str

    def __post_init__(self) -> None:
        _validate_assessment_period(self.assessment_period)
        for name in (
            "opening_payable_fen", "current_accrual_fen", "scheduled_payment_fen",
            "actual_payment_fen", "ending_payable_fen",
        ):
            _require_nonnegative_fen(name, getattr(self, name))
        if self.ending_payable_fen != self.opening_payable_fen + self.current_accrual_fen - self.actual_payment_fen:
            raise TaxRoutingError("TAX_PAYABLE_NOT_RECONCILED", "税费台账期末未缴与期初、本期计算、实际支付不一致。")


@dataclass(frozen=True)
class TaxRoutingDiagnostic:
    severity: str
    code: str
    message_cn: str


@dataclass(frozen=True)
class TaxRoutingRequest:
    """All inputs for F5's frozen tax and routing plan.

    Enterprise cash items are deliberately absent from this public request:
    they are derived in full from I1A and I3C so a caller cannot supply a
    convenient but incomplete subset.
    """

    operating_cycle: OperatingCycleResult
    post_contract_statements: ContractFinancedStatementResult
    observed_account: ObservedAccount
    routing_rules: Mapping[CashCategory, RoutingRule]
    tax_rules: Mapping[TaxType, TaxPaymentRule]
    is_bank_workday: Callable[[date], bool]
    carry_forward_state: TaxCarryForwardState = TaxCarryForwardState()
    quarter_opening_context: TaxQuarterOpeningContext | None = None
    version: str = "syn_b1_i4a1_rc1_tax_routing_1_1"

    def __post_init__(self) -> None:
        if not callable(self.is_bank_workday):
            raise TaxRoutingError("MISSING_BANK_CALENDAR_INTERFACE", "必须提供版本化银行工作日判断接口。")
        if not self.version:
            raise TaxRoutingError("EMPTY_TAX_ROUTING_VERSION", "税费路由版本不能为空。")
        for category, rule in self.routing_rules.items():
            if not isinstance(category, CashCategory) or not isinstance(rule, RoutingRule):
                raise TaxRoutingError("INVALID_ROUTING_RULE", "每类现金路由必须使用冻结类别和规则。")
        if set(self.tax_rules) != set(TaxType):
            raise TaxRoutingError("INCOMPLETE_TAX_RULES", "所得税、模拟增值税和附加税必须分别提供规则。")
        if any(not isinstance(rule, TaxPaymentRule) for rule in self.tax_rules.values()):
            raise TaxRoutingError("INVALID_TAX_RULE", "税费规则类型无效。")
        vat_rule = self.tax_rules[TaxType.SIMULATED_VAT]
        surcharge_rule = self.tax_rules[TaxType.SURCHARGE]
        income_tax_rule = self.tax_rules[TaxType.CORPORATE_INCOME_TAX]
        if vat_rule.effective_vat_burden_rate_bp is None:
            raise TaxRoutingError("MISSING_VAT_RATE", "模拟增值税规则必须提供综合实际税负率。")
        if surcharge_rule.surcharge_rate_on_simulated_vat_bp is None:
            raise TaxRoutingError("MISSING_SURCHARGE_RATE", "附加税规则必须提供附加税比例。")
        if income_tax_rule.effective_vat_burden_rate_bp is not None or income_tax_rule.surcharge_rate_on_simulated_vat_bp is not None:
            raise TaxRoutingError("INVALID_INCOME_TAX_RATE_SOURCE", "所得税规则不得填写增值税或附加税比例。")
        if not income_tax_rule.enabled:
            raise TaxRoutingError(
                "INCOME_TAX_MUST_BE_ENABLED_FOR_STATEMENT_CLOSURE",
                "MVP三张表保留所得税公式，因此所得税必须启用；如不经过当前账户，请改设扣款账户角色。",
            )


@dataclass(frozen=True)
class CashRouteReconciliationRow:
    month: str
    direction: CashDirection
    expected_plan_amount_fen: int
    derived_source_amount_fen: int
    observed_amount_fen: int
    peripheral_amount_fen: int

    @property
    def difference_fen(self) -> int:
        return self.derived_source_amount_fen - self.expected_plan_amount_fen

    @property
    def is_reconciled(self) -> bool:
        return (
            self.difference_fen == 0
            and self.observed_amount_fen + self.peripheral_amount_fen == self.derived_source_amount_fen
        )


@dataclass(frozen=True)
class TaxRoutingResult:
    """Restricted planning output for I4A-2 and I4B, not a CSV or ledger."""

    request: TaxRoutingRequest
    routed_cash_items: tuple[RoutedCashItem, ...]
    route_reconciliation_rows: tuple[CashRouteReconciliationRow, ...]
    tax_ledger_rows: tuple[TaxLedgerRow, ...]
    due_tax_payment_candidates: tuple[TaxPaymentCandidate, ...]
    future_tax_payment_tail: tuple[TaxPaymentCandidate, ...]
    next_carry_forward_state: TaxCarryForwardState
    tax_adjusted_final_named_plan: FinalNamedCashPlanResult
    tax_adjusted_statements: ContractFinancedStatementResult
    diagnostics: tuple[TaxRoutingDiagnostic, ...]

    @property
    def is_route_conserved(self) -> bool:
        return bool(self.route_reconciliation_rows) and all(
            item.observed_amount_fen + item.peripheral_amount_fen == item.source_item.amount_fen
            for item in self.routed_cash_items
        ) and all(row.is_reconciled for row in self.route_reconciliation_rows)


def build_tax_and_routing_plan(request: TaxRoutingRequest) -> TaxRoutingResult:
    """Build a deterministic tax plan and split explicit enterprise cash items.

    The result stops at planned candidates.  Actual payment, transaction
    ordering, balance validation, files, and exports remain outside I4A-1.
    """

    _validate_sources_align(request)
    _require_legacy_combined_tax_disabled(request.operating_cycle)
    months = tuple(row.month for row in request.operating_cycle.monthly_rows)
    _validate_continuation(request.carry_forward_state, months[0])
    _validate_quarter_opening_context(request, months[0])
    base_items = _build_complete_non_tax_cash_items(request)
    _assert_items_match_plan(base_items, request.post_contract_statements.final_named_plan)

    tax_accruals = _monthly_vat_and_surcharge_accruals(request, months)
    zero_payments = {month: 0 for month in months}
    opening_tax_payable = _carried_tax_total(request.carry_forward_state)
    provisional_statements = build_tax_replaced_contract_statements(
        base=request.post_contract_statements,
        final_named_plan=request.post_contract_statements.final_named_plan,
        tax_and_surcharge_accruals_fen=tax_accruals,
        tax_cash_payments_fen=zero_payments,
        income_tax_cash_payments_fen=zero_payments,
        opening_tax_payable_fen=opening_tax_payable,
    )
    income_tax_by_month = {
        row.month: row.income_tax_expense_fen for row in provisional_statements.income_statement_rows
    }
    tax_rows, newly_scheduled, payable_by_type, unassessed = _build_tax_schedule(
        request, months, income_tax_by_month
    )
    start_date, end_date = request.operating_cycle.request.start_date, request.operating_cycle.request.end_date
    pending = (*request.carry_forward_state.pending_payment_candidates, *newly_scheduled)
    _assert_pending_not_paid(pending, request.carry_forward_state.payment_history)
    due, tail = _split_by_horizon(pending, start_date, end_date)
    tax_items = _tax_cash_items(due)
    tax_components = _tax_named_components(due)
    adjusted_plan = build_final_named_cash_plan(
        start_date=start_date,
        end_date=end_date,
        opening_balance_fen=request.post_contract_statements.final_named_plan.monthly_rows[0].opening_balance_fen,
        components=(*request.post_contract_statements.final_named_plan.components, *tax_components),
    )
    payment_by_month = {month: 0 for month in months}
    income_tax_payment_by_month = {month: 0 for month in months}
    for candidate in due:
        month = f"{candidate.scheduled_debit_date.year:04d}-{candidate.scheduled_debit_date.month:02d}"
        if candidate.tax_type is TaxType.CORPORATE_INCOME_TAX:
            income_tax_payment_by_month[month] += candidate.amount_fen
        else:
            payment_by_month[month] += candidate.amount_fen
    adjusted_statements = build_tax_replaced_contract_statements(
        base=request.post_contract_statements,
        final_named_plan=adjusted_plan,
        tax_and_surcharge_accruals_fen=tax_accruals,
        tax_cash_payments_fen=payment_by_month,
        income_tax_cash_payments_fen=income_tax_payment_by_month,
        opening_tax_payable_fen=opening_tax_payable,
    )
    if not adjusted_statements.is_fully_reconciled:
        raise TaxRoutingError("TAX_ADJUSTED_STATEMENTS_NOT_RECONCILED", "替换税费后的月度现金计划和三张表未对平。")
    projected_ending_tax = (
        sum(payable_by_type.values()) + sum(unassessed.values())
        - sum(candidate.amount_fen for candidate in due)
    )
    if adjusted_statements.balance_sheet_rows[-1].tax_payable_fen != projected_ending_tax:
        raise TaxRoutingError(
            "TAX_LEDGER_TO_STATEMENT_NOT_RECONCILED",
            "税费本子扣除本期计划支付后的期末未缴与三张表不一致。",
        )
    routed_items = (
        *_route_enterprise_cash_items(base_items, request.routing_rules),
        *_route_tax_cash_items(tax_items, due),
    )
    route_reconciliation = _route_reconciliation_rows(adjusted_plan, routed_items)
    if not route_reconciliation or not all(row.is_reconciled for row in route_reconciliation):
        raise TaxRoutingError("ENTERPRISE_CASH_ROUTE_NOT_COMPLETE", "企业全部现金事项与最终月度计划没有完整对上。")
    next_state = TaxCarryForwardState(
        processed_through_month=months[-1],
        payable_by_tax_type=payable_by_type,
        unassessed_quarterly_fen=unassessed,
        pending_payment_candidates=tuple(sorted(pending, key=lambda item: (item.scheduled_debit_date, item.candidate_id))),
        payment_history=request.carry_forward_state.payment_history,
    )
    diagnostics = [
        TaxRoutingDiagnostic(
            severity="info",
            code="ACTUAL_TAX_PAYMENT_RESERVED_FOR_I4B",
            message_cn="本阶段只形成税费候选事项；实际扣款、余额检查和支付历史追加由后续唯一账本完成。",
        ),
        TaxRoutingDiagnostic(
            severity="info",
            code="LEGACY_COMBINED_TAX_REPLACED_ONCE",
            message_cn="旧合并税费已被明确禁用；模拟增值税和附加税只通过本结果进入最终月度计划及三张表一次。",
        ),
    ]
    if tail:
        diagnostics.append(
            TaxRoutingDiagnostic(
                severity="info",
                code="FUTURE_TAX_TAIL_PRESERVED",
                message_cn="生成期外税费扣款已保留为跨期尾巴，不会丢失。",
            )
        )
    if any(row.ending_balance_fen < 0 for row in adjusted_plan.monthly_rows):
        diagnostics.append(
            TaxRoutingDiagnostic(
                severity="warning",
                code="MONTH_END_CASH_BELOW_ZERO_REQUIRES_TRANSACTION_STOP",
                message_cn="月度计划已显示资金不足；后续逐笔账本必须在首笔无法支付的流出记账前停止并出具事实单，不能自行补资。",
            )
        )
    return TaxRoutingResult(
        request=request,
        routed_cash_items=routed_items,
        route_reconciliation_rows=route_reconciliation,
        tax_ledger_rows=tax_rows,
        due_tax_payment_candidates=due,
        future_tax_payment_tail=tail,
        next_carry_forward_state=next_state,
        tax_adjusted_final_named_plan=adjusted_plan,
        tax_adjusted_statements=adjusted_statements,
        diagnostics=tuple(diagnostics),
    )


def build_non_tax_operating_cash_items(operating_cycle: OperatingCycleResult) -> tuple[EnterpriseCashItem, ...]:
    """Expose explicit non-tax I1A cash categories without inferring financing.

    Old combined-tax cash is intentionally excluded.  F5 requires I4A-1's
    separated simulated VAT and surcharge to replace it for new work.
    """

    items: list[EnterpriseCashItem] = []
    for row in operating_cycle.monthly_rows:
        entries = (
            ("sales_collection", CashCategory.SALES_COLLECTION, CashDirection.INFLOW, row.cash_collections_fen),
            ("supplier_payment", CashCategory.SUPPLIER_PAYMENT, CashDirection.OUTFLOW, row.supplier_cash_payments_fen),
            ("payroll", CashCategory.PAYROLL, CashDirection.OUTFLOW, row.payroll_cash_payment_fen),
            ("rent_utilities", CashCategory.RENT_AND_UTILITIES, CashDirection.OUTFLOW, row.rent_and_utilities_cash_payment_fen),
            ("other", CashCategory.OTHER_DECLARED_EVENT, CashDirection.OUTFLOW, row.other_operating_cash_payment_fen),
            ("fixed_asset", CashCategory.FIXED_ASSET_PAYMENT, CashDirection.OUTFLOW, row.fixed_asset_cash_payment_fen),
        )
        for suffix, category, direction, amount in entries:
            if amount:
                items.append(EnterpriseCashItem(f"i1a:{row.month}:{suffix}", row.month, category, direction, amount))
    return tuple(items)


def _build_complete_non_tax_cash_items(request: TaxRoutingRequest) -> tuple[EnterpriseCashItem, ...]:
    """Derive the complete non-tax enterprise cash source; callers cannot trim it."""
    items = list(build_non_tax_operating_cash_items(request.operating_cycle))
    for adjustment in request.post_contract_statements.adjustments:
        entries = (
            ("bank_draw", CashCategory.BANK_LOAN_DRAWDOWN, CashDirection.INFLOW, adjustment.bank_loan_drawdown_fen),
            ("shareholder_loan_draw", CashCategory.SHAREHOLDER_FUNDING, CashDirection.INFLOW, adjustment.shareholder_loan_drawdown_fen),
            ("shareholder_capital", CashCategory.SHAREHOLDER_FUNDING, CashDirection.INFLOW, adjustment.shareholder_capital_injection_fen),
            ("bank_repay", CashCategory.BANK_LOAN_REPAYMENT, CashDirection.OUTFLOW, adjustment.bank_principal_repayment_fen),
            ("shareholder_repay", CashCategory.BANK_LOAN_REPAYMENT, CashDirection.OUTFLOW, adjustment.shareholder_principal_repayment_fen),
            ("interest", CashCategory.BANK_INTEREST_PAYMENT, CashDirection.OUTFLOW, adjustment.interest_cash_payment_fen),
        )
        for suffix, category, direction, amount in entries:
            if amount:
                items.append(EnterpriseCashItem(
                    f"i3c:{adjustment.month}:{suffix}", adjustment.month, category, direction, amount
                ))
    return tuple(items)


def _tax_cash_items(candidates: Sequence[TaxPaymentCandidate]) -> tuple[EnterpriseCashItem, ...]:
    category_by_type = {
        TaxType.CORPORATE_INCOME_TAX: CashCategory.CORPORATE_INCOME_TAX_PAYMENT,
        TaxType.SIMULATED_VAT: CashCategory.SIMULATED_VAT_PAYMENT,
        TaxType.SURCHARGE: CashCategory.SURCHARGE_PAYMENT,
    }
    return tuple(
        EnterpriseCashItem(
            f"cash:{candidate.candidate_id}",
            f"{candidate.scheduled_debit_date.year:04d}-{candidate.scheduled_debit_date.month:02d}",
            category_by_type[candidate.tax_type],
            CashDirection.OUTFLOW,
            candidate.amount_fen,
        )
        for candidate in candidates
    )


def _tax_named_components(candidates: Sequence[TaxPaymentCandidate]) -> tuple[NamedMonthlyCashComponent, ...]:
    return tuple(
        NamedMonthlyCashComponent(
            component_id=f"planned_{candidate.candidate_id}",
            month=f"{candidate.scheduled_debit_date.year:04d}-{candidate.scheduled_debit_date.month:02d}",
            cash_flow_class="operating",
            direction=Direction.OUTFLOW,
            amount_fen=candidate.amount_fen,
            source_stage="owner_declared_other",
            truth_visibility="restricted_reason_and_schedule",
        )
        for candidate in candidates
    )


def _validate_sources_align(request: TaxRoutingRequest) -> None:
    operating = request.operating_cycle
    statement_rows = request.post_contract_statements.income_statement_rows
    operating_months = tuple(row.month for row in operating.monthly_rows)
    statement_months = tuple(row.month for row in statement_rows)
    if not operating_months or operating_months != statement_months:
        raise TaxRoutingError("TAX_SOURCE_MONTH_MISMATCH", "经营计划与融资后三表必须使用相同且非空的月份。")
    if request.observed_account.synthetic_account_id != operating.request.synthetic_account_id:
        raise TaxRoutingError("OBSERVED_ACCOUNT_ID_MISMATCH", "观察账户编号必须与虚构企业计划编号一致。")


def _carried_tax_total(state: TaxCarryForwardState) -> int:
    return (
        sum((state.payable_by_tax_type or {}).values())
        + sum((state.unassessed_quarterly_fen or {}).values())
    )


def _require_legacy_combined_tax_disabled(operating: OperatingCycleResult) -> None:
    if operating.request.tax_and_surcharge_rate_bp != 0:
        raise TaxRoutingError(
            "LEGACY_COMBINED_TAX_NOT_DISABLED",
            "新税费路径要求旧合并税费率设为0，不能同时计提两套税费。",
        )
    if operating.opening_state.tax_payable_fen != 0 or any(
        row.tax_and_surcharge_accrual_fen
        or row.tax_cash_payment_fen
        or row.opening_tax_payable_fen
        or row.ending_tax_payable_fen
        for row in operating.monthly_rows
    ):
        raise TaxRoutingError(
            "LEGACY_COMBINED_TAX_BALANCE_PRESENT",
            "旧合并税费仍有计提、支付或应付余额，必须迁入新税费尾巴后再接通。",
        )


def _validate_quarter_opening_context(request: TaxRoutingRequest, first_month: str) -> None:
    quarterly_types = {
        tax_type for tax_type, rule in request.tax_rules.items()
        if rule.enabled and rule.frequency is TaxFrequency.QUARTERLY
    }
    if not quarterly_types or _parse_month_key(first_month)[1] in {1, 4, 7, 10}:
        return
    if request.carry_forward_state.processed_through_month is not None:
        return
    context = request.quarter_opening_context
    if context is None:
        raise TaxRoutingError(
            "MISSING_QUARTER_OPENING_CONTEXT",
            "季度方式从季度中途开始时，必须说明是新设企业还是携带前期税费继续生成。",
        )
    if context is TaxQuarterOpeningContext.NEW_ENTERPRISE_NO_PRIOR_TAX:
        if (
            request.carry_forward_state.payable_by_tax_type
            or request.carry_forward_state.unassessed_quarterly_fen
            or request.carry_forward_state.pending_payment_candidates
        ):
            raise TaxRoutingError("NEW_ENTERPRISE_HAS_PRIOR_TAX", "声明无前期税费时不能同时提供期初税费状态。")
        return
    quarter = _quarter_period(first_month)
    unassessed = request.carry_forward_state.unassessed_quarterly_fen or {}
    if any((tax_type, quarter) not in unassessed for tax_type in quarterly_types):
        raise TaxRoutingError(
            "MISSING_PRIOR_QUARTER_CARRY",
            "连续经营且从季度中途开始时，必须为每个启用的季度税种提供本季度前期累计额，0也要明确填写。",
        )


def _validate_continuation(state: TaxCarryForwardState, first_month: str) -> None:
    if state.processed_through_month is None:
        return
    if _next_month_key(state.processed_through_month) != first_month:
        raise TaxRoutingError("NON_CONTIGUOUS_TAX_CONTINUATION", "税费续生成必须紧接上次已处理月份，不能跳过或重算中间月份。")


def _route_enterprise_cash_items(
    items: Sequence[EnterpriseCashItem], rules: Mapping[CashCategory, RoutingRule]
) -> tuple[RoutedCashItem, ...]:
    result: list[RoutedCashItem] = []
    for item in items:
        if item.category not in rules:
            raise TaxRoutingError("MISSING_CATEGORY_ROUTING", f"现金类别{item.category.value}缺少用户确认的路由规则。")
        rule = rules[item.category]
        observed = item.amount_fen * rule.routed_share_bp // 10_000 if rule.enabled else 0
        result.append(RoutedCashItem(item, observed, item.amount_fen - observed))
    return tuple(result)


def _route_tax_cash_items(
    items: Sequence[EnterpriseCashItem], candidates: Sequence[TaxPaymentCandidate]
) -> tuple[RoutedCashItem, ...]:
    if len(items) != len(candidates):
        raise TaxRoutingError("TAX_ROUTE_SOURCE_MISMATCH", "税费现金事项与扣款候选数量不一致。")
    result: list[RoutedCashItem] = []
    for item, candidate in zip(items, candidates):
        observed = item.amount_fen if candidate.route_destination is RouteDestination.OBSERVED_ACCOUNT else 0
        result.append(RoutedCashItem(item, observed, item.amount_fen - observed))
    return tuple(result)


def _assert_items_match_plan(
    items: Sequence[EnterpriseCashItem], plan: FinalNamedCashPlanResult
) -> None:
    expected: dict[tuple[str, CashDirection], int] = {}
    for row in plan.monthly_rows:
        expected[(row.month, CashDirection.INFLOW)] = row.total_inflow_fen
        expected[(row.month, CashDirection.OUTFLOW)] = row.total_outflow_fen
    actual: dict[tuple[str, CashDirection], int] = {}
    for item in items:
        key = (item.month, item.direction)
        actual[key] = actual.get(key, 0) + item.amount_fen
    if any(actual.get(key, 0) != amount for key, amount in expected.items()) or any(
        key not in expected and amount for key, amount in actual.items()
    ):
        raise TaxRoutingError(
            "DERIVED_CASH_ITEMS_DO_NOT_MATCH_FINAL_PLAN",
            "自动取得的经营、投资和资金合同现金事项与I3C最终月度计划不一致。",
        )


def _route_reconciliation_rows(
    plan: FinalNamedCashPlanResult, routed_items: Sequence[RoutedCashItem]
) -> tuple[CashRouteReconciliationRow, ...]:
    rows: list[CashRouteReconciliationRow] = []
    for plan_row in plan.monthly_rows:
        for direction, expected in (
            (CashDirection.INFLOW, plan_row.total_inflow_fen),
            (CashDirection.OUTFLOW, plan_row.total_outflow_fen),
        ):
            matching = [
                item for item in routed_items
                if item.source_item.month == plan_row.month and item.source_item.direction is direction
            ]
            rows.append(CashRouteReconciliationRow(
                month=plan_row.month,
                direction=direction,
                expected_plan_amount_fen=expected,
                derived_source_amount_fen=sum(item.source_item.amount_fen for item in matching),
                observed_amount_fen=sum(item.observed_amount_fen for item in matching),
                peripheral_amount_fen=sum(item.peripheral_amount_fen for item in matching),
            ))
    return tuple(rows)


def _build_tax_schedule(
    request: TaxRoutingRequest,
    months: Sequence[str],
    income_by_month: Mapping[str, int],
) -> tuple[
    tuple[TaxLedgerRow, ...],
    tuple[TaxPaymentCandidate, ...],
    Mapping[TaxType, int],
    Mapping[tuple[TaxType, str], int],
]:
    sales_by_month = {row.month: row.sales_confirmed_fen for row in request.operating_cycle.monthly_rows}
    opening = {tax_type: (request.carry_forward_state.payable_by_tax_type or {}).get(tax_type, 0) for tax_type in TaxType}
    current_payable = dict(opening)
    unassessed = dict(request.carry_forward_state.unassessed_quarterly_fen or {})
    rows: list[TaxLedgerRow] = []
    candidates: list[TaxPaymentCandidate] = []

    for month in months:
        monthly_accruals = _monthly_tax_accruals(request, month, income_by_month[month], sales_by_month[month])
        for tax_type, accrual in monthly_accruals.items():
            rule = request.tax_rules[tax_type]
            if not rule.enabled:
                continue
            if rule.frequency is TaxFrequency.MONTHLY:
                row, candidate = _assess_tax_period(
                    request, tax_type, month, accrual, current_payable[tax_type]
                )
                rows.append(row)
                current_payable[tax_type] = row.ending_payable_fen
                if candidate is not None:
                    candidates.append(candidate)
                continue
            quarter = _quarter_period(month)
            key = (tax_type, quarter)
            unassessed[key] = unassessed.get(key, 0) + accrual
            if _month_ends_quarter(month):
                quarterly_accrual = unassessed.pop(key)
                row, candidate = _assess_tax_period(
                    request, tax_type, quarter, quarterly_accrual, current_payable[tax_type]
                )
                rows.append(row)
                current_payable[tax_type] = row.ending_payable_fen
                if candidate is not None:
                    candidates.append(candidate)

    return (
        tuple(rows),
        tuple(candidates),
        current_payable,
        unassessed,
    )


def _monthly_vat_and_surcharge_accruals(
    request: TaxRoutingRequest, months: Sequence[str]
) -> Mapping[str, int]:
    sales_by_month = {row.month: row.sales_confirmed_fen for row in request.operating_cycle.monthly_rows}
    result: dict[str, int] = {}
    for month in months:
        vat = (
            sales_by_month[month] * _effective_vat_rate(request) // 10_000
            if request.tax_rules[TaxType.SIMULATED_VAT].enabled else 0
        )
        surcharge = (
            vat * _surcharge_rate(request) // 10_000
            if request.tax_rules[TaxType.SURCHARGE].enabled else 0
        )
        result[month] = vat + surcharge
    return result


def _monthly_tax_accruals(
    request: TaxRoutingRequest, month: str, income_tax_fen: int, sales_fen: int
) -> Mapping[TaxType, int]:
    _require_nonnegative_fen("post_financing_income_tax_fen", income_tax_fen)
    _require_nonnegative_fen("recognized_sales_fen", sales_fen)
    vat_rule = request.tax_rules[TaxType.SIMULATED_VAT]
    surcharge_rule = request.tax_rules[TaxType.SURCHARGE]
    vat = sales_fen * _effective_vat_rate(request) // 10_000 if vat_rule.enabled else 0
    surcharge = vat * _surcharge_rate(request) // 10_000 if surcharge_rule.enabled else 0
    return {
        TaxType.CORPORATE_INCOME_TAX: income_tax_fen if request.tax_rules[TaxType.CORPORATE_INCOME_TAX].enabled else 0,
        TaxType.SIMULATED_VAT: vat,
        TaxType.SURCHARGE: surcharge,
    }


def _effective_vat_rate(request: TaxRoutingRequest) -> int:
    rule = request.tax_rules[TaxType.SIMULATED_VAT]
    rate = rule.effective_vat_burden_rate_bp
    assert rate is not None
    _require_rate("effective_vat_burden_rate_bp", rate)
    return rate


def _surcharge_rate(request: TaxRoutingRequest) -> int:
    rule = request.tax_rules[TaxType.SURCHARGE]
    rate = rule.surcharge_rate_on_simulated_vat_bp
    assert rate is not None
    _require_rate("surcharge_rate_on_simulated_vat_bp", rate)
    return rate


def _assess_tax_period(
    request: TaxRoutingRequest,
    tax_type: TaxType,
    assessment_period: str,
    accrual_fen: int,
    opening_payable_fen: int,
) -> tuple[TaxLedgerRow, TaxPaymentCandidate | None]:
    rule = request.tax_rules[tax_type]
    if accrual_fen == 0:
        return (
            TaxLedgerRow(
                assessment_period=assessment_period,
                tax_type=tax_type,
                opening_payable_fen=opening_payable_fen,
                current_accrual_fen=0,
                scheduled_payment_fen=0,
                actual_payment_fen=0,
                ending_payable_fen=opening_payable_fen,
                scheduled_debit_date=None,
                payer_account_role=rule.payer_account_role,
                route_destination=None,
                status="zero_accrual_no_candidate",
            ),
            None,
        )
    # 不把未知年份的期外日历当作本期经营错误。
    from syn_b1.calendar_adapter import SynB1CalendarError
    # 正常范围仍调用原税费排期和顺延规则。
    calendar_confirmed = True
    # 只有日历不可用且日期在期外时才保留待确认尾项。
    try:
        # 原已公布日期的行为保持不变。
        debit_date = _scheduled_debit_date(rule, assessment_period, request.is_bank_workday)
    # 不吞掉金额、税费或其他技术异常。
    except SynB1CalendarError:
        # 原合同日仅作为期外最低到期日保存。
        debit_date = _scheduled_debit_date(replace(rule, non_bank_day_policy=NonBankDayPolicy.KEEP_CALENDAR_DATE), assessment_period, request.is_bank_workday)
        # 期内缺日历仍须拒绝，不能随意推测。
        if debit_date <= request.operating_cycle.request.end_date:
            # 保留原异常供查明。
            raise
        # 后续续期必须先核对日历，不能直接执行此尾项。
        calendar_confirmed = False
    route = (
        RouteDestination.OBSERVED_ACCOUNT
        if rule.payer_account_role is request.observed_account.account_role
        else RouteDestination.PERIPHERAL_CASH
    )
    candidate = TaxPaymentCandidate(
        candidate_id=f"tax:{tax_type.value}:{assessment_period}",
        tax_type=tax_type,
        assessment_period=assessment_period,
        amount_fen=accrual_fen,
        scheduled_debit_date=debit_date,
        payer_account_role=rule.payer_account_role,
        route_destination=route,
        # 明确区分已排定工作日与期外待确认合同日。
        calendar_confirmed=calendar_confirmed,
    )
    return (
        TaxLedgerRow(
            assessment_period=assessment_period,
            tax_type=tax_type,
            opening_payable_fen=opening_payable_fen,
            current_accrual_fen=accrual_fen,
            scheduled_payment_fen=accrual_fen,
            actual_payment_fen=0,
            ending_payable_fen=opening_payable_fen + accrual_fen,
            scheduled_debit_date=debit_date,
            payer_account_role=rule.payer_account_role,
            route_destination=route,
            # 待确认的期外日历不属于实际付款。
            status="scheduled_not_executed" if calendar_confirmed else "pending_calendar_confirmation",
        ),
        candidate,
    )


def _scheduled_debit_date(
    rule: TaxPaymentRule, assessment_period: str, is_bank_workday: Callable[[date], bool]
) -> date:
    period_end = _assessment_period_end(assessment_period)
    if rule.schedule_mode is TaxScheduleMode.EXPLICIT_PERIOD_DATES:
        assert rule.explicit_period_dates is not None
        if assessment_period not in rule.explicit_period_dates:
            raise TaxRoutingError("MISSING_EXPLICIT_PERIOD_DATE", f"{assessment_period}缺少明确税费扣款日期。")
        candidate = rule.explicit_period_dates[assessment_period]
        if candidate < period_end:
            raise TaxRoutingError("EARLY_EXPLICIT_TAX_DATE", "明确税费扣款日期不能早于对应计税期间结束日。")
    else:
        assert rule.recurring_debit_day is not None
        due_month = _first_day_of_next_month(period_end)
        candidate = date(
            due_month.year,
            due_month.month,
            min(rule.recurring_debit_day, monthrange(due_month.year, due_month.month)[1]),
        )
    if rule.non_bank_day_policy is NonBankDayPolicy.KEEP_CALENDAR_DATE:
        return candidate
    for _ in range(32):
        if is_bank_workday(candidate):
            return candidate
        candidate += timedelta(days=1)
    raise TaxRoutingError("BANK_CALENDAR_NO_WORKDAY", "银行日历接口连续32日未返回工作日，不能安排税费扣款。")


def _split_by_horizon(
    items: Sequence[TaxPaymentCandidate], start_date: date, end_date: date
) -> tuple[tuple[TaxPaymentCandidate, ...], tuple[TaxPaymentCandidate, ...]]:
    due: list[TaxPaymentCandidate] = []
    tail: list[TaxPaymentCandidate] = []
    for item in sorted(items, key=lambda value: (value.scheduled_debit_date, value.candidate_id)):
        if item.scheduled_debit_date < start_date:
            raise TaxRoutingError("OVERDUE_TAX_TAIL", "存在早于本次开始日期且尚未执行的税费尾巴。")
        if item.scheduled_debit_date <= end_date:
            # 续期不能把上期未核定日期直接当成可执行工作日。
            if not item.calendar_confirmed:
                # 明确需要先接入新公布日历并重新核定尾项日期。
                raise TaxRoutingError("TAX_CALENDAR_UNCONFIRMED", "期外税款日历尚未确认，不能纳入本期执行")
            due.append(item)
        else:
            tail.append(item)
    return tuple(due), tuple(tail)


def _assert_pending_not_paid(
    pending: Sequence[TaxPaymentCandidate], history: Sequence[TaxPaymentHistoryEntry]
) -> None:
    pending_ids = [item.candidate_id for item in pending]
    if len(pending_ids) != len(set(pending_ids)):
        raise TaxRoutingError("DUPLICATE_TAX_TAIL", "税费尾巴不能重复保存。")
    if set(pending_ids) & {item.candidate_id for item in history}:
        raise TaxRoutingError("TAX_TAIL_ALREADY_PAID", "已支付税费不能重新作为待处理尾巴。")


def _quarter_period(month: str) -> str:
    year, month_number = _parse_month_key(month)
    return f"{year:04d}-Q{(month_number - 1) // 3 + 1}"


def _month_ends_quarter(month: str) -> bool:
    return _parse_month_key(month)[1] in {3, 6, 9, 12}


def _assessment_period_end(period: str) -> date:
    if _is_quarter_period(period):
        year = int(period[:4])
        month_number = int(period[-1]) * 3
    else:
        year, month_number = _parse_month_key(period)
    return date(year, month_number, monthrange(year, month_number)[1])


def _first_day_of_next_month(value: date) -> date:
    return date(value.year + (value.month == 12), 1 if value.month == 12 else value.month + 1, 1)


def _next_month_key(month: str) -> str:
    value = _first_day_of_next_month(_assessment_period_end(month))
    return f"{value.year:04d}-{value.month:02d}"


def _validate_assessment_period(value: str) -> None:
    if _is_quarter_period(value):
        year = int(value[:4])
        if year < 1:
            raise TaxRoutingError("INVALID_ASSESSMENT_PERIOD", "计税期间年份无效。")
        return
    _validate_month_key(value)


def _is_quarter_period(value: str) -> bool:
    return isinstance(value, str) and len(value) == 7 and value[4:6] == "-Q" and value[-1:] in {"1", "2", "3", "4"} and value[:4].isdigit()


def _validate_month_key(value: str) -> None:
    _parse_month_key(value)


def _parse_month_key(value: str) -> tuple[int, int]:
    if not isinstance(value, str) or len(value) != 7 or value[4] != "-":
        raise TaxRoutingError("INVALID_MONTH", "月份必须使用YYYY-MM。")
    try:
        year, month_number = int(value[:4]), int(value[5:])
    except ValueError as error:
        raise TaxRoutingError("INVALID_MONTH", "月份必须使用YYYY-MM。") from error
    if year < 1 or not 1 <= month_number <= 12:
        raise TaxRoutingError("INVALID_MONTH", "月份无效。")
    return year, month_number


def _require_nonnegative_fen(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TaxRoutingError("INVALID_MONEY", f"{name}必须为非负整数分。")


def _require_positive_fen(name: str, value: int) -> None:
    _require_nonnegative_fen(name, value)
    if value == 0:
        raise TaxRoutingError("ZERO_MONEY", f"{name}必须大于0。")


def _require_rate(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10_000:
        raise TaxRoutingError("INVALID_RATE", f"{name}必须为0至10000之间的整数。")
