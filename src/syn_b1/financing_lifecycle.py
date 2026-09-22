"""SYN-B1-I2: monthly financing lifecycle planning only.

This module receives an already reconciled I1 cash plan and I1B statement
roll-forward.  It adds declared financing *plans* to that path.  It does not
create bank transactions, files, a ledger, or model inputs.  I1 remains the
single owner of the pre-financing operating/investing cash identity; I2 only
adds visible monthly financing movements and reconciles their consequences.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Sequence

from syn_b1.cash_plan import BalanceRange, CashPlanResult, PlanningMode
from syn_b1.statement_rollforward import StatementRollforwardResult


class FinancingLifecycleError(ValueError):
    """The declared financing contract is incomplete or internally invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class FundingKind(str, Enum):
    BANK_LOAN = "bank_loan"
    SHAREHOLDER_LOAN = "shareholder_loan"
    SHAREHOLDER_CAPITAL = "shareholder_capital_injection"


class RepaymentMethod(str, Enum):
    BULLET_PRINCIPAL = "bullet_principal"
    EQUAL_PRINCIPAL = "equal_principal"
    CUSTOM_DECLARED_SCHEDULE = "custom_declared_schedule"


class FixedFinancingPolicy(str, Enum):
    """The two mutually exclusive financing policies allowed in fixed mode."""

    OWNER_DECLARED_SCHEDULE = "owner_declared_schedule"
    MINIMUM_REQUIRED_WITHIN_RESERVE = "minimum_required_within_reserve"


@dataclass(frozen=True)
class PrincipalScheduleShare:
    """One user-declared principal repayment share for a loan instrument."""

    due_date: date
    share_bp: int

    def __post_init__(self) -> None:
        if not isinstance(self.due_date, date):
            raise FinancingLifecycleError("INVALID_REPAYMENT_DATE", "自定义还本日期必须是date。")
        _require_basis_points("share_bp", self.share_bp)


@dataclass(frozen=True)
class FinancingInstrument:
    """A finite fictional financing source selected by the user.

    ``limit_fen`` is a cumulative draw/capital limit, rather than a revolving
    facility.  This deliberately prevents the planner from silently borrowing
    or injecting capital without limit.  Loans need either a fixed maturity
    date or a term in days.  Capital injections have neither interest nor
    repayment.
    """

    instrument_id: str
    kind: FundingKind
    priority: int
    limit_fen: int
    available_from_date: date
    annual_interest_rate_bp: int = 0
    maturity_date: date | None = None
    term_days: int | None = None
    repayment_method: RepaymentMethod | None = None
    minimum_draw_fen: int = 1
    maximum_single_draw_fen: int | None = None
    custom_principal_schedule: tuple[PrincipalScheduleShare, ...] = ()

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise FinancingLifecycleError("EMPTY_INSTRUMENT_ID", "融资工具编号不能为空。")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) or self.priority < 0:
            raise FinancingLifecycleError("INVALID_PRIORITY", "融资优先级必须是非负整数。")
        _require_positive_fen("limit_fen", self.limit_fen)
        if not isinstance(self.available_from_date, date):
            raise FinancingLifecycleError("INVALID_AVAILABLE_DATE", "融资可用日期必须是date。")
        _require_basis_points("annual_interest_rate_bp", self.annual_interest_rate_bp)
        _require_positive_fen("minimum_draw_fen", self.minimum_draw_fen)
        if self.maximum_single_draw_fen is not None:
            _require_positive_fen("maximum_single_draw_fen", self.maximum_single_draw_fen)
            if self.maximum_single_draw_fen < self.minimum_draw_fen:
                raise FinancingLifecycleError("INVALID_DRAW_RANGE", "单笔最大提款额不能低于最小提款额。")

        if self.kind is FundingKind.SHAREHOLDER_CAPITAL:
            if self.annual_interest_rate_bp != 0 or self.maturity_date is not None or self.term_days is not None:
                raise FinancingLifecycleError("INVALID_CAPITAL_TERMS", "股东资本投入不能填写利率、期限或到期日。")
            if self.repayment_method is not None or self.custom_principal_schedule:
                raise FinancingLifecycleError("INVALID_CAPITAL_TERMS", "股东资本投入不能填写还本方式或还本计划。")
            return

        if (self.maturity_date is None) == (self.term_days is None):
            raise FinancingLifecycleError("MISSING_LOAN_TERM", "贷款必须二选一填写到期日或期限天数。")
        if self.term_days is not None:
            if not isinstance(self.term_days, int) or isinstance(self.term_days, bool) or self.term_days <= 0:
                raise FinancingLifecycleError("INVALID_TERM_DAYS", "贷款期限天数必须为正整数。")
        if self.maturity_date is not None and self.maturity_date <= self.available_from_date:
            raise FinancingLifecycleError("INVALID_MATURITY_DATE", "贷款到期日必须晚于可用日期。")
        if self.repayment_method is None:
            raise FinancingLifecycleError("MISSING_REPAYMENT_METHOD", "贷款必须声明还本方式。")
        if self.repayment_method is RepaymentMethod.CUSTOM_DECLARED_SCHEDULE:
            if not self.custom_principal_schedule or sum(item.share_bp for item in self.custom_principal_schedule) != 10_000:
                raise FinancingLifecycleError("INVALID_CUSTOM_REPAYMENT", "自定义还本计划必须非空且比例合计100%。")
            if len({item.due_date for item in self.custom_principal_schedule}) != len(self.custom_principal_schedule):
                raise FinancingLifecycleError("DUPLICATE_CUSTOM_REPAYMENT_DATE", "自定义还本日期不能重复。")
            if any(item.due_date < self.available_from_date for item in self.custom_principal_schedule):
                raise FinancingLifecycleError("RETROACTIVE_CUSTOM_REPAYMENT", "自定义还本日期不能早于可用日期。")
            if self.maturity_date is not None and any(item.due_date > self.maturity_date for item in self.custom_principal_schedule):
                raise FinancingLifecycleError("CUSTOM_REPAYMENT_AFTER_MATURITY", "自定义还本日期不能晚于贷款到期日。")
        elif self.custom_principal_schedule:
            raise FinancingLifecycleError("UNUSED_CUSTOM_REPAYMENT", "只有自定义还本方式可以填写自定义还本计划。")


@dataclass(frozen=True)
class FinancingReservation:
    """Fixed-total space available to named financing; it is not cash."""

    inflow_fen: int
    outflow_fen: int

    def __post_init__(self) -> None:
        _require_nonnegative_fen("融资流入预留", self.inflow_fen)
        _require_nonnegative_fen("融资流出预留", self.outflow_fen)


@dataclass(frozen=True)
class DeclaredFinancingEvent:
    """A user-declared fictional financing receipt or principal payment."""

    date: date
    instrument_id: str
    event_type: str
    amount_fen: int

    def __post_init__(self) -> None:
        if not isinstance(self.date, date):
            raise FinancingLifecycleError("INVALID_DECLARED_EVENT_DATE", "声明融资计划日期必须为date。")
        if not self.instrument_id:
            raise FinancingLifecycleError("EMPTY_DECLARED_EVENT_INSTRUMENT", "声明融资计划必须填写融资工具编号。")
        if self.event_type not in {"drawdown", "capital_injection", "principal_repayment"}:
            raise FinancingLifecycleError("INVALID_DECLARED_EVENT_TYPE", "声明融资计划类型必须为到账、资本投入或还本。")
        _require_positive_fen("声明融资计划金额", self.amount_fen)


@dataclass(frozen=True)
class OpeningLoanState:
    """Instrument-level opening state carried from a prior fictional period."""

    loan_id: str
    instrument_id: str
    funding_kind: FundingKind
    original_principal_fen: int
    outstanding_principal_fen: int
    facility_limit_fen: int
    facility_used_fen: int
    annual_interest_rate_bp: int
    draw_date: date
    interest_accrued_through_date: date
    maturity_date: date
    repayment_method: RepaymentMethod
    remaining_principal_schedule: tuple[tuple[date, int], ...]

    def __post_init__(self) -> None:
        if not self.loan_id:
            raise FinancingLifecycleError("EMPTY_OPENING_LOAN_STATE_ID", "期初融资状态必须填写贷款编号。")
        if not self.instrument_id:
            raise FinancingLifecycleError("EMPTY_OPENING_LOAN_ID", "期初融资状态必须填写工具编号。")
        if self.funding_kind not in {FundingKind.BANK_LOAN, FundingKind.SHAREHOLDER_LOAN}:
            raise FinancingLifecycleError("INVALID_OPENING_FUNDING_KIND", "期初未偿融资只能是银行借款或股东借款。")
        if self.repayment_method not in set(RepaymentMethod):
            raise FinancingLifecycleError("INVALID_OPENING_REPAYMENT_METHOD", "期初融资状态的还本方式无效。")
        for label, value in (
            ("期初原始本金", self.original_principal_fen),
            ("期初未偿本金", self.outstanding_principal_fen),
            ("期初融资总额度", self.facility_limit_fen),
            ("期初已用额度", self.facility_used_fen),
        ):
            _require_nonnegative_fen(label, value)
        if self.outstanding_principal_fen > self.original_principal_fen:
            raise FinancingLifecycleError("OPENING_PRINCIPAL_INVALID", "期初未偿本金不能超过原始本金。")
        if self.facility_used_fen < self.original_principal_fen:
            raise FinancingLifecycleError("OPENING_FACILITY_USED_INVALID", "期初已用额度不能小于原始本金。")
        if self.facility_used_fen > self.facility_limit_fen:
            raise FinancingLifecycleError("OPENING_FACILITY_USED_INVALID", "期初已用额度不能超过融资总额度。")
        _require_basis_points("期初融资年利率", self.annual_interest_rate_bp)
        if not all(isinstance(value, date) for value in (self.draw_date, self.interest_accrued_through_date, self.maturity_date)):
            raise FinancingLifecycleError("INVALID_OPENING_FINANCING_DATE", "期初融资日期必须为date。")
        if self.interest_accrued_through_date < self.draw_date or self.maturity_date <= self.draw_date:
            raise FinancingLifecycleError("OPENING_FINANCING_DATE_ORDER", "期初融资日期顺序无效。")
        if any(not isinstance(due, date) or not isinstance(amount, int) or amount < 0 for due, amount in self.remaining_principal_schedule):
            raise FinancingLifecycleError("INVALID_OPENING_REPAYMENT_SCHEDULE", "期初剩余还本计划无效。")
        if len({due for due, _ in self.remaining_principal_schedule}) != len(self.remaining_principal_schedule):
            raise FinancingLifecycleError("DUPLICATE_OPENING_REPAYMENT_DATE", "期初剩余还本计划日期不能重复。")
        if sum(amount for _, amount in self.remaining_principal_schedule) != self.outstanding_principal_fen:
            raise FinancingLifecycleError("OPENING_REPAYMENT_NOT_EQUAL_PRINCIPAL", "期初剩余还本计划合计必须等于未偿本金。")


@dataclass(frozen=True)
class FinancingOpeningState:
    """Restricted opening financing state for a continued fictional run."""

    loans: tuple[OpeningLoanState, ...] = ()
    interest_payable_fen: int = 0
    shareholder_capital_used_fen: int = 0

    def __post_init__(self) -> None:
        if len({loan.loan_id for loan in self.loans}) != len(self.loans):
            raise FinancingLifecycleError("DUPLICATE_OPENING_LOAN_STATE_ID", "期初融资状态的贷款编号不能重复。")
        _require_nonnegative_fen("期初应付利息", self.interest_payable_fen)
        _require_nonnegative_fen("期初已用股东资本额度", self.shareholder_capital_used_fen)


@dataclass(frozen=True)
class FinancingClosingState:
    """Restricted state that becomes the next period's opening input."""

    loans: tuple[OpeningLoanState, ...]
    interest_payable_fen: int
    shareholder_capital_used_fen: int

    def __post_init__(self) -> None:
        if len({loan.loan_id for loan in self.loans}) != len(self.loans):
            raise FinancingLifecycleError("DUPLICATE_CLOSING_LOAN_STATE_ID", "期末融资状态的贷款编号不能重复。")
        _require_nonnegative_fen("期末应付利息", self.interest_payable_fen)
        _require_nonnegative_fen("期末已用股东资本额度", self.shareholder_capital_used_fen)


@dataclass(frozen=True)
class FinancingLifecycleRequest:
    """Inputs for I2.  Everything here describes a completely fictional case."""

    cash_plan: CashPlanResult
    statements: StatementRollforwardResult
    instruments: tuple[FinancingInstrument, ...] = ()
    planning_mode: PlanningMode | None = None
    ending_balance_target_range: BalanceRange | None = None
    fixed_financing_reservation: FinancingReservation | None = None
    fixed_financing_policy: FixedFinancingPolicy | None = None
    declared_events: tuple[DeclaredFinancingEvent, ...] = ()
    opening_state: FinancingOpeningState = field(default_factory=FinancingOpeningState)

    def __post_init__(self) -> None:
        ids = [item.instrument_id for item in self.instruments]
        if len(ids) != len(set(ids)):
            raise FinancingLifecycleError("DUPLICATE_INSTRUMENT_ID", "融资工具编号不能重复。")
        priorities = [item.priority for item in self.instruments]
        if len(priorities) != len(set(priorities)):
            raise FinancingLifecycleError("DUPLICATE_PRIORITY", "融资优先级不能重复，请明确资金来源顺序。")
        if self.planning_mode is PlanningMode.FIXED_BANK_TOTALS:
            if self.fixed_financing_reservation is None or self.fixed_financing_policy is None:
                raise FinancingLifecycleError("MISSING_FIXED_FINANCING_POLICY", "固定总额模式必须提供融资预留和二选一融资政策。")
        elif self.fixed_financing_reservation is not None or self.fixed_financing_policy is not None:
            raise FinancingLifecycleError("UNUSED_FIXED_FINANCING_POLICY", "只有固定总额模式可以填写融资预留和固定融资政策。")
        if self.planning_mode is not PlanningMode.FIXED_BANK_TOTALS and self.declared_events:
            raise FinancingLifecycleError("DECLARED_EVENTS_ONLY_FOR_FIXED_MODE", "用户声明融资计划只允许用于固定总额模式。")
        if (
            self.planning_mode is PlanningMode.FIXED_BANK_TOTALS
            and self.fixed_financing_policy is FixedFinancingPolicy.MINIMUM_REQUIRED_WITHIN_RESERVE
            and self.declared_events
        ):
            raise FinancingLifecycleError(
                "MIXED_FIXED_FINANCING_POLICY",
                "固定总额选择系统最小必要融资时，不能同时填写用户声明融资计划。",
            )
        instrument_ids = set(ids)
        if any(event.instrument_id not in instrument_ids for event in self.declared_events):
            raise FinancingLifecycleError("UNKNOWN_DECLARED_EVENT_INSTRUMENT", "声明融资计划包含未启用的融资工具。")
        if len({(event.date, event.instrument_id, event.event_type) for event in self.declared_events}) != len(self.declared_events):
            raise FinancingLifecycleError("DUPLICATE_DECLARED_FINANCING_EVENT", "同日期、工具和类型的声明融资计划不能重复。")


@dataclass(frozen=True)
class FinancingScheduleEvent:
    """Restricted monthly-plan event, not a bank transaction or model field."""

    date: date
    month: str
    instrument_id: str
    kind: FundingKind
    event_type: str
    amount_fen: int
    principal_after_event_fen: int


@dataclass(frozen=True)
class InstrumentMonthState:
    month: str
    instrument_id: str
    opening_principal_fen: int
    drawdown_fen: int
    principal_repayment_fen: int
    interest_expense_fen: int
    interest_cash_payment_fen: int
    ending_principal_fen: int
    remaining_capacity_fen: int


@dataclass(frozen=True)
class FinancingCashPlanRow:
    month: str
    opening_bank_cash_fen: int
    base_operating_net_cashflow_fen: int
    base_investing_net_cashflow_fen: int
    bank_loan_drawdown_fen: int
    shareholder_loan_drawdown_fen: int
    shareholder_capital_injection_fen: int
    principal_repayment_fen: int
    interest_cash_payment_fen: int
    financing_net_cashflow_fen: int
    ending_bank_cash_fen: int
    minimum_liquidity_difference_fen: int


@dataclass(frozen=True)
class FinancedIncomeStatementRow:
    month: str
    revenue_fen: int
    cost_of_goods_sold_fen: int
    gross_profit_fen: int
    payroll_expense_fen: int
    rent_and_utilities_expense_fen: int
    other_operating_expense_fen: int
    depreciation_expense_fen: int
    tax_and_surcharge_expense_fen: int
    operating_profit_fen: int
    interest_expense_fen: int
    profit_before_income_tax_fen: int
    income_tax_expense_fen: int
    net_profit_fen: int


@dataclass(frozen=True)
class FinancedCashFlowStatementRow:
    month: str
    opening_bank_cash_fen: int
    operating_net_cashflow_fen: int
    investing_net_cashflow_fen: int
    bank_loan_drawdown_fen: int
    shareholder_loan_drawdown_fen: int
    shareholder_capital_injection_fen: int
    principal_repayment_fen: int
    interest_cash_payment_fen: int
    financing_net_cashflow_fen: int
    ending_bank_cash_fen: int


@dataclass(frozen=True)
class FinancedBalanceSheetRow:
    month: str
    bank_cash_fen: int
    accounts_receivable_fen: int
    inventory_fen: int
    construction_in_progress_fen: int
    gross_fixed_assets_fen: int
    accumulated_depreciation_fen: int
    net_fixed_assets_fen: int
    accounts_payable_fen: int
    capital_expenditure_payable_fen: int
    accrued_payroll_and_utilities_fen: int
    tax_payable_fen: int
    interest_payable_fen: int
    dividend_payable_fen: int
    bank_debt_fen: int
    shareholder_debt_fen: int
    paid_in_capital_fen: int
    retained_earnings_fen: int
    total_assets_fen: int
    total_liabilities_and_equity_fen: int


@dataclass(frozen=True)
class FinancingReconciliationRow:
    month: str
    cash_flow_difference_fen: int
    balance_sheet_difference_fen: int
    bank_debt_rollforward_difference_fen: int
    shareholder_debt_rollforward_difference_fen: int
    retained_earnings_rollforward_difference_fen: int


@dataclass(frozen=True)
class LiquidityFailure:
    date: date
    month: str
    shortfall_fen: int
    conflict_code: str
    exhausted_instrument_ids: tuple[str, ...]
    adjustable_items_cn: tuple[str, ...]


@dataclass(frozen=True)
class FinancingDiagnostic:
    severity: str
    code: str
    message_cn: str


@dataclass(frozen=True)
class FinancingLifecycleResult:
    request: FinancingLifecycleRequest
    is_feasible: bool
    financing_cash_plan_rows: tuple[FinancingCashPlanRow, ...]
    instrument_month_states: tuple[InstrumentMonthState, ...]
    schedule_events: tuple[FinancingScheduleEvent, ...]
    income_statement_rows: tuple[FinancedIncomeStatementRow, ...]
    cash_flow_statement_rows: tuple[FinancedCashFlowStatementRow, ...]
    balance_sheet_rows: tuple[FinancedBalanceSheetRow, ...]
    reconciliation_rows: tuple[FinancingReconciliationRow, ...]
    diagnostics: tuple[FinancingDiagnostic, ...]
    failure: LiquidityFailure | None = None
    closing_state: FinancingClosingState | None = None

    @property
    def is_fully_reconciled(self) -> bool:
        return self.is_feasible and all(
            row.cash_flow_difference_fen == 0
            and row.balance_sheet_difference_fen == 0
            and row.bank_debt_rollforward_difference_fen == 0
            and row.shareholder_debt_rollforward_difference_fen == 0
            and row.retained_earnings_rollforward_difference_fen == 0
            for row in self.reconciliation_rows
        )


@dataclass
class _Tranche:
    loan_id: str
    instrument: FinancingInstrument
    draw_date: date
    original_principal_fen: int
    outstanding_principal_fen: int
    maturity_date: date
    principal_schedule: tuple[tuple[date, int], ...]
    last_interest_payment_date: date


@dataclass(frozen=True)
class _Obligation:
    date: date
    tranche: _Tranche
    event_type: str
    amount_fen: int


def build_financing_lifecycle(request: FinancingLifecycleRequest) -> FinancingLifecycleResult:
    """Plan finite financing around a reconciled pre-financing monthly path.

    A liquidity failure is an expected planning result, not an exception: the
    returned diagnostic names its date, amount, exhausted sources and possible
    user adjustments.  Invalid contracts still raise ``FinancingLifecycleError``.
    """

    _validate_pre_financing_inputs(request)
    instruments = tuple(sorted(request.instruments, key=lambda item: item.priority))
    planning_mode = request.planning_mode or request.cash_plan.request.planning_mode
    target_range = request.ending_balance_target_range or request.cash_plan.request.target_ending_balance_fen
    rows = request.cash_plan.monthly_rows
    statement_by_month = {row.month: row for row in request.statements.cash_flow_statement_rows}
    balance_by_month = {row.month: row for row in request.statements.balance_sheet_rows}
    income_by_month = {row.month: row for row in request.statements.income_statement_rows}

    tranches, total_used = _opening_tranches_and_usage(request, instruments)
    current_cash = request.cash_plan.request.opening_balance_fen
    minimum_liquidity = request.cash_plan.request.minimum_liquidity_fen
    cash_rows: list[FinancingCashPlanRow] = []
    state_rows: list[InstrumentMonthState] = []
    schedule_events: list[FinancingScheduleEvent] = []
    diagnostics: list[FinancingDiagnostic] = []

    declared_by_month: dict[str, tuple[DeclaredFinancingEvent, ...]] = {}
    for item in request.declared_events:
        month = f"{item.date.year:04d}-{item.date.month:02d}"
        declared_by_month[month] = declared_by_month.get(month, ()) + (item,)

    for index, base_row in enumerate(rows):
        month_start, month_end = _period_month_bounds(
            base_row.month, request.cash_plan.request.start_date, request.cash_plan.request.end_date
        )
        opening_principal = _principal_by_instrument(tranches)
        current_tranches = list(tranches)
        draws_by_instrument = {item.instrument_id: 0 for item in instruments}
        capital_by_instrument = {item.instrument_id: 0 for item in instruments}
        declared_repayment_events: list[DeclaredFinancingEvent] = []
        if planning_mode is PlanningMode.FIXED_BANK_TOTALS and request.fixed_financing_policy is FixedFinancingPolicy.OWNER_DECLARED_SCHEDULE:
            _apply_declared_events(
                declared_by_month.get(base_row.month, ()),
                instruments,
                total_used,
                current_tranches,
                draws_by_instrument,
                capital_by_instrument,
                declared_repayment_events,
                schedule_events,
            )

        # Drawing may itself create a first-month interest payment.  Recompute
        # the month's obligations after each draw until the conservative cash
        # checkpoint and, in the final month, the declared ending lower bound
        # are both satisfied.
        draw_attempts = 0
        while True:
            obligations = _month_obligations(current_tranches, month_start, month_end)
            existing_service = sum(item.amount_fen for item in obligations)
            planned_financing_inflow = sum(draws_by_instrument.values()) + sum(capital_by_instrument.values())
            conservative_cash = current_cash - base_row.total_outflow_fen - existing_service + planned_financing_inflow
            required_draw = max(0, minimum_liquidity - conservative_cash)
            projected_ending = (
                current_cash
                + base_row.total_inflow_fen
                - base_row.total_outflow_fen
                + sum(draws_by_instrument.values())
                + sum(capital_by_instrument.values())
                - existing_service
            )
            if index == len(rows) - 1 and planning_mode in {PlanningMode.TARGET_ENDING_BALANCE, PlanningMode.CONSTRAINED_AUTO}:
                target = target_range
                assert target is not None
                required_draw = max(required_draw, target.minimum_fen - projected_ending)
            if required_draw <= 0:
                break

            draw_attempts += 1
            if draw_attempts > 100:
                return _failure_result(
                    request, cash_rows, state_rows, schedule_events, diagnostics,
                    _liquidity_failure(month_start, base_row.month, required_draw, instruments, "DRAW_LOOP_GUARD"),
                )
            if planning_mode is PlanningMode.FIXED_BANK_TOTALS and request.fixed_financing_policy is FixedFinancingPolicy.OWNER_DECLARED_SCHEDULE:
                return _failure_result(
                    request, cash_rows, state_rows, schedule_events, diagnostics,
                    _liquidity_failure(month_start, base_row.month, required_draw, instruments, "FUNDING_SOURCES_EXHAUSTED"),
                )
            candidate = _next_draw_candidate(
                instruments, total_used, month_start, required_draw, current_tranches
            )
            if candidate is None:
                return _failure_result(
                    request, cash_rows, state_rows, schedule_events, diagnostics,
                    _liquidity_failure(month_start, base_row.month, required_draw, instruments, "FUNDING_SOURCES_EXHAUSTED"),
                )
            instrument, amount = candidate
            if instrument.kind is FundingKind.SHAREHOLDER_CAPITAL:
                total_used[instrument.instrument_id] += amount
                capital_by_instrument[instrument.instrument_id] += amount
                schedule_events.append(
                    FinancingScheduleEvent(
                        date=month_start,
                        month=base_row.month,
                        instrument_id=instrument.instrument_id,
                        kind=instrument.kind,
                        event_type="capital_injection",
                        amount_fen=amount,
                        principal_after_event_fen=0,
                    )
                )
                continue

            tranche = _new_tranche(
                instrument,
                month_start,
                amount,
                _new_loan_id(instrument.instrument_id, month_start, current_tranches),
            )
            projected_net = amount - sum(
                item.amount_fen
                for item in _month_obligations((tranche,), month_start, month_end)
            )
            if projected_net <= 0:
                # This instrument cannot help this month under its own declared
                # maturity/repayment terms.  Treat it as unavailable, rather
                # than repeatedly drawing and immediately repaying it.
                total_used[instrument.instrument_id] = instrument.limit_fen
                diagnostics.append(
                    FinancingDiagnostic(
                        severity="warning",
                        code="DRAW_DOES_NOT_INCREASE_MONTHLY_LIQUIDITY",
                        message_cn=f"{instrument.instrument_id}在本月按已声明期限和还本方式提款后不能增加可用现金，未将其作为临时补丁使用。",
                    )
                )
                continue
            total_used[instrument.instrument_id] += amount
            draws_by_instrument[instrument.instrument_id] += amount
            current_tranches.append(tranche)
            schedule_events.append(
                FinancingScheduleEvent(
                    date=month_start,
                    month=base_row.month,
                    instrument_id=instrument.instrument_id,
                    kind=instrument.kind,
                    event_type="drawdown",
                    amount_fen=amount,
                    principal_after_event_fen=amount,
                )
            )

        obligations = _month_obligations(current_tranches, month_start, month_end)
        interest_by_instrument = {item.instrument_id: 0 for item in instruments}
        repayment_by_instrument = {item.instrument_id: 0 for item in instruments}
        for obligation in sorted(obligations, key=lambda item: (item.date, item.event_type, item.tranche.instrument.instrument_id)):
            instrument = obligation.tranche.instrument
            if obligation.event_type == "interest_payment":
                interest_by_instrument[instrument.instrument_id] += obligation.amount_fen
                obligation.tranche.last_interest_payment_date = obligation.date
            else:
                actual_repayment = _consume_scheduled_principal(
                    obligation.tranche, obligation.date, obligation.amount_fen
                )
                if not actual_repayment:
                    continue
                repayment_by_instrument[instrument.instrument_id] += actual_repayment
                obligation.tranche.outstanding_principal_fen -= actual_repayment
                if obligation.tranche.outstanding_principal_fen < 0:
                    raise AssertionError("SYN-B1 I2本金内部滚动错误")
                obligation = _Obligation(
                    obligation.date, obligation.tranche, obligation.event_type, actual_repayment
                )
            schedule_events.append(
                FinancingScheduleEvent(
                    date=obligation.date,
                    month=base_row.month,
                    instrument_id=instrument.instrument_id,
                    kind=instrument.kind,
                    event_type=obligation.event_type,
                    amount_fen=obligation.amount_fen,
                    principal_after_event_fen=obligation.tranche.outstanding_principal_fen,
                )
            )
        for event in sorted(declared_repayment_events, key=lambda item: (item.date, item.instrument_id)):
            if event.date < month_start or event.date > month_end:
                raise AssertionError("SYN-B1 I2声明还本月份内部归类错误")
            actual_repayment = _apply_declared_principal_repayment(
                current_tranches, event.instrument_id, event.amount_fen, event.date
            )
            if actual_repayment != event.amount_fen:
                return _failure_result(
                    request, cash_rows, state_rows, schedule_events, diagnostics,
                    _liquidity_failure(event.date, base_row.month, event.amount_fen, instruments, "DECLARED_REPAYMENT_EXCEEDS_OUTSTANDING"),
                )
            matching = next(
                item for item in current_tranches
                if item.instrument.instrument_id == event.instrument_id and item.outstanding_principal_fen >= 0
            )
            repayment_by_instrument[event.instrument_id] += actual_repayment
            schedule_events.append(
                FinancingScheduleEvent(
                    date=event.date,
                    month=base_row.month,
                    instrument_id=event.instrument_id,
                    kind=matching.instrument.kind,
                    event_type="principal_repayment",
                    amount_fen=actual_repayment,
                    principal_after_event_fen=sum(
                        item.outstanding_principal_fen
                        for item in current_tranches
                        if item.instrument.instrument_id == event.instrument_id
                    ),
                )
            )

        tranches = [item for item in current_tranches if item.outstanding_principal_fen]
        bank_draw = sum(draws_by_instrument[item.instrument_id] for item in instruments if item.kind is FundingKind.BANK_LOAN)
        shareholder_draw = sum(
            draws_by_instrument[item.instrument_id] for item in instruments if item.kind is FundingKind.SHAREHOLDER_LOAN
        )
        capital = sum(capital_by_instrument.values())
        repayment = sum(repayment_by_instrument.values())
        interest = sum(interest_by_instrument.values())
        financing_net = bank_draw + shareholder_draw + capital - repayment - interest
        ending_cash = current_cash + base_row.total_inflow_fen - base_row.total_outflow_fen + financing_net
        conservative_cash = current_cash - base_row.total_outflow_fen - repayment - interest + bank_draw + shareholder_draw + capital
        liquidity_difference = conservative_cash - minimum_liquidity
        if liquidity_difference < 0:
            return _failure_result(
                request, cash_rows, state_rows, schedule_events, diagnostics,
                _liquidity_failure(month_start, base_row.month, -liquidity_difference, instruments, "LIQUIDITY_NOT_COVERED"),
            )

        if index == len(rows) - 1 and planning_mode in {PlanningMode.TARGET_ENDING_BALANCE, PlanningMode.CONSTRAINED_AUTO}:
            target = target_range
            assert target is not None
            if ending_cash > target.maximum_fen:
                return _failure_result(
                    request, cash_rows, state_rows, schedule_events, diagnostics,
                    _liquidity_failure(month_end, base_row.month, ending_cash - target.maximum_fen, instruments, "ENDING_TARGET_UPPER_BOUND_EXCEEDED"),
                )

        cash_rows.append(
            FinancingCashPlanRow(
                month=base_row.month,
                opening_bank_cash_fen=current_cash,
                base_operating_net_cashflow_fen=base_row.operating_inflow_fen - base_row.operating_outflow_fen,
                base_investing_net_cashflow_fen=base_row.investing_inflow_fen - base_row.investing_outflow_fen,
                bank_loan_drawdown_fen=bank_draw,
                shareholder_loan_drawdown_fen=shareholder_draw,
                shareholder_capital_injection_fen=capital,
                principal_repayment_fen=repayment,
                interest_cash_payment_fen=interest,
                financing_net_cashflow_fen=financing_net,
                ending_bank_cash_fen=ending_cash,
                minimum_liquidity_difference_fen=liquidity_difference,
            )
        )
        ending_principal = _principal_by_instrument(tranches)
        for instrument in instruments:
            state_rows.append(
                InstrumentMonthState(
                    month=base_row.month,
                    instrument_id=instrument.instrument_id,
                    opening_principal_fen=opening_principal.get(instrument.instrument_id, 0),
                    drawdown_fen=(
                        capital_by_instrument[instrument.instrument_id]
                        if instrument.kind is FundingKind.SHAREHOLDER_CAPITAL
                        else draws_by_instrument[instrument.instrument_id]
                    ),
                    principal_repayment_fen=repayment_by_instrument[instrument.instrument_id],
                    interest_expense_fen=interest_by_instrument[instrument.instrument_id],
                    interest_cash_payment_fen=interest_by_instrument[instrument.instrument_id],
                    ending_principal_fen=ending_principal.get(instrument.instrument_id, 0),
                    remaining_capacity_fen=instrument.limit_fen - total_used[instrument.instrument_id],
                )
            )
        current_cash = ending_cash

    if planning_mode is PlanningMode.FIXED_BANK_TOTALS:
        fixed_failure = _validate_fixed_reservation(request, cash_rows, instruments)
        if fixed_failure is not None:
            return _failure_result(request, cash_rows, state_rows, schedule_events, diagnostics, fixed_failure)

    income_rows, cashflow_rows, balance_rows, reconciliation_rows = _build_financed_statements(
        request, cash_rows, state_rows, income_by_month, statement_by_month, balance_by_month
    )
    return FinancingLifecycleResult(
        request=request,
        is_feasible=True,
        financing_cash_plan_rows=tuple(cash_rows),
        instrument_month_states=tuple(state_rows),
        schedule_events=tuple(schedule_events),
        income_statement_rows=tuple(income_rows),
        cash_flow_statement_rows=tuple(cashflow_rows),
        balance_sheet_rows=tuple(balance_rows),
        reconciliation_rows=tuple(reconciliation_rows),
        diagnostics=tuple(diagnostics),
        closing_state=_closing_state(
            request,
            instruments,
            tranches,
            total_used,
            balance_rows[-1].interest_payable_fen,
        ),
    )


def _validate_pre_financing_inputs(request: FinancingLifecycleRequest) -> None:
    cash_plan = request.cash_plan
    statements = request.statements
    if statements.request.cash_plan != cash_plan:
        raise FinancingLifecycleError("CASH_PLAN_NOT_MATCHED", "I2必须使用I1B已经核对过的同一份I1现金计划。")
    if not statements.is_fully_reconciled:
        raise FinancingLifecycleError("PRE_FINANCING_STATEMENTS_NOT_RECONCILED", "融资前的三张表未对平，不能安排融资。")
    if cash_plan.request.opening_balance_fen != statements.request.opening_state.bank_cash_fen:
        raise FinancingLifecycleError("OPENING_CASH_MISMATCH", "I1期初现金与I1B期初银行现金不一致。")
    if not cash_plan.monthly_rows:
        raise FinancingLifecycleError("EMPTY_CASH_PLAN", "I1现金计划不能为空。")
    planning_mode = request.planning_mode or cash_plan.request.planning_mode
    if planning_mode is PlanningMode.FIXED_BANK_TOTALS:
        if request.fixed_financing_reservation is None or request.fixed_financing_policy is None:
            raise FinancingLifecycleError("MISSING_FIXED_FINANCING_POLICY", "固定总额模式缺少融资预留或融资政策。")
    if planning_mode in {PlanningMode.TARGET_ENDING_BALANCE, PlanningMode.CONSTRAINED_AUTO}:
        if request.ending_balance_target_range is None and cash_plan.request.target_ending_balance_fen is None:
            raise FinancingLifecycleError("MISSING_TARGET_RANGE", "目标余额或自动模式必须提供期末余额目标范围。")
    _validate_opening_state(request, request.instruments)


def _validate_opening_state(request: FinancingLifecycleRequest, instruments: Sequence[FinancingInstrument]) -> None:
    """Check that carried financing agrees with the opening balance sheet."""

    instrument_by_id = {item.instrument_id: item for item in instruments}
    opening_loans = request.opening_state.loans
    if any(item.instrument_id not in instrument_by_id for item in opening_loans):
        raise FinancingLifecycleError("OPENING_FINANCING_STATE_MISMATCH", "期初融资状态包含未启用的融资工具。")
    bank_debt = 0
    shareholder_debt = 0
    facility_used_by_instrument: dict[str, int] = {}
    for state in opening_loans:
        instrument = instrument_by_id[state.instrument_id]
        if instrument.kind is FundingKind.SHAREHOLDER_CAPITAL or state.funding_kind is FundingKind.SHAREHOLDER_CAPITAL:
            raise FinancingLifecycleError("OPENING_CAPITAL_AS_LOAN", "股东资本投入不能作为未偿借款续接。")
        if (
            state.funding_kind is not instrument.kind
            or state.facility_limit_fen != instrument.limit_fen
            or state.annual_interest_rate_bp != instrument.annual_interest_rate_bp
            or state.repayment_method is not instrument.repayment_method
        ):
            raise FinancingLifecycleError("OPENING_FINANCING_STATE_MISMATCH", "期初融资状态与启用融资工具条款不一致。")
        if state.facility_used_fen > state.facility_limit_fen:
            raise FinancingLifecycleError("OPENING_FINANCING_STATE_MISMATCH", "期初已用融资额度超过工具上限。")
        existing_used = facility_used_by_instrument.setdefault(state.instrument_id, state.facility_used_fen)
        if existing_used != state.facility_used_fen:
            raise FinancingLifecycleError("OPENING_FINANCING_STATE_MISMATCH", "同一融资工具的期初已用额度必须一致。")
        if state.maturity_date != _resolve_maturity(instrument, state.draw_date):
            raise FinancingLifecycleError("OPENING_FINANCING_STATE_MISMATCH", "期初融资到期日与工具条款不一致。")
        if state.interest_accrued_through_date > request.cash_plan.request.start_date:
            raise FinancingLifecycleError("OPENING_FINANCING_STATE_MISMATCH", "期初计息进度不能晚于本期开始日期。")
        if any(due < request.cash_plan.request.start_date for due, _ in state.remaining_principal_schedule):
            raise FinancingLifecycleError("OPENING_FINANCING_STATE_MISMATCH", "期初剩余还本计划不能包含已过期日期。")
        if instrument.kind is FundingKind.BANK_LOAN:
            bank_debt += state.outstanding_principal_fen
        else:
            shareholder_debt += state.outstanding_principal_fen
    opening = request.statements.request.opening_state
    if (
        bank_debt != opening.bank_debt_fen
        or shareholder_debt != opening.shareholder_debt_fen
        or request.opening_state.interest_payable_fen != opening.interest_payable_fen
    ):
        raise FinancingLifecycleError("OPENING_FINANCING_STATE_MISMATCH", "期初融资状态与期初资产负债表借款或应付利息不一致。")
    capital_instruments = [item for item in instruments if item.kind is FundingKind.SHAREHOLDER_CAPITAL]
    if request.opening_state.shareholder_capital_used_fen:
        if len(capital_instruments) != 1 or request.opening_state.shareholder_capital_used_fen > capital_instruments[0].limit_fen:
            raise FinancingLifecycleError("OPENING_FINANCING_STATE_MISMATCH", "期初已用股东资本额度与启用资本工具不一致。")


def _opening_tranches_and_usage(
    request: FinancingLifecycleRequest, instruments: Sequence[FinancingInstrument]
) -> tuple[list[_Tranche], dict[str, int]]:
    instrument_by_id = {item.instrument_id: item for item in instruments}
    total_used = {item.instrument_id: 0 for item in instruments}
    tranches: list[_Tranche] = []
    for state in request.opening_state.loans:
        instrument = instrument_by_id[state.instrument_id]
        total_used[state.instrument_id] = state.facility_used_fen
        if state.outstanding_principal_fen:
            tranches.append(
                _Tranche(
                    loan_id=state.loan_id,
                    instrument=instrument,
                    draw_date=state.draw_date,
                    original_principal_fen=state.original_principal_fen,
                    outstanding_principal_fen=state.outstanding_principal_fen,
                    maturity_date=state.maturity_date,
                    principal_schedule=tuple(state.remaining_principal_schedule),
                    last_interest_payment_date=state.interest_accrued_through_date,
                )
            )
    capital = [item for item in instruments if item.kind is FundingKind.SHAREHOLDER_CAPITAL]
    if capital:
        total_used[capital[0].instrument_id] = request.opening_state.shareholder_capital_used_fen
    return tranches, total_used


def _apply_declared_events(
    events: Sequence[DeclaredFinancingEvent],
    instruments: Sequence[FinancingInstrument],
    total_used: dict[str, int],
    tranches: list[_Tranche],
    draws: dict[str, int],
    capital: dict[str, int],
    declared_repayments: list[DeclaredFinancingEvent],
    schedule_events: list[FinancingScheduleEvent],
) -> None:
    """Apply a fixed-mode owner schedule; no automatic draw is added here."""

    instrument_by_id = {item.instrument_id: item for item in instruments}
    for event in sorted(events, key=lambda item: (item.date, item.instrument_id, item.event_type)):
        instrument = instrument_by_id[event.instrument_id]
        if event.date < instrument.available_from_date and event.event_type != "principal_repayment":
            raise FinancingLifecycleError("DECLARED_EVENT_BEFORE_AVAILABLE", "声明融资到账日期早于工具可用日期。")
        if event.event_type == "capital_injection":
            if instrument.kind is not FundingKind.SHAREHOLDER_CAPITAL:
                raise FinancingLifecycleError("DECLARED_EVENT_KIND_MISMATCH", "资本投入必须使用股东资本工具。")
            if total_used[instrument.instrument_id] + event.amount_fen > instrument.limit_fen:
                raise FinancingLifecycleError("FIXED_FINANCING_INFLOW_RESERVE_EXCEEDED", "声明资本投入超过已启用额度。")
            total_used[instrument.instrument_id] += event.amount_fen
            capital[instrument.instrument_id] += event.amount_fen
            schedule_events.append(FinancingScheduleEvent(event.date, f"{event.date.year:04d}-{event.date.month:02d}", instrument.instrument_id, instrument.kind, event.event_type, event.amount_fen, 0))
        elif event.event_type == "drawdown":
            if instrument.kind is FundingKind.SHAREHOLDER_CAPITAL:
                raise FinancingLifecycleError("DECLARED_EVENT_KIND_MISMATCH", "股东资本工具必须使用资本投入类型。")
            if total_used[instrument.instrument_id] + event.amount_fen > instrument.limit_fen:
                raise FinancingLifecycleError("FIXED_FINANCING_INFLOW_RESERVE_EXCEEDED", "声明融资到账超过已启用额度。")
            tranche = _new_tranche(
                instrument,
                event.date,
                event.amount_fen,
                _new_loan_id(instrument.instrument_id, event.date, tranches),
            )
            total_used[instrument.instrument_id] += event.amount_fen
            draws[instrument.instrument_id] += event.amount_fen
            tranches.append(tranche)
            schedule_events.append(FinancingScheduleEvent(event.date, f"{event.date.year:04d}-{event.date.month:02d}", instrument.instrument_id, instrument.kind, event.event_type, event.amount_fen, event.amount_fen))
        else:
            if instrument.kind is FundingKind.SHAREHOLDER_CAPITAL:
                raise FinancingLifecycleError("DECLARED_EVENT_KIND_MISMATCH", "股东资本工具不能声明还本。")
            declared_repayments.append(event)


def _validate_fixed_reservation(
    request: FinancingLifecycleRequest,
    rows: Sequence[FinancingCashPlanRow],
    instruments: Sequence[FinancingInstrument],
) -> LiquidityFailure | None:
    assert request.fixed_financing_reservation is not None
    inflow = sum(
        row.bank_loan_drawdown_fen + row.shareholder_loan_drawdown_fen + row.shareholder_capital_injection_fen
        for row in rows
    )
    outflow = sum(row.principal_repayment_fen + row.interest_cash_payment_fen for row in rows)
    reserve = request.fixed_financing_reservation
    if inflow > reserve.inflow_fen:
        return _liquidity_failure(request.cash_plan.request.start_date, rows[-1].month, inflow - reserve.inflow_fen, instruments, "FIXED_FINANCING_INFLOW_RESERVE_EXCEEDED")
    if outflow > reserve.outflow_fen:
        return _liquidity_failure(request.cash_plan.request.start_date, rows[-1].month, outflow - reserve.outflow_fen, instruments, "FIXED_FINANCING_OUTFLOW_RESERVE_EXCEEDED")
    if inflow != reserve.inflow_fen or outflow != reserve.outflow_fen:
        return _liquidity_failure(request.cash_plan.request.start_date, rows[-1].month, abs(reserve.inflow_fen - inflow) + abs(reserve.outflow_fen - outflow), instruments, "FIXED_UNEXPLAINED_REMAINDER")
    return None


def _consume_scheduled_principal(tranche: _Tranche, due_date: date, amount_fen: int) -> int:
    """Consume one due instalment and remove it from the carried schedule."""

    updated: list[tuple[date, int]] = []
    remaining_to_consume = amount_fen
    consumed = 0
    for due, scheduled_amount in tranche.principal_schedule:
        if due == due_date and remaining_to_consume:
            applied = min(scheduled_amount, remaining_to_consume)
            scheduled_amount -= applied
            remaining_to_consume -= applied
            consumed += applied
        if scheduled_amount:
            updated.append((due, scheduled_amount))
    if remaining_to_consume:
        raise AssertionError("SYN-B1 I2还本计划与月度偿还义务不一致")
    tranche.principal_schedule = tuple(updated)
    return consumed


def _apply_declared_principal_repayment(
    tranches: Sequence[_Tranche],
    instrument_id: str,
    amount_fen: int,
    repayment_date: date,
) -> int:
    """Apply an owner-declared early repayment without duplicating future due amounts.

    The monthly result has no transaction ledger.  We therefore reduce the
    earliest still-unpaid instalments on the same financing instrument, while
    retaining the declared calendar date in the schedule-event output.
    """

    eligible = sorted(
        (
            item for item in tranches
            if item.instrument.instrument_id == instrument_id and item.outstanding_principal_fen
        ),
        key=lambda item: (item.maturity_date, item.draw_date, item.loan_id),
    )
    if sum(item.outstanding_principal_fen for item in eligible) < amount_fen:
        return 0
    remaining = amount_fen
    for tranche in eligible:
        if not remaining:
            break
        applied = min(tranche.outstanding_principal_fen, remaining)
        _reduce_future_principal_schedule(tranche, applied, repayment_date)
        tranche.outstanding_principal_fen -= applied
        remaining -= applied
    if remaining:
        raise AssertionError("SYN-B1 I2声明还本内部滚动错误")
    return amount_fen


def _reduce_future_principal_schedule(
    tranche: _Tranche, amount_fen: int, repayment_date: date
) -> None:
    """Remove an early repayment from the earliest remaining due instalments."""

    remaining = amount_fen
    updated: list[tuple[date, int]] = []
    for due, scheduled_amount in tranche.principal_schedule:
        if due >= repayment_date and remaining:
            applied = min(scheduled_amount, remaining)
            scheduled_amount -= applied
            remaining -= applied
        if scheduled_amount:
            updated.append((due, scheduled_amount))
    if remaining:
        raise FinancingLifecycleError(
            "DECLARED_REPAYMENT_CANNOT_REDUCE_SCHEDULE",
            "声明还本不能超过尚未到期的剩余还本计划。",
        )
    tranche.principal_schedule = tuple(updated)


def _closing_state(
    request: FinancingLifecycleRequest,
    instruments: Sequence[FinancingInstrument],
    tranches: Sequence[_Tranche],
    total_used: dict[str, int],
    ending_interest_payable_fen: int,
) -> FinancingClosingState:
    loans = tuple(
        OpeningLoanState(
            loan_id=item.loan_id,
            instrument_id=item.instrument.instrument_id,
            funding_kind=item.instrument.kind,
            original_principal_fen=item.original_principal_fen,
            outstanding_principal_fen=item.outstanding_principal_fen,
            facility_limit_fen=item.instrument.limit_fen,
            facility_used_fen=total_used[item.instrument.instrument_id],
            annual_interest_rate_bp=item.instrument.annual_interest_rate_bp,
            draw_date=item.draw_date,
            interest_accrued_through_date=item.last_interest_payment_date,
            maturity_date=item.maturity_date,
            repayment_method=item.instrument.repayment_method,
            remaining_principal_schedule=tuple(item.principal_schedule),
        )
        for item in tranches
    )
    capital = next((item for item in instruments if item.kind is FundingKind.SHAREHOLDER_CAPITAL), None)
    return FinancingClosingState(
        loans=loans,
        interest_payable_fen=ending_interest_payable_fen,
        shareholder_capital_used_fen=total_used[capital.instrument_id] if capital else 0,
    )


def _period_month_bounds(month: str, start: date, end: date) -> tuple[date, date]:
    year, number = _parse_month(month)
    first = date(year, number, 1)
    last = date(year, number, monthrange(year, number)[1])
    return max(first, start), min(last, end)


def _month_obligations(tranches: Sequence[_Tranche], month_start: date, month_end: date) -> tuple[_Obligation, ...]:
    obligations: list[_Obligation] = []
    for tranche in tranches:
        interest_end = min(month_end, tranche.maturity_date)
        if tranche.last_interest_payment_date < interest_end and tranche.outstanding_principal_fen:
            days = (interest_end - tranche.last_interest_payment_date).days
            interest = _interest_fen(
                tranche.outstanding_principal_fen,
                tranche.instrument.annual_interest_rate_bp,
                days,
            )
            if interest:
                obligations.append(_Obligation(interest_end, tranche, "interest_payment", interest))
        for due_date, amount in tranche.principal_schedule:
            if month_start <= due_date <= month_end and amount:
                obligations.append(_Obligation(due_date, tranche, "principal_repayment", amount))
    return tuple(obligations)


def _next_draw_candidate(
    instruments: Sequence[FinancingInstrument],
    total_used: dict[str, int],
    draw_date: date,
    required_fen: int,
    current_tranches: Sequence[_Tranche],
) -> tuple[FinancingInstrument, int] | None:
    del current_tranches  # Future extension point: facility-level utilisation constraints.
    for instrument in instruments:
        if instrument.available_from_date > draw_date:
            continue
        remaining = instrument.limit_fen - total_used[instrument.instrument_id]
        if remaining < instrument.minimum_draw_fen:
            continue
        maximum = min(remaining, instrument.maximum_single_draw_fen or remaining)
        amount = max(required_fen, instrument.minimum_draw_fen)
        amount = min(amount, maximum)
        if amount < instrument.minimum_draw_fen:
            continue
        if instrument.kind is not FundingKind.SHAREHOLDER_CAPITAL:
            maturity = _resolve_maturity(instrument, draw_date)
            if maturity <= draw_date:
                continue
            if instrument.repayment_method is RepaymentMethod.CUSTOM_DECLARED_SCHEDULE and any(
                item.due_date < draw_date for item in instrument.custom_principal_schedule
            ):
                continue
        return instrument, amount
    return None


def _new_loan_id(instrument_id: str, draw_date: date, tranches: Sequence[_Tranche]) -> str:
    """Create a deterministic per-draw loan identity within one fictional run."""

    prefix = f"{instrument_id}@{draw_date.isoformat()}"
    number = 1 + sum(1 for item in tranches if item.loan_id.startswith(prefix + "#"))
    return f"{prefix}#{number}"


def _new_tranche(
    instrument: FinancingInstrument,
    draw_date: date,
    amount_fen: int,
    loan_id: str,
) -> _Tranche:
    maturity = _resolve_maturity(instrument, draw_date)
    method = instrument.repayment_method
    assert method is not None
    if method is RepaymentMethod.BULLET_PRINCIPAL:
        schedule = ((maturity, amount_fen),)
    elif method is RepaymentMethod.EQUAL_PRINCIPAL:
        dates = _monthly_payment_dates(draw_date, maturity)
        schedule = _allocate_amount_to_dates(amount_fen, dates)
    else:
        assert method is RepaymentMethod.CUSTOM_DECLARED_SCHEDULE
        schedule = _allocate_amount_to_shares(amount_fen, instrument.custom_principal_schedule)
        if any(due < draw_date for due, _ in schedule):
            raise FinancingLifecycleError("RETROACTIVE_REPAYMENT", "提款后不能安排已过去的还本日期。")
    return _Tranche(
        loan_id=loan_id,
        instrument=instrument,
        draw_date=draw_date,
        original_principal_fen=amount_fen,
        outstanding_principal_fen=amount_fen,
        maturity_date=maturity,
        principal_schedule=schedule,
        last_interest_payment_date=draw_date,
    )


def _resolve_maturity(instrument: FinancingInstrument, draw_date: date) -> date:
    if instrument.maturity_date is not None:
        return instrument.maturity_date
    assert instrument.term_days is not None
    return date.fromordinal(draw_date.toordinal() + instrument.term_days)


def _monthly_payment_dates(draw_date: date, maturity_date: date) -> tuple[date, ...]:
    dates: list[date] = []
    # A monthly equal-principal plan starts after the draw month.  Otherwise a
    # loan drawn to cover that month's shortage would immediately repay itself
    # at the same month-end and cease to be a usable financing source.
    draw_month_end = date(draw_date.year, draw_date.month, monthrange(draw_date.year, draw_date.month)[1])
    cursor = _next_month_end(draw_month_end)
    while cursor < maturity_date:
        if cursor > draw_date:
            dates.append(cursor)
        cursor = _next_month_end(cursor)
    dates.append(maturity_date)
    return tuple(sorted(set(dates)))


def _allocate_amount_to_dates(amount_fen: int, dates: Sequence[date]) -> tuple[tuple[date, int], ...]:
    if not dates:
        raise FinancingLifecycleError("EMPTY_REPAYMENT_DATES", "贷款没有可用的还本日期。")
    base = amount_fen // len(dates)
    remaining = amount_fen - base * len(dates)
    return tuple((value, base + (1 if index < remaining else 0)) for index, value in enumerate(dates))


def _allocate_amount_to_shares(
    amount_fen: int, shares: Sequence[PrincipalScheduleShare]
) -> tuple[tuple[date, int], ...]:
    ordered = tuple(sorted(shares, key=lambda item: item.due_date))
    remaining = amount_fen
    results: list[tuple[date, int]] = []
    for share in ordered[:-1]:
        allocated = amount_fen * share.share_bp // 10_000
        results.append((share.due_date, allocated))
        remaining -= allocated
    results.append((ordered[-1].due_date, remaining))
    return tuple(results)


def _interest_fen(principal_fen: int, rate_bp: int, days: int) -> int:
    if principal_fen == 0 or rate_bp == 0 or days == 0:
        return 0
    numerator = principal_fen * rate_bp * days
    denominator = 365 * 10_000
    return (numerator + denominator // 2) // denominator


def _principal_by_instrument(tranches: Sequence[_Tranche]) -> dict[str, int]:
    result: dict[str, int] = {}
    for tranche in tranches:
        result[tranche.instrument.instrument_id] = result.get(tranche.instrument.instrument_id, 0) + tranche.outstanding_principal_fen
    return result


def _build_financed_statements(
    request: FinancingLifecycleRequest,
    cash_rows: Sequence[FinancingCashPlanRow],
    state_rows: Sequence[InstrumentMonthState],
    income_by_month,
    cashflow_by_month,
    balance_by_month,
) -> tuple[list[FinancedIncomeStatementRow], list[FinancedCashFlowStatementRow], list[FinancedBalanceSheetRow], list[FinancingReconciliationRow]]:
    instruments_by_id = {item.instrument_id: item for item in request.instruments}
    states_by_month: dict[str, list[InstrumentMonthState]] = {}
    for row in state_rows:
        states_by_month.setdefault(row.month, []).append(row)
    incomes: list[FinancedIncomeStatementRow] = []
    cashflows: list[FinancedCashFlowStatementRow] = []
    balances: list[FinancedBalanceSheetRow] = []
    reconciliations: list[FinancingReconciliationRow] = []
    cumulative_interest = 0
    cumulative_income_tax_adjustment = 0
    cumulative_capital = 0
    previous_bank_debt = request.statements.request.opening_state.bank_debt_fen
    previous_shareholder_debt = request.statements.request.opening_state.shareholder_debt_fen
    previous_retained = request.statements.resolved_opening_retained_earnings_fen

    for cash in cash_rows:
        month_states = states_by_month.get(cash.month, [])
        base_income = income_by_month[cash.month]
        base_cashflow = cashflow_by_month[cash.month]
        base_balance = balance_by_month[cash.month]
        interest = sum(item.interest_expense_fen for item in month_states)
        bank_draw = sum(
            item.drawdown_fen for item in month_states if instruments_by_id[item.instrument_id].kind is FundingKind.BANK_LOAN
        )
        shareholder_draw = sum(
            item.drawdown_fen
            for item in month_states
            if instruments_by_id[item.instrument_id].kind is FundingKind.SHAREHOLDER_LOAN
        )
        capital = sum(
            item.drawdown_fen
            for item in month_states
            if instruments_by_id[item.instrument_id].kind is FundingKind.SHAREHOLDER_CAPITAL
        )
        bank_repay = sum(
            item.principal_repayment_fen
            for item in month_states
            if instruments_by_id[item.instrument_id].kind is FundingKind.BANK_LOAN
        )
        shareholder_repay = sum(
            item.principal_repayment_fen
            for item in month_states
            if instruments_by_id[item.instrument_id].kind is FundingKind.SHAREHOLDER_LOAN
        )
        bank_debt = previous_bank_debt + bank_draw - bank_repay
        shareholder_debt = previous_shareholder_debt + shareholder_draw - shareholder_repay
        cumulative_interest += interest
        cumulative_capital += capital
        financed_profit_before_income_tax = base_income.profit_before_income_tax_fen - interest
        financed_income_tax = max(
            0,
            financed_profit_before_income_tax * request.statements.request.income_tax_rate_bp // 10_000,
        )
        cumulative_income_tax_adjustment += financed_income_tax - base_income.income_tax_expense_fen
        financed_net_profit = financed_profit_before_income_tax - financed_income_tax
        retained = base_balance.retained_earnings_fen - cumulative_interest - cumulative_income_tax_adjustment
        income = FinancedIncomeStatementRow(
            month=cash.month,
            revenue_fen=base_income.revenue_fen,
            cost_of_goods_sold_fen=base_income.cost_of_goods_sold_fen,
            gross_profit_fen=base_income.gross_profit_fen,
            payroll_expense_fen=base_income.payroll_expense_fen,
            rent_and_utilities_expense_fen=base_income.rent_and_utilities_expense_fen,
            other_operating_expense_fen=base_income.other_operating_expense_fen,
            depreciation_expense_fen=base_income.depreciation_expense_fen,
            tax_and_surcharge_expense_fen=base_income.tax_and_surcharge_expense_fen,
            operating_profit_fen=base_income.operating_profit_fen,
            interest_expense_fen=interest,
            profit_before_income_tax_fen=financed_profit_before_income_tax,
            income_tax_expense_fen=financed_income_tax,
            net_profit_fen=financed_net_profit,
        )
        cashflow = FinancedCashFlowStatementRow(
            month=cash.month,
            opening_bank_cash_fen=cash.opening_bank_cash_fen,
            operating_net_cashflow_fen=base_cashflow.operating_net_cashflow_fen,
            investing_net_cashflow_fen=base_cashflow.investing_net_cashflow_fen,
            bank_loan_drawdown_fen=bank_draw,
            shareholder_loan_drawdown_fen=shareholder_draw,
            shareholder_capital_injection_fen=capital,
            principal_repayment_fen=bank_repay + shareholder_repay,
            interest_cash_payment_fen=interest,
            financing_net_cashflow_fen=cash.financing_net_cashflow_fen,
            ending_bank_cash_fen=cash.ending_bank_cash_fen,
        )
        balance = FinancedBalanceSheetRow(
            month=cash.month,
            bank_cash_fen=cash.ending_bank_cash_fen,
            accounts_receivable_fen=base_balance.accounts_receivable_fen,
            inventory_fen=base_balance.inventory_fen,
            construction_in_progress_fen=base_balance.construction_in_progress_fen,
            gross_fixed_assets_fen=base_balance.gross_fixed_assets_fen,
            accumulated_depreciation_fen=base_balance.accumulated_depreciation_fen,
            net_fixed_assets_fen=base_balance.net_fixed_assets_fen,
            accounts_payable_fen=base_balance.accounts_payable_fen,
            capital_expenditure_payable_fen=base_balance.capital_expenditure_payable_fen,
            accrued_payroll_and_utilities_fen=base_balance.accrued_payroll_and_utilities_fen,
            tax_payable_fen=base_balance.tax_payable_fen + cumulative_income_tax_adjustment,
            interest_payable_fen=request.statements.request.opening_state.interest_payable_fen,
            dividend_payable_fen=base_balance.dividend_payable_fen,
            bank_debt_fen=bank_debt,
            shareholder_debt_fen=shareholder_debt,
            paid_in_capital_fen=base_balance.paid_in_capital_fen + cumulative_capital,
            retained_earnings_fen=retained,
            total_assets_fen=0,
            total_liabilities_and_equity_fen=0,
        )
        assets = (
            balance.bank_cash_fen + balance.accounts_receivable_fen + balance.inventory_fen
            + balance.construction_in_progress_fen + balance.net_fixed_assets_fen
        )
        liabilities_equity = (
            balance.accounts_payable_fen + balance.capital_expenditure_payable_fen
            + balance.accrued_payroll_and_utilities_fen + balance.tax_payable_fen
            + balance.interest_payable_fen + balance.dividend_payable_fen + balance.bank_debt_fen
            + balance.shareholder_debt_fen + balance.paid_in_capital_fen + balance.retained_earnings_fen
        )
        balance = FinancedBalanceSheetRow(**{**balance.__dict__, "total_assets_fen": assets, "total_liabilities_and_equity_fen": liabilities_equity})
        cash_difference = (
            cashflow.opening_bank_cash_fen + cashflow.operating_net_cashflow_fen + cashflow.investing_net_cashflow_fen
            + cashflow.financing_net_cashflow_fen - cashflow.ending_bank_cash_fen
        )
        reconciliation = FinancingReconciliationRow(
            month=cash.month,
            cash_flow_difference_fen=cash_difference,
            balance_sheet_difference_fen=assets - liabilities_equity,
            bank_debt_rollforward_difference_fen=bank_debt - (previous_bank_debt + bank_draw - bank_repay),
            shareholder_debt_rollforward_difference_fen=shareholder_debt - (previous_shareholder_debt + shareholder_draw - shareholder_repay),
            retained_earnings_rollforward_difference_fen=retained - (previous_retained + income.net_profit_fen),
        )
        incomes.append(income)
        cashflows.append(cashflow)
        balances.append(balance)
        reconciliations.append(reconciliation)
        previous_bank_debt = bank_debt
        previous_shareholder_debt = shareholder_debt
        previous_retained = retained
    return incomes, cashflows, balances, reconciliations


def _liquidity_failure(
    when: date,
    month: str,
    shortfall_fen: int,
    instruments: Sequence[FinancingInstrument],
    code: str,
) -> LiquidityFailure:
    return LiquidityFailure(
        date=when,
        month=month,
        shortfall_fen=max(0, shortfall_fen),
        conflict_code=code,
        exhausted_instrument_ids=tuple(item.instrument_id for item in instruments),
        adjustable_items_cn=(
            "提高已声明且可用的融资额度或股东投入上限",
            "提前资金来源可用日期或延长贷款期限",
            "降低最低可用余额目标",
            "调整经营、投资或到期付款计划",
        ),
    )


def _failure_result(
    request: FinancingLifecycleRequest,
    cash_rows: Sequence[FinancingCashPlanRow],
    state_rows: Sequence[InstrumentMonthState],
    events: Sequence[FinancingScheduleEvent],
    diagnostics: Sequence[FinancingDiagnostic],
    failure: LiquidityFailure,
) -> FinancingLifecycleResult:
    message = "资金来源不足，已在生成最终账本前停止；请查看缺口日期、额度和可调整项。"
    return FinancingLifecycleResult(
        request=request,
        is_feasible=False,
        financing_cash_plan_rows=tuple(cash_rows),
        instrument_month_states=tuple(state_rows),
        schedule_events=tuple(events),
        income_statement_rows=(),
        cash_flow_statement_rows=(),
        balance_sheet_rows=(),
        reconciliation_rows=(),
        diagnostics=tuple(diagnostics) + (FinancingDiagnostic("error", failure.conflict_code, message),),
        failure=failure,
    )


def _next_month_end(value: date) -> date:
    if value.month == 12:
        year, month = value.year + 1, 1
    else:
        year, month = value.year, value.month + 1
    return date(year, month, monthrange(year, month)[1])


def _parse_month(value: str) -> tuple[int, int]:
    if not isinstance(value, str) or len(value) != 7 or value[4] != "-":
        raise FinancingLifecycleError("INVALID_MONTH", "月度计划月份必须为YYYY-MM。")
    return int(value[:4]), int(value[5:])


def _require_positive_fen(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FinancingLifecycleError("INVALID_MONEY", f"{name}必须为正整数分。")


def _require_nonnegative_fen(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FinancingLifecycleError("INVALID_MONEY", f"{name}必须为非负整数分。")


def _require_basis_points(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10_000:
        raise FinancingLifecycleError("INVALID_RATE", f"{name}必须是0至10000之间的整数。")
