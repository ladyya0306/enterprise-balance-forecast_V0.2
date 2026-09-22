"""SYN-B1-I1B: restricted monthly statements and reconciliations only.

This module turns an I1A operating-cycle result plus an already calculated
I1 cash-plan result into internal management statements.  It deliberately
does not plan financing, create transactions, write files, or expose any
statement values as model inputs.  I1 remains the sole owner of its monthly
cash-plan identity; this module only checks the completed plan against the
named operating and investing cash movements from I1A.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

from syn_b1.cash_plan import CashPlanResult
from syn_b1.operating_cycle import FixedAssetEvent, OperatingCycleResult


class StatementRollforwardError(ValueError):
    """The supplied pre-financing plans cannot form balanced statements."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StatementOpeningState:
    """Complete opening balance-sheet state for one fictional enterprise.

    ``retained_earnings_fen=None`` is the deliberately visible simple-mode
    choice: it is derived as assets minus liabilities minus paid-in capital.
    A supplied retained earnings value, including a negative accumulated loss,
    must make the opening statement balance exactly.

    I1B does not create new financing, but it can carry an already reconciled
    opening financing summary.  I2 validates the instrument-level opening
    state and changes the balances later in the joint path.
    """

    bank_cash_fen: int
    accounts_receivable_fen: int = 0
    inventory_fen: int = 0
    accounts_payable_fen: int = 0
    construction_in_progress_fen: int = 0
    gross_fixed_assets_fen: int = 0
    accumulated_depreciation_fen: int = 0
    capital_expenditure_payable_fen: int = 0
    accrued_payroll_and_utilities_fen: int = 0
    tax_payable_fen: int = 0
    interest_payable_fen: int = 0
    dividend_payable_fen: int = 0
    bank_debt_fen: int = 0
    shareholder_debt_fen: int = 0
    paid_in_capital_fen: int = 0
    retained_earnings_fen: int | None = None
    fixed_asset_remaining_life_months: int | None = None
    fixed_asset_residual_fen: int = 0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if name == "retained_earnings_fen" or name == "fixed_asset_remaining_life_months":
                continue
            _require_nonnegative_fen(name, value)
        if self.accumulated_depreciation_fen > self.gross_fixed_assets_fen:
            raise StatementRollforwardError("INVALID_OPENING_FIXED_ASSET", "期初累计折旧不能超过固定资产原值。")
        if self.fixed_asset_residual_fen > self.net_fixed_assets_fen:
            raise StatementRollforwardError("INVALID_OPENING_FIXED_ASSET", "期初固定资产残值不能超过净额。")
        if self.net_fixed_assets_fen:
            _require_positive_months("fixed_asset_remaining_life_months", self.fixed_asset_remaining_life_months)
        elif self.fixed_asset_remaining_life_months is not None:
            _require_positive_months("fixed_asset_remaining_life_months", self.fixed_asset_remaining_life_months)
        if self.retained_earnings_fen is not None and (
            not isinstance(self.retained_earnings_fen, int) or isinstance(self.retained_earnings_fen, bool)
        ):
            raise StatementRollforwardError("INVALID_RETAINED_EARNINGS", "期初留存收益必须为整数分、可为负，或留空反算。")

    @property
    def net_fixed_assets_fen(self) -> int:
        return self.gross_fixed_assets_fen - self.accumulated_depreciation_fen

    @property
    def assets_before_retained_earnings_fen(self) -> int:
        return (
            self.bank_cash_fen
            + self.accounts_receivable_fen
            + self.inventory_fen
            + self.construction_in_progress_fen
            + self.net_fixed_assets_fen
        )

    @property
    def liabilities_before_retained_earnings_fen(self) -> int:
        return (
            self.accounts_payable_fen
            + self.capital_expenditure_payable_fen
            + self.accrued_payroll_and_utilities_fen
            + self.tax_payable_fen
            + self.interest_payable_fen
            + self.dividend_payable_fen
            + self.bank_debt_fen
            + self.shareholder_debt_fen
            + self.paid_in_capital_fen
        )

    def resolved_retained_earnings_fen(self) -> int:
        if self.retained_earnings_fen is not None:
            return self.retained_earnings_fen
        return self.assets_before_retained_earnings_fen - self.liabilities_before_retained_earnings_fen


@dataclass(frozen=True)
class AssetDepreciationPolicy:
    """Straight-line policy for an I1A fixed-asset event.

    The policy is an explicit fictional scenario setting.  It is required
    only when the event becomes usable inside the I1B planning period.
    """

    event_id: str
    useful_life_months: int
    residual_value_rate_bp: int = 0

    def __post_init__(self) -> None:
        if not self.event_id:
            raise StatementRollforwardError("EMPTY_ASSET_POLICY_ID", "固定资产折旧政策必须填写事件编号。")
        _require_positive_months("useful_life_months", self.useful_life_months)
        if (
            not isinstance(self.residual_value_rate_bp, int)
            or isinstance(self.residual_value_rate_bp, bool)
            or not 0 <= self.residual_value_rate_bp <= 10_000
        ):
            raise StatementRollforwardError("INVALID_RESIDUAL_RATE", "固定资产残值率必须为0至10000之间的整数。")


@dataclass(frozen=True)
class StatementRollforwardRequest:
    """All I1B inputs.  Every amount remains a fictional integer-fen value."""

    operating_cycle: OperatingCycleResult
    cash_plan: CashPlanResult
    opening_state: StatementOpeningState
    income_tax_rate_bp: int
    asset_depreciation_policies: tuple[AssetDepreciationPolicy, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.income_tax_rate_bp, int)
            or isinstance(self.income_tax_rate_bp, bool)
            or not 0 <= self.income_tax_rate_bp <= 10_000
        ):
            raise StatementRollforwardError("INVALID_INCOME_TAX_RATE", "所得税率必须为0至10000之间的整数。")
        policy_ids = [item.event_id for item in self.asset_depreciation_policies]
        if len(policy_ids) != len(set(policy_ids)):
            raise StatementRollforwardError("DUPLICATE_ASSET_POLICY", "固定资产折旧政策事件编号不能重复。")


@dataclass(frozen=True)
class IncomeStatementRow:
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
class CashFlowStatementRow:
    month: str
    opening_bank_cash_fen: int
    cash_collections_fen: int
    supplier_cash_payments_fen: int
    payroll_cash_payment_fen: int
    rent_and_utilities_cash_payment_fen: int
    other_operating_cash_payment_fen: int
    tax_cash_payment_fen: int
    operating_net_cashflow_fen: int
    fixed_asset_cash_payment_fen: int
    investing_net_cashflow_fen: int
    financing_net_cashflow_fen: int
    ending_bank_cash_fen: int


@dataclass(frozen=True)
class BalanceSheetRow:
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
class StatementReconciliationRow:
    month: str
    balance_sheet_difference_fen: int
    cash_flow_to_i1_difference_fen: int
    debt_rollforward_difference_fen: int
    retained_earnings_rollforward_difference_fen: int
    unallocated_cash_budget_fen: int


@dataclass(frozen=True)
class StatementRollforwardResult:
    """Restricted internal result; never a public model-input table."""

    request: StatementRollforwardRequest
    resolved_opening_retained_earnings_fen: int
    income_statement_rows: tuple[IncomeStatementRow, ...]
    cash_flow_statement_rows: tuple[CashFlowStatementRow, ...]
    balance_sheet_rows: tuple[BalanceSheetRow, ...]
    reconciliation_rows: tuple[StatementReconciliationRow, ...]

    @property
    def is_fully_reconciled(self) -> bool:
        return all(
            row.balance_sheet_difference_fen == 0
            and row.cash_flow_to_i1_difference_fen == 0
            and row.debt_rollforward_difference_fen == 0
            and row.retained_earnings_rollforward_difference_fen == 0
            and row.unallocated_cash_budget_fen == 0
            for row in self.reconciliation_rows
        )


@dataclass
class _DepreciableAsset:
    asset_id: str
    remaining_depreciable_fen: int
    residual_fen: int
    remaining_months: int

    def depreciate_one_month(self) -> int:
        if self.remaining_depreciable_fen == 0:
            return 0
        amount = _round_up_div(self.remaining_depreciable_fen, self.remaining_months)
        self.remaining_depreciable_fen -= amount
        self.remaining_months -= 1
        return amount


def build_statement_rollforward(request: StatementRollforwardRequest) -> StatementRollforwardResult:
    """Create monthly restricted statements without creating any financing.

    A matching I1 cash plan is mandatory.  This makes the bank-cash path an
    input from the existing central planner rather than a second cash-planning
    implementation hidden inside the statements module.
    """

    operating = request.operating_cycle
    cash_plan = request.cash_plan
    _validate_inputs_align(operating, cash_plan, request.opening_state)
    policy_by_event = {item.event_id: item for item in request.asset_depreciation_policies}
    events_by_id = {item.event_id: item for item in operating.request.fixed_asset_events}
    _validate_asset_policies(events_by_id, policy_by_event, operating.request.end_date)

    opening_retained = request.opening_state.resolved_retained_earnings_fen()
    if request.opening_state.retained_earnings_fen is not None and _opening_balance_difference(
        request.opening_state, opening_retained
    ) != 0:
        raise StatementRollforwardError("OPENING_BALANCE_NOT_BALANCED", "已填写的期初资产负债表未满足资产等于负债加权益。")

    events_by_purchase_month: dict[str, list[FixedAssetEvent]] = {}
    events_by_commission_month: dict[str, list[FixedAssetEvent]] = {}
    for event in events_by_id.values():
        events_by_purchase_month.setdefault(event.purchase_month, []).append(event)
        if event.ready_for_use_date is not None:
            commission_month = _month_key(event.ready_for_use_date)
            if operating.monthly_rows[0].month <= commission_month <= operating.monthly_rows[-1].month:
                events_by_commission_month.setdefault(commission_month, []).append(event)

    active_assets: list[_DepreciableAsset] = []
    opening = request.opening_state
    if opening.net_fixed_assets_fen:
        active_assets.append(
            _DepreciableAsset(
                asset_id="opening_fixed_assets",
                remaining_depreciable_fen=opening.net_fixed_assets_fen - opening.fixed_asset_residual_fen,
                residual_fen=opening.fixed_asset_residual_fen,
                remaining_months=opening.fixed_asset_remaining_life_months or 1,
            )
        )

    current_gross_fixed_assets = opening.gross_fixed_assets_fen
    current_accumulated_depreciation = opening.accumulated_depreciation_fen
    current_construction = opening.construction_in_progress_fen
    current_retained = opening_retained
    current_cash = opening.bank_cash_fen
    current_income_tax_payable = 0
    current_bank_debt = opening.bank_debt_fen
    current_shareholder_debt = opening.shareholder_debt_fen
    current_interest_payable = opening.interest_payable_fen
    income_rows: list[IncomeStatementRow] = []
    cash_rows: list[CashFlowStatementRow] = []
    balance_rows: list[BalanceSheetRow] = []
    reconciliation_rows: list[StatementReconciliationRow] = []

    plan_rows_by_month = {row.month: row for row in cash_plan.monthly_rows}
    for operating_row in operating.monthly_rows:
        month = operating_row.month
        cash_plan_row = plan_rows_by_month[month]
        commissioned_assets = events_by_commission_month.get(month, [])
        commissioned_cost = sum(item.purchase_amount_fen for item in commissioned_assets)
        for event in commissioned_assets:
            policy = policy_by_event[event.event_id]
            residual = event.purchase_amount_fen * policy.residual_value_rate_bp // 10_000
            active_assets.append(
                _DepreciableAsset(
                    asset_id=event.event_id,
                    remaining_depreciable_fen=event.purchase_amount_fen - residual,
                    residual_fen=residual,
                    remaining_months=policy.useful_life_months,
                )
            )
        depreciation = sum(asset.depreciate_one_month() for asset in active_assets)
        acquisition = sum(item.purchase_amount_fen for item in events_by_purchase_month.get(month, []))
        current_construction += acquisition - commissioned_cost
        current_gross_fixed_assets += commissioned_cost
        current_accumulated_depreciation += depreciation
        if current_construction < 0 or current_accumulated_depreciation > current_gross_fixed_assets:
            raise StatementRollforwardError("INVALID_FIXED_ASSET_ROLLFORWARD", f"{month}固定资产滚动出现负在建工程或累计折旧超过原值。")

        gross_profit = operating_row.sales_confirmed_fen - operating_row.cost_of_goods_sold_fen
        operating_profit = (
            gross_profit
            - operating_row.payroll_expense_fen
            - operating_row.rent_and_utilities_expense_fen
            - operating_row.other_operating_expense_fen
            - depreciation
            - operating_row.tax_and_surcharge_accrual_fen
        )
        profit_before_income_tax = operating_profit
        income_tax = max(0, profit_before_income_tax * request.income_tax_rate_bp // 10_000)
        net_profit = profit_before_income_tax - income_tax
        current_retained += net_profit
        # I1A schedules only the simplified tax-and-surcharge cash payment.
        # I1B therefore keeps income tax as a separately accumulated payable
        # until the later authorised tax/financing lifecycle can schedule it.
        current_income_tax_payable += income_tax

        operating_net_cashflow = (
            operating_row.cash_collections_fen
            - operating_row.supplier_cash_payments_fen
            - operating_row.payroll_cash_payment_fen
            - operating_row.rent_and_utilities_cash_payment_fen
            - operating_row.other_operating_cash_payment_fen
            - operating_row.tax_cash_payment_fen
        )
        investing_net_cashflow = -operating_row.fixed_asset_cash_payment_fen
        statement_ending_cash = current_cash + operating_net_cashflow + investing_net_cashflow
        cash_flow_difference = statement_ending_cash - cash_plan_row.ending_balance_fen
        if cash_flow_difference != 0:
            raise StatementRollforwardError(
                "CASH_PLAN_MISMATCH",
                f"{month}现金流量表期末现金与I1现金计划不一致，差额为{cash_flow_difference}分。",
            )
        current_cash = cash_plan_row.ending_balance_fen

        income_row = IncomeStatementRow(
            month=month,
            revenue_fen=operating_row.sales_confirmed_fen,
            cost_of_goods_sold_fen=operating_row.cost_of_goods_sold_fen,
            gross_profit_fen=gross_profit,
            payroll_expense_fen=operating_row.payroll_expense_fen,
            rent_and_utilities_expense_fen=operating_row.rent_and_utilities_expense_fen,
            other_operating_expense_fen=operating_row.other_operating_expense_fen,
            depreciation_expense_fen=depreciation,
            tax_and_surcharge_expense_fen=operating_row.tax_and_surcharge_accrual_fen,
            operating_profit_fen=operating_profit,
            interest_expense_fen=0,
            profit_before_income_tax_fen=profit_before_income_tax,
            income_tax_expense_fen=income_tax,
            net_profit_fen=net_profit,
        )
        cash_row = CashFlowStatementRow(
            month=month,
            opening_bank_cash_fen=cash_plan_row.opening_balance_fen,
            cash_collections_fen=operating_row.cash_collections_fen,
            supplier_cash_payments_fen=operating_row.supplier_cash_payments_fen,
            payroll_cash_payment_fen=operating_row.payroll_cash_payment_fen,
            rent_and_utilities_cash_payment_fen=operating_row.rent_and_utilities_cash_payment_fen,
            other_operating_cash_payment_fen=operating_row.other_operating_cash_payment_fen,
            tax_cash_payment_fen=operating_row.tax_cash_payment_fen,
            operating_net_cashflow_fen=operating_net_cashflow,
            fixed_asset_cash_payment_fen=operating_row.fixed_asset_cash_payment_fen,
            investing_net_cashflow_fen=investing_net_cashflow,
            financing_net_cashflow_fen=0,
            ending_bank_cash_fen=current_cash,
        )
        balance_row = BalanceSheetRow(
            month=month,
            bank_cash_fen=current_cash,
            accounts_receivable_fen=operating_row.ending_accounts_receivable_fen,
            inventory_fen=operating_row.ending_inventory_fen,
            construction_in_progress_fen=current_construction,
            gross_fixed_assets_fen=current_gross_fixed_assets,
            accumulated_depreciation_fen=current_accumulated_depreciation,
            net_fixed_assets_fen=current_gross_fixed_assets - current_accumulated_depreciation,
            accounts_payable_fen=operating_row.ending_accounts_payable_fen,
            capital_expenditure_payable_fen=operating_row.ending_capital_expenditure_payable_fen,
            accrued_payroll_and_utilities_fen=operating_row.ending_accrued_payroll_and_utilities_fen,
            tax_payable_fen=operating_row.ending_tax_payable_fen + current_income_tax_payable,
            interest_payable_fen=current_interest_payable,
            dividend_payable_fen=opening.dividend_payable_fen,
            bank_debt_fen=current_bank_debt,
            shareholder_debt_fen=current_shareholder_debt,
            paid_in_capital_fen=opening.paid_in_capital_fen,
            retained_earnings_fen=current_retained,
            total_assets_fen=0,
            total_liabilities_and_equity_fen=0,
        )
        balance_row = _with_balance_totals(balance_row)
        reconciliation = StatementReconciliationRow(
            month=month,
            balance_sheet_difference_fen=(
                balance_row.total_assets_fen - balance_row.total_liabilities_and_equity_fen
            ),
            cash_flow_to_i1_difference_fen=cash_flow_difference,
            debt_rollforward_difference_fen=0,
            retained_earnings_rollforward_difference_fen=(
                balance_row.retained_earnings_fen - (current_retained - net_profit) - net_profit
            ),
            unallocated_cash_budget_fen=(
                cash_plan_row.unallocated_inflow_fen + cash_plan_row.unallocated_outflow_fen
            ),
        )
        if reconciliation.balance_sheet_difference_fen != 0:
            raise StatementRollforwardError(
                "BALANCE_SHEET_NOT_BALANCED",
                f"{month}资产负债表未对平，差额为{reconciliation.balance_sheet_difference_fen}分。",
            )
        income_rows.append(income_row)
        cash_rows.append(cash_row)
        balance_rows.append(balance_row)
        reconciliation_rows.append(reconciliation)

    return StatementRollforwardResult(
        request=request,
        resolved_opening_retained_earnings_fen=opening_retained,
        income_statement_rows=tuple(income_rows),
        cash_flow_statement_rows=tuple(cash_rows),
        balance_sheet_rows=tuple(balance_rows),
        reconciliation_rows=tuple(reconciliation_rows),
    )


def _validate_inputs_align(
    operating: OperatingCycleResult,
    cash_plan: CashPlanResult,
    opening: StatementOpeningState,
) -> None:
    if operating.request.synthetic_account_id != cash_plan.request.synthetic_account_id:
        raise StatementRollforwardError("ACCOUNT_ID_MISMATCH", "I1A经营计划与I1现金计划的虚构账户编号不一致。")
    if (
        operating.request.start_date != cash_plan.request.start_date
        or operating.request.end_date != cash_plan.request.end_date
    ):
        raise StatementRollforwardError("PERIOD_MISMATCH", "I1A经营计划与I1现金计划的规划期间不一致。")
    if opening.bank_cash_fen != cash_plan.monthly_rows[0].opening_balance_fen:
        raise StatementRollforwardError("OPENING_CASH_MISMATCH", "期初资产负债表银行现金必须等于I1现金计划期初余额。")
    if opening.accounts_payable_fen != operating.opening_state.accounts_payable_fen:
        raise StatementRollforwardError(
            "OPENING_PAYABLE_MISMATCH",
            "期初资产负债表应付账款必须与I1A期初经营状态的应付账款一致。",
        )
    operating_months = tuple(row.month for row in operating.monthly_rows)
    cash_months = tuple(row.month for row in cash_plan.monthly_rows)
    if operating_months != cash_months:
        raise StatementRollforwardError("MONTH_MISMATCH", "I1A经营计划与I1现金计划的月份序列不一致。")
    for operating_row, cash_row in zip(operating.monthly_rows, cash_plan.monthly_rows, strict=True):
        if cash_row.unallocated_inflow_fen or cash_row.unallocated_outflow_fen:
            raise StatementRollforwardError("UNALLOCATED_CASH_BUDGET", f"{operating_row.month}仍有未命名现金预算，不能进入I1B。")
        if cash_row.operating_inflow_fen != operating_row.cash_collections_fen:
            raise StatementRollforwardError("OPERATING_INFLOW_MISMATCH", f"{operating_row.month}I1经营流入与I1A回款不一致。")
        if cash_row.operating_outflow_fen != operating_row.operating_cash_outflow_fen:
            raise StatementRollforwardError("OPERATING_OUTFLOW_MISMATCH", f"{operating_row.month}I1经营流出与I1A付款不一致。")
        if cash_row.investing_inflow_fen != 0 or cash_row.investing_outflow_fen != operating_row.fixed_asset_cash_payment_fen:
            raise StatementRollforwardError("INVESTING_CASH_MISMATCH", f"{operating_row.month}I1投资现金与I1A设备工程付款不一致。")


def _validate_asset_policies(
    events_by_id: Mapping[str, FixedAssetEvent],
    policies: Mapping[str, AssetDepreciationPolicy],
    end_date: date,
) -> None:
    unknown = set(policies) - set(events_by_id)
    if unknown:
        raise StatementRollforwardError("UNKNOWN_ASSET_POLICY", "折旧政策包含不存在的固定资产事件编号。")
    for event_id, event in events_by_id.items():
        if event.ready_for_use_date is not None and event.ready_for_use_date < _month_start(event.purchase_month):
            raise StatementRollforwardError("INVALID_READY_FOR_USE_DATE", "固定资产达到可使用日期不能早于购置月份。")
        if event.ready_for_use_date is not None and event.ready_for_use_date <= end_date and event_id not in policies:
            raise StatementRollforwardError("MISSING_ASSET_POLICY", "规划期间内达到可使用状态的固定资产必须填写折旧政策。")


def _opening_balance_difference(opening: StatementOpeningState, retained: int) -> int:
    return opening.assets_before_retained_earnings_fen - (opening.liabilities_before_retained_earnings_fen + retained)


def _with_balance_totals(row: BalanceSheetRow) -> BalanceSheetRow:
    assets = (
        row.bank_cash_fen
        + row.accounts_receivable_fen
        + row.inventory_fen
        + row.construction_in_progress_fen
        + row.net_fixed_assets_fen
    )
    liabilities_and_equity = (
        row.accounts_payable_fen
        + row.capital_expenditure_payable_fen
        + row.accrued_payroll_and_utilities_fen
        + row.tax_payable_fen
        + row.interest_payable_fen
        + row.dividend_payable_fen
        + row.bank_debt_fen
        + row.shareholder_debt_fen
        + row.paid_in_capital_fen
        + row.retained_earnings_fen
    )
    return BalanceSheetRow(**{**row.__dict__, "total_assets_fen": assets, "total_liabilities_and_equity_fen": liabilities_and_equity})


def _require_nonnegative_fen(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise StatementRollforwardError("INVALID_MONEY", f"{name}必须为非负整数分。")


def _require_positive_months(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise StatementRollforwardError("INVALID_USEFUL_LIFE", f"{name}必须为正整数月。")


def _round_up_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _month_start(month: str) -> date:
    return date(int(month[:4]), int(month[5:]), 1)
