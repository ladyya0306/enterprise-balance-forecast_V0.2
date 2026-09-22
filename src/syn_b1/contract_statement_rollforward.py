"""SYN-B1-I3C restricted post-contract statement projection.

This additive module preserves the frozen I1B source snapshot.  It owns only
the translation of already declared I3 contracts into post-contract statement
rows; it does not select or execute financing, or own the central cash rule.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from syn_b1.cash_plan import FinalNamedCashPlanResult
from syn_b1.statement_rollforward import (
    BalanceSheetRow, CashFlowStatementRow, IncomeStatementRow,
    StatementReconciliationRow, StatementRollforwardError, StatementRollforwardResult,
)


@dataclass(frozen=True)
class ContractFinancingMonthlyAdjustment:
    month: str
    bank_loan_drawdown_fen: int = 0
    shareholder_loan_drawdown_fen: int = 0
    shareholder_capital_injection_fen: int = 0
    bank_principal_repayment_fen: int = 0
    shareholder_principal_repayment_fen: int = 0
    interest_cash_payment_fen: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.month, str) or len(self.month) != 7 or self.month[4] != "-":
            raise StatementRollforwardError("INVALID_FINANCING_MONTH", "资金合同月度汇总必须使用YYYY-MM格式。")
        try:
            year, month = int(self.month[:4]), int(self.month[5:])
        except ValueError as error:
            raise StatementRollforwardError("INVALID_FINANCING_MONTH", "资金合同月度汇总必须使用YYYY-MM格式。") from error
        if year < 1 or not 1 <= month <= 12:
            raise StatementRollforwardError("INVALID_FINANCING_MONTH", "资金合同月度汇总月份无效。")
        for name, value in self.__dict__.items():
            if name != "month" and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise StatementRollforwardError("INVALID_MONEY", f"{name}必须为非负整数分。")

    @property
    def financing_net_cashflow_fen(self) -> int:
        return (
            self.bank_loan_drawdown_fen + self.shareholder_loan_drawdown_fen
            + self.shareholder_capital_injection_fen - self.bank_principal_repayment_fen
            - self.shareholder_principal_repayment_fen - self.interest_cash_payment_fen
        )


@dataclass(frozen=True)
class ContractFinancedStatementResult:
    """Restricted monthly post-contract statements, not model inputs."""

    pre_financing_statements: StatementRollforwardResult
    final_named_plan: FinalNamedCashPlanResult
    adjustments: tuple[ContractFinancingMonthlyAdjustment, ...]
    income_statement_rows: tuple[IncomeStatementRow, ...]
    cash_flow_statement_rows: tuple[CashFlowStatementRow, ...]
    balance_sheet_rows: tuple[BalanceSheetRow, ...]
    reconciliation_rows: tuple[StatementReconciliationRow, ...]

    @property
    def is_fully_reconciled(self) -> bool:
        return all(
            row.balance_sheet_difference_fen == 0 and row.cash_flow_to_i1_difference_fen == 0
            and row.debt_rollforward_difference_fen == 0 and row.retained_earnings_rollforward_difference_fen == 0
            and row.unallocated_cash_budget_fen == 0 for row in self.reconciliation_rows
        )


def build_contract_financed_statement_rollforward(
    *, pre_financing_statements: StatementRollforwardResult,
    final_named_plan: FinalNamedCashPlanResult,
    adjustments: tuple[ContractFinancingMonthlyAdjustment, ...],
) -> ContractFinancedStatementResult:
    """Reconcile user-declared monthly contract cash into three statements.

    It is a monthly projection, not a daily cash-sufficiency decision: I3B
    remains the sole owner of date-ordered execution.
    """
    if not pre_financing_statements.is_fully_reconciled:
        raise StatementRollforwardError("PRE_FINANCING_STATEMENTS_NOT_RECONCILED", "融资前三张表未对平，不能接入资金合同。")
    months = tuple(row.month for row in pre_financing_statements.income_statement_rows)
    if months != tuple(row.month for row in final_named_plan.monthly_rows) or not months:
        raise StatementRollforwardError("FINAL_PLAN_MONTH_MISMATCH", "最终命名计划与融资前三张表月份不一致。")
    adjustment_by_month = {item.month: item for item in adjustments}
    if len(adjustment_by_month) != len(adjustments) or set(adjustment_by_month) != set(months):
        raise StatementRollforwardError("FINANCING_ADJUSTMENT_MONTH_MISMATCH", "资金合同月度汇总必须与规划月份一一对应。")
    opening = pre_financing_statements.request.opening_state
    if final_named_plan.monthly_rows[0].opening_balance_fen != opening.bank_cash_fen:
        raise StatementRollforwardError("FINAL_PLAN_OPENING_CASH_MISMATCH", "最终计划期初余额与融资前三张表不一致。")
    base_income = {row.month: row for row in pre_financing_statements.income_statement_rows}
    base_cash = {row.month: row for row in pre_financing_statements.cash_flow_statement_rows}
    base_balance = {row.month: row for row in pre_financing_statements.balance_sheet_rows}
    tax_rate = pre_financing_statements.request.income_tax_rate_bp
    bank_debt, shareholder_debt = opening.bank_debt_fen, opening.shareholder_debt_fen
    cumulative_interest = 0
    cumulative_income_tax_adjustment = 0
    cumulative_capital = 0
    income_rows: list[IncomeStatementRow] = []
    cash_rows: list[CashFlowStatementRow] = []
    balance_rows: list[BalanceSheetRow] = []
    reconciliation_rows: list[StatementReconciliationRow] = []
    for final_row in final_named_plan.monthly_rows:
        month, adjustment = final_row.month, adjustment_by_month[final_row.month]
        prior_bank_debt, prior_shareholder_debt = bank_debt, shareholder_debt
        prior_retained = (
            pre_financing_statements.resolved_opening_retained_earnings_fen
            if not balance_rows
            else balance_rows[-1].retained_earnings_fen
        )
        base_income_row = base_income[month]
        base_cash_row = base_cash[month]
        base_balance_row = base_balance[month]
        profit_before_tax = base_income_row.operating_profit_fen - adjustment.interest_cash_payment_fen
        income_tax = max(0, profit_before_tax * tax_rate // 10_000)
        net_profit = profit_before_tax - income_tax
        cumulative_interest += adjustment.interest_cash_payment_fen
        cumulative_income_tax_adjustment += income_tax - base_income_row.income_tax_expense_fen
        cumulative_capital += adjustment.shareholder_capital_injection_fen
        retained = (
            base_balance_row.retained_earnings_fen
            - cumulative_interest
            - cumulative_income_tax_adjustment
        )
        tax_payable = base_balance_row.tax_payable_fen + cumulative_income_tax_adjustment
        paid_capital = base_balance_row.paid_in_capital_fen + cumulative_capital
        bank_debt += adjustment.bank_loan_drawdown_fen - adjustment.bank_principal_repayment_fen
        shareholder_debt += adjustment.shareholder_loan_drawdown_fen - adjustment.shareholder_principal_repayment_fen
        if bank_debt < 0 or shareholder_debt < 0:
            raise StatementRollforwardError("NEGATIVE_DEBT_AFTER_REPAYMENT", f"{month}合同还本超过对应未偿本金。")
        income_row = replace(
            base_income_row,
            interest_expense_fen=adjustment.interest_cash_payment_fen,
            profit_before_income_tax_fen=profit_before_tax,
            income_tax_expense_fen=income_tax,
            net_profit_fen=net_profit,
        )
        cash_row = replace(
            base_cash_row,
            opening_bank_cash_fen=final_row.opening_balance_fen,
            financing_net_cashflow_fen=adjustment.financing_net_cashflow_fen,
            ending_bank_cash_fen=final_row.ending_balance_fen,
        )
        cash_difference = (
            cash_row.opening_bank_cash_fen
            + cash_row.operating_net_cashflow_fen
            + cash_row.investing_net_cashflow_fen
            + cash_row.financing_net_cashflow_fen
            - cash_row.ending_bank_cash_fen
        )
        if cash_difference:
            raise StatementRollforwardError("FINAL_CASH_PLAN_MISMATCH", f"{month}资金合同与I1最终现金计划不一致，差额为{cash_difference}分。")
        balance_row = _with_totals(
            replace(
                base_balance_row,
                bank_cash_fen=final_row.ending_balance_fen,
                tax_payable_fen=tax_payable,
                bank_debt_fen=bank_debt,
                shareholder_debt_fen=shareholder_debt,
                paid_in_capital_fen=paid_capital,
                retained_earnings_fen=retained,
                total_assets_fen=0,
                total_liabilities_and_equity_fen=0,
            )
        )
        reconciliation = StatementReconciliationRow(
            month=month,
            balance_sheet_difference_fen=balance_row.total_assets_fen - balance_row.total_liabilities_and_equity_fen,
            cash_flow_to_i1_difference_fen=cash_difference,
            debt_rollforward_difference_fen=(
                bank_debt
                - prior_bank_debt
                - adjustment.bank_loan_drawdown_fen
                + adjustment.bank_principal_repayment_fen
                + shareholder_debt
                - prior_shareholder_debt
                - adjustment.shareholder_loan_drawdown_fen
                + adjustment.shareholder_principal_repayment_fen
            ),
            retained_earnings_rollforward_difference_fen=retained - prior_retained - net_profit,
            unallocated_cash_budget_fen=0,
        )
        if reconciliation.balance_sheet_difference_fen or reconciliation.debt_rollforward_difference_fen or reconciliation.retained_earnings_rollforward_difference_fen:
            raise StatementRollforwardError("POST_CONTRACT_STATEMENTS_NOT_RECONCILED", f"{month}资金合同后三张表未对平。")
        income_rows.append(income_row)
        cash_rows.append(cash_row)
        balance_rows.append(balance_row)
        reconciliation_rows.append(reconciliation)
    return ContractFinancedStatementResult(
        pre_financing_statements,
        final_named_plan,
        adjustments,
        tuple(income_rows),
        tuple(cash_rows),
        tuple(balance_rows),
        tuple(reconciliation_rows),
    )


def _with_totals(row: BalanceSheetRow) -> BalanceSheetRow:
    assets = (
        row.bank_cash_fen
        + row.accounts_receivable_fen
        + row.inventory_fen
        + row.construction_in_progress_fen
        + row.net_fixed_assets_fen
    )
    liabilities_equity = (
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
    return replace(row, total_assets_fen=assets, total_liabilities_and_equity_fen=liabilities_equity)
