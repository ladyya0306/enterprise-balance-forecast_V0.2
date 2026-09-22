"""Additive tax replacement adapter over the frozen I1/I1B/I3C engines.

The adapter changes only the temporary combined-tax accrual/payment inputs,
then calls the existing cash planner, statement rollforward, and post-contract
statement rollforward.  It therefore does not own a second profit, income-tax,
financing, or cash identity formula.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from syn_b1.cash_plan import (
    CashComponent,
    CashFlowClass,
    Direction,
    FinalNamedCashPlanResult,
    PlanningMode,
    build_final_named_cash_plan,
    build_monthly_cash_plan,
)
from syn_b1.contract_statement_rollforward import (
    ContractFinancedStatementResult,
    build_contract_financed_statement_rollforward,
)
from syn_b1.operating_cycle import OperatingCycleResult
from syn_b1.statement_rollforward import build_statement_rollforward


class TaxProjectionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def build_tax_replaced_contract_statements(
    *,
    base: ContractFinancedStatementResult,
    final_named_plan: FinalNamedCashPlanResult,
    tax_and_surcharge_accruals_fen: Mapping[str, int],
    tax_cash_payments_fen: Mapping[str, int],
    income_tax_cash_payments_fen: Mapping[str, int],
    opening_tax_payable_fen: int,
) -> ContractFinancedStatementResult:
    """Replace I1A's temporary tax inputs and delegate all formulas."""
    if not isinstance(opening_tax_payable_fen, int) or isinstance(opening_tax_payable_fen, bool) or opening_tax_payable_fen < 0:
        raise TaxProjectionError("INVALID_OPENING_TAX_PAYABLE", "期初应付税费必须为非负整数分。")
    pre = base.pre_financing_statements
    months = tuple(row.month for row in pre.income_statement_rows)
    accruals = _validated_month_values("tax_accrual", months, tax_and_surcharge_accruals_fen)
    payments = _validated_month_values("tax_payment", months, tax_cash_payments_fen)
    income_tax_payments = _validated_month_values(
        "income_tax_payment", months, income_tax_cash_payments_fen
    )
    operating = _tax_replaced_operating_cycle(
        pre.request.operating_cycle,
        accruals,
        payments,
        opening_tax_payable_fen,
    )
    cash_plan = _tax_replaced_cash_plan(pre.request.cash_plan, payments)
    opening_state = replace(pre.request.opening_state, tax_payable_fen=opening_tax_payable_fen)
    pre_tax_replaced = build_statement_rollforward(replace(
        pre.request,
        operating_cycle=operating,
        cash_plan=cash_plan,
        opening_state=opening_state,
    ))
    intermediate_plan = _without_income_tax_payment_components(
        final_named_plan,
        base,
        income_tax_payments,
    )
    intermediate = build_contract_financed_statement_rollforward(
        pre_financing_statements=pre_tax_replaced,
        final_named_plan=intermediate_plan,
        adjustments=base.adjustments,
    )
    return _apply_income_tax_payment_delta(
        intermediate,
        final_named_plan,
        income_tax_payments,
    )


def _tax_replaced_operating_cycle(
    operating: OperatingCycleResult,
    accruals: Mapping[str, int],
    payments: Mapping[str, int],
    opening_tax_payable_fen: int,
) -> OperatingCycleResult:
    current = opening_tax_payable_fen
    rows = []
    for row in operating.monthly_rows:
        accrual = accruals[row.month]
        payment = payments[row.month]
        ending = current + accrual - payment
        if ending < 0:
            raise TaxProjectionError(
                "PROJECTED_TAX_PAYMENT_EXCEEDS_PAYABLE",
                f"{row.month}计划税费支付超过期初未缴与本期计提合计。",
            )
        rows.append(replace(
            row,
            tax_and_surcharge_accrual_fen=accrual,
            tax_cash_payment_fen=payment,
            opening_tax_payable_fen=current,
            ending_tax_payable_fen=ending,
        ))
        current = ending
    return replace(
        operating,
        opening_state=replace(operating.opening_state, tax_payable_fen=opening_tax_payable_fen),
        monthly_rows=tuple(rows),
    )


def _tax_replaced_cash_plan(base_plan, payments: Mapping[str, int]):
    total_payment = sum(payments.values())
    if total_payment == 0:
        return base_plan
    component = CashComponent(
        component_id="i4a1_replacement_tax_payment",
        cash_flow_class=CashFlowClass.OPERATING,
        direction=Direction.OUTFLOW,
        monthly_amounts_fen=dict(payments),
    )
    request = base_plan.request
    changes = {"components": (*request.components, component)}
    if request.planning_mode is PlanningMode.FIXED_BANK_TOTALS:
        assert request.fixed_total_outflow_fen is not None
        changes["fixed_total_outflow_fen"] = request.fixed_total_outflow_fen + total_payment
    return build_monthly_cash_plan(replace(request, **changes))


def _without_income_tax_payment_components(
    final_plan: FinalNamedCashPlanResult,
    base: ContractFinancedStatementResult,
    income_tax_payments: Mapping[str, int],
) -> FinalNamedCashPlanResult:
    removed_components = tuple(
        item for item in final_plan.components
        if item.component_id.startswith("planned_tax:corporate_income_tax:")
    )
    components = tuple(
        item for item in final_plan.components
        if not item.component_id.startswith("planned_tax:corporate_income_tax:")
    )
    removed = sum(item.amount_fen for item in removed_components)
    if removed != sum(income_tax_payments.values()):
        raise TaxProjectionError(
            "INCOME_TAX_COMPONENT_MISMATCH",
            "最终计划中的所得税付款项目与逐月所得税支付合计不一致。",
        )
    request = base.pre_financing_statements.request.operating_cycle.request
    return build_final_named_cash_plan(
        start_date=request.start_date,
        end_date=request.end_date,
        opening_balance_fen=final_plan.monthly_rows[0].opening_balance_fen,
        components=components,
    )


def _apply_income_tax_payment_delta(
    intermediate: ContractFinancedStatementResult,
    final_plan: FinalNamedCashPlanResult,
    payments: Mapping[str, int],
) -> ContractFinancedStatementResult:
    final_by_month = {row.month: row for row in final_plan.monthly_rows}
    cumulative_payment = 0
    cash_rows = []
    balance_rows = []
    reconciliation_rows = []
    for cash_row, balance_row, reconciliation in zip(
        intermediate.cash_flow_statement_rows,
        intermediate.balance_sheet_rows,
        intermediate.reconciliation_rows,
        strict=True,
    ):
        month = cash_row.month
        payment = payments[month]
        cumulative_payment += payment
        final_row = final_by_month[month]
        changed_cash = replace(
            cash_row,
            opening_bank_cash_fen=final_row.opening_balance_fen,
            tax_cash_payment_fen=cash_row.tax_cash_payment_fen + payment,
            operating_net_cashflow_fen=cash_row.operating_net_cashflow_fen - payment,
            ending_bank_cash_fen=final_row.ending_balance_fen,
        )
        cash_difference = (
            changed_cash.opening_bank_cash_fen
            + changed_cash.operating_net_cashflow_fen
            + changed_cash.investing_net_cashflow_fen
            + changed_cash.financing_net_cashflow_fen
            - changed_cash.ending_bank_cash_fen
        )
        ending_tax_payable = balance_row.tax_payable_fen - cumulative_payment
        if ending_tax_payable < 0:
            raise TaxProjectionError(
                "INCOME_TAX_PAYMENT_EXCEEDS_TOTAL_PAYABLE",
                f"{month}所得税计划支付超过累计应付税费。",
            )
        changed_balance = replace(
            balance_row,
            bank_cash_fen=final_row.ending_balance_fen,
            tax_payable_fen=ending_tax_payable,
            total_assets_fen=balance_row.total_assets_fen - cumulative_payment,
            total_liabilities_and_equity_fen=(
                balance_row.total_liabilities_and_equity_fen - cumulative_payment
            ),
        )
        changed_reconciliation = replace(
            reconciliation,
            cash_flow_to_i1_difference_fen=cash_difference,
            balance_sheet_difference_fen=(
                changed_balance.total_assets_fen - changed_balance.total_liabilities_and_equity_fen
            ),
        )
        if changed_reconciliation.cash_flow_to_i1_difference_fen or changed_reconciliation.balance_sheet_difference_fen:
            raise TaxProjectionError(
                "INCOME_TAX_PAYMENT_DELTA_NOT_RECONCILED",
                f"{month}所得税付款接回现金计划和资产负债表后未对平。",
            )
        cash_rows.append(changed_cash)
        balance_rows.append(changed_balance)
        reconciliation_rows.append(changed_reconciliation)
    return replace(
        intermediate,
        final_named_plan=final_plan,
        cash_flow_statement_rows=tuple(cash_rows),
        balance_sheet_rows=tuple(balance_rows),
        reconciliation_rows=tuple(reconciliation_rows),
    )


def _validated_month_values(
    label: str,
    months: tuple[str, ...],
    values: Mapping[str, int],
) -> dict[str, int]:
    if set(values) != set(months):
        raise TaxProjectionError("TAX_PROJECTION_MONTH_MISMATCH", "替换税费必须逐月完整提供，不能缺月或多月。")
    result = dict(values)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in result.values()):
        raise TaxProjectionError("INVALID_TAX_PROJECTION_MONEY", f"{label}必须为非负整数分。")
    return result
