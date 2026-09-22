"""One restricted bridge from the final statement plan to the posted account ledger.

This module introduces no cash-flow calculation.  It only proves that the
already calculated enterprise plan, account routing and final posted ledger
describe the same ending cash position.
"""

from __future__ import annotations

from dataclasses import dataclass

from syn_b1.ledger_rollforward import LedgerRollforwardResult
from syn_b1.tax_routing import CashDirection, TaxRoutingResult


@dataclass(frozen=True)
class FinalCashReconciliation:
    """Restricted final-cash evidence; never a model input or public CSV field."""

    execution_status: str
    enterprise_opening_cash_fen: int
    enterprise_ending_cash_fen: int
    observed_opening_cash_fen: int
    observed_source_inflow_fen: int
    observed_source_outflow_fen: int
    observed_expected_ending_cash_fen: int
    observed_ledger_ending_cash_fen: int
    peripheral_net_cash_fen: int
    enterprise_to_route_difference_fen: int
    observed_source_to_ledger_difference_fen: int
    primary_account_statement_to_ledger_difference_fen: int | None

    @property
    def is_reconciled(self) -> bool:
        """Complete runs must close the enterprise-to-route-to-ledger chain exactly."""

        return (
            self.execution_status == "complete"
            and self.enterprise_to_route_difference_fen == 0
            and self.observed_source_to_ledger_difference_fen == 0
            and (
                self.primary_account_statement_to_ledger_difference_fen is None
                or self.primary_account_statement_to_ledger_difference_fen == 0
            )
        )


def build_final_cash_reconciliation(
    tax_routing: TaxRoutingResult,
    ledger: LedgerRollforwardResult,
) -> FinalCashReconciliation:
    """Reconcile the existing final enterprise plan and routed account cash.

    A primary account with every category routed 100% must equal the enterprise
    statement cash directly.  A secondary account is still valid when the
    difference is exactly the net cash deliberately routed through peripheral
    accounts; the bridge makes that distinction explicit rather than hiding it.
    """

    plan = tax_routing.tax_adjusted_final_named_plan
    enterprise_opening = plan.monthly_rows[0].opening_balance_fen
    enterprise_ending = plan.monthly_rows[-1].ending_balance_fen
    observed_inflow = sum(
        item.observed_amount_fen
        for item in tax_routing.routed_cash_items
        if item.source_item.direction is CashDirection.INFLOW
    )
    observed_outflow = sum(
        item.observed_amount_fen
        for item in tax_routing.routed_cash_items
        if item.source_item.direction is CashDirection.OUTFLOW
    )
    peripheral_net = sum(
        item.peripheral_amount_fen if item.source_item.direction is CashDirection.INFLOW else -item.peripheral_amount_fen
        for item in tax_routing.routed_cash_items
    )
    observed_opening = ledger.opening_balance_fen
    observed_expected = observed_opening + observed_inflow - observed_outflow
    observed_ledger = ledger.closing_balance_fen
    enterprise_to_route = enterprise_ending - observed_expected - peripheral_net
    source_to_ledger = observed_expected - observed_ledger
    all_cash_observed = all(item.peripheral_amount_fen == 0 for item in tax_routing.routed_cash_items)
    primary_difference = enterprise_ending - observed_ledger if all_cash_observed else None
    return FinalCashReconciliation(
        execution_status="complete" if ledger.is_complete else "stopped_before_blocked_outflow",
        enterprise_opening_cash_fen=enterprise_opening,
        enterprise_ending_cash_fen=enterprise_ending,
        observed_opening_cash_fen=observed_opening,
        observed_source_inflow_fen=observed_inflow,
        observed_source_outflow_fen=observed_outflow,
        observed_expected_ending_cash_fen=observed_expected,
        observed_ledger_ending_cash_fen=observed_ledger,
        peripheral_net_cash_fen=peripheral_net,
        enterprise_to_route_difference_fen=enterprise_to_route,
        observed_source_to_ledger_difference_fen=source_to_ledger,
        primary_account_statement_to_ledger_difference_fen=primary_difference,
    )
