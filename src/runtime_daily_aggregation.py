"""R5 in-memory aggregation from R4 transactions to continuous daily totals.

This module deliberately has no calendar inference, business-step, target-date,
file, or database responsibility.  It provides the single reusable transaction
to daily aggregation boundary for later runtime cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable

from runtime_transaction_normalizer import NormalizationResult, NormalizedTransaction


CODE_AGGREGATED = "DA_AGGREGATED_001"
CODE_EMPTY_INPUT = "DA_EMPTY_INPUT_001"


@dataclass(frozen=True)
class DailyRecord:
    """An anonymous, derived account-day record without source text or amounts."""

    account_token: str
    calendar_date: str
    debit_total_cny: Decimal
    credit_total_cny: Decimal
    transaction_count: int
    closing_balance_cny: Decimal | None
    source_last_transaction_id: str | None
    reconciliation_status: str


@dataclass(frozen=True)
class DailyQualitySummary:
    transaction_date_count: int
    natural_date_count: int
    untrusted_date_count: int
    reconciliation_failure_count: int


@dataclass(frozen=True)
class DailyAggregationResult:
    code: str
    message_zh: str
    records: tuple[DailyRecord, ...]
    quality_summary: DailyQualitySummary
    write_count: int = 0


def aggregate_daily_records(
    normalized_input: NormalizationResult | Iterable[NormalizedTransaction],
) -> DailyAggregationResult:
    """Aggregate R4-normalized transactions into a continuous natural-date range.

    A R4 reconciliation failure contaminates the two dates joined by that
    failed adjacent-balance check.  Those dates remain in the result, but their
    closing balance is withheld as untrusted.  The function only creates frozen
    dataclasses in memory and never writes or mutates its input.
    """

    transactions, untrusted_dates, failure_count = _input_parts(normalized_input)
    if not transactions:
        return DailyAggregationResult(
            code=CODE_EMPTY_INPUT,
            message_zh="没有可聚合的标准逐笔记录；本步骤未写入文件或数据库。",
            records=(),
            quality_summary=DailyQualitySummary(0, 0, 0, failure_count),
        )

    by_date: dict[str, list[NormalizedTransaction]] = {}
    account_tokens = {transaction.account_token for transaction in transactions}
    if len(account_tokens) != 1:
        raise ValueError("R5日度聚合一次只接受一个账户的标准逐笔记录")
    account_token = next(iter(account_tokens))
    for transaction in transactions:
        by_date.setdefault(transaction.booking_date, []).append(transaction)

    start_date = min(_as_date(value) for value in by_date)
    end_date = max(_as_date(value) for value in by_date)
    records: list[DailyRecord] = []
    prior_trusted_balance: Decimal | None = None
    current_date = start_date
    while current_date <= end_date:
        calendar_date = current_date.isoformat()
        daily_transactions = by_date.get(calendar_date, [])
        if daily_transactions:
            record = _aggregate_transaction_date(
                account_token,
                calendar_date,
                daily_transactions,
                untrusted=calendar_date in untrusted_dates,
            )
            if record.reconciliation_status == "trusted":
                prior_trusted_balance = record.closing_balance_cny
        else:
            record = DailyRecord(
                account_token=account_token,
                calendar_date=calendar_date,
                debit_total_cny=Decimal("0"),
                credit_total_cny=Decimal("0"),
                transaction_count=0,
                closing_balance_cny=prior_trusted_balance,
                source_last_transaction_id=None,
                reconciliation_status=("trusted_carry_forward" if prior_trusted_balance is not None else "untrusted_carry_forward"),
            )
        records.append(record)
        current_date += timedelta(days=1)

    return DailyAggregationResult(
        code=CODE_AGGREGATED,
        message_zh="已按记账日汇总全部流水并补齐自然日；本步骤未推导银行日历，也未写入文件或数据库。",
        records=tuple(records),
        quality_summary=DailyQualitySummary(
            transaction_date_count=len(by_date),
            natural_date_count=len(records),
            untrusted_date_count=sum(1 for record in records if record.reconciliation_status == "untrusted"),
            reconciliation_failure_count=failure_count,
        ),
    )


def _input_parts(
    normalized_input: NormalizationResult | Iterable[NormalizedTransaction],
) -> tuple[tuple[NormalizedTransaction, ...], frozenset[str], int]:
    if isinstance(normalized_input, NormalizationResult):
        untrusted_dates: set[str] = set()
        for failure in normalized_input.reconciliation.failure_examples:
            previous = next(
                (transaction for transaction in normalized_input.transactions if transaction.source_row_number == failure.previous_source_row_number),
                None,
            )
            if previous is not None:
                untrusted_dates.add(previous.booking_date)
            untrusted_dates.add(failure.booking_date)
        return (
            normalized_input.transactions,
            frozenset(untrusted_dates),
            normalized_input.reconciliation.failures,
        )
    return tuple(normalized_input), frozenset(), 0


def _aggregate_transaction_date(
    account_token: str,
    calendar_date: str,
    transactions: list[NormalizedTransaction],
    *,
    untrusted: bool,
) -> DailyRecord:
    if untrusted:
        closing_balance = None
        last_transaction_id = None
        status = "untrusted"
    else:
        last_transaction = transactions[-1]
        closing_balance = last_transaction.post_transaction_balance_cny
        last_transaction_id = last_transaction.transaction_id
        status = "trusted"
    return DailyRecord(
        account_token=account_token,
        calendar_date=calendar_date,
        debit_total_cny=sum((transaction.debit_amount_cny for transaction in transactions), Decimal("0")),
        credit_total_cny=sum((transaction.credit_amount_cny for transaction in transactions), Decimal("0")),
        transaction_count=len(transactions),
        closing_balance_cny=closing_balance,
        source_last_transaction_id=last_transaction_id,
        reconciliation_status=status,
    )


def _as_date(value: str) -> date:
    return date.fromisoformat(value)
