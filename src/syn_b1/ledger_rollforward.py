"""SYN-B1-I4B: post the payable candidate prefix and produce review artifacts.

I4A-2 owns the plan.  This module owns only the current SYN-B1 balance
recurrence, daily roll-up, and explicitly requested temporary review export.
It intentionally does not import the sealed legacy ``synthetic_ledger``.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from syn_b1.generation_stop import CashExhaustionFactSheet
from syn_b1.transaction_planner import TransactionCandidate, TransactionPlanningResult


class LedgerRollforwardError(ValueError):
    """An I4B posting or temporary-review export would violate its contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LedgerRunManifest:
    run_id: str
    account_token: str
    ledger_version: str = "syn_b1_i4b_ledger_review_1_0"

    def __post_init__(self) -> None:
        if not self.run_id or not self.account_token or not self.ledger_version:
            raise LedgerRollforwardError("EMPTY_LEDGER_MANIFEST", "临时复核运行编号、账户代号和版本不能为空。")


@dataclass(frozen=True)
class LedgerTransaction:
    transaction_id: str
    sequence_no: int
    booking_datetime: datetime
    debit_fen: int
    credit_fen: int
    post_transaction_balance_fen: int

    def __post_init__(self) -> None:
        if not self.transaction_id:
            raise LedgerRollforwardError("EMPTY_LEDGER_TRANSACTION_ID", "账本交易编号不能为空。")
        if not isinstance(self.sequence_no, int) or isinstance(self.sequence_no, bool) or self.sequence_no <= 0:
            raise LedgerRollforwardError("INVALID_LEDGER_SEQUENCE", "账本交易顺序必须从1连续编号。")
        if not isinstance(self.booking_datetime, datetime):
            raise LedgerRollforwardError("INVALID_LEDGER_BOOKING_TIME", "账本交易必须提供日期时间。")
        if not isinstance(self.debit_fen, int) or not isinstance(self.credit_fen, int):
            raise LedgerRollforwardError("INVALID_LEDGER_AMOUNT", "账本金额必须为整数分。")
        if (self.debit_fen <= 0) == (self.credit_fen <= 0):
            raise LedgerRollforwardError("INVALID_LEDGER_DIRECTION", "每笔账本交易必须且只能有一个正数方向。")
        if not isinstance(self.post_transaction_balance_fen, int) or self.post_transaction_balance_fen < 0:
            raise LedgerRollforwardError("INVALID_POST_BALANCE", "交易后余额必须为非负整数分。")


@dataclass(frozen=True)
class DailyReviewRow:
    calendar_date: date
    inflow_fen: int
    outflow_fen: int
    transaction_count: int
    ending_balance_fen: int
    execution_status: str

    def __post_init__(self) -> None:
        if self.execution_status not in {"complete", "stopped_before_blocked_outflow"}:
            raise LedgerRollforwardError("INVALID_EXECUTION_STATUS", "日度执行状态无效。")
        if any(
            not isinstance(value, int) or value < 0
            for value in (self.inflow_fen, self.outflow_fen, self.transaction_count, self.ending_balance_fen)
        ):
            raise LedgerRollforwardError("INVALID_DAILY_VALUE", "日度金额、笔数和余额必须为非负整数。")


@dataclass(frozen=True)
class LedgerRollforwardResult:
    manifest: LedgerRunManifest
    opening_balance_fen: int
    transactions: tuple[LedgerTransaction, ...]
    daily_rows: tuple[DailyReviewRow, ...]
    is_complete: bool
    fact_sheet: CashExhaustionFactSheet | None

    @property
    def closing_balance_fen(self) -> int:
        return self.transactions[-1].post_transaction_balance_fen if self.transactions else self.opening_balance_fen

    def model_visible_rows(self) -> tuple[dict[str, str], ...]:
        """Keep audit identifiers and all scenario reason information out of model rows."""
        return tuple({
            "booking_datetime": item.booking_datetime.isoformat(timespec="seconds"),
            "debit_cny": _cny(item.debit_fen),
            "credit_cny": _cny(item.credit_fen),
            "post_transaction_balance_cny": _cny(item.post_transaction_balance_fen),
        } for item in self.transactions)


@dataclass(frozen=True)
class TemporaryReviewBundle:
    output_dir: Path
    transactions_csv: Path
    daily_csv: Path
    fact_sheet_json: Path


def build_ledger_rollforward(
    planning_result: TransactionPlanningResult,
    manifest: LedgerRunManifest,
) -> LedgerRollforwardResult:
    """Post exactly the I4A-2 payable prefix and recalculate all balances."""
    if not planning_result.is_amount_reconciled:
        raise LedgerRollforwardError("CANDIDATE_AMOUNT_NOT_RECONCILED", "逐笔候选金额未勾稽，不能进入账本。")
    opening = planning_result.request.tax_routing_result.request.observed_account.opening_balance_fen
    if not isinstance(opening, int) or opening < 0:
        raise LedgerRollforwardError("INVALID_OPENING_BALANCE", "观察账户期初余额必须为非负整数分。")
    candidates = planning_result.candidates
    completed_ids = tuple(item.movement_id for item in planning_result.cash_preview.completed_movements)
    expected_ids = tuple(item.candidate_id for item in candidates[:len(completed_ids)])
    if completed_ids != expected_ids:
        raise LedgerRollforwardError("NON_PREFIX_COMPLETION", "停止控制器返回的可支付事项必须是候选列表的连续前缀。")
    candidate_by_id = {item.candidate_id: item for item in candidates}
    transactions: list[LedgerTransaction] = []
    balance = opening
    for sequence_no, movement_id in enumerate(completed_ids, start=1):
        candidate = candidate_by_id.get(movement_id)
        if candidate is None:
            raise LedgerRollforwardError("UNKNOWN_COMPLETED_CANDIDATE", "停止控制器返回了不属于候选计划的事项。")
        _assert_candidate_matches_preview(candidate, planning_result)
        debit = candidate.amount_fen if candidate.direction.value == "outflow" else 0
        credit = candidate.amount_fen if candidate.direction.value == "inflow" else 0
        next_balance = balance - debit + credit
        if next_balance < 0:
            raise LedgerRollforwardError("NEGATIVE_POST_BALANCE", "可支付前缀出现负余额，不能记账。")
        transactions.append(LedgerTransaction(
            transaction_id=_review_transaction_id(manifest, sequence_no, candidate.candidate_id),
            sequence_no=sequence_no,
            booking_datetime=candidate.booking_datetime,
            debit_fen=debit,
            credit_fen=credit,
            post_transaction_balance_fen=next_balance,
        ))
        balance = next_balance
    if balance != planning_result.cash_preview.closing_balance_fen:
        raise LedgerRollforwardError("PREVIEW_BALANCE_MISMATCH", "账本期末余额与停止预览余额不一致。")
    _reconcile_transactions(opening, transactions)
    daily_rows = _daily_rollup(opening, transactions, planning_result)
    _reconcile_daily(transactions, daily_rows, balance)
    return LedgerRollforwardResult(
        manifest=manifest,
        opening_balance_fen=opening,
        transactions=tuple(transactions),
        daily_rows=daily_rows,
        is_complete=planning_result.cash_preview.is_complete,
        fact_sheet=planning_result.cash_preview.fact_sheet,
    )


def write_temporary_review_bundle(
    result: LedgerRollforwardResult,
    output_dir: str | Path,
) -> TemporaryReviewBundle:
    """Write an explicitly requested, non-overwriting local review bundle.

    The CSV files are inspection-only.  No source/category/contract/tax reason
    is written, so the bundle cannot accidentally become a hidden-truth export.
    """
    target = Path(output_dir)
    if target.exists():
        raise LedgerRollforwardError("REVIEW_OUTPUT_ALREADY_EXISTS", "临时复核目录已存在，不能覆盖历史结果。")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.writing-{uuid4().hex}"
    temporary.mkdir()
    try:
        transactions_path = temporary / "transactions_review.csv"
        daily_path = temporary / "account_daily_review.csv"
        fact_path = temporary / "generation_fact_sheet.json"
        _write_transactions_csv(transactions_path, result)
        _write_daily_csv(daily_path, result)
        _write_fact_sheet(fact_path, result)
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return TemporaryReviewBundle(
        output_dir=target,
        transactions_csv=target / "transactions_review.csv",
        daily_csv=target / "account_daily_review.csv",
        fact_sheet_json=target / "generation_fact_sheet.json",
    )


def _assert_candidate_matches_preview(candidate: TransactionCandidate, planning_result: TransactionPlanningResult) -> None:
    movement = next(item for item in planning_result.cash_preview.completed_movements if item.movement_id == candidate.candidate_id)
    if (
        movement.booking_datetime != candidate.booking_datetime
        or movement.sequence_no != candidate.sequence_no
        or movement.amount_fen != candidate.amount_fen
        or movement.direction.value != candidate.direction.value
    ):
        raise LedgerRollforwardError("PREVIEW_CANDIDATE_MISMATCH", "候选事项与停止预览内容不一致，不能记账。")


def _reconcile_transactions(opening: int, transactions: list[LedgerTransaction]) -> None:
    balance = opening
    previous_datetime = None
    for expected_sequence, item in enumerate(transactions, start=1):
        if item.sequence_no != expected_sequence:
            raise LedgerRollforwardError("LEDGER_SEQUENCE_MISMATCH", "账本交易顺序不连续。")
        if previous_datetime is not None and item.booking_datetime < previous_datetime:
            raise LedgerRollforwardError("LEDGER_DATETIME_REGRESSION", "账本交易时间倒退。")
        expected = balance - item.debit_fen + item.credit_fen
        if item.post_transaction_balance_fen != expected:
            raise LedgerRollforwardError("LEDGER_BALANCE_MISMATCH", "逐笔交易后余额勾稽失败。")
        balance = item.post_transaction_balance_fen
        previous_datetime = item.booking_datetime


def _daily_rollup(
    opening: int,
    transactions: list[LedgerTransaction],
    planning_result: TransactionPlanningResult,
) -> tuple[DailyReviewRow, ...]:
    status = "complete" if planning_result.cash_preview.is_complete else "stopped_before_blocked_outflow"
    if planning_result.cash_preview.is_complete:
        if not transactions:
            return ()
        start_date = transactions[0].booking_datetime.date()
        cutoff_date = transactions[-1].booking_datetime.date()
    else:
        fact = planning_result.cash_preview.fact_sheet
        if fact is None:
            raise LedgerRollforwardError("MISSING_STOP_FACT", "中断计划必须保留资金不足事实单。")
        start_date = transactions[0].booking_datetime.date() if transactions else fact.blocked_booking_datetime.date()
        cutoff_date = fact.blocked_booking_datetime.date()
    by_date: dict[date, list[LedgerTransaction]] = {}
    for item in transactions:
        by_date.setdefault(item.booking_datetime.date(), []).append(item)
    rows: list[DailyReviewRow] = []
    balance = opening
    current = start_date
    while current <= cutoff_date:
        items = by_date.get(current, [])
        if items:
            balance = items[-1].post_transaction_balance_fen
        rows.append(DailyReviewRow(
            calendar_date=current,
            inflow_fen=sum(item.credit_fen for item in items),
            outflow_fen=sum(item.debit_fen for item in items),
            transaction_count=len(items),
            ending_balance_fen=balance,
            execution_status=status,
        ))
        current += timedelta(days=1)
    return tuple(rows)


def _reconcile_daily(
    transactions: list[LedgerTransaction], daily_rows: tuple[DailyReviewRow, ...], closing_balance: int
) -> None:
    if daily_rows:
        if sum(row.inflow_fen for row in daily_rows) != sum(item.credit_fen for item in transactions):
            raise LedgerRollforwardError("DAILY_INFLOW_MISMATCH", "日度进账合计与逐笔不一致。")
        if sum(row.outflow_fen for row in daily_rows) != sum(item.debit_fen for item in transactions):
            raise LedgerRollforwardError("DAILY_OUTFLOW_MISMATCH", "日度出账合计与逐笔不一致。")
        if sum(row.transaction_count for row in daily_rows) != len(transactions):
            raise LedgerRollforwardError("DAILY_COUNT_MISMATCH", "日度笔数与逐笔不一致。")
        if daily_rows[-1].ending_balance_fen != closing_balance:
            raise LedgerRollforwardError("DAILY_BALANCE_MISMATCH", "日度期末余额与逐笔期末余额不一致。")
    elif transactions:
        raise LedgerRollforwardError("MISSING_DAILY_ROWS", "存在逐笔交易时必须生成日度汇总。")


def _write_transactions_csv(path: Path, result: LedgerRollforwardResult) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "transaction_id", "sequence_no", "booking_datetime", "debit_cny", "credit_cny",
            "post_transaction_balance_cny", "execution_status",
        ))
        writer.writeheader()
        for item in result.transactions:
            writer.writerow({
                "transaction_id": item.transaction_id,
                "sequence_no": item.sequence_no,
                "booking_datetime": item.booking_datetime.isoformat(timespec="seconds"),
                "debit_cny": _cny(item.debit_fen),
                "credit_cny": _cny(item.credit_fen),
                "post_transaction_balance_cny": _cny(item.post_transaction_balance_fen),
                "execution_status": "complete" if result.is_complete else "stopped_before_blocked_outflow",
            })


def _write_daily_csv(path: Path, result: LedgerRollforwardResult) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "calendar_date", "inflow_cny", "outflow_cny", "transaction_count", "ending_balance_cny", "execution_status",
        ))
        writer.writeheader()
        for item in result.daily_rows:
            writer.writerow({
                "calendar_date": item.calendar_date.isoformat(),
                "inflow_cny": _cny(item.inflow_fen),
                "outflow_cny": _cny(item.outflow_fen),
                "transaction_count": item.transaction_count,
                "ending_balance_cny": _cny(item.ending_balance_fen),
                "execution_status": item.execution_status,
            })


def _write_fact_sheet(path: Path, result: LedgerRollforwardResult) -> None:
    fact = result.fact_sheet
    payload: dict[str, object] = {
        "run_id": result.manifest.run_id,
        "ledger_version": result.manifest.ledger_version,
        "execution_status": "complete" if result.is_complete else "stopped_before_blocked_outflow",
        "opening_balance_cny": _cny(result.opening_balance_fen),
        "closing_balance_cny": _cny(result.closing_balance_fen),
        "transaction_count": len(result.transactions),
    }
    if fact is not None:
        payload["fact_sheet"] = {
            "opening_balance_cny": _cny(fact.opening_balance_fen),
            "cumulative_inflow_cny": _cny(fact.cumulative_inflow_fen),
            "cumulative_outflow_cny": _cny(fact.cumulative_outflow_fen),
            "available_balance_before_block_cny": _cny(fact.available_balance_before_block_fen),
            "blocked_transaction_reference": _blocked_reference(result.manifest, fact.blocked_movement_id),
            "blocked_booking_datetime": fact.blocked_booking_datetime.isoformat(timespec="seconds"),
            "blocked_outflow_cny": _cny(fact.blocked_outflow_fen),
            "shortage_cny": _cny(fact.shortage_fen),
            "completed_movement_count": fact.completed_movement_count,
            "planned_movement_count": fact.planned_movement_count,
        }
    with path.open("w", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _cny(amount_fen: int) -> str:
    return f"{amount_fen // 100}.{amount_fen % 100:02d}"


def _review_transaction_id(manifest: LedgerRunManifest, sequence_no: int, candidate_id: str) -> str:
    digest = sha256(f"{manifest.run_id}|{manifest.account_token}|{sequence_no}|{candidate_id}".encode("utf-8")).hexdigest()
    return f"synb1_txn_{digest[:24]}"


def _blocked_reference(manifest: LedgerRunManifest, candidate_id: str) -> str:
    digest = sha256(f"{manifest.run_id}|blocked|{candidate_id}".encode("utf-8")).hexdigest()
    return f"synb1_block_{digest[:24]}"
