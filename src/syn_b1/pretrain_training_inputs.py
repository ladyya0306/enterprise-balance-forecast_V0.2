"""Read-only qualification for frozen SYN-B1 training inputs.

This module deliberately prepares no model feature, target, scaler, model or
database record.  It verifies that each formal enterprise directory can later
be handed to the feature stage without using the human review-note text as an
input.  For a final-test enterprise, callers can request structural-only daily
verification so that no balance label is assembled or reported at this stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


REQUIRED_PUBLIC_FILES = (
    "transactions_total.csv",
    "account_daily_total.csv",
    "cash_flow_review_notes.csv",
    "scenario_profile.json",
    "generation_manifest.json",
    "quality_report.json",
)
TRANSACTION_COLUMNS = (
    "transaction_id",
    "booking_datetime",
    "debit_cny",
    "credit_cny",
    "post_transaction_balance_cny",
)
DAILY_COLUMNS = (
    "calendar_date",
    "inflow_cny",
    "outflow_cny",
    "transaction_count",
    "ending_balance_cny",
    "day_cut_basis",
    "source_last_transaction_id",
    "reconciliation_status",
    "calendar_source",
    "calendar_inference_status",
    "is_bank_workday",
    "business_step",
    "target_date",
    "execution_status",
)
NOTES_COLUMNS = (
    "transaction_id",
    "booking_datetime",
    "direction_cn",
    "amount_cny",
    "cash_flow_type_cn",
    "usage_scope_cn",
)


@dataclass(frozen=True)
class RunQualification:
    sample_id: str
    split_group: str
    run_directory: str
    qualified: bool
    verification_scope: str
    transaction_count: int
    natural_day_count: int
    errors: tuple[str, ...]
    public_file_sha256: dict[str, str]

    def as_row(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "split_group": self.split_group,
            "run_directory": self.run_directory,
            "qualified": str(self.qualified).lower(),
            "verification_scope": self.verification_scope,
            "transaction_count": str(self.transaction_count),
            "natural_day_count": str(self.natural_day_count),
            "errors": " | ".join(self.errors),
            "transactions_total_sha256": self.public_file_sha256.get("transactions_total.csv", ""),
            "account_daily_total_sha256": self.public_file_sha256.get("account_daily_total.csv", ""),
            "cash_flow_review_notes_sha256": self.public_file_sha256.get("cash_flow_review_notes.csv", ""),
        }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    # Formal CSV exports use UTF-8 with BOM for direct Excel opening.  ``utf-8-sig``
    # accepts both that form and ordinary UTF-8 without changing any source byte.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _decimal(value: str, field: str, errors: list[str]) -> Decimal | None:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError):
        errors.append(f"{field}不是有效金额")
        return None


def _expected_note_key(transaction: dict[str, str], errors: list[str]) -> tuple[str, str, str, Decimal] | None:
    debit = _decimal(transaction["debit_cny"], "debit_cny", errors)
    credit = _decimal(transaction["credit_cny"], "credit_cny", errors)
    if debit is None or credit is None:
        return None
    if (debit > 0) == (credit > 0):
        errors.append("逐笔借贷方向不是恰有一个非零金额")
        return None
    direction = "流出" if debit > 0 else "流入"
    return transaction["transaction_id"], transaction["booking_datetime"], direction, debit if debit > 0 else credit


def _continuous_natural_dates(rows: list[dict[str, str]], errors: list[str]) -> None:
    if not rows:
        errors.append("日度表为空")
        return
    try:
        dates = [date.fromisoformat(row["calendar_date"]) for row in rows]
    except ValueError:
        errors.append("日度表含无效calendar_date")
        return
    if len(set(dates)) != len(dates):
        errors.append("日度表calendar_date重复")
    for prior, current in zip(dates, dates[1:]):
        if current != prior + timedelta(days=1):
            errors.append("日度表未连续覆盖自然日")
            break
    steps = [int(row["business_step"]) for row in rows if row["is_bank_workday"] == "True"]
    if steps != list(range(1, len(steps) + 1)):
        errors.append("银行工作日business_step不连续")
    if any(row["reconciliation_status"] != "trusted" for row in rows):
        errors.append("日度表含非trusted勾稽状态")


def qualify_formal_run(
    run_directory: Path,
    *,
    sample_id: str,
    split_group: str,
    expected_run_id: str,
    verify_daily_financial_values: bool,
) -> RunQualification:
    """Check one frozen formal run without producing model inputs or targets."""

    errors: list[str] = []
    hashes: dict[str, str] = {}
    for name in REQUIRED_PUBLIC_FILES:
        path = run_directory / name
        if not path.is_file():
            errors.append(f"缺少{name}")
        else:
            hashes[name] = sha256_file(path)
    if errors:
        return RunQualification(sample_id, split_group, str(run_directory), False, "not_started", 0, 0, tuple(errors), hashes)

    try:
        manifest: dict[str, Any] = json.loads((run_directory / "generation_manifest.json").read_text(encoding="utf-8"))
        quality: dict[str, Any] = json.loads((run_directory / "quality_report.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append("运行清单或质量报告不是有效JSON")
        manifest = {}
        quality = {}

    if manifest.get("sample_id") != sample_id:
        errors.append("运行清单sample_id不匹配")
    if manifest.get("run_id") != expected_run_id:
        errors.append("运行清单run_id不匹配")
    if manifest.get("execution_status") not in {"complete", "stopped_before_blocked_outflow"}:
        errors.append("运行状态不在训练合同允许范围")
    manifest_files = manifest.get("files", {})
    for name, digest in hashes.items():
        if name == "generation_manifest.json":
            # The manifest records payload files, not its own recursive hash.
            continue
        if manifest_files.get(name) != digest:
            errors.append(f"运行清单中文件指纹不匹配：{name}")
    if quality.get("future_training_data_quality_ready") is not True:
        errors.append("质量报告未声明未来训练数据质量就绪")
    notes_policy = quality.get("human_review_notes", {})
    if notes_policy.get("model_input_allowed") is not False or notes_policy.get("database_import_allowed") is not False:
        errors.append("质量报告未隔离人工备注表")
    checks = quality.get("checks", {})
    for name in ("candidate_amount_reconciled", "final_cash_reconciled"):
        if checks.get(name) is not True:
            errors.append(f"质量报告未通过{name}")

    transaction_columns, transactions = _read_csv(run_directory / "transactions_total.csv")
    note_columns, notes = _read_csv(run_directory / "cash_flow_review_notes.csv")
    daily_columns, daily_rows = _read_csv(run_directory / "account_daily_total.csv")
    if tuple(transaction_columns) != TRANSACTION_COLUMNS:
        errors.append("transactions_total.csv字段契约不匹配")
    if tuple(note_columns) != NOTES_COLUMNS:
        errors.append("cash_flow_review_notes.csv字段契约不匹配")
    if tuple(daily_columns) != DAILY_COLUMNS:
        errors.append("account_daily_total.csv字段契约不匹配")
    if len(transactions) != len(notes):
        errors.append("逐笔流水与人工备注表行数不一致")
    transaction_ids = [row.get("transaction_id", "") for row in transactions]
    if len(set(transaction_ids)) != len(transaction_ids) or any(not value for value in transaction_ids):
        errors.append("transactions_total.csv交易编号为空或重复")

    expected_note_keys = [_expected_note_key(row, errors) for row in transactions]
    actual_note_keys: list[tuple[str, str, str, Decimal] | None] = []
    for row in notes:
        amount = _decimal(row.get("amount_cny", ""), "notes.amount_cny", errors)
        actual_note_keys.append(None if amount is None else (row.get("transaction_id", ""), row.get("booking_datetime", ""), row.get("direction_cn", ""), amount))
        if row.get("usage_scope_cn") != "仅供人工查看；不得作为模型输入或导入数据库":
            errors.append("人工备注表用途声明不正确")
            break
    if expected_note_keys != actual_note_keys:
        errors.append("逐笔流水与人工备注表的编号、时间、方向或金额不一一对应")

    _continuous_natural_dates(daily_rows, errors)
    if verify_daily_financial_values:
        transaction_inflow = sum((_decimal(row["credit_cny"], "credit_cny", errors) or Decimal("0")) for row in transactions)
        transaction_outflow = sum((_decimal(row["debit_cny"], "debit_cny", errors) or Decimal("0")) for row in transactions)
        daily_inflow = sum((_decimal(row["inflow_cny"], "daily.inflow_cny", errors) or Decimal("0")) for row in daily_rows)
        daily_outflow = sum((_decimal(row["outflow_cny"], "daily.outflow_cny", errors) or Decimal("0")) for row in daily_rows)
        if transaction_inflow != daily_inflow or transaction_outflow != daily_outflow:
            errors.append("逐笔与日度收付汇总不一致")
        if transactions and daily_rows:
            final_transaction_balance = _decimal(transactions[-1]["post_transaction_balance_cny"], "post_transaction_balance_cny", errors)
            final_daily_balance = _decimal(daily_rows[-1]["ending_balance_cny"], "ending_balance_cny", errors)
            if final_transaction_balance != final_daily_balance:
                errors.append("最终逐笔余额与日度余额不一致")

    scope = "full_daily_reconciliation" if verify_daily_financial_values else "structural_daily_only_final_test_sealed"
    return RunQualification(
        sample_id=sample_id,
        split_group=split_group,
        run_directory=str(run_directory),
        qualified=not errors,
        verification_scope=scope,
        transaction_count=len(transactions),
        natural_day_count=len(daily_rows),
        errors=tuple(dict.fromkeys(errors)),
        public_file_sha256=hashes,
    )
