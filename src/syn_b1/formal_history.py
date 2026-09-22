"""Read and verify an immutable formal SYN-B1 run before continuation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping

from syn_b1.formal_profile import ResolvedFormalProfile, resolve_profile


class FormalHistoryError(ValueError):
    """A claimed formal base run is missing, changed, or cannot be replayed."""


@dataclass(frozen=True)
class VerifiedFormalHistory:
    """The immutable public prefix and saved profile of one completed run."""

    run_directory: Path
    formal_run_id: str
    profile: ResolvedFormalProfile
    transaction_count: int
    transaction_digest: str
    manifest: Mapping[str, Any]


def load_verified_formal_history(directory: str | Path) -> VerifiedFormalHistory:
    """Verify a formal directory before treating it as a continuation base.

    The editable JSON that originally created the run is not trusted.  The
    saved profile and public transaction prefix must both match the saved file
    fingerprints in the original output directory.
    """

    root = Path(directory)
    manifest_path = root / "generation_manifest.json"
    profile_path = root / "scenario_profile.json"
    transactions_path = root / "transactions_total.csv"
    review_notes_path = root / "cash_flow_review_notes.csv"
    if not all((root.is_dir(), manifest_path.is_file(), profile_path.is_file(), transactions_path.is_file(), review_notes_path.is_file())):
        raise FormalHistoryError("原正式运行目录不完整，不能续生成。")
    manifest = _read_json(manifest_path, "运行清单")
    _verify_manifest_files(root, manifest)
    if manifest.get("execution_status") != "complete":
        raise FormalHistoryError("只有完整生成到期末的原正式运行可以续生成。")
    profile_payload = _read_json(profile_path, "保存参数")
    raw_profile = profile_payload.get("input_profile")
    if not isinstance(raw_profile, Mapping):
        raise FormalHistoryError("原正式运行缺少可核验的保存参数。")
    profile = resolve_profile(raw_profile)
    formal_run_id = _text(manifest.get("run_id"), "原正式输出运行编号")
    if manifest.get("sample_id") != profile.sample_id:
        raise FormalHistoryError("原运行清单与保存参数的虚构企业编号不一致。")
    if manifest.get("random_seed") != profile.request.random_seed:
        raise FormalHistoryError("原运行清单与保存参数的随机种子不一致。")
    transaction_digest, transaction_count = _public_transaction_digest(transactions_path)
    _verify_review_notes_pairing(transactions_path, review_notes_path)
    return VerifiedFormalHistory(
        run_directory=root,
        formal_run_id=formal_run_id,
        profile=profile,
        transaction_count=transaction_count,
        transaction_digest=transaction_digest,
        manifest=manifest,
    )


def _verify_manifest_files(root: Path, manifest: Mapping[str, Any]) -> None:
    file_hashes = manifest.get("files")
    if not isinstance(file_hashes, Mapping):
        raise FormalHistoryError("原运行清单缺少文件指纹，不能续生成。")
    required = {
        "transactions_total.csv", "account_daily_total.csv", "cash_flow_review_notes.csv",
        "scenario_profile.json", "quality_report.json",
        "_restricted/three_statements.json", "_restricted/tax_ledger.json",
        "_restricted/funding_tail.json", "_restricted/scenario_truth.json",
    }
    if not required.issubset(file_hashes):
        raise FormalHistoryError("原运行清单缺少必需文件指纹，不能续生成。")
    for relative, expected_hash in file_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise FormalHistoryError("原运行清单的文件指纹格式无效。")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise FormalHistoryError("原运行清单包含越界文件路径，不能续生成。")
        path = root / relative_path
        if not path.is_file() or _sha256(path) != expected_hash:
            raise FormalHistoryError("原正式运行文件已被修改、缺失或与清单不符，已拒绝续生成。")


def _public_transaction_digest(path: Path) -> tuple[str, int]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        required = (
            "transaction_id", "booking_datetime", "debit_cny", "credit_cny",
            "post_transaction_balance_cny",
        )
        if fields != required:
            raise FormalHistoryError("原正式逐笔表字段不符合冻结合同，不能续生成。")
        rows = list(reader)
    payload: list[tuple[str, int, str, int, int, int]] = []
    for sequence_no, row in enumerate(rows, start=1):
        try:
            booking_datetime = datetime.fromisoformat(row["booking_datetime"])
        except (TypeError, ValueError) as error:
            raise FormalHistoryError("原正式逐笔表存在无效交易时间，不能续生成。") from error
        transaction_id = row.get("transaction_id")
        if not isinstance(transaction_id, str) or not transaction_id:
            raise FormalHistoryError("原正式逐笔表存在空交易编号，不能续生成。")
        payload.append((
            transaction_id,
            sequence_no,
            booking_datetime.isoformat(),
            _cny_to_fen(row.get("debit_cny")),
            _cny_to_fen(row.get("credit_cny")),
            _cny_to_fen(row.get("post_transaction_balance_cny")),
        ))
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest(), len(payload)


def _verify_review_notes_pairing(transactions_path: Path, notes_path: Path) -> None:
    """Reject continuation unless the mandatory human table still pairs exactly."""

    with transactions_path.open(encoding="utf-8-sig", newline="") as handle:
        transactions = list(csv.DictReader(handle))
    with notes_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = (
            "transaction_id", "booking_datetime", "direction_cn", "amount_cny",
            "cash_flow_type_cn", "usage_scope_cn",
        )
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise FormalHistoryError("原运行的人工备注表字段无效，不能续生成。")
        notes = list(reader)
    if len(transactions) != len(notes):
        raise FormalHistoryError("原运行的逐笔流水与人工备注行数不一致，不能续生成。")
    for transaction, note in zip(transactions, notes, strict=True):
        debit_fen = _cny_to_fen(transaction.get("debit_cny"))
        credit_fen = _cny_to_fen(transaction.get("credit_cny"))
        expected_direction = "流入" if credit_fen else "流出"
        expected_amount = credit_fen or debit_fen
        if (
            note.get("transaction_id") != transaction.get("transaction_id")
            or note.get("booking_datetime") != transaction.get("booking_datetime")
            or note.get("direction_cn") != expected_direction
            or _cny_to_fen(note.get("amount_cny")) != expected_amount
        ):
            raise FormalHistoryError("原运行的人工备注未按交易编号、时间、方向和金额逐笔对应，不能续生成。")


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalHistoryError(f"无法读取原运行的{label}，不能续生成。") from error
    if not isinstance(value, Mapping):
        raise FormalHistoryError(f"原运行的{label}格式无效，不能续生成。")
    return value


def _cny_to_fen(value: Any) -> int:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise FormalHistoryError("原正式逐笔表金额无效，不能续生成。") from error
    fen = int((decimal * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if fen < 0:
        raise FormalHistoryError("原正式逐笔表金额不能为负数。")
    return fen


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalHistoryError(f"{label}不能为空。")
    return value.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
