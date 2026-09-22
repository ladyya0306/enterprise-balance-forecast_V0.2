"""R3 bank-neutral workbook header inspection and explicit mapping confirmation.

This module identifies fields solely from approved header aliases.  It never
uses amounts, summaries, filenames, counterparties, or account behaviour to
guess a mapping.  It does not normalize transactions or write any mapping to a
database; those responsibilities belong to later task cards.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Sequence

from openpyxl import load_workbook
import yaml


CODE_REQUIRES_CONFIRMATION = "FM_CONFIRM_REQUIRED_001"
CODE_MISSING_FIELD = "FM_REQUIRED_FIELD_MISSING_001"
CODE_AMBIGUOUS_FIELD = "FM_FIELD_AMBIGUOUS_001"
CODE_NO_HEADER = "FM_HEADER_NOT_FOUND_001"
CODE_SHEET_AMBIGUOUS = "FM_SHEET_AMBIGUOUS_001"
CODE_CONFIRMED = "FM_MAPPING_CONFIRMED_001"


RAW_REQUIRED_FIELDS = (
    "booking_datetime_or_booking_date",
    "direction",
    "amount",
    "post_transaction_balance",
)
DAILY_REQUIRED_FIELDS = (
    "calendar_date",
    "daily_inflow_cny",
    "daily_outflow_cny",
    "closing_balance_cny",
    "transaction_count",
)

RAW_ALIASES = {
    "booking_datetime_or_booking_date": ("交易日期", "交易时间", "记账日期", "记账时间", "记账日期时间"),
    "direction": ("交易方向", "收支方向", "借贷方向", "借贷标志", "收入支出方向"),
    "amount": ("交易金额", "发生额", "发生金额", "交易发生额", "金额"),
    "post_transaction_balance": ("账户余额", "交易后余额", "余额", "账户可用余额"),
}
DAILY_ALIASES = {
    "calendar_date": ("calendar_date",),
    "daily_inflow_cny": ("daily_inflow_cny",),
    "daily_outflow_cny": ("daily_outflow_cny",),
    "closing_balance_cny": ("closing_balance_cny",),
    "transaction_count": ("transaction_count",),
}


@dataclass(frozen=True)
class FieldMappingProposal:
    code: str
    message_zh: str
    input_mode: str | None
    sheet_name: str
    header_row_number: int
    headers: tuple[str, ...]
    header_fingerprint: str
    field_to_column: dict[str, str]
    ambiguous_fields: dict[str, tuple[str, ...]]
    missing_fields: tuple[str, ...]
    column_value_kinds: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ConfirmedFieldMapping:
    code: str
    source_format_id: str
    mapping_version: str
    input_mode: str
    sheet_name: str
    header_row_number: int
    header_fingerprint: str
    field_to_column: dict[str, str]


def inspect_workbook_headers(path: Path, *, max_header_rows: int = 10, sample_rows: int = 20) -> FieldMappingProposal:
    """Read workbook headers and small type samples without reading business semantics."""

    workbook = load_workbook(Path(path), read_only=True, data_only=False)
    try:
        candidates: list[tuple[int, str, int, tuple[str, ...], tuple[tuple[object, ...], ...]]] = []
        for sheet in workbook.worksheets:
            rows = sheet.iter_rows(min_row=1, max_row=max_header_rows + sample_rows, values_only=True)
            materialized = list(rows)
            for index, row in enumerate(materialized[:max_header_rows], start=1):
                headers = _as_headers(row)
                score = _header_score(headers)
                if score:
                    candidates.append((score, sheet.title, index, headers, tuple(materialized[index:index + sample_rows])))
        if not candidates:
            return _no_header_proposal()

        candidates.sort(key=lambda item: (-item[0], item[1].casefold(), item[2]))
        best_score = candidates[0][0]
        top = [item for item in candidates if item[0] == best_score]
        if len(top) > 1:
            return _ambiguous_sheet_proposal(tuple(item[1] for item in top))
        _, sheet_name, header_row_number, headers, sample = candidates[0]
        proposal = propose_mapping_from_headers(headers, sheet_name=sheet_name, header_row_number=header_row_number)
        return FieldMappingProposal(
            code=proposal.code,
            message_zh=proposal.message_zh,
            input_mode=proposal.input_mode,
            sheet_name=proposal.sheet_name,
            header_row_number=proposal.header_row_number,
            headers=proposal.headers,
            header_fingerprint=proposal.header_fingerprint,
            field_to_column=proposal.field_to_column,
            ambiguous_fields=proposal.ambiguous_fields,
            missing_fields=proposal.missing_fields,
            column_value_kinds=_profile_value_kinds(headers, sample),
        )
    finally:
        workbook.close()


def propose_mapping_from_headers(
    headers: Sequence[object],
    *,
    sheet_name: str = "<header_only>",
    header_row_number: int = 1,
) -> FieldMappingProposal:
    """Create a proposal from headers only; data values never influence the result."""

    header_tuple = _as_headers(headers)
    raw = _resolve_fields(header_tuple, RAW_ALIASES, RAW_REQUIRED_FIELDS)
    daily = _resolve_fields(header_tuple, DAILY_ALIASES, DAILY_REQUIRED_FIELDS)
    mode, resolved = _choose_mode(raw, daily)
    fingerprint = _header_fingerprint(header_tuple)

    if resolved["ambiguous"]:
        field_names = "、".join(resolved["ambiguous"])
        message = f"发现多个可能的{field_names}列，必须由用户确认映射；本次不读取交易内容，也不写数据库。"
        code = CODE_AMBIGUOUS_FIELD
    elif resolved["missing"]:
        field_names = "、".join(resolved["missing"])
        message = f"缺少必需字段：{field_names}。无法安全继续，请补充或确认正确表头。"
        code = CODE_MISSING_FIELD
    else:
        message = "已根据表头提出字段映射建议，请用户确认后才可进入逐笔核验；本步骤不写数据库。"
        code = CODE_REQUIRES_CONFIRMATION

    return FieldMappingProposal(
        code=code,
        message_zh=message,
        input_mode=mode,
        sheet_name=sheet_name,
        header_row_number=header_row_number,
        headers=header_tuple,
        header_fingerprint=fingerprint,
        field_to_column=resolved["fields"],
        ambiguous_fields=resolved["ambiguous"],
        missing_fields=resolved["missing"],
        column_value_kinds={},
    )


def confirm_mapping(
    proposal: FieldMappingProposal,
    *,
    source_format_id: str,
    mapping_version: str,
) -> ConfirmedFieldMapping:
    """Represent an explicit user confirmation in memory; persistence is later."""

    if proposal.code != CODE_REQUIRES_CONFIRMATION or proposal.input_mode is None:
        raise ValueError("只有完整且无歧义的字段建议才能确认映射")
    if not source_format_id.strip() or not mapping_version.strip():
        raise ValueError("source_format_id和mapping_version不能为空")
    return ConfirmedFieldMapping(
        code=CODE_CONFIRMED,
        source_format_id=source_format_id,
        mapping_version=mapping_version,
        input_mode=proposal.input_mode,
        sheet_name=proposal.sheet_name,
        header_row_number=proposal.header_row_number,
        header_fingerprint=proposal.header_fingerprint,
        field_to_column=dict(proposal.field_to_column),
    )


def load_confirmed_mapping_for_proposal(
    proposal: FieldMappingProposal,
    *,
    registry_path: Path,
) -> ConfirmedFieldMapping:
    """Load only an explicitly registered mapping matching this exact header.

    This is the runtime counterpart of a user's R3 confirmation.  It does not
    create, amend, or infer mappings: an unknown header fingerprint is blocked
    until the registry has been deliberately updated and accepted.
    """

    if proposal.code != CODE_REQUIRES_CONFIRMATION or proposal.input_mode is None:
        raise ValueError("当前工作簿未形成完整、无歧义的字段建议，不能加载确认映射")

    with Path(registry_path).open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    matches: list[dict[str, object]] = []
    for entry in registry.get("mappings", []):
        fingerprints = entry.get("confirmed_header_fingerprints", [])
        if (
            proposal.header_fingerprint in fingerprints
            and entry.get("input_mode") == proposal.input_mode
            and entry.get("header_row_number") == proposal.header_row_number
            and entry.get("field_to_column") == proposal.field_to_column
        ):
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError("当前表头指纹未在确认映射登记表中唯一登记，已阻断逐笔处理")

    entry = matches[0]
    return ConfirmedFieldMapping(
        code=CODE_CONFIRMED,
        source_format_id=str(entry["source_format_id"]),
        mapping_version=str(entry["mapping_version"]),
        input_mode=str(entry["input_mode"]),
        sheet_name=proposal.sheet_name,
        header_row_number=proposal.header_row_number,
        header_fingerprint=proposal.header_fingerprint,
        field_to_column=dict(entry["field_to_column"]),
    )


def _choose_mode(raw: dict[str, object], daily: dict[str, object]) -> tuple[str, dict[str, object]]:
    raw_complete = not raw["missing"] and not raw["ambiguous"]
    daily_complete = not daily["missing"] and not daily["ambiguous"]
    if daily_complete and not raw_complete:
        return "user_reviewed_daily_aggregate_xlsx", daily
    if raw_complete and not daily_complete:
        return "raw_transaction_full_snapshot_xlsx", raw
    if raw_complete and daily_complete:
        # A single header row meeting both contracts would be ambiguous and must
        # be resolved by a later explicit UI; this module does not pick one.
        return "raw_transaction_full_snapshot_xlsx", raw
    raw_coverage = len(raw["fields"])
    daily_coverage = len(daily["fields"])
    if daily_coverage > raw_coverage:
        return "user_reviewed_daily_aggregate_xlsx", daily
    return "raw_transaction_full_snapshot_xlsx", raw


def _resolve_fields(
    headers: tuple[str, ...],
    aliases: dict[str, tuple[str, ...]],
    required: tuple[str, ...],
) -> dict[str, object]:
    normalized_to_original: dict[str, list[str]] = {}
    for header in headers:
        normalized_to_original.setdefault(_normalize_header(header), []).append(header)

    fields: dict[str, str] = {}
    ambiguous: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    for semantic_field in required:
        matches: list[str] = []
        for alias in aliases[semantic_field]:
            matches.extend(normalized_to_original.get(_normalize_header(alias), []))
        # Do not collapse exact duplicates.  Two identically named core
        # columns are still ambiguous: choosing the first one would silently
        # change the source meaning when a bank export layout changes.
        if len(matches) == 1:
            fields[semantic_field] = matches[0]
        elif len(matches) > 1:
            ambiguous[semantic_field] = tuple(matches)
        else:
            missing.append(semantic_field)
    return {"fields": fields, "ambiguous": ambiguous, "missing": tuple(missing)}


def _as_headers(row: Iterable[object]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value).strip() for value in row)


def _normalize_header(value: str) -> str:
    return re.sub(r"[\s\-()（）\[\]【】]", "", value).casefold()


def _header_score(headers: tuple[str, ...]) -> int:
    aliases = {alias for field_aliases in RAW_ALIASES.values() for alias in field_aliases}
    aliases.update(alias for field_aliases in DAILY_ALIASES.values() for alias in field_aliases)
    normalized_aliases = {_normalize_header(alias) for alias in aliases}
    return sum(_normalize_header(header) in normalized_aliases for header in headers if header)


def _header_fingerprint(headers: tuple[str, ...]) -> str:
    payload = json.dumps({"schema": "r3_header_v1", "headers": list(headers)}, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _profile_value_kinds(headers: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> dict[str, tuple[str, ...]]:
    profiles: dict[str, tuple[str, ...]] = {}
    for index, header in enumerate(headers):
        if not header:
            continue
        types = Counter(_value_kind(row[index] if index < len(row) else None) for row in rows)
        profiles[header] = tuple(sorted(kind for kind, count in types.items() if count))
    return profiles


def _value_kind(value: object) -> str:
    if value is None or value == "":
        return "blank"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    return "text"


def _no_header_proposal() -> FieldMappingProposal:
    return FieldMappingProposal(
        code=CODE_NO_HEADER,
        message_zh="未找到可识别的表头行，无法提出字段映射建议。",
        input_mode=None,
        sheet_name="",
        header_row_number=0,
        headers=(),
        header_fingerprint=_header_fingerprint(()),
        field_to_column={},
        ambiguous_fields={},
        missing_fields=(),
        column_value_kinds={},
    )


def _ambiguous_sheet_proposal(sheet_names: tuple[str, ...]) -> FieldMappingProposal:
    return FieldMappingProposal(
        code=CODE_SHEET_AMBIGUOUS,
        message_zh=f"发现多个工作表都像输入表：{'、'.join(sheet_names)}。请先由用户选择工作表。",
        input_mode=None,
        sheet_name="",
        header_row_number=0,
        headers=(),
        header_fingerprint=_header_fingerprint(()),
        field_to_column={},
        ambiguous_fields={},
        missing_fields=(),
        column_value_kinds={},
    )
