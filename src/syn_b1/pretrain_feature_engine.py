"""Frozen 14-bank-workday feature construction for SYN-B1 pretraining.

The module reads only the T1 qualification manifest plus public daily-account
tables.  It never reads review notes, profiles, seeds, transaction narratives,
or restricted-generator material.  Train and validation samples receive labels;
the final-test population receives only a sealed date index so its balance labels
and features remain unopened until the separately authorized final-test stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Iterator


LOOKBACK_BUSINESS_DAYS = 14
BALANCE_FLOOR_CNY = Decimal("1000000.00")
FEATURE_VERSION = "syn_b1_pretrain_t2_14bd_feature_1_0"
SPLIT_WINDOWS = {
    "train": (None, date(2026, 6, 30)),
    "validation": (date(2026, 7, 1), date(2026, 9, 30)),
    "final_test": (date(2026, 10, 1), date(2026, 12, 31)),
}

TREE_INPUT_COLUMNS = (
    "closing_balance_cny",
    "daily_inflow_cny",
    "daily_outflow_cny",
    "daily_net_flow_cny",
    "transaction_count",
    "balance_lag_1bd_cny",
    "balance_change_1bd_cny",
    "balance_change_5bd_cny",
    "inflow_sum_5bd_cny",
    "outflow_sum_5bd_cny",
    "net_flow_sum_5bd_cny",
    "inflow_sum_14bd_cny",
    "outflow_sum_14bd_cny",
    "transaction_count_sum_14bd",
    "weekday_monday_0",
    "is_month_end_business_day",
    "calendar_gap_days_to_next_business_day",
)
SEQUENCE_INPUT_COLUMNS = (
    "calendar_gap_days_from_prior_business_day",
    "closing_balance_cny",
    "daily_inflow_cny",
    "daily_outflow_cny",
    "daily_net_flow_cny",
    "transaction_count",
    "day_has_transaction",
    "weekday_monday_0",
    "is_month_end_business_day",
    "calendar_gap_days_to_next_business_day",
)
TREE_COLUMNS = (
    "sample_key", "enterprise_id", "split_group", "cutoff_date", "target_date", "business_step", "lookback_start_date",
    "lookback_business_days", *TREE_INPUT_COLUMNS,
    "target_balance_cny", "target_change_cny", "target_scaled", "balance_floor_cny", "feature_manifest_version",
)
SEQUENCE_COLUMNS = (
    "sample_key", "enterprise_id", "split_group", "sequence_position", "sequence_length", "calendar_date", "business_step",
    *SEQUENCE_INPUT_COLUMNS, "target_date", "target_balance_cny", "target_change_cny", "target_scaled", "feature_manifest_version",
)
SPLIT_INDEX_COLUMNS = (
    "sample_key", "enterprise_id", "split_group", "cutoff_date", "target_date", "business_step", "lookback_start_date",
    "lookback_business_days", "label_status", "final_test_sealed", "feature_manifest_version",
)


class FeatureConstructionError(ValueError):
    """Raised when frozen input or timing constraints are violated."""


@dataclass(frozen=True)
class DailyBusinessRow:
    calendar_date: date
    inflow_cny: Decimal
    outflow_cny: Decimal
    transaction_count: int
    ending_balance_cny: Decimal
    business_step: int
    target_date: date | None

    @property
    def net_flow_cny(self) -> Decimal:
        return self.inflow_cny - self.outflow_cny


@dataclass(frozen=True)
class EligibleSample:
    enterprise_id: str
    split_group: str
    cutoff: DailyBusinessRow
    target: DailyBusinessRow
    window: tuple[DailyBusinessRow, ...]

    @property
    def sample_key(self) -> str:
        return f"{self.enterprise_id}__bd{self.cutoff.business_step:04d}"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decimal(value: str, field: str) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise FeatureConstructionError(f"{field}不是有效金额：{value!r}") from exc


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _ratio(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))


def _read_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def read_t1_population(qualification_directory: Path, project_root: Path) -> list[dict[str, str]]:
    """Load T1's fixed population and reject stale or unqualified input."""

    report_path = qualification_directory / "training_input_qualification_report.json"
    manifest_path = qualification_directory / "combined_training_population_manifest.csv"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FeatureConstructionError("缺少T1资格审计报告或合并人口清单")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("enterprise_count") != 260:
        raise FeatureConstructionError("T1资格审计不是260户通过状态")
    if report.get("final_test_labels_read_or_assembled") is not False:
        raise FeatureConstructionError("T1报告显示最终测试标签已被读取，禁止继续")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        population = list(csv.DictReader(handle))
    if len(population) != 260 or len({row.get("sample_id") for row in population}) != 260:
        raise FeatureConstructionError("T1人口清单不是260户唯一企业")
    expected_counts = {"train": 156, "validation": 52, "final_test": 52}
    actual_counts = {group: sum(row.get("split_group") == group for row in population) for group in expected_counts}
    if actual_counts != expected_counts:
        raise FeatureConstructionError(f"T1人口切分数量错误：{actual_counts}")
    for row in population:
        if row.get("qualified") != "true":
            raise FeatureConstructionError(f"T1未通过资格的企业：{row.get('sample_id')}")
        if row.get("split_group") not in SPLIT_WINDOWS:
            raise FeatureConstructionError(f"未知切分组：{row.get('split_group')}")
        daily_path = project_root / row["run_directory"] / "account_daily_total.csv"
        if not daily_path.is_file():
            raise FeatureConstructionError(f"缺少日度主表：{row.get('sample_id')}")
        if sha256_file(daily_path) != row.get("account_daily_total_sha256"):
            raise FeatureConstructionError(f"T1之后日度主表发生变化：{row.get('sample_id')}")
    return sorted(population, key=lambda row: int(row["sample_id"].rsplit("_", 1)[1]))


def read_business_daily_rows(daily_path: Path) -> list[DailyBusinessRow]:
    """Read only model-visible public daily fields for train/validation features."""

    rows: list[DailyBusinessRow] = []
    required = {
        "calendar_date", "inflow_cny", "outflow_cny", "transaction_count", "ending_balance_cny",
        "reconciliation_status", "is_bank_workday", "business_step", "target_date",
    }
    for raw in _read_csv(daily_path):
        if not required.issubset(raw):
            raise FeatureConstructionError(f"日度主表字段不足：{daily_path}")
        if raw["reconciliation_status"] != "trusted":
            raise FeatureConstructionError(f"日度主表存在非可信勾稽状态：{daily_path}")
        if raw["is_bank_workday"] != "True":
            continue
        target_value = raw["target_date"]
        rows.append(
            DailyBusinessRow(
                calendar_date=date.fromisoformat(raw["calendar_date"]),
                inflow_cny=_decimal(raw["inflow_cny"], "inflow_cny"),
                outflow_cny=_decimal(raw["outflow_cny"], "outflow_cny"),
                transaction_count=int(raw["transaction_count"]),
                ending_balance_cny=_decimal(raw["ending_balance_cny"], "ending_balance_cny"),
                business_step=int(raw["business_step"]),
                target_date=date.fromisoformat(target_value) if target_value else None,
            )
        )
    if not rows:
        raise FeatureConstructionError(f"日度主表没有银行工作日：{daily_path}")
    if [row.business_step for row in rows] != list(range(1, len(rows) + 1)):
        raise FeatureConstructionError(f"银行工作日序号不连续：{daily_path}")
    return rows


def read_final_test_index_rows(daily_path: Path) -> list[tuple[date, int, date | None]]:
    """Read final-test calendar/index fields only; never assemble balance labels or features."""

    rows: list[tuple[date, int, date | None]] = []
    for raw in _read_csv(daily_path):
        if raw.get("is_bank_workday") != "True":
            continue
        target_value = raw.get("target_date", "")
        rows.append((date.fromisoformat(raw["calendar_date"]), int(raw["business_step"]), date.fromisoformat(target_value) if target_value else None))
    if [step for _, step, _ in rows] != list(range(1, len(rows) + 1)):
        raise FeatureConstructionError(f"最终测试日度索引business_step不连续：{daily_path}")
    return rows


def _target_in_window(split_group: str, target_date: date) -> bool:
    start, end = SPLIT_WINDOWS[split_group]
    return (start is None or target_date >= start) and target_date <= end


def eligible_samples(enterprise_id: str, split_group: str, rows: list[DailyBusinessRow]) -> list[EligibleSample]:
    if split_group == "final_test":
        raise FeatureConstructionError("最终测试不得在T2阶段构造带标签样本")
    by_step = {row.business_step: row for row in rows}
    samples: list[EligibleSample] = []
    for cutoff in rows:
        if cutoff.target_date is None or not _target_in_window(split_group, cutoff.target_date):
            continue
        target = by_step.get(cutoff.business_step + 1)
        if target is None or target.calendar_date != cutoff.target_date:
            raise FeatureConstructionError(f"目标日与下一银行工作日不一致：{enterprise_id} step={cutoff.business_step}")
        start_step = cutoff.business_step - LOOKBACK_BUSINESS_DAYS + 1
        if start_step < 1:
            continue
        window = tuple(by_step[step] for step in range(start_step, cutoff.business_step + 1))
        if len(window) != LOOKBACK_BUSINESS_DAYS:
            raise FeatureConstructionError(f"14银行工作日窗口不足：{enterprise_id} step={cutoff.business_step}")
        samples.append(EligibleSample(enterprise_id, split_group, cutoff, target, window))
    return samples


def _is_month_end_business_day(cutoff: DailyBusinessRow, target: DailyBusinessRow) -> int:
    return int(cutoff.calendar_date.month != target.calendar_date.month)


def tree_row(sample: EligibleSample) -> dict[str, str]:
    window = sample.window
    cutoff = sample.cutoff
    previous = window[-2]
    five_days_ago = window[-6]
    target_change = sample.target.ending_balance_cny - cutoff.ending_balance_cny
    return {
        "sample_key": sample.sample_key,
        "enterprise_id": sample.enterprise_id,
        "split_group": sample.split_group,
        "cutoff_date": cutoff.calendar_date.isoformat(),
        "target_date": sample.target.calendar_date.isoformat(),
        "business_step": str(cutoff.business_step),
        "lookback_start_date": window[0].calendar_date.isoformat(),
        "lookback_business_days": str(LOOKBACK_BUSINESS_DAYS),
        "closing_balance_cny": _money(cutoff.ending_balance_cny),
        "daily_inflow_cny": _money(cutoff.inflow_cny),
        "daily_outflow_cny": _money(cutoff.outflow_cny),
        "daily_net_flow_cny": _money(cutoff.net_flow_cny),
        "transaction_count": str(cutoff.transaction_count),
        "balance_lag_1bd_cny": _money(previous.ending_balance_cny),
        "balance_change_1bd_cny": _money(cutoff.ending_balance_cny - previous.ending_balance_cny),
        "balance_change_5bd_cny": _money(cutoff.ending_balance_cny - five_days_ago.ending_balance_cny),
        "inflow_sum_5bd_cny": _money(sum(row.inflow_cny for row in window[-5:])),
        "outflow_sum_5bd_cny": _money(sum(row.outflow_cny for row in window[-5:])),
        "net_flow_sum_5bd_cny": _money(sum(row.net_flow_cny for row in window[-5:])),
        "inflow_sum_14bd_cny": _money(sum(row.inflow_cny for row in window)),
        "outflow_sum_14bd_cny": _money(sum(row.outflow_cny for row in window)),
        "transaction_count_sum_14bd": str(sum(row.transaction_count for row in window)),
        "weekday_monday_0": str(cutoff.calendar_date.weekday()),
        "is_month_end_business_day": str(_is_month_end_business_day(cutoff, sample.target)),
        "calendar_gap_days_to_next_business_day": str((sample.target.calendar_date - cutoff.calendar_date).days),
        "target_balance_cny": _money(sample.target.ending_balance_cny),
        "target_change_cny": _money(target_change),
        "target_scaled": _ratio(target_change / max(abs(cutoff.ending_balance_cny), BALANCE_FLOOR_CNY)),
        "balance_floor_cny": _money(BALANCE_FLOOR_CNY),
        "feature_manifest_version": FEATURE_VERSION,
    }


def sequence_rows(sample: EligibleSample) -> Iterable[dict[str, str]]:
    target_change = sample.target.ending_balance_cny - sample.cutoff.ending_balance_cny
    for position, row in enumerate(sample.window, start=1):
        previous = sample.window[position - 2] if position > 1 else None
        next_row = sample.window[position] if position < LOOKBACK_BUSINESS_DAYS else sample.target
        yield {
            "sample_key": sample.sample_key,
            "enterprise_id": sample.enterprise_id,
            "split_group": sample.split_group,
            "sequence_position": str(position),
            "sequence_length": str(LOOKBACK_BUSINESS_DAYS),
            "calendar_date": row.calendar_date.isoformat(),
            "business_step": str(row.business_step),
            "calendar_gap_days_from_prior_business_day": str((row.calendar_date - previous.calendar_date).days) if previous else "0",
            "closing_balance_cny": _money(row.ending_balance_cny),
            "daily_inflow_cny": _money(row.inflow_cny),
            "daily_outflow_cny": _money(row.outflow_cny),
            "daily_net_flow_cny": _money(row.net_flow_cny),
            "transaction_count": str(row.transaction_count),
            "day_has_transaction": str(int(row.transaction_count > 0)),
            "weekday_monday_0": str(row.calendar_date.weekday()),
            "is_month_end_business_day": str(_is_month_end_business_day(row, next_row)),
            "calendar_gap_days_to_next_business_day": str((next_row.calendar_date - row.calendar_date).days),
            "target_date": sample.target.calendar_date.isoformat(),
            "target_balance_cny": _money(sample.target.ending_balance_cny),
            "target_change_cny": _money(target_change),
            "target_scaled": _ratio(target_change / max(abs(sample.cutoff.ending_balance_cny), BALANCE_FLOOR_CNY)),
            "feature_manifest_version": FEATURE_VERSION,
        }


def split_index_row(sample: EligibleSample) -> dict[str, str]:
    return {
        "sample_key": sample.sample_key,
        "enterprise_id": sample.enterprise_id,
        "split_group": sample.split_group,
        "cutoff_date": sample.cutoff.calendar_date.isoformat(),
        "target_date": sample.target.calendar_date.isoformat(),
        "business_step": str(sample.cutoff.business_step),
        "lookback_start_date": sample.window[0].calendar_date.isoformat(),
        "lookback_business_days": str(LOOKBACK_BUSINESS_DAYS),
        "label_status": "assembled_train_or_validation",
        "final_test_sealed": "false",
        "feature_manifest_version": FEATURE_VERSION,
    }


def sealed_final_index_rows(enterprise_id: str, rows: list[tuple[date, int, date | None]]) -> Iterable[dict[str, str]]:
    for cutoff_date, business_step, target_date in rows:
        if target_date is None or not _target_in_window("final_test", target_date):
            continue
        if business_step < LOOKBACK_BUSINESS_DAYS:
            continue
        yield {
            "sample_key": f"{enterprise_id}__bd{business_step:04d}",
            "enterprise_id": enterprise_id,
            "split_group": "final_test",
            "cutoff_date": cutoff_date.isoformat(),
            "target_date": target_date.isoformat(),
            "business_step": str(business_step),
            "lookback_start_date": "sealed_until_final_test_open",
            "lookback_business_days": str(LOOKBACK_BUSINESS_DAYS),
            "label_status": "sealed_not_assembled",
            "final_test_sealed": "true",
            "feature_manifest_version": FEATURE_VERSION,
        }


def build_feature_manifest(*, population_manifest_sha256: str, t1_report_sha256: str, output_hashes: dict[str, str], counts: dict[str, int]) -> dict[str, object]:
    return {
        "stage_id": "SYN-B1-PRETRAIN-T2",
        "stage_name_cn": "14银行工作日公共特征、目标与唯一切分",
        "feature_manifest_version": FEATURE_VERSION,
        "lookback_business_days": LOOKBACK_BUSINESS_DAYS,
        "cutoff_included_in_window": True,
        "target_definition_cn": "预测截止日后的下一银行工作日账户期末余额变化；主要标签为target_change_cny，并同时报告target_balance_cny。",
        "split_key": "target_date",
        "split_windows": {group: {"target_date_start": start.isoformat() if start else None, "target_date_end": end.isoformat()} for group, (start, end) in SPLIT_WINDOWS.items()},
        "tree_input_columns": list(TREE_INPUT_COLUMNS),
        "sequence_input_columns": list(SEQUENCE_INPUT_COLUMNS),
        "forbidden_model_inputs": [
            "cash_flow_review_notes.csv", "scenario_profile.json", "generation_manifest.json", "random_seed", "sample_id", "enterprise_id",
            "任何企业画像参数", "_restricted", "future_transactions", "future_cash_events", "future_receivables_or_payables", "financing_contract_or_tail",
            "scenario_truth", "target_date", "任何target_字段",
        ],
        "final_test_policy": {
            "labels_read_or_assembled": False,
            "features_created": False,
            "sealed_index_created": True,
            "explanation_cn": "T2仅登记最终测试的预测截止日与目标日期，不读取或写出其余额标签、目标、特征、预测或指标。",
        },
        "source_t1_population_manifest_sha256": population_manifest_sha256,
        "source_t1_report_sha256": t1_report_sha256,
        "output_sha256": output_hashes,
        "counts": counts,
    }
