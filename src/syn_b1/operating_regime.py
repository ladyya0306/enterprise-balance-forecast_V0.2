"""Version-2 monthly operating-regime drivers for SYN-B1.

This module owns only latent sales, sales seasonality, segment margin and
collection-timing parameters.  The operating-cycle engine remains the single
owner of receivable, inventory, payable, expense and payment rolling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

from syn_b1.operating_cycle import DelayShare


class OperatingRegimeError(ValueError):
    pass


@dataclass(frozen=True)
class OperatingRegimeSegment:
    segment_id: str
    start_month: str
    monthly_linear_change_bp: int
    gross_margin_bp: int

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise OperatingRegimeError("经营阶段编号不能为空。")
        _parse_month(self.start_month)
        if not -2_000 <= self.monthly_linear_change_bp <= 2_000:
            raise OperatingRegimeError("阶段月度变化率必须在-20%至20%之间。")
        if not -10_000 <= self.gross_margin_bp <= 10_000:
            raise OperatingRegimeError("阶段毛利率必须在-100%至100%之间。")


@dataclass(frozen=True)
class OperatingRegimePlan:
    shape: str
    declared_through_month: str
    base_latent_monthly_sales_fen: int
    segments: tuple[OperatingRegimeSegment, ...]
    sales_month_weights_bp: Mapping[int, int]
    collection_delay_adjustment_days_by_sales_month: Mapping[int, int]
    opening_expected_cogs_ratio_bp: int
    procurement_adjustment_speed_bp: int

    def __post_init__(self) -> None:
        allowed = {
            "stable", "sustained_growth", "sustained_contraction",
            "growth_to_contraction", "contraction_to_growth",
        }
        if self.shape not in allowed:
            raise OperatingRegimeError("经营形态无效。")
        _parse_month(self.declared_through_month)
        if self.base_latent_monthly_sales_fen <= 0:
            raise OperatingRegimeError("去季节化月销售锚点必须大于0。")
        expected_count = 2 if "_to_" in self.shape else 1
        if len(self.segments) != expected_count:
            raise OperatingRegimeError("经营形态与阶段数量不一致。")
        if len({item.segment_id for item in self.segments}) != len(self.segments):
            raise OperatingRegimeError("经营阶段编号不能重复。")
        starts = [_month_index(item.start_month) for item in self.segments]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise OperatingRegimeError("经营阶段开始月份必须严格递增。")
        slopes = [item.monthly_linear_change_bp for item in self.segments]
        rules = {
            "stable": slopes == [0],
            "sustained_growth": len(slopes) == 1 and slopes[0] > 0,
            "sustained_contraction": len(slopes) == 1 and slopes[0] < 0,
            "growth_to_contraction": len(slopes) == 2 and slopes[0] > 0 and slopes[1] < 0,
            "contraction_to_growth": len(slopes) == 2 and slopes[0] < 0 and slopes[1] > 0,
        }
        if not rules[self.shape]:
            raise OperatingRegimeError("经营形态与各阶段变化率方向不一致。")
        if set(self.sales_month_weights_bp) != set(range(1, 13)):
            raise OperatingRegimeError("销售季节权重必须完整填写1至12月。")
        if any(not 0 <= value <= 30_000 for value in self.sales_month_weights_bp.values()) or not any(self.sales_month_weights_bp.values()):
            raise OperatingRegimeError("销售季节权重必须在0%至300%之间且至少一个月大于0。")
        if set(self.collection_delay_adjustment_days_by_sales_month) != set(range(1, 13)):
            raise OperatingRegimeError("回款季节调整必须完整填写1至12月。")
        if any(not -60 <= value <= 60 for value in self.collection_delay_adjustment_days_by_sales_month.values()):
            raise OperatingRegimeError("回款季节调整必须在-60至60天之间。")
        if not 0 <= self.opening_expected_cogs_ratio_bp <= 30_000:
            raise OperatingRegimeError("采购初始预期比例必须在0%至300%之间。")
        if not 0 <= self.procurement_adjustment_speed_bp <= 10_000:
            raise OperatingRegimeError("采购调整速度必须在0%至100%之间。")


@dataclass(frozen=True)
class MonthlyRegimeDriver:
    month: str
    segment_id: str
    latent_sales_fen: int
    full_month_sales_fen: int
    gross_margin_bp: int
    collection_delay_adjustment_days: int


def build_monthly_regime_drivers(
    plan: OperatingRegimePlan,
    *,
    start_date: date,
    end_date: date,
) -> tuple[MonthlyRegimeDriver, ...]:
    months = _period_months(start_date, end_date)
    if plan.segments[0].start_month != months[0]:
        raise OperatingRegimeError("第一经营阶段必须从模拟开始月份开始。")
    if _month_index(plan.declared_through_month) < _month_index(months[-1]):
        raise OperatingRegimeError("经营阶段声明截止月份没有覆盖模拟结束日期。")
    if any(_month_index(item.start_month) > _month_index(plan.declared_through_month) for item in plan.segments):
        raise OperatingRegimeError("经营阶段开始月份不能晚于阶段声明截止月份。")

    normalized = _normalized_weights(plan.sales_month_weights_bp)
    segments = plan.segments
    anchors: dict[str, int] = {segments[0].segment_id: plan.base_latent_monthly_sales_fen}
    for previous, current in zip(segments, segments[1:]):
        offset = _month_index(current.start_month) - _month_index(previous.start_month)
        anchor = anchors[previous.segment_id]
        factor = 10_000 + previous.monthly_linear_change_bp * offset
        if factor <= 0:
            raise OperatingRegimeError("转折前销售趋势已使边界销售额不为正。")
        anchors[current.segment_id] = anchor * factor // 10_000

    rows: list[MonthlyRegimeDriver] = []
    for month in months:
        active = segments[0]
        for candidate in segments[1:]:
            if _month_index(candidate.start_month) <= _month_index(month):
                active = candidate
        offset = _month_index(month) - _month_index(active.start_month)
        factor = 10_000 + active.monthly_linear_change_bp * offset
        latent = anchors[active.segment_id] * factor // 10_000
        if latent <= 0:
            raise OperatingRegimeError(f"{month}经营趋势使销售额不为正，请调整阶段变化率或期间。")
        month_number = int(month[5:7])
        sales = latent * normalized[month_number] // 10_000
        rows.append(MonthlyRegimeDriver(
            month=month,
            segment_id=active.segment_id,
            latent_sales_fen=latent,
            full_month_sales_fen=sales,
            gross_margin_bp=active.gross_margin_bp,
            collection_delay_adjustment_days=plan.collection_delay_adjustment_days_by_sales_month[month_number],
        ))
    return tuple(rows)


def adjusted_collection_schedule(
    schedule: Sequence[DelayShare], adjustment_days: int,
) -> tuple[DelayShare, ...]:
    adjusted = tuple(DelayShare(item.delay_days + adjustment_days, item.share_bp) for item in schedule)
    if any(item.delay_days < 0 for item in adjusted):
        raise OperatingRegimeError("回款季节调整使回款早于销售确认，请减小负向调整。")
    return adjusted


def _normalized_weights(weights: Mapping[int, int]) -> dict[int, int]:
    total = sum(weights.values())
    return {month: weights[month] * 120_000 // total for month in range(1, 13)}


def _period_months(start_date: date, end_date: date) -> tuple[str, ...]:
    current = date(start_date.year, start_date.month, 1)
    last = date(end_date.year, end_date.month, 1)
    result: list[str] = []
    while current <= last:
        result.append(f"{current.year:04d}-{current.month:02d}")
        current = date(current.year + (current.month == 12), 1 if current.month == 12 else current.month + 1, 1)
    return tuple(result)


def _parse_month(value: str) -> tuple[int, int]:
    try:
        year_text, month_text = value.split("-")
        year, month = int(year_text), int(month_text)
    except (AttributeError, ValueError) as error:
        raise OperatingRegimeError("月份必须使用YYYY-MM格式。") from error
    if len(value) != 7 or not 1 <= month <= 12 or year < 1:
        raise OperatingRegimeError("月份必须使用YYYY-MM格式。")
    return year, month


def _month_index(value: str) -> int:
    year, month = _parse_month(value)
    return year * 12 + month - 1
