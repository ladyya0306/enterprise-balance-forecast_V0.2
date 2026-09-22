"""R6 in-memory China bank-workday and three-date inference.

This module consumes the continuous ``DailyRecord`` sequence from R5 and
creates a separate immutable calendar result.  It deliberately does not alter
R5 records and has no CSV, SQLite, feature, model, or CLI responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from runtime_daily_aggregation import DailyRecord


CODE_CALENDAR_INFERRED = "BC_CALENDAR_INFERRED_001"
CODE_EMPTY_INPUT = "BC_EMPTY_INPUT_001"

CALENDAR_SOURCE_WEEKDAY_RULE = "weekday_rule"
CALENDAR_SOURCE_GOV_CN_2025 = "gov_cn_2025_holiday_schedule_v1"
CALENDAR_SOURCE_GOV_CN_2026 = "gov_cn_2026_holiday_schedule_v1"
CALENDAR_SOURCE_HISTORICAL_TRANSACTION = "historical_transaction_override"

# The sources and their release versions remain explicit so a later calendar
# update can introduce a new version without silently changing this frozen R6
# implementation.
CALENDAR_SOURCE_REFERENCE_URLS = {
    CALENDAR_SOURCE_GOV_CN_2025: "https://www.gov.cn/zhengce/content/202411/content_6986383.htm",
    CALENDAR_SOURCE_GOV_CN_2026: "https://www.gov.cn/zhengce/zhengceku/202511/content_7047091.htm",
}
CALENDAR_SOURCE_PUBLISHED_DATES = {
    CALENDAR_SOURCE_GOV_CN_2025: date(2024, 11, 12),
    CALENDAR_SOURCE_GOV_CN_2026: date(2025, 11, 4),
}


@dataclass(frozen=True)
class CalendarRecord:
    """One natural date's inferred bank-workday and prediction-date fields."""

    calendar_date: str
    is_bank_workday: bool
    calendar_source: str
    calendar_inference_status: str
    business_step: int | None
    target_date: str | None
    historical_override_source_last_transaction_id: str | None = None
    historical_override_transaction_count: int | None = None


@dataclass(frozen=True)
class CalendarInferenceResult:
    code: str
    message_zh: str
    records: tuple[CalendarRecord, ...]
    write_count: int = 0


# These date sets are the published State Council holiday/makeup schedules,
# limited to the approved 2025--2026 scope.  A date in HOLIDAYS overrides the
# usual Monday--Friday rule; a date in MAKEUP_WORKDAYS overrides the weekend.
_HOLIDAYS_BY_SOURCE: dict[str, frozenset[str]] = {
    CALENDAR_SOURCE_GOV_CN_2025: frozenset(
        {
            "2025-01-01",
            "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",
            "2025-04-04", "2025-04-05", "2025-04-06",
            "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",
            "2025-05-31", "2025-06-01", "2025-06-02",
            "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08",
        }
    ),
    CALENDAR_SOURCE_GOV_CN_2026: frozenset(
        {
            "2026-01-01", "2026-01-02", "2026-01-03",
            "2026-02-15", "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23",
            "2026-04-04", "2026-04-05", "2026-04-06",
            "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
            "2026-06-19", "2026-06-20", "2026-06-21",
            "2026-09-25", "2026-09-26", "2026-09-27",
            "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
        }
    ),
}

_MAKEUP_WORKDAYS_BY_SOURCE: dict[str, frozenset[str]] = {
    CALENDAR_SOURCE_GOV_CN_2025: frozenset({"2025-01-26", "2025-02-08", "2025-04-27", "2025-09-28", "2025-10-11"}),
    CALENDAR_SOURCE_GOV_CN_2026: frozenset({"2026-01-04", "2026-02-14", "2026-02-28", "2026-05-09", "2026-09-20", "2026-10-10"}),
}


def infer_bank_calendar(
    daily_records: Iterable[DailyRecord],
    *,
    prediction_cutoff: str | date,
) -> CalendarInferenceResult:
    """Infer R6 workdays, business steps, and next-workday targets in memory.

    ``prediction_cutoff`` is required to keep future transaction rows from
    leaking into calendar inference.  A transaction dated on or before this
    cutoff can demonstrate that an otherwise non-workday was bookable for the
    account.  Later dates use only the public base calendar and weekday rule.
    """

    source_records = tuple(daily_records)
    if not source_records:
        return CalendarInferenceResult(
            code=CODE_EMPTY_INPUT,
            message_zh="没有R5日度记录可推导银行工作日历；本步骤未写入文件或数据库。",
            records=(),
        )

    cutoff = _as_date(prediction_cutoff)
    dates = tuple(_as_date(record.calendar_date) for record in source_records)
    _validate_ascending_unique_dates(dates)

    preliminary: list[tuple[str, bool, str, str, str | None, int | None]] = []
    for record, calendar_value in zip(source_records, dates, strict=True):
        is_workday, source = _base_calendar(calendar_value, prediction_cutoff=cutoff)
        if (
            calendar_value <= cutoff
            and record.transaction_count > 0
            and not is_workday
        ):
            is_workday = True
            source = CALENDAR_SOURCE_HISTORICAL_TRANSACTION
            status = "historical_transaction_override"
            override_source_last_transaction_id = record.source_last_transaction_id
            override_transaction_count = record.transaction_count
        elif calendar_value > cutoff:
            status = "future_base_calendar"
            override_source_last_transaction_id = None
            override_transaction_count = None
        else:
            status = "base_calendar"
            override_source_last_transaction_id = None
            override_transaction_count = None
        preliminary.append(
            (
                calendar_value.isoformat(),
                is_workday,
                source,
                status,
                override_source_last_transaction_id,
                override_transaction_count,
            )
        )

    business_step = 0
    stepped: list[CalendarRecord] = []
    for (
        calendar_date,
        is_workday,
        source,
        status,
        override_source_last_transaction_id,
        override_transaction_count,
    ) in preliminary:
        if is_workday:
            business_step += 1
            step: int | None = business_step
        else:
            step = None
        stepped.append(
            CalendarRecord(
                calendar_date=calendar_date,
                is_bank_workday=is_workday,
                calendar_source=source,
                calendar_inference_status=status,
                business_step=step,
                target_date=None,
                historical_override_source_last_transaction_id=override_source_last_transaction_id,
                historical_override_transaction_count=override_transaction_count,
            )
        )

    next_workday: str | None = _next_base_workday_after(dates[-1], prediction_cutoff=cutoff).isoformat()
    output: list[CalendarRecord] = []
    for record in reversed(stepped):
        output.append(
            CalendarRecord(
                calendar_date=record.calendar_date,
                is_bank_workday=record.is_bank_workday,
                calendar_source=record.calendar_source,
                calendar_inference_status=record.calendar_inference_status,
                business_step=record.business_step,
                target_date=next_workday if record.is_bank_workday else None,
                historical_override_source_last_transaction_id=record.historical_override_source_last_transaction_id,
                historical_override_transaction_count=record.historical_override_transaction_count,
            )
        )
        if record.is_bank_workday:
            next_workday = record.calendar_date
    output.reverse()

    return CalendarInferenceResult(
        code=CODE_CALENDAR_INFERRED,
        message_zh="已按公开节假日/调休、普通星期规则及历史可记账证据推导三层日期；本步骤未写入文件或数据库。",
        records=tuple(output),
    )


def _base_calendar(calendar_value: date, *, prediction_cutoff: date) -> tuple[bool, str]:
    calendar_date = calendar_value.isoformat()
    for source, holiday_dates in _HOLIDAYS_BY_SOURCE.items():
        if (
            prediction_cutoff >= CALENDAR_SOURCE_PUBLISHED_DATES[source]
            and calendar_date in holiday_dates
        ):
            return False, source
    for source, makeup_dates in _MAKEUP_WORKDAYS_BY_SOURCE.items():
        if (
            prediction_cutoff >= CALENDAR_SOURCE_PUBLISHED_DATES[source]
            and calendar_date in makeup_dates
        ):
            return True, source
    return calendar_value.weekday() < 5, CALENDAR_SOURCE_WEEKDAY_RULE


def is_published_bank_workday(
    calendar_value: date,
    *,
    prediction_cutoff: str | date,
) -> bool:
    """Return the frozen public-calendar result without transaction overrides.

    This is the small public entry point for components, such as SYN-B1, that
    need the published holiday/makeup schedule but must not invent a second
    copy of it.  Historical-account transaction overrides remain exclusively
    inside :func:`infer_bank_calendar`.
    """

    cutoff = _as_date(prediction_cutoff)
    return _base_calendar(calendar_value, prediction_cutoff=cutoff)[0]


def _next_base_workday_after(calendar_value: date, *, prediction_cutoff: date) -> date:
    """Find the next public-calendar workday without reading future transactions."""

    candidate = calendar_value + timedelta(days=1)
    # A full year is far beyond any published holiday interval and avoids an
    # unbounded loop should a future calendar source be configured incorrectly.
    for _ in range(366):
        is_workday, _ = _base_calendar(candidate, prediction_cutoff=prediction_cutoff)
        if is_workday:
            return candidate
        candidate += timedelta(days=1)
    raise ValueError("R6无法在366天内推导下一银行工作日")


def _validate_ascending_unique_dates(dates: tuple[date, ...]) -> None:
    if any(current <= previous for previous, current in zip(dates, dates[1:])):
        raise ValueError("R6日度记录必须按calendar_date严格升序且不得重复")


def _as_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)
