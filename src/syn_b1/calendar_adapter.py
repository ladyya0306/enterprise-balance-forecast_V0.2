"""SYN-B1-I7 adapter to the frozen R6 public bank-calendar implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from runtime_bank_calendar import is_published_bank_workday


FORMAL_CALENDAR_VERSION = "r6_cn_public_calendar_adapter_1_0"
# 新版只由新增小样显式选择，不改旧日历的历史含义。
ORDER_CALENDAR_VERSION = "cn_public_2024_2026_orders_1_0"
# 二〇二四年放假日期来自国务院办公厅通知，春节不包含倡议休假的除夕。
HOLIDAYS_2024 = frozenset(["2024-01-01", "2024-06-10"] + [f"2024-02-{d:02d}" for d in range(10, 18)] + [f"2024-04-{d:02d}" for d in range(4, 7)] + [f"2024-05-{d:02d}" for d in range(1, 6)] + [f"2024-09-{d:02d}" for d in range(15, 18)] + [f"2024-10-{d:02d}" for d in range(1, 8)])
# 调休上班日优先于普通周末。
MAKEUP_2024 = frozenset(["2024-02-04", "2024-02-18", "2024-04-07", "2024-04-28", "2024-05-11", "2024-09-14", "2024-09-29", "2024-10-12"])


# 四年周期另立日历版本，不改变九组历史日表。
CYCLE_CALENDAR_VERSION = "cn_public_2022_2026_cycles_1_0"
# 官方通知来源与公布日随版本保存。
CYCLE_SOURCES = {2022: ("2021-10-25", "https://app.www.gov.cn/govdata/gov/202110/25/477428/article.html"), 2023: ("2022-12-08", "https://app.www.gov.cn/govdata/gov/202212/08/495070/article.html")}
# 只录入官方放假区间，不推测未来年份。
CYCLE_HOLIDAY_SPANS = {2022: [("2022-01-01", "2022-01-03"), ("2022-01-31", "2022-02-06"), ("2022-04-03", "2022-04-05"), ("2022-04-30", "2022-05-04"), ("2022-06-03", "2022-06-05"), ("2022-09-10", "2022-09-12"), ("2022-10-01", "2022-10-07")], 2023: [("2023-01-01", "2023-01-02"), ("2023-01-21", "2023-01-27"), ("2023-04-05", "2023-04-05"), ("2023-04-29", "2023-05-03"), ("2023-06-22", "2023-06-24"), ("2023-09-29", "2023-10-06")]}
# 二〇二二年十二月三十一日本为周六，普通周规则已为休息日。
CYCLE_MAKEUP = {2022: {"2022-01-29", "2022-01-30", "2022-04-02", "2022-04-24", "2022-05-07", "2022-10-08", "2022-10-09"}, 2023: {"2023-01-28", "2023-01-29", "2023-04-23", "2023-05-06", "2023-06-25", "2023-10-07", "2023-10-08"}}


class SynB1CalendarError(ValueError):
    """A formal SYN-B1 calendar instruction is incomplete."""


@dataclass(frozen=True)
class SynB1BankCalendar:
    """Versioned public-calendar view used by tax routing and formal output.

    The public R6 calendar owns holiday and make-up-workday data.  This small
    adapter deliberately adds no second holiday table; it only fixes the
    prediction cutoff as part of the reproducible scenario profile.
    """

    prediction_cutoff: date
    version: str = FORMAL_CALENDAR_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_cutoff, date):
            raise SynB1CalendarError("银行日历截止日必须为日期。")
        if not self.version:
            raise SynB1CalendarError("银行日历版本不能为空。")

    def is_bank_workday(self, value: date) -> bool:
        if not isinstance(value, date):
            raise SynB1CalendarError("待判断银行工作日必须为日期。")
        # 周期版仅补充两个已公布年份，其余复用上一版本。
        if self.version == CYCLE_CALENDAR_VERSION:
            # 对更早年份检查来源是否已经公布。
            if value.year in CYCLE_SOURCES:
                # 截止日前未公布就不能提前使用。
                if self.prediction_cutoff < date.fromisoformat(CYCLE_SOURCES[value.year][0]):
                    # 显示缺失资料，而非假定周规则。
                    raise SynB1CalendarError("所需年度日历在截止日尚未公布。")
                # 调休周末优先，其余排除官方放假日。
                return value.isoformat() in CYCLE_MAKEUP[value.year] or (value.weekday() < 5 and not any(left <= value.isoformat() <= right for left, right in CYCLE_HOLIDAY_SPANS[value.year]))
            # 已支持的年份仍使用原订单日历，未知年份会被明确拒绝。
            return SynB1BankCalendar(self.prediction_cutoff, ORDER_CALENDAR_VERSION).is_bank_workday(value)
        # 新日历严格限定已收录年份，缺资料不得退回普通周规则。
        if self.version == ORDER_CALENDAR_VERSION:
            # 每年通知必须在参数截止日之前已经发布。
            releases = {2024: date(2023, 10, 25), 2025: date(2024, 11, 12), 2026: date(2025, 11, 4)}
            # 防止未收录年份或尚未公布的日历被当成完整资料。
            if value.year not in releases or self.prediction_cutoff < releases[value.year]:
                # 明确报错，不自行猜测节假日。
                raise SynB1CalendarError("新订单日历只接受截止日已公布的2024至2026年资料。")
            # 仅新版补充旧实现未覆盖的二〇二四年。
            if value.year == 2024:
                # 放假、调休和普通星期共同决定公开工作日。
                return value.isoformat() in MAKEUP_2024 or (value.isoformat() not in HOLIDAYS_2024 and value.weekday() < 5)
        return is_published_bank_workday(value, prediction_cutoff=self.prediction_cutoff)

    # 日期推进统一复用本对象的版本化判断。
    def after_workdays(self, value: date, count: int) -> date:
        # 零天保留当天，正数不把起点重复计入。
        while count > 0:
            # 逐自然日向后查看。
            value = date.fromordinal(value.toordinal() + 1)
            # 只累计已公开的工作日。
            count -= int(self.is_bank_workday(value))
        # 返回唯一约定日期。
        return value

    # 新输出日表直接呈现公开工作日，不用自然日交易倒推工作日。
    def public_records(self, daily_records):
        # 复用既有字段类型，保持日表列名不变。
        from runtime_bank_calendar import CalendarRecord
        # 累计工作日序号与输出记录。
        step, result = 0, []
        # 日期来自原日度账务记录，金额完全不修改。
        for record in daily_records:
            # 读取唯一自然日键。
            value = date.fromisoformat(record.calendar_date)
            # 不用当日是否发生交易改变公开工作日。
            workday = self.is_bank_workday(value)
            # 只给工作日累计编号。
            step += int(workday)
            # 次一工作日严格复用新日历版本。
            # 截止日之外的下一工作日属于未知公开信息，留空而不猜测。
            target = self.after_workdays(value, 1).isoformat() if workday and value < self.prediction_cutoff else None
            # 公开口径写入来源字段，避免误认为旧版历史交易覆盖。
            result.append(CalendarRecord(record.calendar_date, workday, self.version, "public_schedule_without_transaction_override", step if workday else None, target))
        # 所有记录已按输入自然日顺序排列。
        return tuple(result)
