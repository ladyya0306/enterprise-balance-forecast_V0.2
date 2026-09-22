# 周期订单只确定业务日期与预算，全部现金仍交给原账务模块。
"""严格旬、月、季度、年度贸易订单，先保存再记账。"""
# 自然周期以月份天数划分。
from calendar import monthrange
# 日期不借用预测结果。
from datetime import date
# 原订单类型及成本所有者保持唯一。
from syn_b1.supplementary_orders import SupplementaryOrder, order_cost, order_payload
# 新日历显式选择，不改变历史日历。
from syn_b1.calendar_adapter import CYCLE_CALENDAR_VERSION

# 本轮只接受已经确认的业务预算。
POLICY = {"version": "cycle_trade_20260908_1_0", "monthly_sales_fen": 100000000, "margin_bp": 250, "receipt": "first_workday", "supplier": "last_workday", "delivery": "same_day_trade"}

# 自然旬和季度边界不按工作日数替代。
def periods(frequency, start, end):
    # 拒绝残缺观察月，不隐瞒缺失周期。
    if frequency not in ("ten", "month", "quarter", "year") or start.day != 1 or end.day != monthrange(end.year, end.month)[1] or start > end:
        # 输入不完整时停止预排，不凑日期。
        raise ValueError("周期名称或完整月份范围无效。")
    # 保存全部自然周期。
    result = []
    # 每个月按自然顺序检查。
    for ordinal in range(start.year * 12 + start.month - 1, end.year * 12 + end.month):
        # 月序号转换为年月。
        year, month = ordinal // 12, ordinal % 12 + 1
        # 旬内金额把余分按日期顺序分配。
        if frequency == "ten":
            # 每月固定三个真实旬。
            for index, (left, right) in enumerate(((1, 10), (11, 20), (21, monthrange(year, month)[1]))):
                # 月预算到分相加保持完全相同。
                result.append((date(year, month, left), date(year, month, right), 100000000 // 3 + int(index < 100000000 % 3)))
        # 其他周期从相应首月开始。
        elif frequency == "month" or (frequency == "quarter" and month in (1, 4, 7, 10)) or (frequency == "year" and month == 1):
            # 月、季、年分别承接相应月数的经营预算。
            length = {"month": 1, "quarter": 3, "year": 12}[frequency]
            # 周期结束均位于同一自然年。
            result.append((date(year, month, 1), date(year, month + length - 1, monthrange(year, month + length - 1)[1]), 100000000 * length))
    # 开头结尾都必须恰好覆盖观察期间。
    if not result or result[0][0] != start or result[-1][1] != end:
        # 不将残缺季度或年度称作完整周期。
        raise ValueError("观察期间必须由完整目标周期组成。")
    # 返回冻结顺序。
    return result

# 计划不接受余额参数，金额完全来自固定经营预算。
def build_cycle_plan(frequency, start, end, calendar):
    # 新周期只能用明确覆盖所需年份的新版日历。
    if calendar.version != CYCLE_CALENDAR_VERSION:
        # 不允许落回旧版自然日覆盖日历。
        raise ValueError("严格周期需要新版公开日历。")
    # 订单及周期核查同时生成。
    orders, spans = [], []
    # 所有周期都在账本生成前排好。
    for index, (left, right, sale) in enumerate(periods(frequency, start, end), 1):
        # 枚举本周期真正工作日。
        workdays = [date.fromordinal(n) for n in range(left.toordinal(), right.toordinal() + 1) if calendar.is_bank_workday(date.fromordinal(n))]
        # 至少两天才能分开主要收付款。
        if len(workdays) < 2:
            # 明确指出业务无法安排的周期。
            raise ValueError(f"{left}至{right}不足两个工作日。")
        # 贸易交接及验收付款同日，采购款周期末结算。
        order = SupplementaryOrder(f"cycle-{frequency}-{index:03d}", "regular", sale, order_cost(sale, POLICY["margin_bp"]), workdays[0], workdays[0], workdays[-1], workdays[0])
        # 原订单格式不包含生成后余额。
        orders.append(order_payload(order))
        # 日期表保留周期边界及可用日数。
        spans.append({"order_id": order.order_id, "start": left.isoformat(), "end": right.isoformat(), "workday_count": len(workdays)})
    # 明确这批订单替代旧主业，避免重复销售。
    return {"policy": dict(POLICY), "frequency": frequency, "periods": spans, "orders": orders}

# 参数文件中的冻结清单必须与规则逐字对应。
def load_cycle_orders(raw, start, end, calendar):
    # 旧画像保持旧行为。
    if raw is None:
        # 没有新周期就没有新订单。
        return ()
    # 重建只用于验证，不读取或修改现金。
    if raw != build_cycle_plan(raw["frequency"], start, end, calendar):
        # 篡改金额、日期或业务规则都报错。
        raise ValueError("严格周期订单清单与已确认规则不一致。")
    # 日期恢复为原订单数据类型。
    return tuple(SupplementaryOrder(**{k: date.fromisoformat(v) if k.endswith("_date") else v for k, v in item.items()}) for item in raw["orders"])
