# 本模块只预排业务，不读取余额，也不计算另一套账本。
"""已批准九组小样的补充订单计划与逐月驱动。"""
# 数据类用于冻结每张订单。
from dataclasses import dataclass, asdict
# 日期用于备货、交货和收付合同。
from datetime import date, datetime, time
# 哈希派生互不干扰的可复现随机编号。
from hashlib import sha256
# 精确十进制规则集中计算成本到分。
from decimal import Decimal, ROUND_HALF_UP
# 随机数只在运行前生成业务清单。
from random import Random
# 日期合法性由同一新版日历判断。
from syn_b1.calendar_adapter import ORDER_CALENDAR_VERSION

# 三种收付条款仅改变现金日期。
TERMS = ("cash_at_delivery", "advance_supplier", "settlement")
# 所有可调值集中保存在参数文件，载入时逐项检查。
DEFAULT_POLICY = {"version": "supplementary_orders_1_0", "seed": 202609080201, "margin_bp": 250, "small_share_bp": 7000, "quiet_share_bp": 2000, "busy_days": 5, "busy_weight": 3, "amount_ranges_fen": {"small": [50000, 300000], "regular": [500000, 2000000], "large": [5000000, 30000000]}, "delivery_days": [3, 8], "supplier_after_delivery_days": [3, 8], "client_after_delivery_days": [5, 15], "settlement_days": [6, 16, 26], "large_per_complete_quarter": 1}

# 所有半分和半张都使用同一四舍五入定义。
def half_up(value):
    # 避免浮点数和银行家舍入改变订单数或一分钱。
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

# 成本金额只有这里拥有公式，下游只引用冻结值。
def order_cost(sale_fen, margin_bp=250):
    # 先按整数分乘毛利互补比例，再四舍五入到分。
    return half_up(Decimal(sale_fen) * Decimal(10000 - margin_bp) / 10000)

# 每个随机用途独立派生，改收付款方式不会重新抽金额。
def seeded(seed, *parts):
    # 保存整数编号可供任何一次复现追查。
    number = int.from_bytes(sha256(":".join(map(str, (seed, *parts))).encode()).digest()[:8], "big")
    # 返回独立随机对象及其编号。
    return Random(number), number

# 固定订单中的金额全部以分保存。
@dataclass(frozen=True)
# 订单保存业务事实计划，不用于模型特征。
class SupplementaryOrder:
    # 订单号在同密度的三种条款间一致。
    order_id: str
    # 类型决定人工可读备注。
    kind: str
    # 成交价不随账户余额调整。
    sale_fen: int
    # 成本直接引用唯一计算结果。
    cost_fen: int
    # 备货增加库存及应付。
    stock_date: date
    # 交货确认销售、结转成本及应收。
    delivery_date: date
    # 供应商收款日期决定现金流出。
    supplier_date: date
    # 客户付款日期决定现金流入。
    client_date: date

    # 直接调用经营程序也必须遵守金额和时间先后，不能只依赖画像入口。
    def __post_init__(self):
        # 类型与整数金额应在最早入口拒绝。
        if not self.order_id or self.kind not in ("small", "regular", "large") or any(type(v) is not int or v <= 0 for v in (self.sale_fen, self.cost_fen)):
            # 零金额或负金额不是本轮完整订单。
            raise ValueError("订单编号、类型或整数分金额无效。")
        # 日期对象必须完整，避免字符串被不小心按文字排序。
        if any(type(d) is not date for d in (self.stock_date, self.delivery_date, self.supplier_date, self.client_date)):
            # 无日期就无法确认收入或约定现金。
            raise ValueError("订单必须提供四个完整日期。")
        # 本轮没有预收账款，也没有采购之前的供应商预付。
        if not (self.stock_date <= self.delivery_date <= self.client_date and self.supplier_date >= self.stock_date):
            # 拒绝会形成负应收应付的错误日期。
            raise ValueError("订单日期违反备货、交货及约定收付先后。")

# 原始对象转换为能保存和复核的清单。
def order_payload(order):
    # 日期统一保存为年月日文字，不丢精度。
    return {key: value.isoformat() if isinstance(value, date) else value for key, value in asdict(order).items()}

# 客户集中结算选择交货日及以后的第一个实际结算日。
def settlement_date(delivery, calendar, days):
    # 从交货所在月开始寻找。
    month = delivery.replace(day=1)
    # 合同的结算日期每个月都有，遇假日只顺延。
    while True:
        # 每月按自然日先后检查全部结算日。
        for day in days:
            # 自然月固定日期是合同约定，不按余额选日。
            due = month.replace(day=day)
            # 休息日顺延到公开工作日。
            while not calendar.is_bank_workday(due):
                # 每次只前进一个自然日。
                due = date.fromordinal(due.toordinal() + 1)
            # 恰逢交货日可以当日结算。
            if due >= delivery:
                # 返回最早符合约定的到账日。
                return due
        # 跨月继续寻找，不提前收未来货款。
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)

# 预排全部订单与忙闲清单；禁止传入余额或模型结果。
def build_order_plan(policy, density, term, start, end, calendar):
    # 本轮只接受一次批准的规则，变更另存新版本。
    if policy != DEFAULT_POLICY or density not in (1, 3, 6) or term not in TERMS or calendar.version != ORDER_CALENDAR_VERSION:
        # 不默默接受未批准的业务解释。
        raise ValueError("订单规则、密度、条款或日历版本不符合九组小样约定。")
    # 先列期间内实际工作日，所有后续订单只从此选备货日。
    months = {}
    # 自然日逐日枚举，不遗漏调休周末。
    for ordinal in range(start.toordinal(), end.toordinal() + 1):
        # 转换当前自然日期。
        value = date.fromordinal(ordinal)
        # 只将工作日列入本月可选集合。
        if calendar.is_bank_workday(value):
            # 按年月保存已按日期排序的列表。
            months.setdefault(value.strftime("%Y-%m"), []).append(value)
    # 完整季度的大单月份先于所有逐单安排固定。
    large_months = set()
    # 遍历每个已有年月，季度起点只处理一次。
    for month in months:
        # 解析月序号方便判断完整季度。
        year, number = map(int, month.split("-"))
        # 只在季度第一月判断这个季度是否完整覆盖。
        if number in (1, 4, 7, 10):
            # 下一季度开始日前一天就是本季度结尾。
            next_quarter = date(year + (number == 10), 1 if number == 10 else number + 3, 1)
            # 不为残缺季度硬塞季度大单。
            if start <= date(year, number, 1) and end.toordinal() >= next_quarter.toordinal() - 1:
                # 季度大单月份独立抽取一次。
                rng, _ = seeded(policy["seed"], density, month, "large_month")
                # 每完整季度只在三个现有月份中选一个。
                large_months.add(f"{year}-{rng.choice(range(number, number + 3)):02d}")
    # 保存逐月忙闲和每单抽签编号。
    evidence, output = [], []
    # 月份按日历顺序处理，不按结果排序。
    for month, workdays in months.items():
        # 少于五天不满足已批准的忙碌段约定。
        if len(workdays) < policy["busy_days"]:
            # 参数不足直接拒绝，不改时间范围。
            raise ValueError("月份不足五个工作日，无法安排忙碌段。")
        # 一张订单有一次进账和一次出账。
        count = half_up(Decimal(len(workdays) * density) / 2)
        # 安静日只限制新备货，不限制已到期现金。
        rng, quiet_seed = seeded(policy["seed"], density, month, "quiet")
        # 至少一天没有新增备货。
        quiet = set(rng.sample(workdays, max(1, half_up(Decimal(len(workdays) * policy["quiet_share_bp"]) / 10000))))
        # 连续五个工作日独立选择，不要求连续自然日。
        rng, busy_seed = seeded(policy["seed"], density, month, "busy")
        # 忙碌段必须留在同一个自然月。
        busy_start = rng.randrange(len(workdays) - policy["busy_days"] + 1)
        # 保存真实抽出的五个工作日。
        busy = workdays[busy_start:busy_start + policy["busy_days"]]
        # 大单占用既有订单位置，不能额外追加。
        rng, large_seed = seeded(policy["seed"], density, month, "large_position")
        # 非大单月份没有大单位置。
        large_pos = rng.randrange(count) if month in large_months else None
        # 按完整非大单数确定小单数。
        small_count = half_up(Decimal(count - int(large_pos is not None)) * policy["small_share_bp"] / 10000)
        # 订单类型位置随机打散，避免小单全部挤在固定编号前。
        rng, kinds_seed = seeded(policy["seed"], density, month, "kinds")
        # 小单位置不会和已固定的大单位置重复。
        small_positions = set(rng.sample([i for i in range(count) if i != large_pos], small_count))
        # 实际可备货日期排除安静日。
        eligible = [d for d in workdays if d not in quiet]
        # 保存本月的预先安排供复审。
        evidence.append({"month": month, "bankdays": len(workdays), "order_count": count, "quiet_dates": [d.isoformat() for d in sorted(quiet)], "busy_dates": [d.isoformat() for d in busy], "seeds": [quiet_seed, busy_seed, large_seed, kinds_seed]})
        # 每单抽取独立金额、日期和条款间隔。
        for index in range(count):
            # 三种现金方式共用同一订单类型。
            kind = "large" if index == large_pos else ("small" if index in small_positions else "regular")
            # 准备保存所有逐单种子。
            seeds = {}
            # 单次抽签函数不接收任何余额。
            def draw(purpose, low, high):
                # 按用途确定本单唯一随机编号。
                generator, number_seed = seeded(policy["seed"], density, month, index, purpose)
                # 留下复现依据。
                seeds[purpose] = number_seed
                # 在明确整数范围中只抽一次。
                return generator.randint(low, high)
            # 成交金额以整数分直接抽取。
            sale = draw("sale", *policy["amount_ranges_fen"][kind])
            # 备货日期独立按忙闲权重抽取。
            rng, seeds["stock"] = seeded(policy["seed"], density, month, index, "stock")
            # 忙碌段内可选日期权重为三，其他为一。
            stock = rng.choices(eligible, weights=[policy["busy_weight"] if d in busy else 1 for d in eligible], k=1)[0]
            # 交货日期只由备货日期及生产用时决定。
            delivery = calendar.after_workdays(stock, draw("delivery", *policy["delivery_days"]))
            # 两种可能的账期都预先抽好，三组均保存同一抽签依据。
            supplier_lag = draw("supplier", *policy["supplier_after_delivery_days"])
            # 客户延期只适用于垫款条款。
            client_lag = draw("client", *policy["client_after_delivery_days"])
            # 供应商给短账期时，交货后才付款；其余备货当日付款。
            supplier = calendar.after_workdays(delivery, supplier_lag) if term == "cash_at_delivery" else stock
            # 客户支付按已声明的三种业务安排执行。
            client = delivery if term == "cash_at_delivery" else (calendar.after_workdays(delivery, client_lag) if term == "advance_supplier" else settlement_date(delivery, calendar, policy["settlement_days"]))
            # 每张订单只调用一次成本计算所有者。
            order = SupplementaryOrder(f"extra-{density}-{month}-{index:04d}", kind, sale, order_cost(sale, policy["margin_bp"]), stock, delivery, supplier, client)
            # 序列化时保留全部抽签与实际日期，不只保存随机种子。
            output.append({**order_payload(order), "derived_seeds": seeds})
    # 同时返回业务清单与月计划，生成账本之前先保存。
    return {"policy": policy, "density": density, "term": term, "monthly_design": evidence, "orders": output}

# 从已保存清单载入，不在记账时重新抽取订单。
def load_orders(raw, start, end, calendar):
    # 旧画像完全没有新字段时保持旧行为。
    if raw is None:
        # 空元组不会增加任何业务。
        return ()
    # 按相同规则重建只用于核对冻结参数与清单，没有读取现金。
    expected = build_order_plan(raw["policy"], raw["density"], raw["term"], start, end, calendar)
    # 清单中的任何金额、日期、重复编号和派生编号被修改都拒绝。
    if raw != expected:
        # 要改规则必须另存版本，不能悄悄改已固定订单。
        raise ValueError("补充订单清单与保存的规则、种子或日期不一致。")
    # 显式转换日期并只接纳数据类字段。
    return tuple(SupplementaryOrder(**{k: date.fromisoformat(v) if k.endswith("_date") else v for k, v in item.items() if k != "derived_seeds"}) for item in raw["orders"])

# 月度经营所有者消费这些原始业务驱动，不另算余额。
def month_drivers(orders, month):
    # 对四个业务动作分别按实际约定月份加总。
    return {"sales": sum(o.sale_fen for o in orders if o.delivery_date.strftime("%Y-%m") == month), "cogs": sum(o.cost_fen for o in orders if o.delivery_date.strftime("%Y-%m") == month), "purchases": sum(o.cost_fen for o in orders if o.stock_date.strftime("%Y-%m") == month), "inventory_before": sum(o.cost_fen for o in orders if o.stock_date.strftime("%Y-%m") < month <= o.delivery_date.strftime("%Y-%m"))}

# 现金事件仍挂在原月度经营来源之下，防止重复记现金。
def order_cash_events(orders, start, end):
    # 局部导入避免经营模块与逐笔模块形成循环导入。
    from syn_b1.transaction_planner import DatedSourceEvent
    # 收款与付款分别保留原始订单编号。
    events = []
    # 顺序使用预排清单，不随现金结果排序。
    for order in orders:
        # 客户进账与供应商出账分开处理。
        for suffix, due, amount in (("sales_collection", order.client_date, order.sale_fen), ("supplier_payment", order.supplier_date, order.cost_fen)):
            # 未来未到期的事件不进入本期现金候选。
            if start <= due <= end:
                # 实际备注只描述已发生的收付，不透露未来计划。
                note = {"small": "零散小单", "regular": "常规订单", "large": "偶发集中订单"}[order.kind] + ("货款到账" if suffix == "sales_collection" else "采购付款")
                # 使用固定上午十点，保留同日多单而不合并后重复。
                events.append(DatedSourceEvent(f"{order.order_id}:{suffix}", f"i1a:{due:%Y-%m}:{suffix}", datetime.combine(due, time(10)), amount, business_order_id=order.order_id, historical_note_cn=note))
    # 日期与金额已冻结，后续禁止再次随机拆分。
    return tuple(events)
