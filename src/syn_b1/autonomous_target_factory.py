"""将24个形状目标编译为具有独立客户、采购和岗位来源的精确合同预算。"""
from __future__ import annotations

# 导入月末天数工具，用于校验显式合同期的结束日。
from calendar import monthrange
from datetime import date, timedelta
from decimal import Decimal
from random import Random

from syn_b1.formal_profile import default_profile, resolve_profile


MONTHS = tuple(f"{2024 + i // 12}-{i % 12 + 1:02d}" for i in range(24))


# 校验日期合同并按输入范围生成完整的24个月序列。
def _schedule(start_date: str, end_date: str, prediction_cutoff: str) -> tuple[date, date, tuple[str, ...]]:
    """校验并生成两年合同月序列；默认值保持历史画像不变。"""
    # 把三个ISO字符串解析为可比较的日期对象。
    start, end, cutoff = (date.fromisoformat(value) for value in (start_date, end_date, prediction_cutoff))
    # 合同期必须覆盖完整自然月，避免生成半月节点。
    if start.day != 1 or end.day != monthrange(end.year, end.month)[1]:
        raise ValueError("经营期间必须从月初开始并在月末结束")
    # 公开预测截止日必须与经营期间末日一致。
    if cutoff != end:
        raise ValueError("prediction_cutoff必须等于经营期间截止日")
    # 按月偏移重建合同月名，不再引用固定2024—2025常量。
    months = tuple(f"{start.year + (start.month - 1 + index) // 12}-{(start.month - 1 + index) % 12 + 1:02d}"
                   for index in range((end.year - start.year) * 12 + end.month - start.month + 1))
    # 成熟自主机制仍按24个月合同结构运行。
    if len(months) != 24:
        raise ValueError("自主机制工厂目前要求连续24个月经营期间")
    # 返回统一的日期边界和合同月序列给画像编译器。
    return start, end, months

# 同一走势不等于同一原因；名称同时是对外可读的机制标识，不参与余额计算。
MECHANISMS = {
    "T01": ("annual_service_renewal", "年度维保续约逐批增加，服务人员与备件采购随新增覆盖增加"),
    "T02": ("consumable_replenishment_growth", "客户装机扩大，按耗用补货合同逐批增加"),
    "T03": ("retainer_balance", "存量合规订阅与固定响应班组基本平衡"),
    "T04": ("exchange_parts_balance", "循环部件替换订单与常驻维修支出基本平衡"),
    "T05": ("installed_base_runoff", "客户设备逐季退出，按剩余装机耗用补货"),
    "T06": ("sunset_channel_support", "渠道收尾合同减少，分拣与场地服务持续"),
    "T07": ("early_acceptance_recovery", "早期验收完成后，已签服务合同开始持续交付"),
    "T08": ("late_certification_recovery", "认证在后期完成，认证后客户分批启动订单"),
    "T09": ("early_inventory_release", "早期逐步去库存，随后续约与补货恢复"),
    "T10": ("late_channel_recovery", "渠道库存较晚消化，随后补货合同逐步恢复"),
    "T11": ("early_rollout_completion", "早期集中上线完成，后续新增项目减少"),
    "T12": ("late_buildout_completion", "后期项目建设完成，建设订单结束后留下运维"),
    "T13": ("early_premium_service_end", "早期临时高毛利响应服务到期，转为常规服务"),
    "T14": ("late_promotional_end", "后期促销与高附加服务结束，常规订单回落"),
    "T15": ("two_wave_delivery", "首轮项目收尾、第二轮交付启动、后续项目再次收尾"),
    "T16": ("staggered_contract_recovery", "客户合同错峰退出、续签恢复、末段再次退出"),
    "T17": ("inventory_rebuild_then_runoff", "先消耗安全库存、再补建库存，最终收缩采购"),
    "T18": ("two_wave_tender", "首轮招标交付、资源占用下降、第二轮招标启动"),
    "T19": ("maintenance_upgrade_cycle", "维护升级启动、旧批合同结束、新批升级上线"),
    "T20": ("channel_launch_and_consolidation", "渠道启动、整合期放缓、后续区域扩展"),
    "T21": ("alternating_bid_portfolios", "四组独立客户招标订单依次启动和收尾"),
    "T22": ("service_and_rollout_rotation", "服务续签和分批铺设合同交替产生现金贡献"),
    "T23": ("repair_shutdown_recovery", "检修约束、恢复交付、第二轮整合和后续恢复"),
    "T24": ("channel_drawdown_restock", "渠道去库存、回补、再去库存和最终恢复"),
}


def _money(fen: int) -> str:
    return f"{fen // 100}.{fen % 100:02d}"


def _split(total: int, count: int, rng: Random) -> list[int]:
    weights = [rng.randint(8, 28) for _ in range(count)]
    values = [total * weight // sum(weights) for weight in weights]
    values[-1] += total - sum(values)
    return values


# 构建可选显式日期合同的自主画像，默认值保持旧调用兼容。
def build_target_profile(target: dict, revision: int, *, seed_base: int, stage_order_adjustment: tuple[float, ...] | None = None,
                         mechanism_route: dict | None = None, start_date: str = "2024-01-01",
                         end_date: str = "2025-12-31", prediction_cutoff: str = "2025-12-31"):
    """冻结一份独立经营假设；订单量而非余额是唯一的形状驱动。"""
    # 先校验日期合同，后续资产、节点和事件均只使用该结果。
    start, end, months = _schedule(start_date, end_date, prediction_cutoff)
    target_id = target["target_id"]
    # A route is a frozen contract-plan input.  Copy it before adding derived
    # defaults so a caller's archived route is never mutated in place.
    route = dict(mechanism_route or {})
    if route and route.get("target_id") not in {None, target_id}:
        raise ValueError("机制路线目标与形状目标不一致")
    mechanism_id, reason = MECHANISMS[target_id]
    mechanism_id = route.get("mechanism_id", mechanism_id)
    reason = route.get("business_reason", reason)
    # Retries alter only the stage-order-volume group.  Contract population,
    # counterparties and payment rhythm must not drift merely because a retry
    # number changed.
    route_key = sum((index + 1) * ord(char) for index, char in enumerate(mechanism_id))
    rng = Random(seed_base + int(target_id[1:]) * 100 + route_key)
    profile = default_profile(f"autonomous_{target_id.lower()}_{mechanism_id}_r{revision:02d}", "autonomous_target_v2")
    # 将画像期间写为显式日期，而不是历史常量。
    profile.update(schema_version="syn_b1_formal_profile_2_0", start_date=start.isoformat(), end_date=end.isoformat(), random_seed=seed_base + int(target_id[1:]) * 100 + route_key)
    # 将同一截止日写入公开银行日历配置。
    profile["calendar"]["prediction_cutoff"] = prediction_cutoff
    # 期初资金在每户预算前冻结；不依据后续余额临时追加。
    profile["initial_capital"]["amount_cny"] = "3600000.00"
    operating = profile["operating"]
    for key in ("base_monthly_sales_cny", "sales_monthly_growth_percent", "sales_month_weights_percent", "gross_margin_percent", "monthly_payroll_cny", "monthly_rent_and_utilities_cny", "monthly_other_operating_expense_cny"):
        operating.pop(key, None)
    for key in ("collection_schedule", "supplier_payment_schedule", "payroll_payment_schedule", "rent_and_utilities_payment_schedule", "other_operating_expense_payment_schedule"):
        operating[key] = [{"delay_days": 0, "share_percent": "100"}]
    # 固定资产的采购月和投用日随显式合同起点重建。
    operating["fixed_assets"] = [{"event_id": "EQ01", "asset_type": "equipment", "purchase_month": months[0], "purchase_amount_cny": "60000.00", "payment_schedule": [{"delay_days": 0, "share_percent": "100"}], "ready_for_use_date": start.isoformat(), "useful_life_months": 60, "residual_value_percent": "5"}]
    # 每户客户、供应商、岗位的组成独立；客户数与集中度随机制决定。
    project = route.get("project", target_id in {"T07", "T08", "T11", "T12", "T15", "T16", "T17", "T18", "T19", "T20", "T21", "T22", "T23", "T24"})
    low_dispersion = target["volatility"] == "低"
    customer_range = route.get("customer_count_range", [3, 7] if project else [18, 30] if low_dispersion else [10, 18])
    supplier_range = route.get("supplier_count_range", [5, 10] if low_dispersion else [3, 8])
    if len(customer_range) != 2 or len(supplier_range) != 2 or not 1 <= customer_range[0] <= customer_range[1] <= 30 or not 1 <= supplier_range[0] <= supplier_range[1] <= 12:
        raise ValueError("客户或供应商结构超出受控范围")
    customer_count = rng.randint(*customer_range); supplier_count = rng.randint(*supplier_range)
    # The legacy mode is retained for frozen predecessors.  New mechanism
    # routes may instead declare a customer-specific service/acceptance
    # calendar and supplier statement calendar before any cashflow is made.
    # These are contract facts, not post-generation date jitter.
    calendar_pattern = route.get("contract_calendar_pattern", "legacy_uniform_monthly")
    if calendar_pattern not in {"legacy_uniform_monthly", "staggered_service_and_statement_calendar_v1"}:
        raise ValueError("未知的合同日历机制")
    customer_windows = supplier_windows = None
    customer_credit_keys = supplier_credit_keys = None
    if calendar_pattern == "staggered_service_and_statement_calendar_v1":
        # Each customer has its own pre-signed monthly service window; each
        # supplier has its own delivery/statement window.  The complete
        # two-year calendar is fixed now and written into the event facts.
        # A dedicated deterministic stream keeps the calendar mechanism from
        # silently changing price, cost, staffing or batch amounts.
        calendar_rng = Random(profile["random_seed"] + 9173)
        # 客户窗口按重建后的合同月数生成，避免遗留固定月份。
        customer_windows = [[calendar_rng.randint(1, 20) for _ in months] for _ in range(customer_count)]
        # 供应商窗口也按同一合同月数生成。
        supplier_windows = [[calendar_rng.randint(1, 18) for _ in months] for _ in range(supplier_count)]
        customer_credit_keys = [calendar_rng.randint(0, 11) for _ in range(customer_count)]
        supplier_credit_keys = [calendar_rng.randint(0, 11) for _ in range(supplier_count)]
    # Service hours / replenishment packs are smaller contractual units than a
    # whole project.  This keeps stage demand responsive without inventing cash
    # entries or moving any settlement date on a retry.
    price = rng.choice(route.get("unit_price_fen_options", [500000, 700000, 900000]))
    cost_range = route.get("cost_to_price_range", [.50, .68])
    if len(cost_range) != 2 or not .45 <= cost_range[0] <= cost_range[1] <= .80:
        raise ValueError("成本率超出45%至80%业务范围")
    cost = int(price * rng.uniform(*cost_range))
    payroll_range = route.get("payroll_fen_range", [13500000, 17500000])
    rent_range = route.get("rent_fen_range", [3300000, 4800000])
    other_range = route.get("other_fen_range", [5000000, 8500000])
    payroll = rng.randint(*payroll_range); rent = rng.randint(*rent_range); other = rng.randint(*other_range)
    fixed = payroll + rent + other
    margin = (price - cost) / price
    tax_allowance = .025
    # 最后阶段边界取实际合同月数，保持24月默认结果不变。
    cuts = [0] + target["breakpoint_months"] + [len(months)]
    desired = target["monthly_net_cash_slope_cny"]
    adjustments = stage_order_adjustment or tuple(1.0 for _ in desired)
    if len(adjustments) != len(desired):
        raise ValueError("阶段订单调整数必须与目标阶段一致")
    nodes, events, order_facts, supplier_facts, external_service_contracts = [], [], [], [], []
    # These two mechanisms begin with a larger, explicitly declared contracted
    # demand footprint.  It is selected before generation (not after seeing a
    # chart) so their high positive middle/last stages remain inside the
    # documented 0.65--1.45 stage-demand control range.
    demand_footprint = route.get("demand_footprint", {"T16": 1.30, "T23": 1.20, "T24": 1.40}.get(target_id, 1.0))
    # Stage-specific external work is a real signed service contract (channel
    # handover, remediation inspection, or reverse-logistics), not an
    # after-the-fact balance adjustment.  It must therefore be known before
    # customer order volume is solved; the previous implementation added it
    # after that calculation and systematically made T16/T23/T24 tails too
    # negative.
    default_external = {"T16": [0, 0, 4000000], "T23": [0, 0, 10000000, 0], "T24": [0, 0, 17500000, 0]}.get(target_id, [0] * len(desired))
    stage_external = list(route.get("stage_external_service_fen", default_external))
    if len(stage_external) != len(desired) or any(not isinstance(amount, int) or amount < 0 or amount > 20000000 for amount in stage_external):
        raise ValueError("分阶段外部服务合同必须逐段列示，且金额须在0至20万元之间")
    route["stage_external_service_fen"] = stage_external
    settlement_slices = route.get("settlement_slices", 12 if low_dispersion else 4 if project else 1)
    supplier_slices = route.get("supplier_slices", 4 if low_dispersion else 1)
    if not .65 <= demand_footprint <= 1.45 or not 1 <= settlement_slices <= 12 or not 1 <= supplier_slices <= 12:
        raise ValueError("机制路线的需求或批次参数超出冻结范围")

    def add(category, month, due, amount, suffix, note, asset=""):
        if amount:
            events.append(dict(event_id=f"{target_id}-{month}-{category}-{suffix}", category=category, recognition_month=month,
                               asset_id=asset, due_date=due.isoformat(), amount_fen=amount, note_cn=note))

    # 逐个重建后的合同月生成订单、付款和结算事实。
    for index, month in enumerate(months):
        stage = next(i for i in range(len(cuts) - 1) if cuts[i] <= index < cuts[i + 1])
        # 销售总额由该段目标净贡献及固定承诺反推，再转换为真实标准件订单。
        target_net = desired[stage]
        stage_other = other + stage_external[stage]
        # The external service agreement is part of the stage commitment and
        # therefore belongs in the pre-generation cash budget.
        sales_cny = max(180000, int((stage_other / 100 + payroll / 100 + rent / 100 + target_net) / max(.12, margin - tax_allowance) * demand_footprint * adjustments[stage]))
        units = max(1, round(sales_cny * 100 / price))
        sales = units * price
        variable_cost = units * cost
        # 有明确阶段机制的订单强度变化，而不是按日余额反推。
        external_note = {"T16": "渠道撤销及场地交接服务合同", "T23": "检修整改与恢复交付前的外部检测合同", "T24": "渠道去库存阶段的回购、重新包装与物流合同"}.get(target_id, "分阶段外部服务合同")
        stage_reason = reason if not stage_external[stage] else reason + "；" + external_note + "仍在履行"
        node = dict(month=month, stage_id=f"{target_id}-{mechanism_id}-S{stage + 1}", business_reason=stage_reason,
                    sales_units=units, unit_price_cny=_money(price), unit_variable_cost_cny=_money(cost), target_inventory_cny="0.00",
                    payroll_cny=_money(payroll), rent_cny=_money(rent), other_expense_cny=_money(stage_other))
        nodes.append(node)
        month_start = date.fromisoformat(month + "-01")
        # 分拆交付不靠随机日期抖动：每个客户有预先不同服务窗口和账期。
        active_customers = min(customer_count, units)
        stage_settlement_slices = route.get("stage_settlement_slices", [settlement_slices] * len(desired))
        stage_supplier_slices = route.get("stage_supplier_slices", [supplier_slices] * len(desired))
        if len(stage_settlement_slices) != len(desired) or len(stage_supplier_slices) != len(desired):
            raise ValueError("分阶段合同批次必须覆盖全部目标阶段")
        active_settlement_slices = stage_settlement_slices[stage]
        active_supplier_slices = stage_supplier_slices[stage]
        if not 1 <= active_settlement_slices <= 12 or not 1 <= active_supplier_slices <= 12:
            raise ValueError("分阶段合同批次超出受控范围")
        for customer, amount in enumerate(_split(sales, active_customers, rng), 1):
            delivery_day = (2 + ((customer * 3 + int(target_id[1:]) + index * 2) % 20)
                            if calendar_pattern == "legacy_uniform_monthly" else customer_windows[customer - 1][index])
            delivery = month_start + timedelta(days=delivery_day)
            # T24末段是已签的回补订单，采用较短的到货验收账期；这不是
            # 重试时挪日期，而是该机制冻结的合同条款。
            stage_terms = route.get("stage_collection_credit_days")
            terms = stage_terms[stage] if stage_terms else route.get("collection_credit_days", [0, 7, 14] if target_id in {"T23", "T24"} and stage == 3 else [7, 14, 21, 30, 45])
            if not terms or any(day < 0 or day > 60 for day in terms):
                raise ValueError("客户合同账期必须在0至60日")
            credit_key = customer + index if calendar_pattern == "legacy_uniform_monthly" else customer_credit_keys[customer - 1]
            credit = terms[credit_key % len(terms)]
            for batch, batch_amount in enumerate(_split(amount, active_settlement_slices, rng), 1):
                batch_delivery = delivery + timedelta(days=(batch - 1) * 4)
                add("collection", month, batch_delivery + timedelta(days=credit), batch_amount, f"C{customer:02d}B{batch}", "客户交付验收后的合同回款")
                order_facts.append(dict(month=month, customer_id=f"C{customer:02d}", batch=batch, delivery_date=batch_delivery.isoformat(), credit_days=credit, amount_fen=batch_amount))
        for supplier, amount in enumerate(_split(variable_cost, supplier_count, rng), 1):
            arrival_day = (1 + ((supplier * 4 + index + int(target_id[1:])) % 18)
                           if calendar_pattern == "legacy_uniform_monthly" else supplier_windows[supplier - 1][index])
            arrival = month_start + timedelta(days=arrival_day)
            stage_supplier_terms = route.get("stage_supplier_credit_days")
            supplier_terms = stage_supplier_terms[stage] if stage_supplier_terms else route.get("supplier_credit_days", [0, 7, 15, 30])
            if not supplier_terms or any(day < 0 or day > 45 for day in supplier_terms):
                raise ValueError("供应商合同账期必须在0至45日")
            credit_key = supplier + index if calendar_pattern == "legacy_uniform_monthly" else supplier_credit_keys[supplier - 1]
            credit = supplier_terms[credit_key % len(supplier_terms)]
            for batch, batch_amount in enumerate(_split(amount, active_supplier_slices, rng), 1):
                batch_arrival = arrival + timedelta(days=(batch - 1) * 3)
                add("supplier_payment", month, batch_arrival + timedelta(days=credit), batch_amount, f"S{supplier:02d}B{batch}", "备件到货后按供应合同付款")
                supplier_facts.append(dict(month=month, supplier_id=f"S{supplier:02d}", batch=batch, arrival_date=batch_arrival.isoformat(), credit_days=credit, amount_fen=batch_amount))
        add("payroll_payment", month, month_start + timedelta(days=6), payroll, "team", "持续服务班组工资")
        add("rent_payment", month, month_start + timedelta(days=2), rent, "lease", "固定场地租赁付款")
        add("other_expense_payment", month, month_start + timedelta(days=24), stage_other, "support", "合规、技术支持及外部服务费" if not stage_external[stage] else external_note)
        if stage_external[stage]:
            external_service_contracts.append(dict(month=month, stage_id=f"{target_id}-{mechanism_id}-S{stage + 1}", due_date=(month_start + timedelta(days=24)).isoformat(), amount_fen=stage_external[stage], note_cn=external_note))
    # 设备付款固定在显式首月的第三天，避免保留旧年份日期。
    add("fixed_asset_payment", months[0], start + timedelta(days=2), 6000000, "EQ01", "投用设备购买款", "EQ01")
    operating["business_nodes"] = dict(version="physical_contract_business_v1", nodes=nodes, settlements=events,
                                        asset_capacities=[dict(event_id="EQ01", monthly_units=2000)],
                                        capacity_basis="自有设备月产能2000标准件；客户订单与供应商合同分别记录。")
    # 预算先在中央输入验证，不能把解释文字代替参数合法性。
    resolve_profile(profile)
    basis = dict(target_id=target_id, family_id=f"autonomous_family_{target_id}_{mechanism_id}", mechanism_id=mechanism_id,
                 business_reason=reason, revision=revision, seed=profile["random_seed"], customer_count=customer_count,
                 supplier_count=supplier_count, unit_price_fen=price, unit_cost_fen=cost, payroll_fen=payroll, rent_fen=rent,
                 other_expense_fen=other, stage_order_adjustment=list(adjustments), order_facts=order_facts, supplier_facts=supplier_facts,
                 demand_footprint=demand_footprint, settlement_slices=settlement_slices, supplier_slices=supplier_slices,
                 contract_calendar_pattern=calendar_pattern,
                 mechanism_route=route, external_service_contracts=external_service_contracts,
                 causal_statement="阶段订单总量由客户合同启动、到期或项目完成机制改变；客户和供应商的日期分别由交付/到货加各自合同账期形成。")
    return profile, basis
