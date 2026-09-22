# 合同节点只提供经营金额和到期日期，余额仍由原中央账本计算。
"""显式月度合同、租赁经营说明和分期结算适配。"""
# 不可变对象保存已审参数。
from dataclasses import dataclass
# 日期与时刻均在生成前确定。
from datetime import date, datetime, time
# 检查完整自然月。
from calendar import monthrange
from functools import cached_property

# 分类与原经营月度现金来源保持一一对应。
SUFFIX = {"collection": "sales_collection", "supplier_payment": "supplier_payment", "payroll_payment": "payroll", "rent_payment": "rent_utilities", "other_expense_payment": "other", "fixed_asset_payment": "fixed_asset"}

# 明确金额模式不虚构商品数量与单价。
@dataclass(frozen=True)
# 混合产品企业可以按已签订单金额预算。
class ContractMonthlyNode:
    # 连续月份。
    month: str
    # 经营阶段追溯编号。
    stage_id: str
    # 已审经营解释。
    business_reason: str
    # 本月交付确认金额，单位分。
    sales_fen: int
    # 本月材料外协耗用，单位分。
    variable_cost_fen: int
    # 月末库存，中央据此计算采购。
    target_inventory_fen: int
    # 工资确认金额。
    payroll_fen: int
    # 租赁与水电确认金额。
    rent_fen: int
    # 其他费用确认金额。
    other_expense_fen: int
    # 不可撤销采购下限，不能因停止预期而取消。
    minimum_purchase_fen: int

# 每笔分期对应一个确认月份，期外部分保留为尾款。
@dataclass(frozen=True)
# 到期合同不是已经发生的银行交易。
class ContractSettlement:
    # 在本户范围内唯一。
    event_id: str
    # 与原经营分类一致。
    category: str
    # 销售、采购或费用确认月。
    recognition_month: str
    # 固定资产付款对应具体资产。
    asset_id: str
    # 不因实际余额而移动日期。
    due_date: date
    # 严格整数分。
    amount_fen: int
    # 银行备注只说明已发生的支付类型。
    note_cn: str
    # 仅供应商合同可声明的、预先协商且有上限的等待回款宽限天数。
    negotiated_grace_days: int = 0
    # 触发等待的真实客户回款合同编号，未开启机制时保持为空。
    dependent_receipt_event_id: str | None = None
    # 审计中保留协商原因，不能由余额不足时临时编造。
    defer_reason_cn: str | None = None

# 原经营模块通过相同节点接口使用这一计划。
@dataclass(frozen=True)
# 不持有现金余额，也不提供补资逻辑。
class ContractBusinessPlan:
    # 完整月度经营驱动。
    nodes: tuple
    # 已签收付清单。
    settlements: tuple
    # 租赁工位与外协来源的明确说明。
    capacity_basis: str
    # 该版本按合同金额建模，不伪装成物理产能模拟。
    capacity_mode: str
    # 默认关闭；只有预算明确声明后才允许供应商在宽限内等指定回款。
    receipt_dependent_supplier_payment_enabled: bool = False
    # 新版数量合同保留原设备产能；旧金额合同默认不使用此字段。
    asset_capacities: tuple = ()

    @cached_property
    def settlements_by_source(self):
        # 一次分组供各月现金排期复用，避免数百户时逐来源扫描全部合同。
        groups = {}
        for event in self.settlements:
            key = (event.category, event.recognition_month, event.asset_id)
            groups.setdefault(key, []).append(event)
        return {key: tuple(events) for key, events in groups.items()}

    # 直接调用中央也须检查完整计划。
    def validate_period(self, start, end, assets):
        # 复用中央的月份枚举。
        from syn_b1.operating_cycle import _period_months
        # 日期范围必须完整、顺序确定。
        if start.day != 1 or end.day != monthrange(end.year, end.month)[1] or tuple(n.month for n in self.nodes) != _period_months(start, end):
            # 不能漏月或用短期预算冒充完整期间。
            raise ValueError("合同节点必须覆盖完整且连续的自然月份")
        # 不允许把租赁设备变成免费自有资产。
        if self.capacity_mode == "owned_assets_with_physical_nodes":
            # 同一套数量、期间、投用日期和设备产能门禁在定日合同下继续执行。
            from syn_b1.business_nodes import BusinessNodePlan
            BusinessNodePlan(self.nodes, self.asset_capacities).validate_period(start, end, assets)
        elif self.capacity_mode != "leased_workstations_with_declared_outsourcing":
            raise ValueError("合同节点产能模式无效")
        if not self.capacity_basis.strip():
            # 新经营模式必须明确说明能力来源。
            raise ValueError("合同金额模式须明确租赁工位及外协边界")
        # 资产和事件身份必须唯一。
        if len({a.event_id for a in assets}) != len(assets) or len({e.event_id for e in self.settlements}) != len(self.settlements):
            # 重复合同可能导致重复付款。
            raise ValueError("资产或分期结算编号重复")
        # 月度实际应结算金额由业务确认计算。
        expected, inventory = {}, 0
        # 全部节点在看余额之前验证。
        for n in self.nodes:
            # 本月采购仍采用中央同一库存恒等式进行验证。
            purchase = n.variable_cost_fen + n.target_inventory_fen - inventory
            # 不允许突破已签最低采购承诺。
            if purchase < n.minimum_purchase_fen:
                # 这是合同矛盾，不是经营停止。
                raise ValueError(f"{n.month}采购低于不可撤销承诺")
            # 租赁开工的费用不能隐含为零。
            if n.sales_fen and n.rent_fen == 0:
                # 免费场地等其他模式须明确扩展，不能默认猜测。
                raise ValueError(f"{n.month}租赁经营有交付却未声明租赁费用")
            # 对每项业务保留应结算总额。
            for category, amount in (("collection", n.sales_fen), ("supplier_payment", purchase), ("payroll_payment", n.payroll_fen), ("rent_payment", n.rent_fen), ("other_expense_payment", n.other_expense_fen)):
                # 固定资产以外的来源按月份分类唯一。
                expected[(category, n.month, "")] = amount
            # 库存只用于核查，不创建第二套现金账。
            inventory = n.target_inventory_fen
        # 资产分期与原资产确认事件逐项对上。
        for a in assets:
            # 资产确认金额由中央固定资产事项所有。
            expected[("fixed_asset_payment", a.purchase_month, a.event_id)] = a.purchase_amount_fen
        # 每项实际合同金额只累计一次。
        actual = {}
        # 已协商的依赖必须指向本合同清单中真实存在的客户回款。
        collection_ids = {e.event_id for e in self.settlements if e.category == "collection"}
        # 允许期外尾款，但不允许早于业务确认月。
        for e in self.settlements:
            # 直接中央调用也须拒绝负款、零款、真假值及空说明。
            if type(e.amount_fen) is not int or e.amount_fen <= 0 or not e.event_id or not e.note_cn.strip():
                raise ValueError("结算身份、说明或金额无效")
            # 类别与日期均有可执行意义。
            if e.category not in SUFFIX or e.due_date < date.fromisoformat(e.recognition_month + "-01") or e.due_date < start:
                # 本版本不将预收预付假装应收应付。
                raise ValueError("结算类别无效或涉及未声明的跨月预收预付")
            # 宽限只能是预先协商的有限自然日，不能成为无限延期开关。
            if type(e.negotiated_grace_days) is not int or not 0 <= e.negotiated_grace_days <= 31:
                raise ValueError("协商宽限必须为0至31个自然日")
            # 工资、租赁等刚性款项以及未启用计划均不能带回款依赖字段。
            has_dependency = e.dependent_receipt_event_id is not None or e.defer_reason_cn is not None or e.negotiated_grace_days != 0
            if has_dependency and (not self.receipt_dependent_supplier_payment_enabled or e.category != "supplier_payment"):
                raise ValueError("只有已启用机制的供应商付款可以声明回款依赖")
            if has_dependency and (not e.dependent_receipt_event_id or not e.defer_reason_cn or e.negotiated_grace_days == 0):
                raise ValueError("供应商回款依赖必须同时声明回款合同、宽限和原因")
            if e.dependent_receipt_event_id is not None and e.dependent_receipt_event_id not in collection_ids:
                raise ValueError("供应商回款依赖必须引用真实客户回款合同")
            # 核查归属，不用付款日当收入确认月。
            key = (e.category, e.recognition_month, e.asset_id)
            # 未声明的采购费用不静默计入。
            if key not in expected:
                # 指出没有对应确认来源。
                raise ValueError(f"结算没有业务确认来源：{e.event_id}")
            # 分期金额相加必须等于确认总额。
            actual[key] = actual.get(key, 0) + e.amount_fen
        # 零金额可以没有分期，非零不能缺付款或回款。
        if any(actual.get(k, 0) != v for k, v in expected.items()):
            # 材料错误必须在产生任何流水之前发现。
            raise ValueError("合同分期总额与销售、采购、费用或设备确认金额不一致")
        # 原中央按月份消费节点。
        return {n.month: n for n in self.nodes}

# 从完整YAML读取显式合同金额模式。
def load_contract_business(raw, start, end, assets, physical_plan=None):
    # 拒绝拼错字段或混入未来余额。
    if set(raw) - {"version", "nodes", "settlements", "capacity_basis", "capacity_mode", "receipt_dependent_supplier_payment_enabled"} or not {"version", "nodes", "settlements", "capacity_basis", "capacity_mode"} <= set(raw):
        # 防止未知配置看似保存却未执行。
        raise ValueError("合同节点字段不完整或包含未实现字段")
    # 金额必须是非负整数分，不接受布尔值。
    def money(v):
        # 不悄悄四舍五入改变冻结金额。
        if type(v) is not int or v < 0:
            # 直接拒绝无效金额。
            raise ValueError("合同金额必须是非负整数分")
        # 保留精确分。
        return v
    # 建立只读月度节点。
    nodes = []
    # 输入字段严格对应数据类。
    for row in raw["nodes"] if physical_plan is None else ():
        # 所有金额字段都在构造前核验。
        node = ContractMonthlyNode(**{k: money(v) if k.endswith("_fen") else v for k, v in row.items()})
        # 经营原因不能缺失。
        if not node.stage_id.strip() or not node.business_reason.strip():
            # 纯粹的曲线目标不代替经营说明。
            raise ValueError("缺少经营阶段或原因")
        # 保存原顺序。
        nodes.append(node)
    # 分期对象独立于银行候选。
    events = []
    # 日期不按余额结果调整。
    for row in raw["settlements"]:
        # 显式构造并校验字段。
        event = ContractSettlement(**{**row, "due_date": date.fromisoformat(row["due_date"]), "amount_fen": money(row["amount_fen"])})
        # 正式结算必须有非零金额与身份。
        if not event.event_id or not event.note_cn.strip() or event.amount_fen == 0:
            # 零项不应伪装成银行记录。
            raise ValueError("结算身份、说明或金额无效")
        # 保存全部期内期外结算。
        events.append(event)
    # 统一验证，外层不另写账务计算。
    # 未声明时保持旧版本完全关闭，避免旧画像出现日期变化。
    if physical_plan is not None and raw["capacity_mode"] != "owned_assets_with_physical_nodes":
        raise ValueError("数量合同必须保留自有设备产能模式")
    if physical_plan is None and raw["capacity_mode"] == "owned_assets_with_physical_nodes":
        raise ValueError("自有设备合同缺少数量和产能节点")
    plan = ContractBusinessPlan(tuple(nodes) if physical_plan is None else physical_plan.nodes, tuple(events), raw["capacity_basis"], raw["capacity_mode"], raw.get("receipt_dependent_supplier_payment_enabled", False), () if physical_plan is None else physical_plan.asset_capacities)
    # 解析时完成所有静态核对。
    plan.validate_period(start, end, assets)
    # 返回给唯一中央运行链。
    return plan

# 日期排入原月度日程，不额外增加现金来源。
def schedule_node_cash(schedules, category, month, amount, default, plan, asset_id=""):
    # 旧模式使用原有延迟份额公式。
    from syn_b1.operating_cycle import _schedule_amount
    # 只有明确的新合同模式才取合同日期。
    if not isinstance(plan, ContractBusinessPlan):
        # 保证历史画像结果不变。
        return _schedule_amount(schedules[category], month, amount, default)
    # 合同必须同时属于当前类别、确认月和资产。
    events = plan.settlements_by_source.get((category, month, asset_id), ())
    # 中央实际计算的采购金额还要再次对上合同。
    if sum(e.amount_fen for e in events) != amount:
        # 不用临时差额流水弥补错误。
        raise ValueError("中央业务确认与合同分期不一致")
    # 期外月份仍排入原尾款日程。
    for e in events:
        # 只汇总日期所属月，不改金额。
        due = e.due_date.strftime("%Y-%m")
        # 同一月可以有多笔不同合同。
        schedules[category][due] = schedules[category].get(due, 0) + e.amount_fen

# 月度金额已经对平之后提供精确日期。
def contract_cash_events(plan, start, end):
    # 原逐笔定日类型负责统一候选校验。
    from syn_b1.transaction_planner import DatedSourceEvent
    # 未选择合同节点时不改变旧订单。
    if not isinstance(plan, ContractBusinessPlan):
        # 旧模式不增加任何记录。
        return ()
    # 工资等付款上午，客户回款下午；期初注资由原融资模块先入账。
    return tuple(DatedSourceEvent(e.event_id, f"i1a:{e.due_date:%Y-%m}:{SUFFIX[e.category]}", datetime.combine(e.due_date, time(15 if e.category == "collection" else 9)), e.amount_fen, business_order_id=e.event_id, historical_note_cn=e.note_cn, original_due_booking_datetime=datetime.combine(e.due_date, time(15 if e.category == "collection" else 9)), negotiated_grace_days=e.negotiated_grace_days, dependent_receipt_event_id=e.dependent_receipt_event_id, payment_deferral_reason_cn=e.defer_reason_cn, receipt_dependency_enabled=plan.receipt_dependent_supplier_payment_enabled) for e in plan.settlements if start <= e.due_date <= end)
