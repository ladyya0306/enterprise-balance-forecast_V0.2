# 本模块只定义显式经营节点，不持有现金余额或创建流水。
"""经过用户预算认可的月度经营节点与投用产能校验。"""
# 数据类让已解析节点在运行中不可改写。
from dataclasses import dataclass
# 日期用于检查完整月份和设备投用时间。
from datetime import date
# 月末日数用于拒绝未声明的部分月份折算。
from calendar import monthrange

# 每个月完整声明数量、价格、材料外协耗用和费用。
@dataclass(frozen=True)
# 不按即时现金或目标曲线计算任何经营参数。
class BusinessMonthlyNode:
    # 月份必须连续覆盖整个生成期间。
    month: str
    # 阶段编号仅供受限经营解释追溯。
    stage_id: str
    # 经营原因与具体参数一起保存，不进入预测特征。
    business_reason: str
    # 等效标准件交付数量。
    sales_units: int
    # 售价统一使用整数分。
    unit_price_fen: int
    # 单位成本仅含材料外协，工资和折旧另列。
    unit_variable_cost_fen: int
    # 用户预先声明期末库存金额。
    target_inventory_fen: int
    # 本月工资费用及应付款来源。
    payroll_fen: int
    # 本月房租水电费用及应付款来源。
    rent_fen: int
    # 本月其他经营费用及应付款来源。
    other_expense_fen: int

    # 直接程序调用也须接受同一份类型与金额检查。
    def __post_init__(self):
        # 明确禁止把真假值当成金额或数量。
        for value in (self.sales_units, self.unit_price_fen, self.unit_variable_cost_fen, self.target_inventory_fen, self.payroll_fen, self.rent_fen, self.other_expense_fen):
            # 所有金额和数量均须是非负整数。
            if type(value) is not int or value < 0:
                # 拒绝负采购诱因和无效金额，而非悄悄取绝对值。
                raise ValueError("经营节点金额和数量必须为非负整数。")
        # 每个节点都应有可解释的阶段和业务原因。
        if not self.stage_id.strip() or not self.business_reason.strip():
            # 缺少经营含义的节点不能作为有效样本输入。
            raise ValueError("经营节点必须填写阶段编号和经营原因。")

    # 销售是明确数量乘以明确单价，不由余额反推。
    @property
    # 保留金额到分，不通过毛利率舍入二次重算成本。
    def sales_fen(self):
        # 单价和数量的乘积就是本月确认销售。
        return self.sales_units * self.unit_price_fen

    # 变动耗用独立于人工、房租和折旧。
    @property
    # 使用明确单位成本精确计算材料外协耗用。
    def variable_cost_fen(self):
        # 采购及库存递推仍留在中央经营模块。
        return self.sales_units * self.unit_variable_cost_fen

    @property
    def minimum_purchase_fen(self):
        # 数量模式仍由库存恒等式决定采购，不引入未签订的最低采购承诺。
        return 0

# 一份计划包含完整月份和每项设备的有效产能增量。
@dataclass(frozen=True)
# 计划本身不创建付款、融资或银行余额。
class BusinessNodePlan:
    # 以只读顺序保存全部月份节点。
    nodes: tuple[BusinessMonthlyNode, ...]
    # 设备编号与有效月产能逐项对应。
    asset_capacities: tuple[tuple[str, int], ...]

    # 验证运行期间及设备约束，续期也必须重新覆盖。
    def validate_period(self, start, end, assets):
        # 本轮明确只接受整月，避免暗设部分月份产销折算。
        if start.day != 1 or end.day != monthrange(end.year, end.month)[1]:
            # 日期粒度变化需要另行声明，不能静默折算。
            raise ValueError("月度经营节点首户只支持完整月份。")
        # 复用既有月份枚举，避免不同边界解释。
        from syn_b1.operating_cycle import _period_months
        # 检查漏月、重复月、倒序或多余月份。
        if tuple(node.month for node in self.nodes) != _period_months(start, end):
            # 用户输入必须一次完整描述所需期间。
            raise ValueError("经营节点必须按顺序逐月完整覆盖生成期间，不得重复或缺月。")
        # 读取已声明设备，产能不能凭空出现。
        by_id = {event.event_id: event for event in assets}
        # 保留设备编号列表以检测重复。
        ids = [item[0] for item in self.asset_capacities]
        # 要求自购设备生产企业每项设备恰好声明一次产能。
        if not ids or len(ids) != len(set(ids)) or set(ids) != set(by_id):
            # 禁止找不到对应设备合同的生产能力。
            raise ValueError("经营节点的设备产能必须与固定资产事项逐项对应且不重复。")
        # 检查产能单位和设备投用粒度。
        for event_id, units in self.asset_capacities:
            # 本轮仅支持月初投用设备，并允许辅助设备增量为零。
            if type(units) is not int or units < 0 or by_id[event_id].ready_for_use_date is None or by_id[event_id].ready_for_use_date.day != 1:
                # 不把月末投用冒充整个月均有能力。
                raise ValueError("设备产能须为非负整数，且本轮设备投用日必须为月初。")
        # 逐月验证交付能力。
        for node in self.nodes:
            # 当月月初只有已经投用的设备提供产能。
            available = sum(units for event_id, units in self.asset_capacities if by_id[event_id].ready_for_use_date <= date.fromisoformat(node.month + "-01"))
            # 不自动扩产或把缺口改成委外订单。
            if node.sales_units > available:
                # 提示具体违反约束的月份。
                raise ValueError(f"{node.month}交付数量超过已投用设备产能。")
        # 返回唯一的月份索引，供中央模块读取输入。
        return {node.month: node for node in self.nodes}
