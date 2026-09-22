# 中文教学注释：本行执行既有S5流程的机械步骤。
"""S5 的薄编排：登记预算、调用既有工厂并保存不可覆盖的检查回执。"""
# 中文逐行注释：导入命令行参数工具以区分只计划与后续获准的实际执行。
# 中文教学注释：本行执行既有S5流程的机械步骤。
import argparse
# 中文逐行注释：导入 CSV 读取器以核验中央生成器实际写出的日表。
# 中文教学注释：本行执行既有S5流程的机械步骤。
import csv
# 中文逐行注释：导入哈希工具以封存计划、输入日表和候选文件指纹。
# 中文教学注释：本行执行既有S5流程的机械步骤。
import hashlib
# 中文逐行注释：导入 JSON 工具以保存可恢复的状态和回执。
# 中文教学注释：本行执行既有S5流程的机械步骤。
import json
# 中文逐行注释：导入计时工具以记录每次完整生成的实际耗时。
# 中文教学注释：本行执行既有S5流程的机械步骤。
import time
# 中文逐行注释：导入系统路径模块以加载项目src中的成熟生成器。
# 中文教学注释：本行执行既有S5流程的机械步骤。
import sys
# 中文逐行注释：导入深拷贝以在不修改冻结24目标的前提下建立Q4合同预算骨架。
# 中文教学注释：本行执行既有S5流程的机械步骤。
from copy import deepcopy
# 中文逐行注释：导入日期类型以严格截取 2026 年第四季度。
# 中文教学注释：本行执行既有S5流程的机械步骤。
from datetime import date
# 中文逐行注释：导入路径类型以避免字符串路径混用。
# 中文教学注释：本行执行既有S5流程的机械步骤。
from pathlib import Path
# 中文逐行注释：导入中位数以实现 S4 冻结的自然周余额口径。
# 中文教学注释：本行执行既有S5流程的机械步骤。
from statistics import median

# 中文逐行注释：定位项目根目录，所有输出均相对该根目录。
# 中文教学注释：本行执行既有S5流程的机械步骤。
ROOT = Path(__file__).resolve().parents[1]
# 中文逐行注释：注册项目src目录，供独立执行时导入成熟工厂和生成器。
# 中文教学注释：本行执行既有S5流程的机械步骤。
sys.path.insert(0, str(ROOT / "src"))
# 中文逐行注释：固定 S4 预算文件，S5 不修改这个历史冻结件。
# 中文教学注释：本行执行既有S5流程的机械步骤。
BUDGET_PATH = ROOT / "data" / "research_round2_v4" / "s4" / "generation_budget.json"
# 中文逐行注释：固定 S5 的新证据目录。
# 中文教学注释：本行执行既有S5流程的机械步骤。
S5_ROOT = ROOT / "data" / "research_round2_v4" / "s5"
# 中文逐行注释：固定首批六类各一身份的事前登记文件。
# 中文教学注释：本行执行既有S5流程的机械步骤。
PLAN_PATH = S5_ROOT / "candidate_plan_first6_v5.json"

# 中文逐行注释：登记六个目标类别及其真实经营事件；这些不是余额修补项。
# 中文教学注释：本行执行既有S5流程的机械步骤。
FIRST_SIX = (
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    ("总体上升", "T01", "annual_service_renewal", "10月两份维保续约生效，11月新增两户年度服务验收，12月续约回款覆盖扩大。"),
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    ("总体下降", "T05", "installed_base_runoff", "10月两户退出停止补货，11月再退出一户且保留质保支出，12月只结清既有小额应收。"),
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    ("总体平稳", "T03", "retainer_balance", "10至12月固定订阅服务与常驻响应班组逐月对等续签、交付和付款。"),
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    ("先升后降", "T11", "early_rollout_completion", "10月上线验收并集中回款，11月项目收尾，12月转入低量运维及固定交付支出。"),
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    ("先降后升", "T07", "early_acceptance_recovery", "10月验收前整改和工资照常支付，11月认证完成，12月已签服务合同分批启动并回款。"),
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    ("反复起伏", "T21", "alternating_bid_portfolios", "10月第一组招标验收回款，11月第二组采购和履约付款，12月第三组验收回款；事件须在周级日表形成两次十个百分点转折。"),
# 中文教学注释：本行执行既有S5流程的机械步骤。
)


# 中文逐行注释：计算文件 SHA256，供不可变证据互相引用。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def sha256_file(path: Path) -> str:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """返回文件内容哈希，不读取或解释旧 52 户数据。"""
    # 中文逐行注释：按字节读取当前明确许可的文件。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 中文逐行注释：读取 S4 冻结预算，拒绝任何非预期版本。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def load_budget() -> dict:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """读取并核对 S4 的 109、218、6 小时和 3 GiB 上限。"""
    # 中文逐行注释：解析已冻结 JSON 而不写回它。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    # 中文逐行注释：确保本编排只服务当前冻结版本。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if budget["version"] != "round2_v4_s4_generation_budget_20260915_v1":
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise ValueError("S4生成预算版本不匹配")
    # 中文逐行注释：防止调用方用错误预算悄悄扩大尝试次数。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if budget["single_process_limits"]["maximum_all_generation_bundles"] != 218:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise ValueError("S5只允许218次完整流水生成")
    # 中文逐行注释：防止调用方把冻结的单进程约束改成并行。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if budget["single_process_limits"]["processes"] != 1:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise ValueError("S5只允许单进程")
    # 中文逐行注释：返回已验证的冻结预算。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return budget


# 中文逐行注释：创建六户事前计划；本函数不调用预算器或流水生成器。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def build_first_six_plan() -> dict:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """把类别、工厂入口、Q4合同事件和唯一可修订变量组冻结到计划。"""
    # 中文逐行注释：读取冻结预算以逐类关联配额。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    budget = load_budget()
    # 中文逐行注释：建立类别到目标人数的查询表。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    quota = {row["shape"]: row["target_accepted"] for row in budget["shape_quota_budget"]}
    # 中文逐行注释：逐项登记六个互不替代的候选身份。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    rows = []
    # 中文逐行注释：按冻结首批顺序处理每一个类别。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    for ordinal, (shape, target_id, mechanism_id, event_plan) in enumerate(FIRST_SIX, start=1):
        # 中文逐行注释：记录工厂需要实现的 Q4 合同事件接口，而非事后余额结果。
        # 中文逐行注释：为Q4逐月阶段准备目标骨架，T21在十至十二月保留上降上三段合同量。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        q4_target = {"T01": ([21], [0, 450000]), "T05": ([21], [0, -350000]), "T03": ([21], [0, 0]), "T11": ([21, 22], [0, 500000, -550000]), "T07": ([21, 22], [0, -450000, 1000000]), "T21": ([21, 22, 23], [0, 500000, -550000, 600000])}[target_id]
        # 中文逐行注释：为每户冻结传入成熟工厂的真实合同参数。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        route = {"target_id": target_id, "contract_calendar_pattern": "staggered_service_and_statement_calendar_v1", "stage_external_service_fen": ([0, 20000000] if target_id == "T05" else [0, 0, 20000000] if target_id == "T11" else [0, 20000000, 0] if target_id == "T07" else [0] * len(q4_target[1])), "stage_collection_credit_days": ([[0, 7], [0, 7]] if target_id == "T05" else [[7, 14, 21] for _ in q4_target[1]]), "stage_supplier_credit_days": ([[0, 7], [0, 7]] if target_id == "T05" else [[7, 15] for _ in q4_target[1]]), "stage_settlement_slices": ([2, 1] if target_id == "T05" else [2, 12, 12] if target_id == "T07" else [2, 3, 2, 3][:len(q4_target[1])]), "stage_supplier_slices": ([2, 1] if target_id == "T05" else [2, 12, 12] if target_id == "T07" else [2, 3, 2, 3][:len(q4_target[1])])}
        # 中文逐行注释：移除不适用于单阶段机制的空可选参数。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        route = {key: value for key, value in route.items() if value is not None}
        # 中文逐行注释：登记完整计划行，实际预算将在后续静态预检命令写入独立目录。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        rows.append({"pilot_ordinal": ordinal, "candidate_identity": f"s5_{target_id.lower()}_001", "planned_shape": shape,
                     # 中文教学注释：本行执行既有S5流程的机械步骤。
                     "shape_quota": quota[shape], "target_id": target_id, "mechanism_id": mechanism_id,
                     # 中文教学注释：本行执行既有S5流程的机械步骤。
                     "factory": "syn_b1.autonomous_target_factory.build_target_profile",
                     # 中文教学注释：本行执行既有S5流程的机械步骤。
                     "factory_date_arguments": {"start_date": "2025-01-01", "end_date": "2026-12-31", "prediction_cutoff": "2026-12-31"},
                     # 中文教学注释：本行执行既有S5流程的机械步骤。
                     "mechanism_route": {"stage_order_adjustment": "冻结24月阶段订单量参数；Q4位于第22至24月", "stage_external_service_fen": "仅在预先声明的机制阶段登记外部履约合同", "stage_collection_credit_days": "客户验收账期按已签条款，不在生成后改日期", "stage_supplier_credit_days": "供应到货账期按已签条款"},
                     # 中文教学注释：本行执行既有S5流程的机械步骤。
                     "effective_target_q4_stages": {"breakpoint_months": q4_target[0], "monthly_net_cash_slope_cny": q4_target[1]},
                     # 中文教学注释：本行执行既有S5流程的机械步骤。
                     "factory_route_values": route,
                     # 中文教学注释：本行执行既有S5流程的机械步骤。
                     "q4_contract_event_plan": event_plan, "revision_allowed": 1,
                     # 中文教学注释：本行执行既有S5流程的机械步骤。
                     "only_revision_variable_group": "declared_q4_contract_delivery_or_settlement_volume",
                     # 中文教学注释：本行执行既有S5流程的机械步骤。
                     "pre_generation_gate": ["factory_date_contract", "budget_compiler_or_resolve_profile", "planned_contract_event_receipt", "no_model_scoring"]})
    # 中文逐行注释：返回计划本身及其明确的禁止事项。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return {"version": "round2_v4_s5_first6_candidate_plan_v5", "status": "planned_not_generated", "s4_budget_sha256": sha256_file(BUDGET_PATH),
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            "generation_rule": "任何完整 generate_from_profile 或 preview_from_profile 调用均登记为一次尝试；不得把预演当免费预算检查。",
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            "rows": rows, "not_run": ["generate_from_profile", "preview_from_profile", "model_inference", "model_scoring", "training"]}


# 中文逐行注释：一次性写计划，已有计划内容不同即拒绝覆盖。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def write_plan_once() -> Path:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """落盘首批计划且保留既有同内容文件，供主代理验收后再执行。"""
    # 中文逐行注释：确保目标目录存在但不触碰其他 S5 证据。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 中文逐行注释：规范化 JSON 字节以便比较已有文件。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    payload = json.dumps(build_first_six_plan(), ensure_ascii=False, indent=2) + "\n"
    # 中文逐行注释：已有同内容计划可安全复用，不重复创建。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if PLAN_PATH.exists():
        # 中文逐行注释：拒绝覆盖不同计划，防止候选在执行前被替换。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if PLAN_PATH.read_text(encoding="utf-8") != payload:
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            raise FileExistsError("首批候选计划已存在且内容不同，必须新建版本并人工复核")
        # 中文逐行注释：返回已验证的同内容文件。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        return PLAN_PATH
    # 中文逐行注释：首次写入事前计划。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    PLAN_PATH.write_text(payload, encoding="utf-8")
    # 中文逐行注释：返回新建文件路径。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return PLAN_PATH


# 中文逐行注释：从冻结计划获得某一首批户的实际24月目标，且不修改源目标列表。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def effective_target(row: dict, targets_by_id: dict) -> dict:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """将仅Q4需要转折的计划转换成成熟工厂认可的24月阶段合同骨架。"""
    # 中文逐行注释：复制原目标以保持其冻结内存对象不变。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    target = deepcopy(targets_by_id[row["target_id"]])
    # 中文逐行注释：读取可选的Q4阶段替换信息。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    override = row["effective_target_q4_stages"]
    # 中文逐行注释：仅在计划明确替换时写入断点和订单净贡献骨架。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if isinstance(override, dict):
        # 中文逐行注释：写入已登记的24月断点。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        target["breakpoint_months"] = override["breakpoint_months"]
        # 中文逐行注释：写入已登记的各阶段订单贡献而非余额。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        target["monthly_net_cash_slope_cny"] = override["monthly_net_cash_slope_cny"]
    # 中文逐行注释：返回供预算工厂编译的独立副本。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return target


# 中文逐行注释：从已编译合同结算日计算预算代理余额，避免为预算检查调用完整流水生成器。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def planned_q4_shape(profile: dict) -> dict:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """以合同收付款逐日累计形成预算代理；该代理通过后仍须实际流水复核。"""
    # 中文逐行注释：读取预先冻结的期初资金。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    cash = int(round(float(profile["initial_capital"]["amount_cny"]) * 100))
    # 中文逐行注释：把合同结算按日期聚合为真实方向的预算现金事件。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    events = {}
    # 中文逐行注释：遍历工厂已经生成的合同结算节点。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    for event in profile["operating"]["business_nodes"]["settlements"]:
        # 中文逐行注释：客户回款为正，其余付款为负。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        sign = 1 if event["category"] == "collection" else -1
        # 中文逐行注释：累加同日合同事件而不改动任一事件。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        events[event["due_date"]] = events.get(event["due_date"], 0) + sign * event["amount_fen"]
    # 中文逐行注释：初始化完整Q4日余额容器。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    q4 = []
    # 中文逐行注释：从合同起点逐日累计至截止日。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    current = date.fromisoformat(profile["start_date"])
    # 中文逐行注释：解析合同截止日。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    end = date.fromisoformat(profile["end_date"])
    # 中文逐行注释：逐日执行预算层的已签收付款累计。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    while current <= end:
        # 中文逐行注释：加入当日所有合同事件。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        cash += events.get(current.isoformat(), 0)
        # 中文逐行注释：只保存冻结考试季度。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if current >= date(2026, 10, 1):
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            q4.append((current, cash / 100))
        # 中文逐行注释：移到下一自然日。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        current = date.fromordinal(current.toordinal() + 1)
    # 中文逐行注释：拒绝资金预算已经为负或Q4日期不完整的画像。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if len(q4) != 92 or cash < 0:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise ValueError("合同预算缺少完整Q4或截止日资金为负")
    # 中文逐行注释：按冻结周口径聚合预算代理余额。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    weeks = {}
    # 中文逐行注释：逐日归类到周一。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    for day, balance in q4:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        weeks.setdefault(day.toordinal() - day.weekday(), []).append(balance)
    # 中文逐行注释：得到排序周中位数。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    medians = [median(weeks[key]) for key in sorted(weeks)]
    # 中文逐行注释：用冻结纯函数评价预算代理走势。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    from audit_old_balance_shapes_20260908 import long_shape, swings
    # 中文逐行注释：计算冻结参考余额。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    reference = max(0.01, median(balance for _, balance in q4))
    # 中文逐行注释：计算标签和转折次数。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    label = long_shape(medians, reference)
    # 中文逐行注释：计算十个百分点门槛下的转折事实。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    turns, initial = swings(medians, 0.10 * reference)
    # 中文逐行注释：映射到S4六配额名称。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    mapped = {"整体平稳": "总体平稳", "持续上行": "总体上升", "持续下行": "总体下降", "先升后降": "先升后降", "先降后升": "先降后升", "反复起伏": "反复起伏"}.get(label, "其他混合")
    # 中文逐行注释：返回代理证据，明确不把它称为实际流水结果。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return {"planned_exam_window_shape": mapped, "weekly_median_cny": medians, "reference_cny": reference, "turn_count": turns, "initial_direction": initial, "proxy_only_not_actual_ledger": True}


# 中文逐行注释：静态编译首6合同预算；resolve_profile在工厂内执行，不调用完整流水生成器。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def prepare_first6_budgets() -> list[dict]:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """调用成熟预算工厂生成2025--2026合同画像，作为实际生成前的唯一预算预检。"""
    # 中文逐行注释：确保计划已写入且内容已冻结。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    plan_path = write_plan_once()
    # 中文逐行注释：读取计划 JSON。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    # 中文逐行注释：导入成熟24目标目录，未调用其任何历史基线函数。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    from run_shape_guided_budget_search_20260913 import targets
    # 中文逐行注释：导入唯一自主机制预算工厂。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    from syn_b1.autonomous_target_factory import build_target_profile
    # 中文逐行注释：建立目标编号索引。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    targets_by_id = {item["target_id"]: item for item in targets()}
    # 中文逐行注释：建立静态预算输出目录。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    budget_root = S5_ROOT / "budget_preflight_first6_v4"
    # 中文逐行注释：初始化回执行集合。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    receipts = []
    # 中文逐行注释：按固定首批顺序逐户预检。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    for row in plan["rows"]:
        # 中文逐行注释：构造Q4阶段化的合同目标。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        target = effective_target(row, targets_by_id)
        # 中文逐行注释：读取冻结路线值。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        route = row["factory_route_values"]
        # 中文逐行注释：使用全1阶段倍率；唯一修订变量仅在失败后由执行器改此组。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        adjustments = tuple(1.0 for _ in target["monthly_net_cash_slope_cny"])
        # 中文逐行注释：调用预算工厂和其内部resolve_profile，不调用preview或generate。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        profile, basis = build_target_profile(target, 0, seed_base=2026091500, stage_order_adjustment=adjustments, mechanism_route=route, **row["factory_date_arguments"])
        # 中文逐行注释：在任何完整生成前核验合同预算代理是否已达到计划Q4类别。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        proxy = planned_q4_shape(profile)
        # 中文逐行注释：代理不匹配时阻断预算，不允许把schema通过当成可生成。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if proxy["planned_exam_window_shape"] != row["planned_shape"]:
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            raise ValueError(f"合同预算Q4代理未达标：{row['candidate_identity']}={proxy['planned_exam_window_shape']}")
        # 中文逐行注释：把候选身份改为S5登记名称以避免与历史目录混淆。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        profile["sample_id"] = row["candidate_identity"]
        # 中文逐行注释：给本轮静态预算一个独立运行编号。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        profile["run_id"] = "round2_v4_s5_budget_preflight_v1"
        # 中文逐行注释：定位不可覆盖的预算文件。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        profile_path = budget_root / "profiles" / f"{row['candidate_identity']}.json"
        # 中文逐行注释：定位不可覆盖的合同事实文件。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        basis_path = budget_root / "contract_facts" / f"{row['candidate_identity']}.json"
        # 中文逐行注释：创建父目录。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        # 中文逐行注释：创建事实父目录。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        basis_path.parent.mkdir(parents=True, exist_ok=True)
        # 中文逐行注释：规范化画像字节，便于安全恢复。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        profile_text = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
        # 中文逐行注释：规范化合同事实字节，便于安全恢复。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        basis_text = json.dumps(basis, ensure_ascii=False, indent=2) + "\n"
        # 中文逐行注释：拒绝覆盖不同已登记预算。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if profile_path.exists() and profile_path.read_text(encoding="utf-8") != profile_text:
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            raise FileExistsError(f"预算画像冲突：{profile_path}")
        # 中文逐行注释：首次保存或保留同字节预算画像。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if not profile_path.exists():
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            profile_path.write_text(profile_text, encoding="utf-8")
        # 中文逐行注释：拒绝覆盖不同合同事实。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if basis_path.exists() and basis_path.read_text(encoding="utf-8") != basis_text:
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            raise FileExistsError(f"合同事实冲突：{basis_path}")
        # 中文逐行注释：首次保存或保留同字节合同事实。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if not basis_path.exists():
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            basis_path.write_text(basis_text, encoding="utf-8")
        # 中文逐行注释：登记此操作未触发完整流水的事实和文件指纹。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        receipts.append({"candidate_identity": row["candidate_identity"], "planned_shape": row["planned_shape"], "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "profile_sha256": sha256_file(profile_path), "contract_facts": str(basis_path.relative_to(ROOT)).replace("\\", "/"), "contract_facts_sha256": sha256_file(basis_path), "planned_q4_proxy": proxy, "complete_generation_attempts_used": 0, "budget_preflight": "resolve_profile_and_contract_q4_proxy_passed"})
    # 中文逐行注释：定位总回执文件。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    receipt_path = budget_root / "receipt.json"
    # 中文逐行注释：规范化总回执内容。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    receipt_text = json.dumps({"status": "static_budget_preflight_passed_not_generated", "rows": receipts, "complete_generation_attempts_used": 0}, ensure_ascii=False, indent=2) + "\n"
    # 中文逐行注释：拒绝覆盖与当前计划不一致的总回执。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if receipt_path.exists() and receipt_path.read_text(encoding="utf-8") != receipt_text:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise FileExistsError("首6预算总回执已存在且内容不同")
    # 中文逐行注释：首次写入总回执。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if not receipt_path.exists():
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        receipt_path.write_text(receipt_text, encoding="utf-8")
    # 中文逐行注释：返回每户静态预算结果。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return receipts


# 中文逐行注释：复用冻结曲线内重复政策，不平滑实际日表。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def within_curve_quality(daily_csv: Path) -> dict:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """对完整实际日表执行 within_curve_repetition_v2，并记录政策代码指纹。"""
    # 中文逐行注释：导入既有重复波形检查和其冻结政策。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    from curve_repetition_check import POLICY, check
    # 中文逐行注释：按真实日期顺序读取实际日表。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    rows = sorted(csv.DictReader(daily_csv.open(encoding="utf-8-sig", newline="")), key=lambda row: row["calendar_date"])
    # 中文逐行注释：调用既有检查，绝不平滑或改写余额数据。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    result = check(rows)
    # 中文逐行注释：附加源码指纹以便将来定位规则版本。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    result["policy_source_sha256"] = sha256_file(ROOT / "tools" / "curve_repetition_check.py")
    # 中文逐行注释：附加政策版本以便执行器作为硬门读取。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    result["policy_version"] = POLICY["version"]
    # 中文逐行注释：返回原始检查结论。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return result


# 中文逐行注释：执行首6冻结画像的单进程生成和最小质量门。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def execute_first6() -> list[dict]:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """逐户调用既有中央生成器；已有完整成功目录绝不重跑。"""
    # 中文逐行注释：读取静态预算回执以取得唯一冻结画像清单。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    receipt = json.loads((S5_ROOT / "budget_preflight_first6_v4" / "receipt.json").read_text(encoding="utf-8"))
    # 中文逐行注释：读取首6计划以比对实际Q4类别。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    plan = {row["candidate_identity"]: row for row in json.loads((S5_ROOT / "candidate_plan_first6_v5.json").read_text(encoding="utf-8"))["rows"]}
    # 中文逐行注释：导入唯一中央流水生成器。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    from syn_b1.formal_generator import generate_from_profile
    # 中文逐行注释：建立独立实际输出与不可覆盖尝试账本目录。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    output = S5_ROOT / "first6_actual_v1"
    # 中文逐行注释：建立调用记录目录。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    attempts_dir = output / "attempts"
    # 中文逐行注释：创建目录。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    attempts_dir.mkdir(parents=True, exist_ok=True)
    # 中文逐行注释：初始化结果集合。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    results = []
    # 中文逐行注释：保持预算回执中的固定顺序和单进程宽度一。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    for ordinal, row in enumerate(receipt["rows"], start=1):
        # 中文逐行注释：冻结候选身份。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        identity = row["candidate_identity"]
        # 中文逐行注释：读取对应画像绝对路径。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        profile_path = ROOT / row["profile"]
        # 中文逐行注释：定位该身份唯一的中央输出目录。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        run_dir = output / "flows" / identity / "round2_v4_s5_budget_preflight_v1"
        # 中文逐行注释：定位不可覆盖的尝试账本记录。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        attempt_path = attempts_dir / f"{ordinal:02d}_{identity}.json"
        # 中文逐行注释：已有完整输出只复用，不调用生成器。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if (run_dir / "generation_manifest.json").exists():
            # 中文逐行注释：加载已有成功或失败输出状态。
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            manifest = json.loads((run_dir / "generation_manifest.json").read_text(encoding="utf-8"))
            # 中文逐行注释：登记恢复行为不耗费新尝试。
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            results.append({"candidate_identity": identity, "status": "reused_existing_output", "execution_status": manifest.get("execution_status"), "new_generation_attempt": 0, "directory": str(run_dir.relative_to(ROOT)).replace("\\", "/")})
            # 中文逐行注释：继续下一户。
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            continue
        # 中文逐行注释：任何第七次调用都被冻结首批上限拒绝。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if ordinal > 6:
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            raise ValueError("首批最多六次完整生成")
        # 中文逐行注释：在调用前写入不可变attempt_started记录。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        attempt_path.write_text(json.dumps({"candidate_identity": identity, "attempt": 1, "status": "started_before_generate", "profile_sha256": sha256_file(profile_path)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # 中文逐行注释：记录调用起始时间。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        started = time.monotonic()
        # 中文逐行注释：调用既有中央生成器且明确固定输出运行号。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        bundle = generate_from_profile(profile_path, output_root=output / "flows", output_run_id="round2_v4_s5_budget_preflight_v1")
        # 中文逐行注释：计算本次完整调用耗时。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        elapsed = time.monotonic() - started
        # 中文逐行注释：由生成结果定位实际目录。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        directory = run_dir
        # 中文逐行注释：执行成熟账务评估函数而不重造账务引擎。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        from shape_contract_search import _ledger_assessment
        # 中文逐行注释：读取实际日表用于Q4形状与曲线质量门。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        daily = directory / "account_daily_total.csv"
        # 中文逐行注释：计算账务检查。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        ledger = _ledger_assessment(directory)
        # 中文逐行注释：计算实际Q4冻结形状。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        shape = exam_window_shape(daily)
        # 中文逐行注释：计算不平滑日表的曲线内重复检查。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        curve = within_curve_quality(daily)
        # 中文逐行注释：检查实际类别与冻结计划一致。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        shape_passed = shape["exam_window_shape"] == plan[identity]["planned_shape"]
        # 中文逐行注释：计算实际输出目录总字节数。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        bytes_used = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
        # 中文逐行注释：形成硬门结论。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        passed = bool(ledger["passed"] and shape_passed and curve["passed"] and curve["sufficient_observation"])
        # 中文逐行注释：用最终回执覆盖开始记录，开始状态已被中央输出和本字段保留。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        final = {"candidate_identity": identity, "attempt": 1, "status": "passed_first6_pending_cross_duplicate" if passed else "failed_quality_gate", "elapsed_seconds": elapsed, "generation_directory": str(directory.relative_to(ROOT)).replace("\\", "/"), "bytes_used": bytes_used, "ledger": ledger, "q4_shape": shape, "curve_repetition": curve, "shape_passed": shape_passed, "complete_generation_attempts_used": 1}
        # 中文逐行注释：写入实际回执。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        attempt_path.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # 中文逐行注释：把结果加入返回集合。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        results.append(final)
    # 中文逐行注释：写入本批汇总，后续跨样本重复检查可在此基础上追加。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    (output / "summary.json").write_text(json.dumps({"status": "first6_generated_pending_cross_duplicate", "rows": results, "attempts_used": sum(row.get("complete_generation_attempts_used", 0) for row in results)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 中文逐行注释：返回首6实际结果。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return results


# 中文逐行注释：提取一份 2026 日表签名，显式替代旧工具固定 2024--2025 的 DAY_INDEX。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def signature_2026(directory: Path) -> dict:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """构造 compare 可复用的全年数组，同时保存日期平移不敏感的拷贝指纹。"""
    # 中文逐行注释：读取唯一中央生成器的日表。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    daily_path = directory / "account_daily_total.csv"
    # 中文逐行注释：按日期排序，绝不依赖 CSV 行号。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    rows = sorted(csv.DictReader(daily_path.open(encoding="utf-8-sig", newline="")), key=lambda row: row["calendar_date"])
    # 中文逐行注释：仅允许完整自然年 2026 参与 S5 明显重复比较。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    rows = [row for row in rows if row["calendar_date"].startswith("2026-")]
    # 中文逐行注释：拒绝缺少全年日期的输出，避免无交集被误写为通过。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if len(rows) != 365 or len({row["calendar_date"] for row in rows}) != 365:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise ValueError("2026重复检查需要完整365个自然日")
    # 中文逐行注释：延迟导入数组包，保持只写计划时没有沉重运行时依赖。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    import numpy as np
    # 中文逐行注释：组装旧 compare 所需字段，使相关公式保持复用。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    balance = np.asarray([float(row["ending_balance_cny"]) for row in rows], dtype=float)
    # 中文逐行注释：组装真实日净收支，不以余额反推现金流。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    cash = np.asarray([float(row["inflow_cny"]) - float(row["outflow_cny"]) for row in rows], dtype=float)
    # 中文逐行注释：定位正式生成必须具有的现金流复核表。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    notes_path = directory / "cash_flow_review_notes.csv"
    # 中文逐行注释：缺少融资分类证据时失败，禁止把全NaN经营流当作通过。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if not notes_path.exists():
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise FileNotFoundError("明显重复检查缺少cash_flow_review_notes.csv")
    # 中文逐行注释：建立全年日期到数组下标的精确映射。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    day_index = {row["calendar_date"]: index for index, row in enumerate(rows)}
    # 中文逐行注释：初始化经营净流为零并在下方排除融资后累加。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    operating = np.zeros(365, dtype=float)
    # 中文逐行注释：逐笔读取中央生成器写出的分类复核记录。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    for note in csv.DictReader(notes_path.open(encoding="utf-8-sig", newline="")):
        # 中文逐行注释：提取记账日期并忽略2026以外记录。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        booking_date = note["booking_datetime"][:10]
        # 中文逐行注释：仅保留已映射的2026自然日。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        index = day_index.get(booking_date)
        # 中文逐行注释：识别已声明融资项目，和成熟signature保持同一排除口径。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        financing = any(word in note["cash_flow_type_cn"] for word in ("股东", "银行借款", "贷款", "借款本金", "利息"))
        # 中文逐行注释：非融资记录按方向累加实际经营现金流。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if index is not None and not financing:
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            operating[index] += float(note["amount_cny"]) * (1 if note["direction_cn"] == "流入" else -1)
    # 中文逐行注释：使用日期无关的金额序列哈希拦截日期整体平移的复制件。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    shifted = [[round(float(row["ending_balance_cny"]) - balance[0], 2), round(float(row["inflow_cny"]), 2), round(float(row["outflow_cny"]), 2)] for row in rows]
    # 中文逐行注释：返回 compare 兼容数组及额外的克隆指纹。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return {"balance": balance, "cash": cash, "operating": operating, "date_shift_clone_sha256": hashlib.sha256(json.dumps(shifted, separators=(",", ":")).encode()).hexdigest(), "daily_sha256": sha256_file(daily_path)}


# 中文逐行注释：复用成熟 compare 的相关系数门槛并补上日期平移克隆拒绝规则。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def obvious_duplicate_2026(left: dict, right: dict) -> dict:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """返回 2026 同年明显重复检查；绝不把零共同日期当成通过。"""
    # 中文逐行注释：从旧成熟工具只导入 compare 公式，不调用其 history_index 或 main。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    from shape_contract_search import compare
    # 中文逐行注释：使用 compare 的余额与原始现金相关计算。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    compared = compare(left, right)
    # 中文逐行注释：同日完整数组必须有365天共同记录。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    complete_common = compared["common_days"] == 365
    # 中文逐行注释：相同的日期无关金额序列直接判为复制，即使日历整体平移。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    shifted_clone = left["date_shift_clone_sha256"] == right["date_shift_clone_sha256"]
    # 中文逐行注释：保留成熟经营流阈值并拒绝缺失经营相关性的静默通过。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    strong = bool(complete_common and compared["operating_strong"])
    # 中文逐行注释：输出可审计的拒绝原因。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return {"compare": compared, "complete_common_365": complete_common, "date_shift_clone": shifted_clone, "obvious_duplicate": bool(strong or shifted_clone), "passed": bool(complete_common and not strong and not shifted_clone)}


# 中文逐行注释：按 S4 的周中位数规则给实际 Q4 日表分类。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def exam_window_shape(daily_csv: Path) -> dict:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """只读实际余额并调用冻结 long_shape 与 swings；不推理、不评分。"""
    # 中文逐行注释：导入 S4 明确许可的两个纯函数，不运行该文件 main。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    from audit_old_balance_shapes_20260908 import long_shape, swings
    # 中文逐行注释：读取 Q4 每个自然日的实际日末余额。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    values = []
    # 中文逐行注释：遍历中央生成器日表。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    for row in csv.DictReader(daily_csv.open(encoding="utf-8-sig", newline="")):
        # 中文逐行注释：解析日期以应用闭区间。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        current = date.fromisoformat(row["calendar_date"])
        # 中文逐行注释：只保留冻结的 10 月 1 日至 12 月 31 日。
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        if date(2026, 10, 1) <= current <= date(2026, 12, 31):
            # 中文教学注释：本行执行既有S5流程的机械步骤。
            values.append((current, float(row["ending_balance_cny"])))
    # 中文逐行注释：完整 Q4 是门槛的一部分，缺日不得计入配额。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if len(values) != 92:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise ValueError("exam_window_shape_v1要求2026Q4完整92个自然日")
    # 中文逐行注释：按周一至周日聚合可用日末余额。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    weeks = {}
    # 中文逐行注释：逐日归入 ISO 周一日期。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    for current, balance in values:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        weeks.setdefault(current.toordinal() - current.weekday(), []).append(balance)
    # 中文逐行注释：按周起始日期排序并取中位数。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    medians = [median(weeks[key]) for key in sorted(weeks)]
    # 中文逐行注释：冻结协议至少需要十个周中位数。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if len(medians) < 10:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise ValueError("周中位数少于10个")
    # 中文逐行注释：计算全季参考余额并设定0.01元下限。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    reference = max(0.01, median([balance for _, balance in values]))
    # 中文逐行注释：按冻结十个百分点阈值计算转折和初始方向。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    turns, initial = swings(medians, 0.10 * reference)
    # 中文逐行注释：按冻结纯函数得到未映射标签。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    raw_label = long_shape(medians, reference)
    # 中文逐行注释：将旧函数中文标签映射为 S4 配额标签。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    mapping = {"整体平稳": "总体平稳", "持续上行": "总体上升", "持续下行": "总体下降", "先升后降": "先升后降", "先降后升": "先降后升", "反复起伏": "反复起伏"}
    # 中文逐行注释：返回所有协议要求登记的输入和结果。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    return {"exam_window_shape": mapping.get(raw_label, "其他混合"), "weekly_median_cny": medians, "reference_cny": reference, "threshold_cny": 0.10 * reference, "turn_count": turns, "initial_direction": initial, "input_daily_sha256": sha256_file(daily_csv)}


# 中文逐行注释：提供显式命令行入口，默认只写计划，杜绝误触发生成。
# 中文教学注释：本行执行既有S5流程的机械步骤。
def main() -> None:
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    """写入待验收计划；实际生成入口须由后续已审核版本另行启用。"""
    # 中文逐行注释：建立参数解析器。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    parser = argparse.ArgumentParser()
    # 中文逐行注释：只允许明确的计划写入操作。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    parser.add_argument("--write-first6-plan", action="store_true")
    # 中文逐行注释：允许明确启动不生成流水的首6静态预算预检。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    parser.add_argument("--prepare-first6-budgets", action="store_true")
    # 中文逐行注释：允许在预算审定后显式运行首6中央生成。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    parser.add_argument("--execute-first6", action="store_true")
    # 中文逐行注释：解析调用参数。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    args = parser.parse_args()
    # 中文逐行注释：未给计划参数时失败，避免默认产生任何文件或运行。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if not args.write_first6_plan and not args.prepare_first6_budgets and not args.execute_first6:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        raise SystemExit("仅支持计划或静态预算预检；本版本不执行生成")
    # 中文逐行注释：写入不可覆盖计划并输出可供脚本读取的路径。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if args.write_first6_plan:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        print(json.dumps({"candidate_plan": str(write_plan_once()), "status": "planned_not_generated"}, ensure_ascii=False))
    # 中文逐行注释：只在显式请求时执行工厂内resolve_profile静态预算预检。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if args.prepare_first6_budgets:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        print(json.dumps({"rows": prepare_first6_budgets(), "status": "static_budget_preflight_passed_not_generated"}, ensure_ascii=False))
    # 中文逐行注释：只在主流程明确给出执行参数时调用中央生成器。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    if args.execute_first6:
        # 中文教学注释：本行执行既有S5流程的机械步骤。
        print(json.dumps({"rows": execute_first6()}, ensure_ascii=False))


# 中文逐行注释：仅在脚本直接执行时进入命令行入口。
# 中文教学注释：本行执行既有S5流程的机械步骤。
if __name__ == "__main__":
    # 中文逐行注释：执行受限的只计划入口。
    # 中文教学注释：本行执行既有S5流程的机械步骤。
    main()
