"""POST-T6-C2A：构造日均余额共同配对数据，并提供概率区间公共评价公式。"""

# 导入CSV标准库；它流式读取和写出较大的预测配对文件。
import csv
# 导入随机数标准库；企业整块自助抽样用固定随机源形成可复现置信区间。
import random
# 从collections导入defaultdict；它按企业累计区间指标。
from collections import defaultdict
# 从decimal导入Decimal；金额、比例和区间评分使用精确十进制计算。
from decimal import Decimal
# 从pathlib导入Path；路径对象让读取函数不依赖当前工作目录。
from pathlib import Path

# 导入YAML；它读取冻结的C1合同并核验本卡边界。
import yaml

# 复用既有比例格式和SHA256工具，确保指标显示和指纹口径统一。
from syn_b1.post_t6_a3_xgboost_windows import ratio, sha256_file

# 固定C2A实现版本；改变配对字段或公式必须新建版本目录。
C2A_VERSION = "syn_b1_post_t6_c2a_average_interval_foundation_1_0"
# 固定C1绑定的B2锁定配置，禁止根据新指标重新选择配置。
SELECTED_CONFIGURATION = "lstm_direct_selected_window_h16_scaled_l1"
# 固定三种原始训练种子，后续不得删去表现较弱的种子。
TRAINING_SEEDS = (2026083102, 2026083103, 2026083104)
# 固定四个开发池滚动验证折。
FOLDS = ("A", "B", "C", "D")
# 固定两个用户关心的日均余额期限。
AVERAGE_HORIZONS = (10, 30)
# 固定名义80%区间的漏出惩罚系数，即2除以0.20等于10。
INTERVAL_MISS_PENALTY = Decimal("10")
# 固定企业整块自助抽样的次数；只在C2E调用，不在C2A产物上形成候选结论。
BOOTSTRAP_REPLICATES = 2000
# 固定企业整块自助抽样种子，保证同输入同结果。
BOOTSTRAP_SEED = 2026090105


# 声明C2A专用异常，使数据缺陷不会被误解成模型优劣。
class C2AFoundationError(ValueError):
    """当共同配对数据、角色隔离或区间公式不满足冻结条件时抛出。"""


# 将一个完整30日逐日预测组转换为10日和30日两条共同配对记录。
def build_average_pair_rows(rows: list[dict[str, str]], balance_scale: Decimal) -> list[dict[str, str]]:
    # 要求每个预测起点都有完整30个未来银行工作日。
    if len(rows) != 30:
        # 阻断缺日或多日路径。
        raise C2AFoundationError("共同配对数据要求每个预测起点恰有30个未来日")
    # 读取并验证未来步号必须连续从1到30。
    steps = tuple(int(row["forecast_step"]) for row in rows)
    # 拒绝乱序、重复或断裂路径。
    if steps != tuple(range(1, 31)):
        # 指出未来步号问题。
        raise C2AFoundationError("共同配对数据的未来步号必须连续为1至30")
    # 账户尺度必须为正，才能计算归一化宽度与评分。
    if balance_scale <= 0:
        # 阻断无定义比例。
        raise C2AFoundationError("共同配对数据的账户尺度必须大于0")
    # 读取组内不变的基础字段。
    first_row = rows[0]
    # 读取折角色。
    fold_role = first_row["fold_role"]
    # 只允许C1规定的calibration或validation角色。
    if fold_role not in ("calibration", "validation"):
        # 阻断fit、未知或最终测试角色。
        raise C2AFoundationError("共同配对数据只允许calibration或validation角色")
    # 验证同一组30行的连接字段完全一致。
    for row in rows:
        # 组合每行应该保持不变的身份字段。
        invariant = (row["fold_id"], row["fold_role"], row["training_seed"], row["sample_key"], row["enterprise_id"], row["configuration_id"])
        # 与首行的同类字段比较。
        expected = (first_row["fold_id"], first_row["fold_role"], first_row["training_seed"], first_row["sample_key"], first_row["enterprise_id"], first_row["configuration_id"])
        # 不一致说明CSV分组被破坏。
        if invariant != expected:
            # 阻断混入其他样本或配置。
            raise C2AFoundationError("同一30日预测组混入了不同折、角色、种子、样本、企业或配置")
    # 初始化将返回的两条记录。
    pair_rows: list[dict[str, str]] = []
    # 依次计算10日和30日日均余额。
    for horizon in AVERAGE_HORIZONS:
        # 截取当前期限前的同一批未来日。
        horizon_rows = rows[:horizon]
        # 将期限天数转换成精确分母。
        divisor = Decimal(horizon)
        # 计算真实日均余额。
        actual_average = sum((Decimal(row["actual_target_balance_cny"]) for row in horizon_rows), Decimal("0")) / divisor
        # 计算锁定LSTM点预测的日均余额。
        lstm_average = sum((Decimal(row["predicted_target_balance_cny"]) for row in horizon_rows), Decimal("0")) / divisor
        # 计算余额保持基线的日均余额。
        baseline_average = sum((Decimal(row["locked_baseline_predicted_balance_cny"]) for row in horizon_rows), Decimal("0")) / divisor
        # 记录当前期限的共同配对数据。
        pair_rows.append({
            # 保存滚动折，供后续同折校准和验证隔离。
            "fold_id": first_row["fold_id"],
            # 保存角色，供后续禁止validation参与校准。
            "fold_role": fold_role,
            # 保存原始训练种子，供后续三种子中位数计算。
            "training_seed": first_row["training_seed"],
            # 保存样本连接键，不作为任何模型输入。
            "sample_key": first_row["sample_key"],
            # 保存企业连接键，供企业等权和整块抽样使用。
            "enterprise_id": first_row["enterprise_id"],
            # 保存日均期限。
            "average_horizon_business_days": str(horizon),
            # 保存截止点可见账户尺度。
            "balance_scale_cny": f"{balance_scale:.2f}",
            # 保存真实日均余额，只用于calibration或validation评价。
            "actual_average_balance_cny": f"{actual_average:.2f}",
            # 保存既有锁定LSTM日均点预测。
            "locked_lstm_point_average_balance_cny": f"{lstm_average:.2f}",
            # 保存余额保持日均点预测。
            "balance_hold_point_average_balance_cny": f"{baseline_average:.2f}",
            # 保存C2A版本。
            "c2a_version": C2A_VERSION,
        })
    # 返回两个期限的共同配对记录。
    return pair_rows


# 计算一条真实值与预测区间的覆盖、归一化宽度和归一化80%区间评分。
def interval_measurement(actual: Decimal, lower: Decimal, upper: Decimal, balance_scale: Decimal) -> tuple[bool, Decimal, Decimal]:
    # 拒绝反向区间，防止下限大于上限却继续评价。
    if lower > upper:
        # 抛出明确异常。
        raise C2AFoundationError("预测区间下限不能大于上限")
    # 拒绝非正尺度，防止比例无定义。
    if balance_scale <= 0:
        # 抛出明确异常。
        raise C2AFoundationError("区间评价的账户尺度必须大于0")
    # 判断真实日均余额是否落在闭区间内。
    covered = lower <= actual <= upper
    # 计算原始人民币区间宽度。
    raw_width = upper - lower
    # 初始化真实值低于下限时的漏出距离。
    below_distance = Decimal("0")
    # 初始化真实值高于上限时的漏出距离。
    above_distance = Decimal("0")
    # 真实值低于下限时记录差距。
    if actual < lower:
        # 计算下穿距离。
        below_distance = lower - actual
    # 真实值高于上限时记录差距。
    if actual > upper:
        # 计算上穿距离。
        above_distance = actual - upper
    # 按冻结名义80%公式计算原始区间评分。
    raw_score = raw_width + INTERVAL_MISS_PENALTY * below_distance + INTERVAL_MISS_PENALTY * above_distance
    # 返回覆盖、归一化宽度和归一化评分。
    return covered, raw_width / balance_scale, raw_score / balance_scale


# 按企业等权汇总候选与基线区间评分，供C2E比较方法时复用。
def enterprise_macro_interval_summary(records: list[dict[str, Decimal | str]]) -> dict[str, Decimal | int]:
    # 初始化企业到候选与基线累计指标的映射。
    by_enterprise: dict[str, list[Decimal]] = defaultdict(lambda: [Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0")])
    # 逐条读取已经配对的候选与基线区间记录。
    for record in records:
        # 将企业编号转为字符串键。
        enterprise_id = str(record["enterprise_id"])
        # 读取真实日均余额。
        actual = Decimal(str(record["actual_average_balance_cny"]))
        # 读取账户尺度。
        scale = Decimal(str(record["balance_scale_cny"]))
        # 读取候选区间下限。
        candidate_lower = Decimal(str(record["candidate_lower_cny"]))
        # 读取候选区间上限。
        candidate_upper = Decimal(str(record["candidate_upper_cny"]))
        # 读取基线区间下限。
        baseline_lower = Decimal(str(record["baseline_lower_cny"]))
        # 读取基线区间上限。
        baseline_upper = Decimal(str(record["baseline_upper_cny"]))
        # 评价候选区间。
        candidate_covered, candidate_width, candidate_score = interval_measurement(actual, candidate_lower, candidate_upper, scale)
        # 评价基线区间。
        baseline_covered, baseline_width, baseline_score = interval_measurement(actual, baseline_lower, baseline_upper, scale)
        # 取得当前企业累计桶。
        bucket = by_enterprise[enterprise_id]
        # 累加候选覆盖次数。
        bucket[0] += Decimal(int(candidate_covered))
        # 累加基线覆盖次数。
        bucket[1] += Decimal(int(baseline_covered))
        # 累加候选归一化宽度。
        bucket[2] += candidate_width
        # 累加基线归一化宽度。
        bucket[3] += baseline_width
        # 累加候选归一化区间评分。
        bucket[4] += candidate_score
        # 累加基线归一化区间评分。
        bucket[5] += baseline_score
        # 累加该企业记录数。
        bucket[6] += Decimal("1")
    # 拒绝空记录集。
    if not by_enterprise:
        # 抛出明确异常。
        raise C2AFoundationError("无法汇总空的区间记录集")
    # 初始化企业等权候选覆盖率列表。
    candidate_coverages: list[Decimal] = []
    # 初始化企业等权基线覆盖率列表。
    baseline_coverages: list[Decimal] = []
    # 初始化企业等权候选宽度列表。
    candidate_widths: list[Decimal] = []
    # 初始化企业等权基线宽度列表。
    baseline_widths: list[Decimal] = []
    # 初始化企业等权候选评分列表。
    candidate_scores: list[Decimal] = []
    # 初始化企业等权基线评分列表。
    baseline_scores: list[Decimal] = []
    # 初始化候选评分优于基线的企业数。
    improved_enterprise_count = 0
    # 逐企业计算内部平均，确保大企业不凭更多日期或金额主导结果。
    for bucket in by_enterprise.values():
        # 读取该企业记录数。
        record_count = bucket[6]
        # 计算候选覆盖率。
        candidate_coverage = bucket[0] / record_count
        # 计算基线覆盖率。
        baseline_coverage = bucket[1] / record_count
        # 计算候选平均宽度。
        candidate_width = bucket[2] / record_count
        # 计算基线平均宽度。
        baseline_width = bucket[3] / record_count
        # 计算候选平均评分。
        candidate_score = bucket[4] / record_count
        # 计算基线平均评分。
        baseline_score = bucket[5] / record_count
        # 保存六项企业级平均值。
        candidate_coverages.append(candidate_coverage)
        # 保存基线覆盖率。
        baseline_coverages.append(baseline_coverage)
        # 保存候选宽度。
        candidate_widths.append(candidate_width)
        # 保存基线宽度。
        baseline_widths.append(baseline_width)
        # 保存候选评分。
        candidate_scores.append(candidate_score)
        # 保存基线评分。
        baseline_scores.append(baseline_score)
        # 候选评分严格更低时记为改善企业。
        if candidate_score < baseline_score:
            # 增加改善企业数。
            improved_enterprise_count += 1
    # 将企业数转为精确分母。
    enterprise_count = Decimal(len(by_enterprise))
    # 计算企业等权候选覆盖率。
    candidate_coverage_macro = sum(candidate_coverages, Decimal("0")) / enterprise_count
    # 计算企业等权基线覆盖率。
    baseline_coverage_macro = sum(baseline_coverages, Decimal("0")) / enterprise_count
    # 计算企业等权候选宽度。
    candidate_width_macro = sum(candidate_widths, Decimal("0")) / enterprise_count
    # 计算企业等权基线宽度。
    baseline_width_macro = sum(baseline_widths, Decimal("0")) / enterprise_count
    # 计算企业等权候选区间评分。
    candidate_score_macro = sum(candidate_scores, Decimal("0")) / enterprise_count
    # 计算企业等权基线区间评分。
    baseline_score_macro = sum(baseline_scores, Decimal("0")) / enterprise_count
    # 拒绝零基线评分，避免Skill Score无定义。
    if baseline_score_macro == 0:
        # 抛出明确异常。
        raise C2AFoundationError("基线区间评分为0，无法计算区间Skill Score")
    # 返回全部公共汇总指标。
    return {
        # 保存企业数量。
        "enterprise_count": int(enterprise_count),
        # 保存候选覆盖率。
        "candidate_empirical_coverage": candidate_coverage_macro,
        # 保存基线覆盖率。
        "baseline_empirical_coverage": baseline_coverage_macro,
        # 保存候选宽度。
        "candidate_normalized_width": candidate_width_macro,
        # 保存基线宽度。
        "baseline_normalized_width": baseline_width_macro,
        # 保存候选区间评分。
        "candidate_normalized_interval_score": candidate_score_macro,
        # 保存基线区间评分。
        "baseline_normalized_interval_score": baseline_score_macro,
        # 保存候选相对基线的区间Skill Score。
        "interval_skill_score": Decimal("1") - candidate_score_macro / baseline_score_macro,
        # 保存改善企业数。
        "improved_enterprise_count": improved_enterprise_count,
        # 保存改善企业比例。
        "improved_enterprise_rate": Decimal(improved_enterprise_count) / enterprise_count,
    }


# 按企业整块重复抽样，返回候选相对基线区间Skill的95%百分位区间。
def enterprise_block_bootstrap_skill(records: list[dict[str, Decimal | str]], replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> tuple[Decimal, Decimal]:
    # 拒绝非正抽样次数。
    if replicates <= 0:
        # 抛出清晰异常。
        raise C2AFoundationError("企业整块自助抽样次数必须大于0")
    # 按企业保存原始记录，抽样时整户进入而不是抽散日期。
    grouped: dict[str, list[dict[str, Decimal | str]]] = defaultdict(list)
    # 逐条归入对应企业。
    for record in records:
        # 使用企业连接键归组。
        grouped[str(record["enterprise_id"])].append(record)
    # 至少需要两个企业才有企业间不确定性意义。
    if len(grouped) < 2:
        # 阻断单企业伪置信区间。
        raise C2AFoundationError("企业整块自助抽样至少需要两个企业")
    # 固定随机数发生器，避免同一输入每次得到不同结论。
    generator = random.Random(seed)
    # 固定企业键顺序，避免字典写入顺序影响结果。
    enterprise_ids = tuple(sorted(grouped))
    # 初始化全部重复样本的Skill列表。
    skills: list[Decimal] = []
    # 重复既定次数。
    for _ in range(replicates):
        # 有放回抽取与原企业数相同数量的企业键。
        sampled_ids = [generator.choice(enterprise_ids) for _ in enterprise_ids]
        # 初始化本次重复样本的记录列表。
        sampled_records: list[dict[str, Decimal | str]] = []
        # 逐次处理抽中的整户，重复抽中的同一户必须保留重复权重。
        for draw_index, enterprise_id in enumerate(sampled_ids):
            # 为本次抽样创建不与原企业相同的临时企业键。
            bootstrap_enterprise_id = f"{enterprise_id}__bootstrap_draw_{draw_index}"
            # 逐条复制该户完整记录。
            for record in grouped[enterprise_id]:
                # 只替换用于企业等权的临时键，其它评价金额完全不变。
                sampled_records.append({**record, "enterprise_id": bootstrap_enterprise_id})
        # 使用同一企业等权公式评价该重复样本。
        summary = enterprise_macro_interval_summary(sampled_records)
        # 保存当前重复样本的区间Skill。
        skills.append(Decimal(str(summary["interval_skill_score"])))
    # 从小到大排序，便于取百分位边界。
    ordered = sorted(skills)
    # 固定2.5%分位位置，使用向下取整的保守索引。
    lower_index = int(Decimal(replicates - 1) * Decimal("0.025"))
    # 固定97.5%分位位置，使用向下取整索引。
    upper_index = int(Decimal(replicates - 1) * Decimal("0.975"))
    # 返回95%区间下界和上界。
    return ordered[lower_index], ordered[upper_index]


# 读取A1标签中的账户尺度；它只用于评价归一化，不进入模型。
def read_balance_scales(label_path: Path) -> dict[str, Decimal]:
    # 初始化样本键与账户尺度映射。
    scales: dict[str, Decimal] = {}
    # 以UTF-8和CSV换行规则读取标签文件。
    with label_path.open("r", encoding="utf-8", newline="") as handle:
        # 创建按表头取值的流式读取器。
        reader = csv.DictReader(handle)
        # 逐样本读取尺度。
        for row in reader:
            # 读取样本连接键。
            sample_key = row["sample_key"]
            # 拒绝重复样本键。
            if sample_key in scales:
                # 抛出清晰异常。
                raise C2AFoundationError(f"A1标签存在重复样本键：{sample_key}")
            # 保存截止日可见账户尺度。
            scales[sample_key] = Decimal(row["balance_scale_cny"])
    # 拒绝空标签文件。
    if not scales:
        # 抛出清晰异常。
        raise C2AFoundationError("A1标签文件为空")
    # 返回全部尺度。
    return scales


# 核验C1合同身份、冻结状态和禁止边界。
def validate_c1_contract(contract_path: Path) -> dict[str, object]:
    # 使用安全加载读取YAML合同。
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    # 要求正确任务编号与冻结状态。
    if contract.get("stage_id") != "POST-T6-C1" or contract.get("status") != "frozen":
        # 阻断草稿或错误合同。
        raise C2AFoundationError("C2A只能使用冻结的POST-T6-C1合同")
    # 要求合同绑定B2锁定配置。
    if contract["input_evidence"]["b2_selected_configuration_id"] != SELECTED_CONFIGURATION:
        # 阻断事后换模型。
        raise C2AFoundationError("C1合同绑定的B2配置与C2A固定配置不一致")
    # 要求产品目标仍是10日和30日日均余额。
    if contract["research_question"]["primary_target"] != "future_10_business_day_average_closing_balance_cny" or contract["research_question"]["secondary_target"] != "future_30_business_day_average_closing_balance_cny":
        # 阻断目标漂移。
        raise C2AFoundationError("C1合同的日均余额目标不完整")
    # 要求当前卡只允许冻结范围内的工作。
    if "训练、微调或重选任何模型" not in contract["current_stage_scope"]["forbidden_now"]:
        # 阻断被修改的当前范围。
        raise C2AFoundationError("C1合同缺少当前阶段禁止训练边界")
    # 返回已核验合同。
    return contract


# 流式构造共同配对数据，并返回逐角色行数和样本组数。
def write_average_pair_records(prediction_path: Path, output_path: Path, balance_scales: dict[str, Decimal]) -> dict[str, int]:
    # 固定共同配对CSV列顺序。
    fields = ("fold_id", "fold_role", "training_seed", "sample_key", "enterprise_id", "average_horizon_business_days", "balance_scale_cny", "actual_average_balance_cny", "locked_lstm_point_average_balance_cny", "balance_hold_point_average_balance_cny", "c2a_version")
    # 初始化角色到输出行数的字典。
    row_counts = {"calibration": 0, "validation": 0}
    # 初始化角色到完整样本组数的字典。
    group_counts = {"calibration": 0, "validation": 0}
    # 记录已经完整关闭的预测组，防止同组在文件后部重复出现。
    closed_groups: set[tuple[str, str, int, str]] = set()
    # 初始化当前连续预测组键。
    current_key: tuple[str, str, int, str] | None = None
    # 初始化当前组的30条逐日记录。
    current_rows: list[dict[str, str]] = []
    # 定义结束当前组并写出两条日均记录的内部函数。
    def finalize_group(writer: csv.DictWriter) -> None:
        # 声明要读取和重置的外层变量。
        nonlocal current_key, current_rows
        # 没有当前组时不需要处理。
        if current_key is None:
            # 直接返回。
            return
        # 拒绝重复组，避免样本被重复计权。
        if current_key in closed_groups:
            # 抛出明确异常。
            raise C2AFoundationError(f"B2预测文件重复出现同一完整组：{current_key}")
        # 解包折、角色、种子和样本键。
        fold_id, fold_role, training_seed, sample_key = current_key
        # 样本键必须存在于A1开发池标签。
        if sample_key not in balance_scales:
            # 阻断未知样本或最终测试样本。
            raise C2AFoundationError(f"B2预测样本不在A1开发池标签中：{sample_key}")
        # 构造10日和30日日均共同配对记录。
        pair_rows = build_average_pair_rows(current_rows, balance_scales[sample_key])
        # 再次核验内部记录的关键字段与组键一致。
        if any(row["fold_id"] != fold_id or row["fold_role"] != fold_role or int(row["training_seed"]) != training_seed or row["sample_key"] != sample_key for row in pair_rows):
            # 阻断构造过程意外改写连接键。
            raise C2AFoundationError("共同配对记录与原始预测组键不一致")
        # 写入两个期限的配对记录。
        writer.writerows(pair_rows)
        # 增加当前角色的输出行数。
        row_counts[fold_role] += len(pair_rows)
        # 增加当前角色的完整样本组数。
        group_counts[fold_role] += 1
        # 标记当前组已关闭。
        closed_groups.add(current_key)
        # 清空当前组缓冲。
        current_rows = []
    # 以UTF-8和CSV标准换行写入新文件。
    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        # 创建固定字段写入器。
        writer = csv.DictWriter(output_handle, fieldnames=fields)
        # 写入表头。
        writer.writeheader()
        # 以UTF-8流式打开B2逐日预测明细。
        with prediction_path.open("r", encoding="utf-8", newline="") as prediction_handle:
            # 创建按表头读取的CSV迭代器。
            reader = csv.DictReader(prediction_handle)
            # 逐行扫描全部B2配置。
            for row in reader:
                # 跳过未被B2锁定的其它内部配置。
                if row["configuration_id"] != SELECTED_CONFIGURATION:
                    # 继续读取下一行。
                    continue
                # 读取折角色。
                fold_role = row["fold_role"]
                # 只允许calibration或validation进入本卡。
                if fold_role not in row_counts:
                    # 阻断未知角色或越界数据。
                    raise C2AFoundationError("B2锁定配置包含非calibration/validation角色")
                # 将种子转为整数。
                training_seed = int(row["training_seed"])
                # 读取折号。
                fold_id = row["fold_id"]
                # 拒绝未登记种子或折。
                if training_seed not in TRAINING_SEEDS or fold_id not in FOLDS:
                    # 阻断实验矩阵漂移。
                    raise C2AFoundationError("B2锁定配置包含未登记种子或滚动折")
                # 组成连续预测组键。
                row_key = (fold_id, fold_role, training_seed, row["sample_key"])
                # 组切换时先结算上一完整组。
                if current_key is not None and row_key != current_key:
                    # 结算上一组。
                    finalize_group(writer)
                # 新组开始时保存其键。
                if current_key != row_key:
                    # 更新当前组键。
                    current_key = row_key
                # 保存当前逐日预测行。
                current_rows.append(row)
        # 文件末尾结算最后一个组。
        finalize_group(writer)
    # 要求两个角色均至少产生一条记录。
    if row_counts["calibration"] == 0 or row_counts["validation"] == 0:
        # 阻断缺少校准或验证角色。
        raise C2AFoundationError("共同配对数据缺少calibration或validation角色")
    # 返回角色行数和完整组数。
    return {
        # 保存calibration日均行数。
        "calibration_pair_rows": row_counts["calibration"],
        # 保存validation日均行数。
        "validation_pair_rows": row_counts["validation"],
        # 保存calibration完整样本组数。
        "calibration_sample_groups": group_counts["calibration"],
        # 保存validation完整样本组数。
        "validation_sample_groups": group_counts["validation"],
    }


# 核验输出配对表中两个角色、四折、三种子和两个期限均完整。
def validate_pair_records(pair_path: Path) -> dict[str, int]:
    # 初始化种子、折、角色、期限组合的行数。
    combinations: dict[tuple[int, str, str, int], int] = defaultdict(int)
    # 初始化总行数。
    total_rows = 0
    # 以UTF-8和CSV换行规则读取输出文件。
    with pair_path.open("r", encoding="utf-8", newline="") as handle:
        # 创建按表头读取的迭代器。
        reader = csv.DictReader(handle)
        # 逐行核验记录。
        for row in reader:
            # 读取种子。
            seed = int(row["training_seed"])
            # 读取折号。
            fold_id = row["fold_id"]
            # 读取角色。
            role = row["fold_role"]
            # 读取期限。
            horizon = int(row["average_horizon_business_days"])
            # 拒绝不在冻结矩阵中的字段。
            if seed not in TRAINING_SEEDS or fold_id not in FOLDS or role not in ("calibration", "validation") or horizon not in AVERAGE_HORIZONS:
                # 阻断输出漂移。
                raise C2AFoundationError("共同配对表包含未登记种子、折、角色或期限")
            # 验证当前版本准确。
            if row["c2a_version"] != C2A_VERSION:
                # 阻断混入其它版本输出。
                raise C2AFoundationError("共同配对表版本不一致")
            # 增加当前组合行数。
            combinations[(seed, fold_id, role, horizon)] += 1
            # 增加总行数。
            total_rows += 1
    # 遍历完整的3×4×2×2组合。
    for seed in TRAINING_SEEDS:
        # 遍历折号。
        for fold_id in FOLDS:
            # 遍历两个角色。
            for role in ("calibration", "validation"):
                # 遍历两个期限。
                for horizon in AVERAGE_HORIZONS:
                    # 每个组合必须至少有一行。
                    if combinations[(seed, fold_id, role, horizon)] == 0:
                        # 阻断缺组合。
                        raise C2AFoundationError("共同配对表缺少完整种子、折、角色或期限组合")
    # 返回总行数和组合数。
    return {"pair_row_count": total_rows, "seed_fold_role_horizon_combination_count": len(combinations)}
