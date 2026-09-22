"""SYN-B1 T4 的固定参数 XGBoost 候选训练、验证与公平比较逻辑。"""

# 导入 csv 标准库；它读取 T3 锁定基线明细并生成可人工复核的预测表。
import csv
# 导入 json 标准库；它读取阶段报告和锁定合同，不依赖数据库。
import json
# 从 dataclasses 导入 dataclass；它把固定训练参数表达成字段清楚的不可变对象。
from dataclasses import asdict, dataclass
# 从 decimal 导入 Decimal 和金额舍入规则；指标因此不会受二进制浮点展示误差干扰。
from decimal import Decimal, ROUND_HALF_UP
# 从 pathlib 导入 Path；它负责跨平台定位已经冻结的 T2、T3 文件。
from pathlib import Path

# 导入 numpy；它把训练样本转换为 XGBoost 接受的二维数值矩阵。
import numpy as np
# 导入 xgboost；本模块只使用原生 Booster 接口，不额外依赖 scikit-learn。
import xgboost as xgb

# 复用 T3 已冻结的17项特征、样本结构和输入校验；这样不会出现两套特征口径。
from syn_b1.pretrain_baselines import BaselineContractError, RIDGE_FEATURE_COLUMNS, TreeSample, read_tree_samples, sha256_file, validate_t2_report

# 固定本阶段实现版本；所有输出都会写入该标识以防未来结果混用。
XGBOOST_VERSION = "syn_b1_pretrain_t4_xgboost_1_0"
# 固定项目随机种子；它来自既有实验合同，不允许根据验证集成绩更换。
FROZEN_RANDOM_SEED = 2026083001
# 固定 T4 候选名称；后续 LSTM 可在相同验证集合上与它公平比较。
MODEL_NAME = "xgboost_tabular_candidate"
# 再次冻结允许进入模型的17项特征；元组复用可证明没有身份、日期、画像或备注列。
MODEL_FEATURE_COLUMNS = RIDGE_FEATURE_COLUMNS


# 声明 T4 专用契约异常；它区分数据泄漏或配对错误与模型效果不佳。
class XGBoostContractError(BaselineContractError):
    """当 T2/T3 血缘、最终测试密封或配对评价不满足时抛出。"""


# 用不可变数据类保存唯一训练配置；所有值在看到 T4 验证结果之前固定。
@dataclass(frozen=True)
class XGBoostTrainingConfig:
    # 设置平方误差回归目标；模型预测下一银行工作日余额变化金额。
    objective: str = "reg:squarederror"
    # 设置直方图建树方法；它在当前17维表格数据上速度快且结果可复现。
    tree_method: str = "hist"
    # 设置树的最大深度为4；小树限制复杂度，降低合成样本过拟合风险。
    max_depth: int = 4
    # 设置学习率为0.03；较小步长配合固定轮数逐步拟合。
    eta: float = 0.03
    # 设置每棵树使用80%训练行；随机过程由固定种子控制。
    subsample: float = 0.80
    # 设置每棵树使用80%特征；它降低少数金额字段独占所有分裂的风险。
    colsample_bytree: float = 0.80
    # 设置叶节点最小权重为5；它避免只为极少数样本生成过细规则。
    min_child_weight: float = 5.0
    # 设置最小损失下降为0；首轮候选不增加额外剪枝超参数。
    gamma: float = 0.0
    # 设置 L1 正则为0；当前不通过验证集搜索稀疏强度。
    reg_alpha: float = 0.0
    # 设置 L2 正则为1；它是固定的保守默认约束。
    reg_lambda: float = 1.0
    # 设置训练轮数为400；不使用验证集早停，避免用验证结果间接选择轮数。
    num_boost_round: int = 400
    # 设置单机线程数为4；它控制资源使用，不改变实验样本和评价规则。
    nthread: int = 4
    # 保存固定种子；训练配置和模型文件都能据此复现。
    seed: int = FROZEN_RANDOM_SEED


# 把浮点预测转换为 Decimal；先转字符串可避免直接携带二进制浮点尾差。
def _decimal_from_float(value: float) -> Decimal:
    # 返回十进制对象，后续金额和比例都用同一精确计算路径。
    return Decimal(str(float(value)))


# 把金额统一显示到分；它只影响输出显示，不改变模型内部浮点训练。
def money(value: Decimal) -> str:
    # 使用普通财务四舍五入规则，确保 CSV 能直接供用户查看。
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# 把比例统一显示为八位小数；账户归一化误差和 Skill Score 使用该格式。
def ratio(value: Decimal) -> str:
    # 用定点格式避免科学计数法影响 Excel 阅读。
    return f"{value.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP):f}"


# 读取 UTF-8 或 UTF-8-BOM CSV；函数只用于冻结派生文件，不接触人工备注表。
def read_csv(path: Path) -> list[dict[str, str]]:
    # 以 utf-8-sig 打开可兼容普通 UTF-8 与 Excel 友好 BOM。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # 将小规模验证明细完整读入，便于严格检查样本键是否一一对应。
        return list(csv.DictReader(handle))


# 核验 T3 已通过、基线已锁定且最终测试仍未开启。
def validate_t3_contract(t3_directory: Path, t2_tree_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    # 构造 T3 报告路径；缺少报告说明不能证明前置阶段通过。
    report_path = t3_directory / "baseline_run_report.json"
    # 构造锁定基线路径；T4 的 Skill Score 必须使用这里唯一指定的方法。
    locked_path = t3_directory / "locked_baseline.json"
    # 同时要求两个文件存在；任一缺失都停止而不是自行重算基线。
    if not report_path.is_file() or not locked_path.is_file():
        # 抛出清晰错误，要求先恢复已验收的 T3 产物。
        raise XGBoostContractError("缺少 T3 报告或锁定基线文件")
    # 解析 T3 报告；该文件只含计数、状态和指纹。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 解析锁定基线；该文件不含任何最终测试标签。
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    # 要求 T3 状态为通过，否则不能训练复杂候选。
    if report.get("status") != "passed":
        # 抛出契约错误，防止失败基线被用作比较分母。
        raise XGBoostContractError("T3 基线阶段未通过")
    # 要求最终测试标签未读取且最终测试未开启。
    if report.get("final_test_labels_read_or_assembled") is not False or report.get("final_test_opened") is not False:
        # 一旦密封标志不成立，本任务必须停止。
        raise XGBoostContractError("T3 报告显示最终测试已开启或标签已读取")
    # 要求锁定范围明确为验证集，避免从最终测试反向选择基线。
    if locked.get("selection_scope") != "validation_only":
        # 拒绝来源不明或评价范围改变的基线。
        raise XGBoostContractError("锁定基线不是仅依据验证集产生")
    # 重新计算 T2 树表指纹，并与 T3 保存值比较。
    if locked.get("tree_features_14bd_sha256") != sha256_file(t2_tree_path):
        # 阻止在不同训练快照上比较基线和 XGBoost。
        raise XGBoostContractError("T2 树特征表与 T3 锁定基线的输入指纹不一致")
    # 返回两个已核验对象，供运行报告记录输入血缘。
    return report, locked


# 把 TreeSample 列表转换成 XGBoost 矩阵；标签只取冻结的下一日余额变化。
def make_dmatrix(samples: list[TreeSample], include_label: bool) -> xgb.DMatrix:
    # 要求至少一条样本，避免训练器在空切分上产生误导性模型。
    if not samples:
        # 抛出契约错误并明确问题是空输入。
        raise XGBoostContractError("XGBoost 输入样本为空")
    # 逐行提取17维冻结特征，并使用 float64 保留较大人民币金额的有效精度。
    feature_matrix = np.asarray([sample.features for sample in samples], dtype=np.float64)
    # 要求矩阵恰好有17列；这也是对禁止画像、编号和日期入模的硬检查。
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] != len(MODEL_FEATURE_COLUMNS):
        # 抛出异常，避免特征列变更后静默训练另一种模型。
        raise XGBoostContractError("XGBoost 特征列数不等于冻结的17项")
    # 检查全部特征为有限数；缺失、正负无穷都不得静默进入首轮候选。
    if not np.isfinite(feature_matrix).all():
        # 抛出契约错误，要求先修复派生数据而不是让模型自行猜测。
        raise XGBoostContractError("XGBoost 特征含缺失值或无穷值")
    # 仅在训练时创建标签数组；预测路径不需要也不接受标签。
    label_array = np.asarray([float(sample.target_change_cny) for sample in samples], dtype=np.float64) if include_label else None
    # 创建原生 DMatrix，并附上中文合同已冻结的英文特征名供重要性表追溯。
    return xgb.DMatrix(feature_matrix, label=label_array, feature_names=list(MODEL_FEATURE_COLUMNS))


# 使用唯一固定配置拟合 XGBoost；函数不接收验证集，因此不可能按验证结果早停或调参。
def fit_xgboost(train_samples: list[TreeSample], config: XGBoostTrainingConfig) -> xgb.Booster:
    # 把训练样本转换成含标签矩阵；只有训练企业会进入此对象。
    train_matrix = make_dmatrix(train_samples, include_label=True)
    # 将不可变配置转成字典，随后移除属于训练循环而非 Booster 的轮数字段。
    parameters = asdict(config)
    # 从参数字典取出固定轮数；xgb.train 通过单独参数接收它。
    num_boost_round = int(parameters.pop("num_boost_round"))
    # 调用原生训练器；不传验证集和早停参数，确保验证集只用于训练后的单次评价。
    return xgb.train(parameters, train_matrix, num_boost_round=num_boost_round, verbose_eval=False)


# 读取 T3 锁定方法的逐样本预测，并检查它与当前验证集完全配对。
def read_locked_baseline_rows(path: Path, locked_method: str, validation_samples: list[TreeSample]) -> dict[str, dict[str, str]]:
    # 从 T3 明细中只保留唯一锁定方法，其他四类基线不参与 Skill Score 分母。
    rows = [row for row in read_csv(path) if row.get("baseline_method") == locked_method]
    # 建立样本键映射；重复键会导致字典行数小于原列表并在后面被拒绝。
    by_key = {row["sample_key"]: row for row in rows}
    # 提取当前 T2 验证样本键，形成公平比较的唯一集合。
    expected_keys = {sample.sample_key for sample in validation_samples}
    # 同时核验行数和键集合；缺失、重复、额外样本都会失败。
    if len(rows) != len(by_key) or set(by_key) != expected_keys:
        # 抛出契约错误，禁止模型与基线比较不同企业或不同日期。
        raise XGBoostContractError("锁定基线与 XGBoost 验证样本没有逐条一一对应")
    # 返回配对映射，供每条模型预测同时核对真实标签和计算分母。
    return by_key


# 生成每条验证预测及公平比较字段；模型输出只来自固定训练模型和截止日前17项特征。
def build_validation_predictions(booster: xgb.Booster, samples: list[TreeSample], baseline_by_key: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    # 创建不含标签的验证矩阵；标签只保留在 TreeSample 中用于随后评价。
    validation_matrix = make_dmatrix(samples, include_label=False)
    # 一次完成全部验证预测；顺序与传入 samples 的冻结 CSV 顺序一致。
    predicted_changes = booster.predict(validation_matrix)
    # 要求预测数与样本数一致，避免底层异常造成静默漏行。
    if len(predicted_changes) != len(samples):
        # 抛出契约错误并停止写出不完整结果。
        raise XGBoostContractError("XGBoost 验证预测行数与样本数不一致")
    # 创建结果列表；每条验证样本只写一条 XGBoost 预测。
    records: list[dict[str, str]] = []
    # 配对遍历样本和模型预测，计算变化、余额与账户归一化误差。
    for sample, predicted_value in zip(samples, predicted_changes):
        # 将模型浮点输出转换为 Decimal，统一后续金额计算口径。
        predicted_change = _decimal_from_float(float(predicted_value))
        # 用截止日已知余额加预测变化，得到下一银行工作日余额预测。
        predicted_balance = sample.closing_balance_cny + predicted_change
        # 计算目标变化绝对误差；余额误差在本定义下金额相同但仍单独展示。
        change_error = abs(predicted_change - sample.target_change_cny)
        # 计算目标余额绝对误差，便于业务用户直接理解偏差金额。
        balance_error = abs(predicted_balance - sample.target_balance_cny)
        # 只用截止日已知余额与冻结下限计算归一化分母，不读取未来余额作为尺度。
        normalized_error = change_error / max(abs(sample.closing_balance_cny), sample.balance_floor_cny)
        # 读取同一样本的锁定基线明细，准备核对标签并输出配对误差。
        baseline = baseline_by_key[sample.sample_key]
        # 核对基线真实变化与当前 T2 标签一致；金额展示均到分，因此按分比较。
        if Decimal(baseline["actual_target_change_cny"]) != sample.target_change_cny.quantize(Decimal("0.01")):
            # 拒绝标签不一致的 Skill Score，避免把不同版本结果相除。
            raise XGBoostContractError(f"锁定基线标签与 T2 不一致：{sample.sample_key}")
        # 写入一条可审计明细；身份和日期只用于配对，训练矩阵没有这些列。
        records.append({
            # 记录固定候选名称，避免和未来 LSTM 明细混淆。
            "model_name": MODEL_NAME,
            # 记录样本唯一键，仅供公平配对与审计。
            "sample_key": sample.sample_key,
            # 记录企业编号，仅供逐户评价，不进入模型。
            "enterprise_id": sample.enterprise_id,
            # 记录验证切分；出现其他值会在读取阶段被隔离。
            "split_group": sample.split_group,
            # 记录预测截止日，证明特征只到该日。
            "cutoff_date": sample.cutoff_date,
            # 记录目标银行工作日，供时间隔离检查。
            "target_date": sample.target_date,
            # 记录实际目标余额，来源仅限验证标签。
            "actual_target_balance_cny": money(sample.target_balance_cny),
            # 记录实际目标变化，作为主监督目标。
            "actual_target_change_cny": money(sample.target_change_cny),
            # 记录模型预测余额。
            "predicted_target_balance_cny": money(predicted_balance),
            # 记录模型预测变化。
            "predicted_target_change_cny": money(predicted_change),
            # 记录变化金额绝对误差。
            "absolute_error_target_change_cny": money(change_error),
            # 记录余额金额绝对误差。
            "absolute_error_target_balance_cny": money(balance_error),
            # 记录账户归一化绝对误差。
            "account_normalized_absolute_error": ratio(normalized_error),
            # 记录同一样本锁定基线的变化绝对误差，便于逐行复核 Skill Score。
            "locked_baseline_absolute_error_target_change_cny": baseline["absolute_error_target_change_cny"],
            # 记录实现版本，防止未来文件混用。
            "feature_manifest_version": XGBOOST_VERSION,
        })
    # 返回完整验证明细，调用方会先汇总再写入新目录。
    return records


# 汇总 XGBoost 整体验证指标并计算相对锁定基线的 Skill Score。
def summarize_predictions(rows: list[dict[str, str]]) -> dict[str, str]:
    # 要求至少一条记录，空验证集不能形成任何放行结论。
    if not rows:
        # 抛出异常并停止输出。
        raise XGBoostContractError("XGBoost 验证预测为空")
    # 用 Decimal 汇总模型变化绝对误差。
    model_error_sum = sum((Decimal(row["absolute_error_target_change_cny"]) for row in rows), Decimal("0"))
    # 用同一批样本汇总锁定基线变化绝对误差。
    baseline_error_sum = sum((Decimal(row["locked_baseline_absolute_error_target_change_cny"]) for row in rows), Decimal("0"))
    # 汇总余额误差，供业务阅读。
    balance_error_sum = sum((Decimal(row["absolute_error_target_balance_cny"]) for row in rows), Decimal("0"))
    # 汇总账户归一化误差，保持与 T3 主指标相同口径。
    normalized_error_sum = sum((Decimal(row["account_normalized_absolute_error"]) for row in rows), Decimal("0"))
    # 锁定基线总误差必须大于零，否则 Skill Score 分母无定义。
    if baseline_error_sum <= Decimal("0"):
        # 抛出异常，避免用零分母制造成绩。
        raise XGBoostContractError("锁定基线总绝对误差不是正数，无法计算 Skill Score")
    # 把样本数量转成 Decimal，保证平均值仍按十进制计算。
    count = Decimal(len(rows))
    # 计算 Skill Score；正数代表比锁定基线减少误差，负数代表更差。
    skill_score = Decimal("1") - model_error_sum / baseline_error_sum
    # 返回单一模型级指标字典；所有值都能写入 JSON 或 CSV。
    return {
        # 记录候选名称。
        "model_name": MODEL_NAME,
        # 记录验证样本数。
        "validation_sample_count": str(len(rows)),
        # 记录变化金额平均绝对误差。
        "mae_target_change_cny": money(model_error_sum / count),
        # 记录余额金额平均绝对误差。
        "mae_target_balance_cny": money(balance_error_sum / count),
        # 记录与 T3 一致的账户归一化平均绝对误差。
        "account_normalized_mae": ratio(normalized_error_sum / count),
        # 记录模型总绝对误差，作为 Skill Score 分子来源。
        "sum_absolute_error_target_change_cny": money(model_error_sum),
        # 记录锁定基线总绝对误差，作为 Skill Score 分母来源。
        "locked_baseline_sum_absolute_error_target_change_cny": money(baseline_error_sum),
        # 记录相对锁定基线的技能分数。
        "skill_score_vs_locked_baseline": ratio(skill_score),
        # 明确评价范围只有验证集。
        "evaluation_scope": "validation_only",
        # 记录实现版本。
        "feature_manifest_version": XGBOOST_VERSION,
    }


# 按企业汇总验证误差；这用于识别模型在哪些企业表现较弱，不参与重新调参。
def summarize_by_enterprise(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 创建按企业聚集的字典；企业编号只用于报告，不作为训练特征。
    grouped: dict[str, list[dict[str, str]]] = {}
    # 遍历全部验证明细，并将每行放入对应企业列表。
    for row in rows:
        # setdefault 在企业第一次出现时创建空列表，随后追加当前记录。
        grouped.setdefault(row["enterprise_id"], []).append(row)
    # 创建逐户结果列表；后续按企业编号排序保证输出稳定。
    results: list[dict[str, str]] = []
    # 按企业编号排序遍历，避免字典插入次序影响文件指纹。
    for enterprise_id in sorted(grouped):
        # 读取该企业全部验证样本。
        enterprise_rows = grouped[enterprise_id]
        # 汇总该企业模型和锁定基线误差。
        model_sum = sum((Decimal(row["absolute_error_target_change_cny"]) for row in enterprise_rows), Decimal("0"))
        # 汇总同一企业同一日期的基线误差。
        baseline_sum = sum((Decimal(row["locked_baseline_absolute_error_target_change_cny"]) for row in enterprise_rows), Decimal("0"))
        # 汇总账户归一化误差。
        normalized_sum = sum((Decimal(row["account_normalized_absolute_error"]) for row in enterprise_rows), Decimal("0"))
        # 把该企业样本数转成 Decimal 用于平均。
        count = Decimal(len(enterprise_rows))
        # 基线误差为零时 Skill Score 无定义，用中文状态而不是伪造数值。
        enterprise_skill = "基线误差为0，无法计算" if baseline_sum == Decimal("0") else ratio(Decimal("1") - model_sum / baseline_sum)
        # 添加一条逐户指标记录。
        results.append({
            # 写入企业编号供用户筛选。
            "enterprise_id": enterprise_id,
            # 写入该企业验证样本数。
            "validation_sample_count": str(len(enterprise_rows)),
            # 写入该企业变化金额平均绝对误差。
            "mae_target_change_cny": money(model_sum / count),
            # 写入该企业账户归一化平均绝对误差。
            "account_normalized_mae": ratio(normalized_sum / count),
            # 写入相对同企业锁定基线的 Skill Score 或不可计算说明。
            "skill_score_vs_locked_baseline": enterprise_skill,
        })
    # 返回稳定排序的逐户指标。
    return results


# 生成特征重要性表；该表只解释训练后树分裂，不把重要性用于本轮调参。
def feature_importance_rows(booster: xgb.Booster) -> list[dict[str, str]]:
    # 取得每项特征的累计增益；未被使用的特征在字典中不存在。
    gain_by_feature = booster.get_score(importance_type="total_gain")
    # 计算总增益，供输出归一化占比；全部为零时使用1避免除零。
    total_gain = sum(gain_by_feature.values()) or 1.0
    # 为17项冻结特征全部写行，未使用项也明确显示为0。
    rows = [
        {
            # 写入特征名；名称与 T2 树表和模型文件保持一致。
            "feature_name": feature_name,
            # 写入累计增益并保留八位小数。
            "total_gain": f"{float(gain_by_feature.get(feature_name, 0.0)):.8f}",
            # 写入增益占比，便于人眼比较不同金额尺度特征。
            "gain_share": f"{float(gain_by_feature.get(feature_name, 0.0)) / total_gain:.8f}",
        }
        # 按冻结特征顺序遍历，保证即使未参与分裂也不漏项。
        for feature_name in MODEL_FEATURE_COLUMNS
    ]
    # 按增益从高到低、同分按特征名排序，方便用户阅读且保持确定性。
    return sorted(rows, key=lambda row: (-float(row["total_gain"]), row["feature_name"]))
