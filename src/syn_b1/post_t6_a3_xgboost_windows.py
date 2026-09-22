"""POST-T6-A3：在A1开发池比较三个预登记XGBoost观察窗口。"""

# 导入csv标准库；本模块只读取A1/A2派生CSV并写出可审计的预测结果。
import csv
# 导入hashlib标准库；它为输入和输出生成不可变的SHA-256指纹。
import hashlib
# 导入json标准库；它读取前置报告并写出A3的锁定和验收记录。
import json
# 从collections导入defaultdict；它按折、窗口、指标和企业累计等权误差。
from collections import defaultdict
# 从dataclasses导入dataclass；它用不可变对象表达标签和窗口特征。
from dataclasses import dataclass
# 从decimal导入Decimal与舍入规则；评价金额和比例保持财务展示精度。
from decimal import Decimal, ROUND_HALF_UP
# 从pathlib导入Path；它以跨平台对象定位冻结输入和版本化输出。
from pathlib import Path
# 从typing导入Iterable；它标注通用CSV写入器的流式输入。
from typing import Iterable

# 导入numpy；它把CSV数值特征转换为XGBoost可读取的二维矩阵。
import numpy as np
# 从A2复用A1血缘核验所用的异常、A0读取和文件哈希；避免A3私自重写前置边界。
from syn_b1.post_t6_a2_strong_baselines import A2ContractError, load_a0_contract, sha256_file

# 固定A3实现版本；未来改变训练、评价或锁定规则必须新建版本。
A3_VERSION = "syn_b1_post_t6_a3_xgboost_windows_1_0"
# 固定完整未来预测路径长度；A0要求每个模型同时覆盖T+1至T+30。
FORECAST_PATH_BUSINESS_DAYS = 30
# 固定三个首轮窗口；它们正是A0六个XGBoost候选的前三个。
WINDOWS = (14, 28, 60)
# 固定窗口候选编号；输出、锁定和后续结构比较必须使用这些预登记名称。
WINDOW_CONFIGURATION_IDS = {
    14: "xgb_direct_w14_l1",
    28: "xgb_direct_w28_l1",
    60: "xgb_direct_w60_l1",
}
# 固定四个评价项目；T+5必须评价完整五日路径而非只看第五日端点。
REGISTERED_MEASURES = (
    ("t_plus_1_endpoint", 1),
    ("t_plus_5_path", 5),
    ("t_plus_10_endpoint", 10),
    ("t_plus_30_endpoint", 30),
)
# 固定A3只预测的折角色；fit角色用于训练而不写出训练集内预测。
PREDICTION_ROLES = ("calibration", "validation")


# 声明A3专用异常；它把血缘、泄漏或候选清单错误同普通模型效果区分开。
class A3ContractError(A2ContractError):
    """当A3违反A0合同、A1/A2血缘或最终测试密封边界时抛出。"""


# 保存一个样本的公开标签和可见账户尺度；它不含画像、种子或人工备注。
@dataclass(frozen=True)
class TargetPath:
    # 保存样本键；它只连接特征、标签、折角色和输出审计。
    sample_key: str
    # 保存企业编号；它只用于企业等权评价，绝不进入训练矩阵。
    enterprise_id: str
    # 保存截止日可见余额；它用于把预测变化还原为余额并构造锁定基线。
    cutoff_balance_cny: Decimal
    # 保存截止日可见账户尺度；它只用于账户内归一化误差。
    balance_scale_cny: Decimal
    # 保存30日真实余额路径；它只用于开发池训练标签与开发评价。
    target_balances_cny: tuple[Decimal, ...]
    # 保存30日真实余额变化路径；直接树模型以它作为每个期限的训练标签。
    target_changes_cny: tuple[Decimal, ...]


# 将Decimal金额按人民币分写入CSV；模型训练仍使用浮点矩阵，展示不影响训练。
def money(value: Decimal) -> str:
    # 用普通财务舍入保留两位小数。
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# 将比例按八位小数写入CSV；它便于复核归一化误差与Skill Score。
def ratio(value: Decimal) -> str:
    # 使用定点格式避免Excel以科学计数法显示小比例。
    return f"{value.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP):f}"


# 以UTF-8或UTF-8-BOM流式读取CSV；读取器不接触任何原始企业目录或备注文件。
def iter_csv(path: Path) -> Iterable[dict[str, str]]:
    # 打开冻结派生文件；newline为空保留csv模块的跨平台换行处理。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # 逐行产生字典，避免把百万行预测文件一次读入内存。
        yield from csv.DictReader(handle)


# 读取A0的首轮XGBoost窗口合同，并拒绝任何候选或共享参数漂移。
def validate_a0_xgboost_contract(contract_path: Path) -> dict[str, object]:
    # 先调用A2已使用的A0通用核验，确认合同编号和冻结状态。
    contract = load_a0_contract(contract_path)
    # 读取XGBoost有限参数矩阵；缺失说明不能证明A3的候选来源。
    matrix = contract["limited_parameter_matrix"]["xgboost"]
    # 读取前三个窗口候选；A3绝不提前运行后续结构或损失比较候选。
    first_three = matrix["configurations"][:3]
    # 提取候选编号，顺序也是预登记的窗口比较顺序。
    ids = tuple(item["id"] for item in first_three)
    # 要求编号与本模块的14/28/60映射完全一致。
    if ids != tuple(WINDOW_CONFIGURATION_IDS[window] for window in WINDOWS):
        # 阻断错误合同或代码偷偷换窗口。
        raise A3ContractError("A0前三个XGBoost窗口候选与A3固定清单不一致")
    # 逐个窗口核验对应的历史长度、直接结构和L1损失。
    for item, window in zip(first_three, WINDOWS, strict=True):
        # 要求窗口长度精确匹配。
        if item["lookback_business_days"] != window:
            # 阻断错配的窗口配置。
            raise A3ContractError("A0 XGBoost窗口长度与A3不一致")
        # 要求结构是每期限一棵直接预测树。
        if item["structure"] != "direct":
            # 阻断未获授权的两阶段结构提前进入A3。
            raise A3ContractError("A3只能运行直接预测结构")
        # 要求损失是绝对误差。
        if item["loss"] != "absolute_error":
            # 阻断未获授权的Huber或尺度加权损失。
            raise A3ContractError("A3只能运行绝对误差损失")
    # 读取共享固定参数，后续训练器不接受命令行调参。
    shared = matrix["shared_parameters"]
    # 要求A0指定直接预测和禁止验证集早停。
    if shared["prediction_style"] != "one_direct_model_per_future_business_day" or shared["early_stopping_forbidden"] is not True:
        # 阻断训练结构或验证集边界改变。
        raise A3ContractError("A0没有冻结A3所需的直接预测或禁用早停规则")
    # 返回完整合同供调用方记录指纹和固定参数。
    return contract


# 核验A2的强基线输出、指纹与密封声明，并返回四个已锁定方法。
def validate_a2_contract(a2_directory: Path, contract_path: Path, a1_directory: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    # 定位A2报告、锁定结果和汇总指标；三者缺一不可。
    report_path = a2_directory / "a2_strong_baseline_report.json"
    # 定位A2锁定文件。
    locked_path = a2_directory / "locked_strong_baselines.json"
    # 定位A2汇总指标文件。
    summary_path = a2_directory / "baseline_summary_metrics.csv"
    # 要求每份A2证据都存在。
    if not report_path.is_file() or not locked_path.is_file() or not summary_path.is_file():
        # 阻断不完整强基线证据。
        raise A3ContractError("缺少A2强基线报告、锁定文件或汇总指标")
    # 读取A2报告。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 读取A2锁定结果。
    locked_payload = json.loads(locked_path.read_text(encoding="utf-8"))
    # 要求A2明确通过。
    if report.get("status") != "passed":
        # 阻断未通过的基线阶段。
        raise A3ContractError("A2强基线阶段未通过")
    # 逐项要求A2没有开启最终测试或训练复杂模型。
    for key in ("final_test_run_directories_opened", "final_test_daily_files_read", "final_test_labels_read_or_assembled", "final_test_features_created", "models_trained_or_tuned", "candidate_selected", "final_test_opened"):
        # 任一标志不是false即说明前置边界已被破坏。
        if report.get(key) is not False:
            # 阻断继续训练并报告触发字段。
            raise A3ContractError(f"A2报告边界不通过：{key}")
    # 要求A2的A0合同指纹与当前冻结合同一致。
    if report["input_sha256"]["a0_contract"] != sha256_file(contract_path):
        # 阻断在不同实验合同上沿用基线。
        raise A3ContractError("A2绑定的A0合同指纹不一致")
    # 要求A2绑定的A1报告指纹与当前A1目录一致。
    if report["input_sha256"]["a1_report"] != sha256_file(a1_directory / "a1_derivative_report.json"):
        # 阻断A1数据版本漂移。
        raise A3ContractError("A2绑定的A1报告指纹不一致")
    # 逐项重算A2已哈希的输出，防止锁定文件或指标被静默改写。
    for name, expected_hash in report["output_sha256"].items():
        # 要求当前文件哈希等于A2报告保存的哈希。
        if sha256_file(a2_directory / name) != expected_hash:
            # 阻断被改写的强基线结果。
            raise A3ContractError(f"A2输出指纹不一致：{name}")
    # 读取四期限锁定字典。
    locked = locked_payload.get("locked_baselines")
    # 要求锁定结果是字典且恰有四个预登记项目。
    if not isinstance(locked, dict) or set(locked) != {name for name, _ in REGISTERED_MEASURES}:
        # 阻断缺期限或额外期限的锁定文件。
        raise A3ContractError("A2锁定基线期限集合不完整")
    # A2实际锁定的全部方法均是余额保持；A3据此在同一截止日逐条重建配对基线。
    if any(item.get("locked_baseline_method") != "last_balance_path" for item in locked.values()):
        # 阻断尚未实现的其他锁定方法，避免错误地用余额保持冒充它。
        raise A3ContractError("当前A3只支持A2实际锁定的余额保持基线")
    # 返回报告、锁定结果与汇总指标路径信息供主流程使用。
    return report, locked_payload, {"summary_path": summary_path.name}


# 读取A1滚动索引并将每折三个角色分别保存为样本键列表。
def read_fold_roles(path: Path) -> dict[str, dict[str, list[str]]]:
    # 初始化四折角色字典。
    folds: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"fit": [], "calibration": [], "validation": []})
    # 初始化每折已见样本键集合，防止同折同角色重复。
    seen: dict[str, set[str]] = defaultdict(set)
    # 逐行读取不含金额的A1索引。
    for row in iter_csv(path):
        # 读取折编号。
        fold_id = row["fold_id"]
        # 读取角色名称。
        role = row["fold_role"]
        # 读取样本键。
        sample_key = row["sample_key"]
        # 要求角色只能是A1声明的三种。
        if role not in folds[fold_id]:
            # 阻断未知角色。
            raise A3ContractError(f"A1滚动索引存在未知角色：{role}")
        # 要求同一折不会重复同一截止点。
        if sample_key in seen[fold_id]:
            # 阻断角色重叠或重复索引。
            raise A3ContractError(f"A1滚动索引同折重复样本：{fold_id} {sample_key}")
        # 记录当前样本键已见。
        seen[fold_id].add(sample_key)
        # 追加当前角色列表。
        folds[fold_id][role].append(sample_key)
    # 要求恰好A/B/C/D四折。
    if set(folds) != {"A", "B", "C", "D"}:
        # 阻断折集合漂移。
        raise A3ContractError("A1滚动索引不是A/B/C/D四折")
    # 逐折要求fit、校准和验证均非空。
    for fold_id, roles in folds.items():
        # 检查三类列表。
        if any(not roles[role] for role in ("fit", "calibration", "validation")):
            # 阻断空角色，避免少跑困难季度。
            raise A3ContractError(f"A1滚动索引存在空角色：{fold_id}")
    # 返回按折角色分组的样本键。
    return dict(folds)


# 读取开发池30日标签表；它只来自A1而不是任何最终测试文件。
def read_targets(path: Path) -> dict[str, TargetPath]:
    # 初始化样本键到标签对象的映射。
    targets: dict[str, TargetPath] = {}
    # 逐行读取102,964条开发池标签。
    for row in iter_csv(path):
        # 读取样本键。
        sample_key = row["sample_key"]
        # 要求样本键不重复。
        if sample_key in targets:
            # 阻断重复标签。
            raise A3ContractError(f"A1标签表样本键重复：{sample_key}")
        # 解析30日真实余额路径。
        balances = tuple(Decimal(row[f"target_balance_t_plus_{step:02d}_cny"]) for step in range(1, FORECAST_PATH_BUSINESS_DAYS + 1))
        # 解析30日真实余额变化路径。
        changes = tuple(Decimal(row[f"target_change_t_plus_{step:02d}_cny"]) for step in range(1, FORECAST_PATH_BUSINESS_DAYS + 1))
        # 保存不含画像和备注的标签对象。
        targets[sample_key] = TargetPath(sample_key, row["enterprise_id"], Decimal(row["cutoff_balance_cny"]), Decimal(row["balance_scale_cny"]), balances, changes)
    # 要求标签表非空。
    if not targets:
        # 阻断空标签文件。
        raise A3ContractError("A1开发池标签表为空")
    # 返回完整开发池标签映射。
    return targets


# 读取一个窗口的公开树特征；只接收A1特征清单允许的真实模型列。
def read_window_features(path: Path, model_columns: tuple[str, ...], expected_keys: set[str]) -> dict[str, np.ndarray]:
    # 初始化样本键到浮点特征向量的映射。
    features: dict[str, np.ndarray] = {}
    # 逐行读取当前窗口的102,964条公开树特征。
    for row in iter_csv(path):
        # 读取样本键。
        sample_key = row["sample_key"]
        # 要求样本键不重复。
        if sample_key in features:
            # 阻断重复特征行。
            raise A3ContractError(f"A1树特征样本键重复：{sample_key}")
        # 将冻结列按固定顺序转换为float64向量。
        values = np.asarray([float(row[column]) for column in model_columns], dtype=np.float64)
        # 要求没有缺失或无穷数值。
        if not np.isfinite(values).all():
            # 阻断不可训练特征。
            raise A3ContractError(f"A1树特征存在缺失或无穷：{sample_key}")
        # 保存当前样本特征。
        features[sample_key] = values
    # 要求当前窗口的样本键与标签表完全一一对应。
    if set(features) != expected_keys:
        # 阻断漏行、额外行或错版本特征。
        raise A3ContractError("A1树特征样本键与标签表不完全一致")
    # 返回已验证的窗口特征映射。
    return features


# 把指定样本键列表转换为训练矩阵；企业编号、日期和角色不会进入矩阵。
def make_feature_matrix(keys: list[str], features: dict[str, np.ndarray]) -> np.ndarray:
    # 逐样本按索引顺序取出公开数值特征。
    matrix = np.asarray([features[key] for key in keys], dtype=np.float64)
    # 要求二维且至少一行一列。
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        # 阻断空训练或错误矩阵。
        raise A3ContractError("XGBoost特征矩阵为空或维度错误")
    # 返回冻结列顺序的数值矩阵。
    return matrix


# 从A0共享参数构造原生XGBoost训练参数；不接受命令行覆盖或验证集早停。
def xgboost_parameters(contract: dict[str, object]) -> dict[str, object]:
    # 读取A0共享参数。
    shared = contract["limited_parameter_matrix"]["xgboost"]["shared_parameters"]
    # 返回直接L1回归所需的固定参数字典。
    return {
        # 使用绝对误差回归目标，对应A0的absolute_error。
        "objective": "reg:absoluteerror",
        # 使用CPU直方图建树，保证本机可复现与较快训练。
        "tree_method": "hist",
        # 设置冻结树深。
        "max_depth": int(shared["max_depth"]),
        # 设置冻结学习率。
        "eta": float(shared["learning_rate"]),
        # 设置冻结子样本比例。
        "subsample": float(shared["subsample"]),
        # 设置冻结列采样比例。
        "colsample_bytree": float(shared["colsample_bytree"]),
        # 设置冻结最小叶权重。
        "min_child_weight": float(shared["min_child_weight"]),
        # 设置冻结L2正则。
        "reg_lambda": float(shared["reg_lambda"]),
        # 设置固定随机种子。
        "seed": int(shared["random_seed"]),
        # 固定四线程，控制运行资源且不改变实验变量。
        "nthread": 4,
    }


# 初始化按折、窗口、指标、企业累计模型与基线误差的嵌套桶。
def new_aggregates() -> dict[str, dict[int, dict[str, dict[str, list[Decimal]]]]]:
    # 返回默认字典；最内层四格依次为模型误差和、基线误差和、计数和预测次数。
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0")]))))


# 用一个验证样本的完整30日预测路径累计四项企业等权指标。
def update_validation_aggregates(aggregates: dict[str, dict[int, dict[str, dict[str, list[Decimal]]]]], fold_id: str, window: int, target: TargetPath, predicted_balances: tuple[Decimal, ...]) -> None:
    # 逐个预登记指标更新。
    for measure_name, horizon in REGISTERED_MEASURES:
        # T+5路径评价第1至5日，其他项目只评价相应端点。
        steps = range(1, horizon + 1) if measure_name == "t_plus_5_path" else (horizon,)
        # 逐个需要评价的未来日累计。
        for step in steps:
            # 计算模型预测余额绝对误差。
            model_error = abs(predicted_balances[step - 1] - target.target_balances_cny[step - 1]) / target.balance_scale_cny
            # 余额保持基线预测就是截止日余额，计算同一样本同一日的配对误差。
            baseline_error = abs(target.cutoff_balance_cny - target.target_balances_cny[step - 1]) / target.balance_scale_cny
            # 取得当前企业累计桶。
            bucket = aggregates[fold_id][window][measure_name][target.enterprise_id]
            # 累加模型归一化误差。
            bucket[0] += model_error
            # 累加配对基线归一化误差。
            bucket[1] += baseline_error
            # 累加当前企业的预测日计数。
            bucket[2] += Decimal("1")
            # 累加审计用预测次数。
            bucket[3] += Decimal("1")


# 把企业累计桶变成四折、三窗口、四指标的可写入CSV指标行。
def fold_metric_rows(aggregates: dict[str, dict[int, dict[str, dict[str, list[Decimal]]]]]) -> list[dict[str, str]]:
    # 初始化结果行列表。
    rows: list[dict[str, str]] = []
    # 按折号稳定输出。
    for fold_id in sorted(aggregates):
        # 按预登记窗口稳定输出。
        for window in WINDOWS:
            # 按预登记指标稳定输出。
            for measure_name, _ in REGISTERED_MEASURES:
                # 读取当前企业累计桶。
                enterprise_buckets = aggregates[fold_id][window][measure_name]
                # 要求每折每窗口每指标都有企业。
                if not enterprise_buckets:
                    # 阻断少跑任何困难折或指标。
                    raise A3ContractError(f"A3缺少折{fold_id}窗口{window}指标{measure_name}的验证企业")
                # 逐企业计算各自平均模型误差。
                model_means = [values[0] / values[2] for values in enterprise_buckets.values()]
                # 逐企业计算各自平均基线误差。
                baseline_means = [values[1] / values[2] for values in enterprise_buckets.values()]
                # 再让每家企业等权平均，防止样本多的企业支配结果。
                model_macro = sum(model_means, Decimal("0")) / Decimal(len(model_means))
                # 同样计算基线企业等权误差，保证Skill Score同口径。
                baseline_macro = sum(baseline_means, Decimal("0")) / Decimal(len(baseline_means))
                # 基线误差为零时无法定义相对技能，直接拒绝而不是伪造数值。
                if baseline_macro == 0:
                    # 阻断无定义Skill Score。
                    raise A3ContractError("A3验证折的余额保持基线误差为零")
                # 计算相对同折同窗口基线的技能分数。
                skill = Decimal("1") - model_macro / baseline_macro
                # 写入完整审计指标行。
                rows.append({
                    "fold_id": fold_id,
                    "configuration_id": WINDOW_CONFIGURATION_IDS[window],
                    "lookback_business_days": str(window),
                    "registered_measure": measure_name,
                    "enterprise_count": str(len(enterprise_buckets)),
                    "prediction_count": str(sum(int(values[3]) for values in enterprise_buckets.values())),
                    "model_enterprise_macro_account_normalized_mae": ratio(model_macro),
                    "locked_baseline_enterprise_macro_account_normalized_mae": ratio(baseline_macro),
                    "skill_score_vs_locked_baseline": ratio(skill),
                    "a3_version": A3_VERSION,
                })
    # 返回固定48行指标。
    return rows


# 计算Decimal列表中位数；窗口选择按A0要求以四折中位误差决定。
def decimal_median(values: list[Decimal]) -> Decimal:
    # 要求至少一项输入。
    if not values:
        # 阻断空指标集合。
        raise A3ContractError("空列表不能计算四折中位数")
    # 对数值排序。
    ordered = sorted(values)
    # 取得中点位置。
    middle = len(ordered) // 2
    # 奇数列表直接返回中点。
    if len(ordered) % 2 == 1:
        # 返回唯一中位数。
        return ordered[middle]
    # 四折取中间两项的平均。
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")


# 汇总每窗口四折指标，并只依据T+5主指标锁定一个窗口。
def summarize_and_lock(fold_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    # 初始化指标到窗口到折行的分组。
    grouped: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    # 逐行归入对应指标和窗口。
    for row in fold_rows:
        # 将窗口文本还原为整数键。
        grouped[row["registered_measure"]][int(row["lookback_business_days"])].append(row)
    # 初始化汇总行。
    summary: list[dict[str, str]] = []
    # 初始化T+5窗口分数。
    t5_scores: dict[int, Decimal] = {}
    # 按四个指标输出所有窗口汇总。
    for measure_name, horizon in REGISTERED_MEASURES:
        # 按预登记窗口依次处理。
        for window in WINDOWS:
            # 取得当前窗口四折结果。
            rows = grouped[measure_name][window]
            # 要求恰好四折，不能遗漏不利季度。
            if len(rows) != 4:
                # 阻断缺折结果。
                raise A3ContractError(f"A3窗口{window}指标{measure_name}不是完整四折")
            # 提取四折模型误差。
            model_values = [Decimal(row["model_enterprise_macro_account_normalized_mae"]) for row in rows]
            # 提取四折基线误差。
            baseline_values = [Decimal(row["locked_baseline_enterprise_macro_account_normalized_mae"]) for row in rows]
            # 计算四折模型误差中位数。
            model_median = decimal_median(model_values)
            # 计算四折基线误差中位数。
            baseline_median = decimal_median(baseline_values)
            # 基线中位数必须非零。
            if baseline_median == 0:
                # 阻断无定义总体Skill Score。
                raise A3ContractError("A3汇总基线误差为零")
            # 计算基于同口径中位误差的Skill Score。
            skill = Decimal("1") - model_median / baseline_median
            # 写入当前指标和窗口的汇总行。
            summary.append({
                "registered_measure": measure_name,
                "horizon_business_days": str(horizon),
                "configuration_id": WINDOW_CONFIGURATION_IDS[window],
                "lookback_business_days": str(window),
                "fold_count": "4",
                "median_model_enterprise_macro_account_normalized_mae": ratio(model_median),
                "median_locked_baseline_enterprise_macro_account_normalized_mae": ratio(baseline_median),
                "median_skill_score_vs_locked_baseline": ratio(skill),
                "minimum_fold_model_macro_mae": ratio(min(model_values)),
                "maximum_fold_model_macro_mae": ratio(max(model_values)),
                "selection_metric": "four_fold_median_enterprise_macro_account_normalized_mae",
                "a3_version": A3_VERSION,
            })
            # T+5是唯一窗口选择依据，保存它的模型误差中位数。
            if measure_name == "t_plus_5_path":
                # 保存当前窗口主指标分数。
                t5_scores[window] = model_median
    # 以误差最小为先、窗口更短为同分规则锁定唯一窗口。
    selected_window = min(WINDOWS, key=lambda window: (t5_scores[window], window))
    # 返回全部汇总行和机器可读的唯一窗口锁定记录。
    return summary, {
        "selection_scope": "A1四折validation角色，仅208户开发池",
        "selection_metric": "four_fold_median_enterprise_macro_account_normalized_mae_on_t_plus_5_path",
        "tie_breaker_cn": "T+5四折中位误差完全相同则按14、28、60日顺序选择更短窗口；不得看结构、损失、LSTM或最终测试。",
        "selected_configuration_id": WINDOW_CONFIGURATION_IDS[selected_window],
        "selected_lookback_business_days": selected_window,
        "selected_t_plus_5_median_model_error": ratio(t5_scores[selected_window]),
    }


# 写入严格字段顺序的CSV；输出目录必须由调用方保证全新。
def write_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    # 新建UTF-8 CSV。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 创建严格列清单写入器。
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # 写出表头。
        writer.writeheader()
        # 写出所有行。
        writer.writerows(rows)


# 计算本模块代码文件指纹；报告用它明确记录实际运行的实现版本。
def module_sha256() -> str:
    # 对当前源文件的原始字节计算SHA-256。
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
