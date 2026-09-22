"""POST-T6-B1：小型LSTM的预登记14、28、60日观察窗口比较规则。"""

# 导入CSV标准库；它读取A5的完成证据并写出固定列顺序结果。
import csv
# 导入JSON标准库；它核验前置阶段报告和锁定文件。
import json
# 从collections导入defaultdict；它按配置、折和评价指标分组。
from collections import defaultdict
# 从decimal导入Decimal；窗口选择使用精确十进制误差。
from decimal import Decimal
# 从pathlib导入Path；它定位冻结的合同与前置输出。
from pathlib import Path
# 从typing导入Iterable；它标注通用CSV写入函数。
from typing import Iterable

# 导入YAML；它读取A0的LSTM候选清单。
import yaml

# 复用A3的评价期限、显示格式和文件哈希，避免改变误差口径。
from syn_b1.post_t6_a3_xgboost_windows import REGISTERED_MEASURES, ratio, sha256_file

# 固定B1实现版本；任何训练或选择口径改变必须创建新版本。
B1_VERSION = "syn_b1_post_t6_b1_lstm_windows_1_0"
# 固定A0预登记的三个窗口长度。
WINDOWS = (14, 28, 60)
# 固定A0预登记的三个训练种子；必须全部训练并保留。
TRAINING_SEEDS = (2026083102, 2026083103, 2026083104)
# 固定三个窗口候选的可追溯编号。
WINDOW_CONFIGURATION_IDS = {
    14: "lstm_direct_w14_h16_l1",
    28: "lstm_direct_w28_h16_l1",
    60: "lstm_direct_w60_h16_l1",
}


# 声明B1专用异常；它使血缘或密封错误不能被误作训练结果。
class B1ContractError(ValueError):
    """当B1输入血缘、候选矩阵或最终测试密封边界不符合合同要求时抛出。"""


# 读取并核验A0前三个LSTM窗口候选和共同固定训练参数。
def validate_a0_lstm_contract(contract_path: Path) -> dict[str, object]:
    # 读取冻结YAML文本并解析为字典。
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    # 要求合同身份及状态均准确，防止使用未冻结草稿。
    if contract.get("stage_id") != "POST-T6-A0" or contract.get("status") != "frozen":
        # 阻断不受控合同。
        raise B1ContractError("B1只能使用冻结的POST-T6-A0合同")
    # 读取小型LSTM有限参数矩阵。
    matrix = contract["limited_parameter_matrix"]["small_lstm"]
    # 读取前三个窗口候选；B1不得提前运行后续隐藏层、双头或损失候选。
    configurations = matrix["configurations"][:3]
    # 要求候选数量完整。
    if len(configurations) != len(WINDOWS):
        # 阻断不完整窗口矩阵。
        raise B1ContractError("A0没有完整登记三个LSTM窗口候选")
    # 按预登记顺序逐项核验三个窗口。
    for configuration, window in zip(configurations, WINDOWS, strict=True):
        # 要求编号、窗口、隐藏维度、直接路径和L1损失完全相同。
        valid = (
            configuration["id"] == WINDOW_CONFIGURATION_IDS[window]
            and configuration["lookback_business_days"] == window
            and configuration["hidden_size"] == 16
            and configuration["structure"] == "direct_path"
            and configuration["loss"] == "absolute_error"
        )
        # 任一候选漂移都禁止继续。
        if not valid:
            # 指出候选不符合冻结清单。
            raise B1ContractError("A0 LSTM窗口候选与B1固定清单不一致")
    # 读取共同训练设置。
    shared = matrix["shared_parameters"]
    # 要求输出长度、网络层数和CPU设备正是A0冻结的第一轮研究范围。
    if shared["output_business_days"] != 30 or shared["num_layers"] != 1 or shared["device"] != "cpu":
        # 阻断结构或运行设备漂移。
        raise B1ContractError("A0 LSTM共同训练参数不符合B1范围")
    # 要求三个种子没有被改变。
    if tuple(matrix["training_seeds"]) != TRAINING_SEEDS:
        # 阻断种子漂移或只挑单种子。
        raise B1ContractError("A0 LSTM训练种子与B1固定清单不一致")
    # 返回合同，供训练器读取全部冻结超参数。
    return contract


# 核验A5已经完成且仍未触碰最终测试，然后绑定A0合同指纹。
def validate_a5_boundary(a5_directory: Path, contract_path: Path) -> dict[str, object]:
    # 定位A5阶段报告。
    report_path = a5_directory / "a5_xgboost_loss_report.json"
    # 定位A5的损失锁定记录。
    lock_path = a5_directory / "locked_xgboost_loss.json"
    # 要求两份完成证据存在。
    if not report_path.is_file() or not lock_path.is_file():
        # 阻断缺少前置阶段证据的训练。
        raise B1ContractError("缺少A5报告或XGBoost损失锁定记录")
    # 读取A5报告。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 读取A5锁定记录。
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    # 要求A5通过，且它只完成XGBoost内部选择。
    if report.get("status") != "passed" or lock.get("selected_configuration_id") != "xgb_direct_w60_l1":
        # 阻断不完整或漂移的前置结果。
        raise B1ContractError("A5未通过或没有锁定既定XGBoost内部配置")
    # 要求A5记录的A0指纹仍匹配当前冻结合同。
    if report["input_sha256"]["a0"] != sha256_file(contract_path):
        # 阻断合同版本漂移。
        raise B1ContractError("A5绑定的A0合同指纹不一致")
    # 列出任何一个为真都代表不允许继续的密封边界。
    blocked_keys = (
        "final_test_opened",
        "final_test_labels_read_or_assembled",
        "final_test_features_created",
        "cash_flow_review_notes_read_or_loaded",
        "lstm_trained",
        "unique_candidate_selected",
    )
    # 逐项检查前置报告。
    for key in blocked_keys:
        # 只接受明确False，缺失字段也不被当作安全。
        if report.get(key) is not False:
            # 阻断任何密封越界。
            raise B1ContractError(f"A5边界不通过：{key}")
    # 返回读取后的前置证据。
    return {"report": report, "lock": lock}


# 将严格字段顺序的行写入新的UTF-8 CSV；调用方必须确保目录尚未存在。
def write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    # 以newline空值打开，交由csv库处理跨平台换行。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 固定字段顺序，防止不同运行写出不同列顺序。
        writer = csv.DictWriter(handle, fieldnames=fields)
        # 写出表头。
        writer.writeheader()
        # 写出所有结果行。
        writer.writerows(rows)


# 将三个种子的同折指标合成为“种子中位数”折指标，供窗口选择使用。
def seed_median_fold_rows(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 初始化按窗口、折和期限分组的映射。
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    # 将种子级行按共同实验单元归组。
    for row in seed_rows:
        # 键不含种子，因而每组应恰有三个种子。
        grouped[(row["configuration_id"], row["fold_id"], row["registered_measure"])].append(row)
    # 初始化合成后的48行折指标。
    rows: list[dict[str, str]] = []
    # 以稳定顺序处理配置、折和指标。
    for configuration_id, fold_id, measure_name in sorted(grouped):
        # 读取三种子行。
        members = grouped[(configuration_id, fold_id, measure_name)]
        # 要求恰好三个且种子集合完整。
        if len(members) != len(TRAINING_SEEDS) or {int(row["training_seed"]) for row in members} != set(TRAINING_SEEDS):
            # 阻断遗漏或重复种子。
            raise B1ContractError("B1折指标没有完整三种子")
        # 从小到大排序三种子模型误差，中位数为中间项。
        model_values = sorted(Decimal(row["model_enterprise_macro_account_normalized_mae"]) for row in members)
        # 同样读取基线误差；理论相同，仍按中位数写出以便独立复核。
        baseline_values = sorted(Decimal(row["locked_baseline_enterprise_macro_account_normalized_mae"]) for row in members)
        # 取中间种子模型误差。
        model_median = model_values[1]
        # 取中间种子基线误差。
        baseline_median = baseline_values[1]
        # 基线为零时技能分数没有定义。
        if baseline_median == 0:
            # 阻断无意义比较。
            raise B1ContractError("B1折指标的锁定基线误差为零")
        # 写出种子中位数折指标。
        rows.append({
            "fold_id": fold_id,
            "configuration_id": configuration_id,
            "registered_measure": measure_name,
            "enterprise_count": members[0]["enterprise_count"],
            "prediction_count": members[0]["prediction_count"],
            "median_over_training_seeds": "3",
            "median_model_enterprise_macro_account_normalized_mae": ratio(model_median),
            "median_locked_baseline_enterprise_macro_account_normalized_mae": ratio(baseline_median),
            "median_skill_score_vs_locked_baseline": ratio(Decimal("1") - model_median / baseline_median),
            "b1_version": B1_VERSION,
        })
    # 三窗口×四折×四期限应当恰为48行。
    if len(rows) != len(WINDOWS) * 4 * len(REGISTERED_MEASURES):
        # 阻断少窗口、少折或少期限。
        raise B1ContractError("B1种子中位折指标数量不完整")
    # 返回可供窗口选择的折指标。
    return rows


# 汇总三窗口的四折种子中位数指标，并且只按T+5路径选择一个窗口。
def summarize_and_lock_windows(fold_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    # 初始化“指标→窗口→折行”的嵌套分组。
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    # 逐行归组。
    for row in fold_rows:
        # 使用候选编号作为窗口键。
        grouped[row["registered_measure"]][row["configuration_id"]].append(row)
    # 初始化汇总输出。
    summary: list[dict[str, str]] = []
    # 初始化T+5的窗口误差字典。
    t_plus_5_scores: dict[str, Decimal] = {}
    # 按预登记期限处理。
    for measure_name, horizon_days in REGISTERED_MEASURES:
        # 按短窗口优先的预登记顺序处理候选。
        for window in WINDOWS:
            # 取得当前窗口的候选编号。
            configuration_id = WINDOW_CONFIGURATION_IDS[window]
            # 取得四折的种子中位数行。
            members = grouped[measure_name][configuration_id]
            # 要求完整四折，不能删掉困难季度后选择窗口。
            if len(members) != 4:
                # 阻断不完整四折。
                raise B1ContractError("B1窗口汇总不是完整四折")
            # 从小到大排序四折模型误差。
            model_values = sorted(Decimal(row["median_model_enterprise_macro_account_normalized_mae"]) for row in members)
            # 从小到大排序四折基线误差。
            baseline_values = sorted(Decimal(row["median_locked_baseline_enterprise_macro_account_normalized_mae"]) for row in members)
            # 四折中位数为中间两个误差的平均。
            model_median = (model_values[1] + model_values[2]) / Decimal("2")
            # 同口径计算基线中位数。
            baseline_median = (baseline_values[1] + baseline_values[2]) / Decimal("2")
            # 防止技能分数分母为零。
            if baseline_median == 0:
                # 明确拒绝无定义值。
                raise B1ContractError("B1窗口汇总的锁定基线误差为零")
            # 计算相对冻结基线的技能分数。
            skill_score = Decimal("1") - model_median / baseline_median
            # 写出窗口汇总行。
            summary.append({
                "registered_measure": measure_name,
                "horizon_business_days": str(horizon_days),
                "configuration_id": configuration_id,
                "lookback_business_days": str(window),
                "fold_count": "4",
                "seed_count_per_fold": "3",
                "median_model_enterprise_macro_account_normalized_mae": ratio(model_median),
                "median_locked_baseline_enterprise_macro_account_normalized_mae": ratio(baseline_median),
                "median_skill_score_vs_locked_baseline": ratio(skill_score),
                "b1_version": B1_VERSION,
            })
            # 仅登记T+5成绩作为窗口选择依据。
            if measure_name == "t_plus_5_path":
                # 保存当前窗口误差。
                t_plus_5_scores[configuration_id] = model_median
    # 先按最小T+5误差选择；精确同分时按14、28、60日选择较短窗口。
    selected = min(WINDOWS, key=lambda window: (t_plus_5_scores[WINDOW_CONFIGURATION_IDS[window]], window))
    # 返回所有窗口诊断与锁定记录；该锁定尚不是跨家族唯一候选。
    return summary, {
        "selected_configuration_id": WINDOW_CONFIGURATION_IDS[selected],
        "selected_lookback_business_days": selected,
        "selected_t_plus_5_median_model_error": ratio(t_plus_5_scores[WINDOW_CONFIGURATION_IDS[selected]]),
        "tie_breaker_cn": "T+5四折种子中位误差相同则选择更短窗口：14日、28日、60日；不得看后续结构、损失、XGBoost或最终测试。",
    }
