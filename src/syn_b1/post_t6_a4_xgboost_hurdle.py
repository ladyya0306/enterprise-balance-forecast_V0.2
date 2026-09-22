"""POST-T6-A4：在A3锁定60日窗口上比较XGBoost两阶段结构。"""

# 导入csv标准库；本模块读取A3指标并写出严格字段顺序的比较表。
import csv
# 导入json标准库；它核验A3报告并写出结构锁定记录。
import json
# 从collections导入defaultdict；它按指标和结构汇总四折结果。
from collections import defaultdict
# 从decimal导入Decimal；指标汇总保持财务精度。
from decimal import Decimal
# 从pathlib导入Path；它定位冻结合同和A3证据目录。
from pathlib import Path
# 从typing导入Iterable；它标注CSV写入器的通用行输入。
from typing import Iterable

# 导入yaml；它读取A0中唯一允许的两阶段候选定义。
import yaml

# 从A3复用A3异常和显示函数；A4不私自改写A3评价口径。
from syn_b1.post_t6_a3_xgboost_windows import A3ContractError, REGISTERED_MEASURES, ratio, sha256_file

# 固定A4实现版本；改变两阶段组合规则必须升级版本。
A4_VERSION = "syn_b1_post_t6_a4_xgboost_hurdle_1_0"
# 固定A3已锁定的60日窗口；A4不得重新比较14或28日。
SELECTED_WINDOW = 60
# 固定直接结构的继承候选编号。
DIRECT_CONFIGURATION_ID = "xgb_direct_w60_l1"
# 固定A0预登记两阶段候选编号。
HURDLE_CONFIGURATION_ID = "xgb_hurdle_selected_window_l1"
# 固定二元事件标签：变化恰为零是无事件，非零是有变化。
ZERO_CHANGE = Decimal("0")


# 声明A4专用异常；它区分A3血缘错误与两阶段训练结果。
class A4ContractError(A3ContractError):
    """当A4违反结构比较合同、继承窗口或最终测试密封时抛出。"""


# 核验A0第四个XGBoost候选恰为已登记两阶段结构，且前三个窗口候选未被重新启用。
def validate_a0_hurdle_contract(contract_path: Path) -> dict[str, object]:
    # 安全读取冻结YAML合同。
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    # 要求合同阶段和状态正确。
    if contract.get("stage_id") != "POST-T6-A0" or contract.get("status") != "frozen":
        # 阻断未冻结或错误合同。
        raise A4ContractError("A4只能使用冻结的POST-T6-A0合同")
    # 读取XGBoost候选清单。
    configurations = contract["limited_parameter_matrix"]["xgboost"]["configurations"]
    # 要求第四项精确为A0预登记的两阶段候选。
    if len(configurations) < 4 or configurations[3]["id"] != HURDLE_CONFIGURATION_ID:
        # 阻断候选顺序或名称漂移。
        raise A4ContractError("A0没有按预登记清单提供A4两阶段候选")
    # 读取两阶段候选字段。
    hurdle = configurations[3]
    # 要求窗口来源必须是前三个窗口锁定结果。
    if hurdle["lookback_business_days"] != "selected_from_first_three_only":
        # 阻断A4擅自尝试新窗口。
        raise A4ContractError("A4两阶段候选没有绑定前三窗口的锁定结果")
    # 要求结构和损失精确匹配。
    if hurdle["structure"] != "event_probability_plus_nonzero_amount" or hurdle["loss"] != "binary_logloss_plus_absolute_error":
        # 阻断错误模型结构或损失。
        raise A4ContractError("A4两阶段候选结构或损失与A0不一致")
    # 返回已核验合同。
    return contract


# 核验A3报告、锁定窗口、输出指纹及最终测试密封，并读取A3直接结构折指标。
def validate_a3_and_read_direct_rows(a3_directory: Path, contract_path: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    # 定位A3报告。
    report_path = a3_directory / "a3_xgboost_window_report.json"
    # 定位A3窗口锁定文件。
    lock_path = a3_directory / "locked_xgboost_window.json"
    # 定位A3折指标文件。
    metrics_path = a3_directory / "xgboost_window_fold_metrics.csv"
    # 要求三份证据存在。
    if not report_path.is_file() or not lock_path.is_file() or not metrics_path.is_file():
        # 阻断不完整A3输出。
        raise A4ContractError("缺少A3报告、窗口锁定或折指标文件")
    # 解析A3报告。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 解析A3窗口锁定文件。
    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    # 要求A3已通过且锁定60日直接结构。
    if report.get("status") != "passed" or locked.get("selected_configuration_id") != DIRECT_CONFIGURATION_ID or locked.get("selected_lookback_business_days") != SELECTED_WINDOW:
        # 阻断未通过或未锁定的窗口结果。
        raise A4ContractError("A3没有锁定60日直接预测窗口")
    # 要求A3绑定同一A0合同。
    if report["input_sha256"]["a0_contract"] != sha256_file(contract_path):
        # 阻断合同版本漂移。
        raise A4ContractError("A3绑定的A0合同指纹不一致")
    # 逐项要求A3最终测试、备注和LSTM边界保持关闭。
    for key in ("final_test_run_directories_opened", "final_test_daily_files_read", "final_test_labels_read_or_assembled", "final_test_features_created", "final_test_opened", "cash_flow_review_notes_read_or_loaded", "lstm_trained", "unique_candidate_selected"):
        # 任一不是false均停止A4。
        if report.get(key) is not False:
            # 报告越界字段。
            raise A4ContractError(f"A3报告边界不通过：{key}")
    # 重算A3主要输出文件哈希。
    for name, expected_hash in report["output_sha256"].items():
        # 要求当前输出未被改写。
        if sha256_file(a3_directory / name) != expected_hash:
            # 阻断被篡改证据。
            raise A4ContractError(f"A3输出指纹不一致：{name}")
    # 流式读取A3折指标。
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        # 仅保留A3锁定60日直接结构的48行中的16行。
        direct_rows = [row for row in csv.DictReader(handle) if row["configuration_id"] == DIRECT_CONFIGURATION_ID]
    # 要求直接参考恰为四折乘四指标。
    if len(direct_rows) != 16:
        # 阻断少折、少期限或混入其他窗口。
        raise A4ContractError("A3继承直接结构不是完整4折×4指标")
    # 返回A3报告和继承的直接结构指标。
    return report, direct_rows


# 将一个真实余额变化转换为二元事件标签；金额为零时为0，否则为1。
def event_label(change: Decimal) -> float:
    # 返回可供XGBoost二元分类器训练的浮点标签。
    return float(change != ZERO_CHANGE)


# 将事件概率与条件非零金额预测相乘，得到无条件的余额变化预测。
def combine_hurdle_prediction(event_probability: float, nonzero_amount: float) -> Decimal:
    # 将两个浮点先转字符串再转Decimal，避免二进制尾差进入报告。
    probability = Decimal(str(float(event_probability)))
    # 将条件金额转成Decimal。
    amount = Decimal(str(float(nonzero_amount)))
    # 返回“发生概率×发生时金额”的无条件期望变化。
    return probability * amount


# 汇总直接结构与两阶段结构的四折指标，并按T+5锁定唯一结构。
def summarize_and_lock_structure(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    # 初始化指标到结构到折行的分组。
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    # 逐行分组。
    for row in rows:
        # 追加当前结构折指标。
        grouped[row["registered_measure"]][row["configuration_id"]].append(row)
    # 初始化汇总行列表。
    summary: list[dict[str, str]] = []
    # 初始化T+5结构分数。
    t5_scores: dict[str, Decimal] = {}
    # 按四个预登记指标处理。
    for measure_name, horizon in REGISTERED_MEASURES:
        # 按“直接结构优先、两阶段结构其次”的预登记复杂度顺序处理。
        for configuration_id in (DIRECT_CONFIGURATION_ID, HURDLE_CONFIGURATION_ID):
            # 读取当前结构的四折行。
            fold_rows = grouped[measure_name][configuration_id]
            # 要求恰好四折。
            if len(fold_rows) != 4:
                # 阻断缺失困难季度。
                raise A4ContractError(f"A4结构{configuration_id}指标{measure_name}不是完整四折")
            # 提取四折模型误差。
            model_values = sorted(Decimal(row["model_enterprise_macro_account_normalized_mae"]) for row in fold_rows)
            # 提取四折锁定基线误差。
            baseline_values = sorted(Decimal(row["locked_baseline_enterprise_macro_account_normalized_mae"]) for row in fold_rows)
            # 计算四折中位数。
            model_median = (model_values[1] + model_values[2]) / Decimal("2")
            # 计算基线中位数。
            baseline_median = (baseline_values[1] + baseline_values[2]) / Decimal("2")
            # 要求分母非零。
            if baseline_median == 0:
                # 阻断无定义Skill Score。
                raise A4ContractError("A4结构汇总的锁定基线误差为零")
            # 计算相对同期限基线的技能分数。
            skill = Decimal("1") - model_median / baseline_median
            # 写入当前结构汇总行。
            summary.append({
                "registered_measure": measure_name,
                "horizon_business_days": str(horizon),
                "configuration_id": configuration_id,
                "fold_count": "4",
                "median_model_enterprise_macro_account_normalized_mae": ratio(model_median),
                "median_locked_baseline_enterprise_macro_account_normalized_mae": ratio(baseline_median),
                "median_skill_score_vs_locked_baseline": ratio(skill),
                "minimum_fold_model_macro_mae": ratio(min(model_values)),
                "maximum_fold_model_macro_mae": ratio(max(model_values)),
                "selection_metric": "four_fold_median_enterprise_macro_account_normalized_mae",
                "a4_version": A4_VERSION,
            })
            # 保存T+5中位模型误差用于结构锁定。
            if measure_name == "t_plus_5_path":
                # 记录当前结构T+5分数。
                t5_scores[configuration_id] = model_median
    # 以最小T+5误差为先，精确同分时优先更简单的直接结构。
    selected = min((DIRECT_CONFIGURATION_ID, HURDLE_CONFIGURATION_ID), key=lambda item: (t5_scores[item], (DIRECT_CONFIGURATION_ID, HURDLE_CONFIGURATION_ID).index(item)))
    # 返回汇总行和锁定结构记录。
    return summary, {
        "selection_scope": "A1四折validation角色，仅208户开发池",
        "selection_metric": "four_fold_median_enterprise_macro_account_normalized_mae_on_t_plus_5_path",
        "tie_breaker_cn": "T+5四折中位误差完全相同则选择更简单的直接预测结构；不得看损失、LSTM或最终测试。",
        "selected_configuration_id": selected,
        "selected_t_plus_5_median_model_error": ratio(t5_scores[selected]),
    }


# 将严格字段顺序的CSV写入全新输出目录。
def write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    # 新建UTF-8 CSV。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 建立字段固定的写入器。
        writer = csv.DictWriter(handle, fieldnames=fields)
        # 写入表头。
        writer.writeheader()
        # 写入全部行。
        writer.writerows(rows)
