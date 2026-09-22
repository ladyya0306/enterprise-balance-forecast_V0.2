"""POST-T6-B2：在锁定28日窗口上比较LSTM隐藏维度、双头结构与损失。"""

# 导入CSV标准库；它读取B1种子折指标并写出B2固定列顺序表。
import csv
# 导入JSON标准库；它核验B1报告和窗口锁定记录。
import json
# 从collections导入defaultdict；它按配置、折和期限组织指标。
from collections import defaultdict
# 从decimal导入Decimal；配置选择保持精确十进制口径。
from decimal import Decimal
# 从pathlib导入Path；它定位冻结合同和前置证据。
from pathlib import Path

# 导入YAML；它读取A0预登记配置。
import yaml

# 复用A3的四个评价期限、比例格式和SHA256函数。
from syn_b1.post_t6_a3_xgboost_windows import REGISTERED_MEASURES, ratio, sha256_file

# 固定B2实现版本；训练或选择规则变化必须新建版本。
B2_VERSION = "syn_b1_post_t6_b2_lstm_variants_1_0"
# 固定B1锁定的28银行工作日窗口。
LOCKED_WINDOW = 28
# 固定B1继承的单层16维直接L1配置。
BASE_CONFIGURATION = "lstm_direct_w28_h16_l1"
# 固定A0登记的32维直接L1配置。
HIDDEN32_CONFIGURATION = "lstm_direct_selected_window_h32_l1"
# 固定A0登记的事件概率与金额双头配置。
TWO_HEAD_CONFIGURATION = "lstm_two_head_selected_window_h16"
# 固定A0登记的Huber配置。
HUBER_CONFIGURATION = "lstm_direct_selected_window_h16_huber"
# 固定A0登记的账户尺度加权L1配置。
SCALED_L1_CONFIGURATION = "lstm_direct_selected_window_h16_scaled_l1"
# 固定所有参与B2比较的配置顺序；精确同分时越靠前越优先。
CONFIGURATION_ORDER = (
    BASE_CONFIGURATION,
    HIDDEN32_CONFIGURATION,
    TWO_HEAD_CONFIGURATION,
    HUBER_CONFIGURATION,
    SCALED_L1_CONFIGURATION,
)
# 固定三个独立训练种子。
TRAINING_SEEDS = (2026083102, 2026083103, 2026083104)


# 声明B2专用异常；它使合同或密封错误不能混为模型负结果。
class B2ContractError(ValueError):
    """当B2候选、前置血缘或最终测试密封不满足时抛出。"""


# 核验A0后四个LSTM配置正是B2允许的新比较项。
def validate_a0_variants(contract_path: Path) -> dict[str, object]:
    # 读取并安全解析冻结YAML。
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    # 要求合同身份与冻结状态准确。
    if contract.get("stage_id") != "POST-T6-A0" or contract.get("status") != "frozen":
        # 阻断草稿或错误合同。
        raise B2ContractError("B2只能使用冻结的POST-T6-A0合同")
    # 读取全部七个LSTM配置。
    configurations = contract["limited_parameter_matrix"]["small_lstm"]["configurations"]
    # 要求完整七项，防止切片漏项。
    if len(configurations) != 7:
        # 阻断配置矩阵漂移。
        raise B2ContractError("A0 LSTM配置数量不是完整七项")
    # 写出后四项必须逐项匹配的冻结字段。
    expected = (
        (HIDDEN32_CONFIGURATION, 32, "direct_path", "absolute_error"),
        (TWO_HEAD_CONFIGURATION, 16, "event_probability_plus_amount_path", "0.25*binary_cross_entropy+1.00*account_scaled_absolute_error"),
        (HUBER_CONFIGURATION, 16, "direct_path", "huber"),
        (SCALED_L1_CONFIGURATION, 16, "direct_path", "account_scale_weighted_absolute_error"),
    )
    # 逐项核验A0第四至第七项配置。
    for configuration, required in zip(configurations[3:7], expected, strict=True):
        # 解包预登记编号、隐藏维度、结构与损失。
        identifier, hidden_size, structure, loss_name = required
        # 同时要求窗口来源只能是B1锁定结果。
        valid = (
            configuration["id"] == identifier
            and configuration["lookback_business_days"] == "selected_from_first_three_only"
            and configuration["hidden_size"] == hidden_size
            and configuration["structure"] == structure
            and configuration["loss"] == loss_name
        )
        # 任一字段漂移都拒绝继续。
        if not valid:
            # 指明配置不符合A0。
            raise B2ContractError("B2配置与A0预登记后四项不一致")
    # 要求种子集合继续与B1一致。
    if tuple(contract["limited_parameter_matrix"]["small_lstm"]["training_seeds"]) != TRAINING_SEEDS:
        # 阻断挑选单种子或换种子。
        raise B2ContractError("B2训练种子与A0不一致")
    # 返回已核验合同。
    return contract


# 核验B1锁定28日窗口、输出指纹和最终测试密封，并读取继承的基准配置指标。
def validate_b1_and_read_base_rows(b1_directory: Path, contract_path: Path) -> dict[str, object]:
    # 定位B1报告。
    report_path = b1_directory / "b1_lstm_window_report.json"
    # 定位B1窗口锁定文件。
    lock_path = b1_directory / "locked_lstm_window.json"
    # 定位B1种子级折指标。
    metric_path = b1_directory / "lstm_window_seed_fold_metrics.csv"
    # 三份证据缺一不可。
    if not report_path.is_file() or not lock_path.is_file() or not metric_path.is_file():
        # 阻断不完整前置结果。
        raise B2ContractError("缺少B1报告、窗口锁定或种子折指标")
    # 读取B1报告。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 读取B1锁定记录。
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    # 要求B1通过且锁定预期28日基准配置。
    if report.get("status") != "passed" or lock.get("selected_configuration_id") != BASE_CONFIGURATION or lock.get("selected_lookback_business_days") != LOCKED_WINDOW:
        # 阻断窗口血缘漂移。
        raise B2ContractError("B1没有锁定28日LSTM基准配置")
    # 要求B1绑定当前A0合同。
    if report["input_sha256"]["a0_contract"] != sha256_file(contract_path):
        # 阻断合同被改写。
        raise B2ContractError("B1绑定的A0合同指纹不一致")
    # 重算B1所有主要输出哈希。
    for filename, expected_hash in report["output_sha256"].items():
        # 当前文件必须与B1报告记录相同。
        if sha256_file(b1_directory / filename) != expected_hash:
            # 阻断被改写的B1结果。
            raise B2ContractError(f"B1输出指纹不一致：{filename}")
    # 列出必须保持关闭的边界。
    boundary_keys = (
        "final_test_run_directories_opened",
        "final_test_daily_files_read",
        "final_test_labels_read_or_assembled",
        "final_test_features_created",
        "final_test_opened",
        "cash_flow_review_notes_read_or_loaded",
        "unique_candidate_selected",
    )
    # 逐项核验B1边界。
    for key in boundary_keys:
        # 缺字段或非False都不接受。
        if report.get(key) is not False:
            # 阻断越界来源。
            raise B2ContractError(f"B1边界不通过：{key}")
    # 读取B1种子级折指标。
    with metric_path.open("r", encoding="utf-8", newline="") as handle:
        # 只继承28日基准配置的三种子×四折×四期限48行。
        base_rows = [row for row in csv.DictReader(handle) if row["configuration_id"] == BASE_CONFIGURATION]
    # 要求基准结果完整。
    if len(base_rows) != 48:
        # 阻断少种子、少折或少期限。
        raise B2ContractError("B1继承基准配置指标不是完整48行")
    # 返回报告、锁定和基准行。
    return {"report": report, "lock": lock, "base_rows": base_rows}


# 将五配置的种子级折指标合成为每折三种子中位数。
def seed_median_fold_rows(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 初始化配置、折和期限分组。
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    # 逐行归组。
    for row in seed_rows:
        # 种子不进入键，因而每组必须有三行。
        grouped[(row["configuration_id"], row["fold_id"], row["registered_measure"])].append(row)
    # 初始化80条种子中位折指标。
    rows: list[dict[str, str]] = []
    # 按稳定字典键顺序处理。
    for configuration_id, fold_id, measure_name in sorted(grouped):
        # 读取三种子成员。
        members = grouped[(configuration_id, fold_id, measure_name)]
        # 要求种子数量和集合准确。
        if len(members) != 3 or {int(row["training_seed"]) for row in members} != set(TRAINING_SEEDS):
            # 阻断缺失、重复或替换种子。
            raise B2ContractError("B2折指标没有完整三种子")
        # 排序三种子模型误差。
        model_values = sorted(Decimal(row["model_enterprise_macro_account_normalized_mae"]) for row in members)
        # 排序三种子基线误差。
        baseline_values = sorted(Decimal(row["locked_baseline_enterprise_macro_account_normalized_mae"]) for row in members)
        # 三项中位数就是中间项。
        model_median = model_values[1]
        # 取得基线中位数。
        baseline_median = baseline_values[1]
        # 拒绝零基线分母。
        if baseline_median == 0:
            # 阻断无定义技能分数。
            raise B2ContractError("B2折指标基线误差为零")
        # 写出种子中位折指标。
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
            "b2_version": B2_VERSION,
        })
    # 五配置×四折×四期限必须恰为80行。
    if len(rows) != len(CONFIGURATION_ORDER) * 4 * len(REGISTERED_MEASURES):
        # 阻断缺配置或缺折。
        raise B2ContractError("B2种子中位折指标数量不完整")
    # 返回折指标。
    return rows


# 汇总五配置四折指标，并只按T+5选择一个LSTM内部配置。
def summarize_and_lock_variants(fold_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    # 初始化期限到配置的折行分组。
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    # 逐行归组。
    for row in fold_rows:
        # 保存当前折行。
        grouped[row["registered_measure"]][row["configuration_id"]].append(row)
    # 初始化20条汇总结果。
    summary: list[dict[str, str]] = []
    # 初始化T+5配置误差。
    t_plus_5_scores: dict[str, Decimal] = {}
    # 按四个注册期限处理。
    for measure_name, horizon_days in REGISTERED_MEASURES:
        # 按固定同分顺序处理五配置。
        for configuration_id in CONFIGURATION_ORDER:
            # 取得当前配置完整四折。
            members = grouped[measure_name][configuration_id]
            # 拒绝少折结果。
            if len(members) != 4:
                # 阻断不公平比较。
                raise B2ContractError("B2配置汇总不是完整四折")
            # 排序四折模型误差。
            model_values = sorted(Decimal(row["median_model_enterprise_macro_account_normalized_mae"]) for row in members)
            # 排序四折基线误差。
            baseline_values = sorted(Decimal(row["median_locked_baseline_enterprise_macro_account_normalized_mae"]) for row in members)
            # 四折中位数取中间两项平均。
            model_median = (model_values[1] + model_values[2]) / Decimal("2")
            # 同口径计算基线中位数。
            baseline_median = (baseline_values[1] + baseline_values[2]) / Decimal("2")
            # 拒绝零基线分母。
            if baseline_median == 0:
                # 阻断无定义技能分数。
                raise B2ContractError("B2汇总基线误差为零")
            # 计算相对强基线的技能分数。
            skill_score = Decimal("1") - model_median / baseline_median
            # 写出配置汇总行。
            summary.append({
                "registered_measure": measure_name,
                "horizon_business_days": str(horizon_days),
                "configuration_id": configuration_id,
                "lookback_business_days": str(LOCKED_WINDOW),
                "fold_count": "4",
                "seed_count_per_fold": "3",
                "median_model_enterprise_macro_account_normalized_mae": ratio(model_median),
                "median_locked_baseline_enterprise_macro_account_normalized_mae": ratio(baseline_median),
                "median_skill_score_vs_locked_baseline": ratio(skill_score),
                "b2_version": B2_VERSION,
            })
            # 只有T+5进入配置选择。
            if measure_name == "t_plus_5_path":
                # 保存T+5误差。
                t_plus_5_scores[configuration_id] = model_median
    # 先选T+5误差最低者，精确同分才按固定顺序选更简单者。
    selected = min(CONFIGURATION_ORDER, key=lambda identifier: (t_plus_5_scores[identifier], CONFIGURATION_ORDER.index(identifier)))
    # 返回诊断汇总和内部锁定记录。
    return summary, {
        "selected_configuration_id": selected,
        "selected_lookback_business_days": LOCKED_WINDOW,
        "selected_t_plus_5_median_model_error": ratio(t_plus_5_scores[selected]),
        "tie_breaker_cn": "T+5四折种子中位误差完全相同，按28日16维直接L1、32维直接L1、双头、Huber、账户尺度L1顺序选择；不得看XGBoost或最终测试。",
    }
