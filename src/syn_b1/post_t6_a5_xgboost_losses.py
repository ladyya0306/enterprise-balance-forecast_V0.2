"""POST-T6-A5：在已锁定的60日直接树结构上比较三种预登记损失。"""

# 导入CSV标准库；它读取A4折指标并写出A5严格字段顺序的结果表。
import csv
# 导入JSON标准库；它核验A4的报告和锁定文件。
import json
# 从collections导入defaultdict；它按指标与候选自动建立汇总分组。
from collections import defaultdict
# 从decimal导入Decimal；财务误差的中位数和技能分数须避免浮点尾差。
from decimal import Decimal
# 从pathlib导入Path；它以跨平台方式定位冻结证据文件。
from pathlib import Path
# 从typing导入Iterable；它标注CSV写入函数可接受任意可迭代行。
from typing import Iterable

# 导入YAML；它读取A0已经冻结的候选清单。
import yaml

# 从A3复用四个注册评价期限、数值格式化和SHA256函数；A5不得私自改口径。
from syn_b1.post_t6_a3_xgboost_windows import REGISTERED_MEASURES, ratio, sha256_file

# 固定本模块版本；任何选择、口径或血缘规则改变都必须产生新版本。
A5_VERSION = "syn_b1_post_t6_a5_xgboost_losses_1_0"
# 固定从A4继承的普通绝对误差直接树编号。
DIRECT_L1 = "xgb_direct_w60_l1"
# 固定A0预登记的伪Huber直接树编号。
HUBER = "xgb_direct_selected_window_huber"
# 固定A0预登记的账户尺度加权绝对误差直接树编号。
SCALED_L1 = "xgb_direct_selected_window_scaled_l1"
# 固定A4锁定的可见历史窗口长度。
WINDOW = 60
# 固定损失选择的复杂度同分顺序；越靠前越简单，不能看结果后改变。
LOSS_TIE_ORDER = (DIRECT_L1, HUBER, SCALED_L1)


# 声明A5专用异常；调用方可据此区分合同越界和普通运行错误。
class A5ContractError(ValueError):
    """当损失比较的血缘、候选或最终测试密封边界不满足时抛出。"""


# 核验A0最后两项XGBoost候选正是本阶段允许比较的两种损失。
def validate_a0_losses(path: Path) -> dict[str, object]:
    # 以UTF-8安全读取冻结合同。
    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    # 要求阶段编号和冻结状态没有漂移。
    if contract.get("stage_id") != "POST-T6-A0" or contract.get("status") != "frozen":
        # 阻断非冻结合同，防止事后扩展候选。
        raise A5ContractError("A5只能使用冻结的A0合同")
    # 读取A0的XGBoost配置清单。
    configurations = contract["limited_parameter_matrix"]["xgboost"]["configurations"]
    # 要求清单足够长，避免切片导致少项而没有报错。
    if len(configurations) < 6:
        # 阻断不完整候选矩阵。
        raise A5ContractError("A0没有完整登记A5的两种损失候选")
    # 写出不依赖结果的预登记编号和损失名称。
    expected = ((HUBER, "pseudo_huber"), (SCALED_L1, "account_scale_weighted_absolute_error"))
    # 逐项核验A0中的第五和第六候选。
    for configuration, expected_pair in zip(configurations[4:6], expected, strict=True):
        # 解包当前候选必须使用的编号与损失。
        identifier, loss_name = expected_pair
        # 同时限制直接结构、锁定窗口来源和损失，防止换入未登记模型。
        valid = (configuration["id"] == identifier and configuration["structure"] == "direct" and configuration["lookback_business_days"] == "selected_from_first_three_only" and configuration["loss"] == loss_name)
        # 任一字段漂移即拒绝执行。
        if not valid:
            # 报告候选与合同不一致。
            raise A5ContractError("A5损失候选与A0冻结清单不一致")
    # 返回已核验合同，供训练器读取共同固定参数。
    return contract


# 核验A4锁定的结构、输出指纹和密封边界，并读出继承的16行普通L1指标。
def validate_a4_lock(directory: Path, contract_path: Path) -> dict[str, object]:
    # 定位A4阶段报告。
    report_path = directory / "a4_xgboost_hurdle_report.json"
    # 定位A4结构锁定记录。
    lock_path = directory / "locked_xgboost_structure.json"
    # 定位A4的结构折指标表。
    metrics_path = directory / "xgboost_structure_fold_metrics.csv"
    # 三项都是A5继承普通L1结果的必要证据。
    if not report_path.is_file() or not lock_path.is_file() or not metrics_path.is_file():
        # 缺任一证据都不能进行后续比较。
        raise A5ContractError("缺少A4报告、结构锁定或折指标")
    # 读取A4报告内容。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 读取A4结构锁定内容。
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    # A4必须已经通过且把直接60日L1锁为唯一结构。
    structure_ok = (report.get("status") == "passed" and lock.get("selected_configuration_id") == DIRECT_L1 and lock.get("locked_lookback_business_days") == WINDOW)
    # 不允许在A4未锁定时让A5绕过结构选择。
    if not structure_ok:
        # 阻断错误结构血缘。
        raise A5ContractError("A4没有锁定60日直接L1结构")
    # A4绑定的A0合同哈希必须仍等于当前冻结合同哈希。
    if report["input_sha256"]["a0_contract"] != sha256_file(contract_path):
        # 阻断合同被改写后仍复用旧结果。
        raise A5ContractError("A4绑定的A0合同指纹不一致")
    # 重算A4所有主要输出哈希，防止继承被改写的普通L1指标。
    for filename, expected_hash in report["output_sha256"].items():
        # 当前文件哈希必须等于A4报告记录的哈希。
        if sha256_file(directory / filename) != expected_hash:
            # 指明被篡改或损坏的证据文件。
            raise A5ContractError(f"A4输出指纹不一致：{filename}")
    # 逐项要求最终测试、人工备注、LSTM及唯一候选选择保持关闭。
    boundary_keys = ("final_test_run_directories_opened", "final_test_daily_files_read", "final_test_labels_read_or_assembled", "final_test_features_created", "final_test_opened", "cash_flow_review_notes_read_or_loaded", "lstm_trained", "unique_candidate_selected")
    # 逐一验证边界布尔值。
    for key in boundary_keys:
        # 只接受明确的False，缺字段亦视为不通过。
        if report.get(key) is not False:
            # 阻断任何越界的A4来源。
            raise A5ContractError(f"A4边界不通过：{key}")
    # 打开严格CSV，仅筛出继承的普通L1结构行。
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        # 从32行结构比较中选出直接结构的四折乘四期限16行。
        direct_rows = [row for row in csv.DictReader(handle) if row["configuration_id"] == DIRECT_L1]
    # 要求继承行恰好完整。
    if len(direct_rows) != 16:
        # 阻断少折、少期限或混入错误模型的结果。
        raise A5ContractError("A4继承直接结构指标不是完整4折×4期限")
    # 返回报告和继承行，避免A5再次训练普通L1而改变可比性。
    return {"report": report, "direct_rows": direct_rows}


# 汇总三种损失，并且只按四折T+5主指标中位误差选择一种损失。
def summarize_and_lock(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    # 初始化“评价指标→候选→折行”的嵌套分组。
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    # 逐行放进相应分组。
    for row in rows:
        # 追加当前折指标行。
        grouped[row["registered_measure"]][row["configuration_id"]].append(row)
    # 初始化输出汇总表。
    summary: list[dict[str, str]] = []
    # 初始化只供选择T+5的三种损失误差字典。
    t_plus_5_scores: dict[str, Decimal] = {}
    # 按A0预登记的四个期限逐一生成汇总。
    for measure_name, horizon_days in REGISTERED_MEASURES:
        # 按固定的复杂度顺序处理三个候选，结果顺序可复现。
        for configuration_id in LOSS_TIE_ORDER:
            # 取出当前指标和候选的四折行。
            fold_rows = grouped[measure_name][configuration_id]
            # 只接受完整四折，不能以少折结果占优。
            if len(fold_rows) != 4:
                # 阻断不完整比较。
                raise A5ContractError("A5损失比较不是完整四折")
            # 读取并从小到大排序模型误差。
            model_values = sorted(Decimal(row["model_enterprise_macro_account_normalized_mae"]) for row in fold_rows)
            # 读取并从小到大排序相同折的冻结基线误差。
            baseline_values = sorted(Decimal(row["locked_baseline_enterprise_macro_account_normalized_mae"]) for row in fold_rows)
            # 四折为偶数，取中间两个误差的算术平均作为中位数。
            model_median = (model_values[1] + model_values[2]) / Decimal("2")
            # 同口径计算基线四折中位数。
            baseline_median = (baseline_values[1] + baseline_values[2]) / Decimal("2")
            # 技能分数需要非零基线分母。
            if baseline_median == 0:
                # 避免生成无定义的相对改善。
                raise A5ContractError("A5锁定基线中位误差为零，技能分数无定义")
            # 计算相对冻结基线的改善；正数代表模型误差较低。
            skill_score = Decimal("1") - model_median / baseline_median
            # 写入每个期限与候选的可读汇总行。
            summary.append({"registered_measure": measure_name, "horizon_business_days": str(horizon_days), "configuration_id": configuration_id, "fold_count": "4", "median_model_enterprise_macro_account_normalized_mae": ratio(model_median), "median_locked_baseline_enterprise_macro_account_normalized_mae": ratio(baseline_median), "median_skill_score_vs_locked_baseline": ratio(skill_score), "a5_version": A5_VERSION})
            # T+5是预登记的主选择指标，其它期限只能作为诊断。
            if measure_name == "t_plus_5_path":
                # 保存当前候选的T+5中位误差。
                t_plus_5_scores[configuration_id] = model_median
    # 先选T+5误差最小者；精确同分才使用固定复杂度顺序。
    selected = min(LOSS_TIE_ORDER, key=lambda identifier: (t_plus_5_scores[identifier], LOSS_TIE_ORDER.index(identifier)))
    # 返回所有汇总行及锁定记录；锁定记录不代表跨模型家族的唯一候选。
    return summary, {"selected_configuration_id": selected, "selected_t_plus_5_median_model_error": ratio(t_plus_5_scores[selected]), "tie_breaker_cn": "T+5四折中位误差同分时按直接L1、Huber、账户尺度L1顺序选择，不得看LSTM或最终测试。"}


# 将严格字段顺序的行写成一个新的UTF-8 CSV文件。
def write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    # 以换行控制方式新建文件，避免Windows产生空白行。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 固定列顺序，防止不同Python版本写出不同表头。
        writer = csv.DictWriter(handle, fieldnames=fields)
        # 先写表头。
        writer.writeheader()
        # 再写全部数据行。
        writer.writerows(rows)
