"""POST-T6-C2C：为B2锁定LSTM点预测建立名义80%的分割保形区间。"""

# 导入CSV标准库；它读取共同配对和锁定基线文件并写出候选区间。
import csv
# 从collections导入defaultdict；它按种子、滚动折和期限累计残差与指标。
from collections import defaultdict
# 从decimal导入Decimal；金额、半径、宽度和评分保持精确十进制。
from decimal import Decimal
# 从pathlib导入Path；它让函数接收明确的本地文件路径。
from pathlib import Path

# 复用C2A的共同期限、种子、折、区间评分和企业等权汇总公式。
from syn_b1.post_t6_c2a_average_interval_foundation import AVERAGE_HORIZONS, FOLDS, SELECTED_CONFIGURATION, TRAINING_SEEDS, enterprise_macro_interval_summary, interval_measurement
# 复用C2B的有限样本保形排名、半径和稳定中位数，避免复制统计公式。
from syn_b1.post_t6_c2b_balance_hold_interval import C2BBaselineError, conformal_radius, decimal_median
# 复用统一比例格式函数。
from syn_b1.post_t6_a3_xgboost_windows import ratio

# 固定C2C实现版本；改变区间中心或校准公式必须新建版本。
C2C_VERSION = "syn_b1_post_t6_c2c_locked_lstm_interval_1_0"
# 固定C1登记的简单候选方法编号。
LSTM_METHOD_ID = "locked_lstm_scaled_split_conformal"
# 固定C2B锁定基线方法编号。
BASELINE_METHOD_ID = "balance_hold_scaled_split_conformal"


# 声明C2C专用异常，使共同记录错配与模型效果问题区分开。
class C2CLockedLstmError(ValueError):
    """当锁定LSTM残差、基线配对或评价矩阵不满足C1/C2C条件时抛出。"""


# 组成共同配对与基线区间使用的唯一连接键。
def pair_key(row: dict[str, str]) -> tuple[str, str, int, str, int]:
    # 返回折、角色、种子、样本键和期限组成的元组。
    return (row["fold_id"], row["fold_role"], int(row["training_seed"]), row["sample_key"], int(row["average_horizon_business_days"]))


# 读取C2A校准角色，并只根据锁定LSTM点预测计算归一化绝对残差。
def read_lstm_calibration_residuals(pair_path: Path) -> dict[tuple[int, str, int], list[dict[str, str | Decimal]]]:
    # 初始化组合到校准残差记录的映射。
    grouped: dict[tuple[int, str, int], list[dict[str, str | Decimal]]] = defaultdict(list)
    # 以UTF-8和CSV标准换行读取共同配对表。
    with pair_path.open("r", encoding="utf-8", newline="") as handle:
        # 创建按表头取值的读取器。
        reader = csv.DictReader(handle)
        # 逐条读取共同配对记录。
        for row in reader:
            # 仅calibration角色可以决定LSTM区间半径。
            if row["fold_role"] != "calibration":
                # 跳过validation记录，避免用考试答案校准。
                continue
            # 读取种子。
            seed = int(row["training_seed"])
            # 读取滚动折。
            fold_id = row["fold_id"]
            # 读取日均期限。
            horizon = int(row["average_horizon_business_days"])
            # 拒绝未登记组合。
            if seed not in TRAINING_SEEDS or fold_id not in FOLDS or horizon not in AVERAGE_HORIZONS:
                # 阻断输入漂移。
                raise C2CLockedLstmError("C2A配对表含未登记种子、折或期限")
            # 读取真实日均余额。
            actual = Decimal(row["actual_average_balance_cny"])
            # 读取B2锁定LSTM日均点预测；本卡唯一候选中心。
            point = Decimal(row["locked_lstm_point_average_balance_cny"])
            # 读取账户尺度。
            scale = Decimal(row["balance_scale_cny"])
            # 账户尺度必须为正。
            if scale <= 0:
                # 阻断无定义残差。
                raise C2CLockedLstmError("C2A配对表的账户尺度必须大于0")
            # 计算账户尺度归一化绝对残差。
            normalized_residual = abs(actual - point) / scale
            # 保存只含候选校准所需字段的记录。
            grouped[(seed, fold_id, horizon)].append({
                # 保存样本键。
                "sample_key": row["sample_key"],
                # 保存企业键。
                "enterprise_id": row["enterprise_id"],
                # 保存账户尺度。
                "balance_scale_cny": scale,
                # 保存真实日均。
                "actual_average_balance_cny": actual,
                # 保存锁定LSTM点预测。
                "locked_lstm_point_average_balance_cny": point,
                # 保存归一化残差。
                "normalized_absolute_residual": normalized_residual,
            })
    # 遍历完整实验矩阵。
    for seed in TRAINING_SEEDS:
        # 遍历四个折。
        for fold_id in FOLDS:
            # 遍历两种期限。
            for horizon in AVERAGE_HORIZONS:
                # 每个组合必须有校准记录。
                if not grouped[(seed, fold_id, horizon)]:
                    # 阻断缺校准组合。
                    raise C2CLockedLstmError("锁定LSTM校准残差缺少完整种子、折或期限")
    # 返回完整校准残差。
    return grouped


# 计算24个锁定LSTM保形半径，并写出可审计的分位数行。
def build_lstm_quantile_rows(grouped: dict[tuple[int, str, int], list[dict[str, str | Decimal]]]) -> tuple[dict[tuple[int, str, int], Decimal], list[dict[str, str]]]:
    # 初始化组合到半径的映射。
    radii: dict[tuple[int, str, int], Decimal] = {}
    # 初始化24条审计行。
    rows: list[dict[str, str]] = []
    # 按固定种子顺序处理。
    for seed in TRAINING_SEEDS:
        # 按固定折顺序处理。
        for fold_id in FOLDS:
            # 按固定期限顺序处理。
            for horizon in AVERAGE_HORIZONS:
                # 取得当前校准记录。
                records = grouped[(seed, fold_id, horizon)]
                # 提取全部归一化绝对残差。
                residuals = [Decimal(str(record["normalized_absolute_residual"])) for record in records]
                # 复用C2B的有限样本保形公式。
                rank, radius = conformal_radius(residuals)
                # 保存当前半径。
                radii[(seed, fold_id, horizon)] = radius
                # 写出审计行。
                rows.append({
                    # 保存种子。
                    "training_seed": str(seed),
                    # 保存折号。
                    "fold_id": fold_id,
                    # 保存期限。
                    "average_horizon_business_days": str(horizon),
                    # 保存校准记录数。
                    "calibration_record_count": str(len(records)),
                    # 保存不同企业数。
                    "calibration_enterprise_count": str(len({str(record['enterprise_id']) for record in records})),
                    # 保存有限样本排名。
                    "finite_sample_conformal_rank": str(rank),
                    # 保存归一化半径。
                    "normalized_conformal_radius": ratio(radius),
                    # 保存候选方法编号。
                    "interval_method_id": LSTM_METHOD_ID,
                    # 保存版本。
                    "c2c_version": C2C_VERSION,
                })
    # 要求恰有24行。
    if len(rows) != 24:
        # 阻断缺组合。
        raise C2CLockedLstmError("锁定LSTM分位数审计行不是完整24行")
    # 返回半径和审计行。
    return radii, rows


# 读取C2B区间表并建立每条共同配对记录的锁定基线区间映射。
def read_baseline_intervals(baseline_pair_path: Path) -> dict[tuple[str, str, int, str, int], dict[str, Decimal]]:
    # 初始化连接键到基线字段的映射。
    intervals: dict[tuple[str, str, int, str, int], dict[str, Decimal]] = {}
    # 以UTF-8和CSV标准换行读取C2B区间表。
    with baseline_pair_path.open("r", encoding="utf-8", newline="") as handle:
        # 创建按表头读取器。
        reader = csv.DictReader(handle)
        # 逐条读取基线区间。
        for row in reader:
            # 生成唯一连接键。
            key = pair_key(row)
            # 拒绝重复键。
            if key in intervals:
                # 阻断重复计权。
                raise C2CLockedLstmError("C2B基线区间表存在重复共同配对键")
            # 要求基线方法编号准确。
            if row["interval_method_id"] != BASELINE_METHOD_ID:
                # 阻断混入其它方法。
                raise C2CLockedLstmError("C2B基线区间表的方法编号不一致")
            # 保存后续公平比较所需字段。
            intervals[key] = {
                # 保存真实日均用于同一行核对。
                "actual": Decimal(row["actual_average_balance_cny"]),
                # 保存账户尺度用于同一行核对。
                "scale": Decimal(row["balance_scale_cny"]),
                # 保存基线下限。
                "lower": Decimal(row["p10_lower_bound_cny"]),
                # 保存基线点预测。
                "point": Decimal(row["p50_point_estimate_cny"]),
                # 保存基线上限。
                "upper": Decimal(row["p90_upper_bound_cny"]),
            }
    # 拒绝空基线表。
    if not intervals:
        # 抛出明确异常。
        raise C2CLockedLstmError("C2B基线区间表为空")
    # 返回完整基线映射。
    return intervals


# 用锁定LSTM点预测和同折半径生成一个候选区间，并配对同一行C2B基线。
def build_lstm_interval_row(row: dict[str, str], radius: Decimal, baseline: dict[str, Decimal]) -> tuple[dict[str, str], dict[str, Decimal | str]]:
    # 读取真实日均。
    actual = Decimal(row["actual_average_balance_cny"])
    # 读取账户尺度。
    scale = Decimal(row["balance_scale_cny"])
    # 读取锁定LSTM点预测中心。
    point = Decimal(row["locked_lstm_point_average_balance_cny"])
    # 确保C2B基线与当前共同配对记录是同一个真实值。
    if baseline["actual"] != actual:
        # 阻断错配行。
        raise C2CLockedLstmError("C2A与C2B同一键的真实日均余额不一致")
    # 确保C2B基线与当前共同配对记录使用同一个尺度。
    if baseline["scale"] != scale:
        # 阻断错配尺度。
        raise C2CLockedLstmError("C2A与C2B同一键的账户尺度不一致")
    # 将归一化半径还原为当前账户的人民币半径。
    amount_radius = radius * scale
    # 计算候选P10下限。
    lower = point - amount_radius
    # 计算候选P90上限。
    upper = point + amount_radius
    # 计算候选覆盖、宽度和评分。
    candidate_covered, candidate_width, candidate_score = interval_measurement(actual, lower, upper, scale)
    # 计算同一记录C2B基线覆盖、宽度和评分。
    baseline_covered, baseline_width, baseline_score = interval_measurement(actual, baseline["lower"], baseline["upper"], scale)
    # 构造可写入CSV的全字段行。
    output_row = {
        # 保存折号。
        "fold_id": row["fold_id"],
        # 保存角色。
        "fold_role": row["fold_role"],
        # 保存种子。
        "training_seed": row["training_seed"],
        # 保存样本键。
        "sample_key": row["sample_key"],
        # 保存企业键。
        "enterprise_id": row["enterprise_id"],
        # 保存期限。
        "average_horizon_business_days": row["average_horizon_business_days"],
        # 保存账户尺度。
        "balance_scale_cny": f"{scale:.2f}",
        # 保存真实日均。
        "actual_average_balance_cny": f"{actual:.2f}",
        # 保存锁定LSTM点预测。
        "locked_lstm_point_average_balance_cny": f"{point:.2f}",
        # 保存候选半径。
        "normalized_conformal_radius": ratio(radius),
        # 保存候选P10。
        "candidate_p10_lower_bound_cny": f"{lower:.2f}",
        # 保存候选P50。
        "candidate_p50_point_estimate_cny": f"{point:.2f}",
        # 保存候选P90。
        "candidate_p90_upper_bound_cny": f"{upper:.2f}",
        # 保存候选覆盖。
        "candidate_covered_by_interval": str(candidate_covered).lower(),
        # 保存候选宽度。
        "candidate_normalized_interval_width": ratio(candidate_width),
        # 保存候选评分。
        "candidate_normalized_interval_score": ratio(candidate_score),
        # 保存C2B基线P10。
        "baseline_p10_lower_bound_cny": f"{baseline['lower']:.2f}",
        # 保存C2B基线P50。
        "baseline_p50_point_estimate_cny": f"{baseline['point']:.2f}",
        # 保存C2B基线P90。
        "baseline_p90_upper_bound_cny": f"{baseline['upper']:.2f}",
        # 保存基线覆盖。
        "baseline_covered_by_interval": str(baseline_covered).lower(),
        # 保存基线宽度。
        "baseline_normalized_interval_width": ratio(baseline_width),
        # 保存基线评分。
        "baseline_normalized_interval_score": ratio(baseline_score),
        # 保存候选方法编号。
        "candidate_interval_method_id": LSTM_METHOD_ID,
        # 保存基线方法编号。
        "baseline_interval_method_id": BASELINE_METHOD_ID,
        # 保存版本。
        "c2c_version": C2C_VERSION,
    }
    # 构造企业等权汇总函数所需的精确十进制记录。
    metric_record = {
        # 保存企业键。
        "enterprise_id": row["enterprise_id"],
        # 保存真实日均。
        "actual_average_balance_cny": actual,
        # 保存账户尺度。
        "balance_scale_cny": scale,
        # 保存候选下限。
        "candidate_lower_cny": lower,
        # 保存候选上限。
        "candidate_upper_cny": upper,
        # 保存基线下限。
        "baseline_lower_cny": baseline["lower"],
        # 保存基线上限。
        "baseline_upper_cny": baseline["upper"],
    }
    # 返回CSV行和汇总记录。
    return output_row, metric_record


# 将C2A共同配对数据应用LSTM半径，并要求逐条匹配C2B基线。
def write_lstm_interval_pairs(pair_path: Path, output_path: Path, radii: dict[tuple[int, str, int], Decimal], baseline_intervals: dict[tuple[str, str, int, str, int], dict[str, Decimal]]) -> dict[tuple[int, str, int], list[dict[str, Decimal | str]]]:
    # 固定区间配对CSV列顺序。
    fields = ("fold_id", "fold_role", "training_seed", "sample_key", "enterprise_id", "average_horizon_business_days", "balance_scale_cny", "actual_average_balance_cny", "locked_lstm_point_average_balance_cny", "normalized_conformal_radius", "candidate_p10_lower_bound_cny", "candidate_p50_point_estimate_cny", "candidate_p90_upper_bound_cny", "candidate_covered_by_interval", "candidate_normalized_interval_width", "candidate_normalized_interval_score", "baseline_p10_lower_bound_cny", "baseline_p50_point_estimate_cny", "baseline_p90_upper_bound_cny", "baseline_covered_by_interval", "baseline_normalized_interval_width", "baseline_normalized_interval_score", "candidate_interval_method_id", "baseline_interval_method_id", "c2c_version")
    # 初始化仅validation角色的评价桶。
    validation_records: dict[tuple[int, str, int], list[dict[str, Decimal | str]]] = defaultdict(list)
    # 记录共同配对表实际使用的基线键。
    used_baseline_keys: set[tuple[str, str, int, str, int]] = set()
    # 以UTF-8和CSV标准换行写候选区间表。
    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        # 创建固定字段写入器。
        writer = csv.DictWriter(output_handle, fieldnames=fields)
        # 写入表头。
        writer.writeheader()
        # 以UTF-8和CSV标准换行读取共同配对表。
        with pair_path.open("r", encoding="utf-8", newline="") as pair_handle:
            # 创建按表头取值的读取器。
            reader = csv.DictReader(pair_handle)
            # 逐条建立候选区间。
            for row in reader:
                # 读取种子。
                seed = int(row["training_seed"])
                # 读取折号。
                fold_id = row["fold_id"]
                # 读取期限。
                horizon = int(row["average_horizon_business_days"])
                # 生成共同基线键。
                key = pair_key(row)
                # C2B必须为当前行提供同一基线区间。
                if key not in baseline_intervals:
                    # 阻断任何共同记录缺失。
                    raise C2CLockedLstmError("C2B基线区间缺少C2A共同配对记录")
                # 读取当前种子、折、期限的锁定LSTM半径。
                radius = radii[(seed, fold_id, horizon)]
                # 形成CSV行与精确评价记录。
                output_row, metric_record = build_lstm_interval_row(row, radius, baseline_intervals[key])
                # 写出当前行。
                writer.writerow(output_row)
                # 标记该基线键已经使用。
                used_baseline_keys.add(key)
                # 只有validation角色进入候选对基线效果比较。
                if row["fold_role"] == "validation":
                    # 追加当前评价记录。
                    validation_records[(seed, fold_id, horizon)].append(metric_record)
    # C2A和C2B的共同配对键集合必须完全相等。
    if used_baseline_keys != set(baseline_intervals):
        # 阻断多行、少行或角色不一致。
        raise C2CLockedLstmError("C2A与C2B共同配对键集合不完全一致")
    # 核验每个验证组合均非空。
    for seed in TRAINING_SEEDS:
        # 遍历四折。
        for fold_id in FOLDS:
            # 遍历两期限。
            for horizon in AVERAGE_HORIZONS:
                # 当前组合为空即拒绝。
                if not validation_records[(seed, fold_id, horizon)]:
                    # 抛出明确异常。
                    raise C2CLockedLstmError("锁定LSTM区间缺少完整validation种子、折或期限")
    # 返回validation评价桶。
    return validation_records


# 将24个种子折候选对基线比较汇总成指标行。
def build_seed_fold_metrics(validation_records: dict[tuple[int, str, int], list[dict[str, Decimal | str]]], quantile_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 建立量化审计行索引。
    quantile_by_key = {(int(row["training_seed"]), row["fold_id"], int(row["average_horizon_business_days"])): row for row in quantile_rows}
    # 初始化24条指标行。
    rows: list[dict[str, str]] = []
    # 遍历冻结种子。
    for seed in TRAINING_SEEDS:
        # 遍历四折。
        for fold_id in FOLDS:
            # 遍历期限。
            for horizon in AVERAGE_HORIZONS:
                # 取得验证记录。
                records = validation_records[(seed, fold_id, horizon)]
                # 用C2A公共公式计算候选相对C2B基线的企业等权指标。
                summary = enterprise_macro_interval_summary(records)
                # 读取校准审计行。
                quantile = quantile_by_key[(seed, fold_id, horizon)]
                # 写出当前种子折指标。
                rows.append({
                    # 保存种子。
                    "training_seed": str(seed),
                    # 保存折号。
                    "fold_id": fold_id,
                    # 保存期限。
                    "average_horizon_business_days": str(horizon),
                    # 保存验证企业数。
                    "validation_enterprise_count": str(summary["enterprise_count"]),
                    # 保存预测起点数。
                    "validation_forecast_origin_count": str(len(records)),
                    # 保存校准记录数。
                    "calibration_record_count": quantile["calibration_record_count"],
                    # 保存保形排名。
                    "finite_sample_conformal_rank": quantile["finite_sample_conformal_rank"],
                    # 保存候选半径。
                    "normalized_conformal_radius": quantile["normalized_conformal_radius"],
                    # 保存候选覆盖率。
                    "candidate_enterprise_macro_empirical_coverage": ratio(Decimal(str(summary["candidate_empirical_coverage"]))),
                    # 保存基线覆盖率。
                    "baseline_enterprise_macro_empirical_coverage": ratio(Decimal(str(summary["baseline_empirical_coverage"]))),
                    # 保存候选宽度。
                    "candidate_enterprise_macro_normalized_interval_width": ratio(Decimal(str(summary["candidate_normalized_width"]))),
                    # 保存基线宽度。
                    "baseline_enterprise_macro_normalized_interval_width": ratio(Decimal(str(summary["baseline_normalized_width"]))),
                    # 保存候选评分。
                    "candidate_enterprise_macro_normalized_interval_score": ratio(Decimal(str(summary["candidate_normalized_interval_score"]))),
                    # 保存基线评分。
                    "baseline_enterprise_macro_normalized_interval_score": ratio(Decimal(str(summary["baseline_normalized_interval_score"]))),
                    # 保存候选相对基线的区间Skill。
                    "interval_skill_score_vs_baseline": ratio(Decimal(str(summary["interval_skill_score"]))),
                    # 保存评分改善企业比例。
                    "improved_enterprise_rate": ratio(Decimal(str(summary["improved_enterprise_rate"]))),
                    # 保存候选方法编号。
                    "candidate_interval_method_id": LSTM_METHOD_ID,
                    # 保存基线方法编号。
                    "baseline_interval_method_id": BASELINE_METHOD_ID,
                    # 保存版本。
                    "c2c_version": C2C_VERSION,
                })
    # 要求完整24行。
    if len(rows) != 24:
        # 阻断缺组合。
        raise C2CLockedLstmError("锁定LSTM区间种子折指标不是完整24行")
    # 返回指标行。
    return rows


# 将三种子中位成每个折的候选对基线指标。
def build_fold_metrics(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 初始化8条折级行。
    rows: list[dict[str, str]] = []
    # 遍历四折。
    for fold_id in FOLDS:
        # 遍历两期限。
        for horizon in AVERAGE_HORIZONS:
            # 筛选当前折期限的三种子行。
            members = [row for row in seed_rows if row["fold_id"] == fold_id and int(row["average_horizon_business_days"]) == horizon]
            # 要求完整三种子。
            if len(members) != 3 or {int(row["training_seed"]) for row in members} != set(TRAINING_SEEDS):
                # 阻断缺种子。
                raise C2CLockedLstmError("锁定LSTM区间折指标缺少完整三种子")
            # 写出种子中位折行。
            rows.append({
                # 保存折号。
                "fold_id": fold_id,
                # 保存期限。
                "average_horizon_business_days": str(horizon),
                # 保存种子数。
                "seed_count": "3",
                # 保存候选覆盖率中位数。
                "median_candidate_enterprise_macro_empirical_coverage": ratio(decimal_median([Decimal(row["candidate_enterprise_macro_empirical_coverage"]) for row in members])),
                # 保存基线覆盖率中位数。
                "median_baseline_enterprise_macro_empirical_coverage": ratio(decimal_median([Decimal(row["baseline_enterprise_macro_empirical_coverage"]) for row in members])),
                # 保存候选宽度中位数。
                "median_candidate_enterprise_macro_normalized_interval_width": ratio(decimal_median([Decimal(row["candidate_enterprise_macro_normalized_interval_width"]) for row in members])),
                # 保存基线宽度中位数。
                "median_baseline_enterprise_macro_normalized_interval_width": ratio(decimal_median([Decimal(row["baseline_enterprise_macro_normalized_interval_width"]) for row in members])),
                # 保存候选评分中位数。
                "median_candidate_enterprise_macro_normalized_interval_score": ratio(decimal_median([Decimal(row["candidate_enterprise_macro_normalized_interval_score"]) for row in members])),
                # 保存基线评分中位数。
                "median_baseline_enterprise_macro_normalized_interval_score": ratio(decimal_median([Decimal(row["baseline_enterprise_macro_normalized_interval_score"]) for row in members])),
                # 保存区间Skill。
                "interval_skill_score_vs_baseline": ratio(decimal_median([Decimal(row["interval_skill_score_vs_baseline"]) for row in members])),
                # 保存改善企业比例中位数。
                "median_improved_enterprise_rate": ratio(decimal_median([Decimal(row["improved_enterprise_rate"]) for row in members])),
                # 保存候选方法。
                "candidate_interval_method_id": LSTM_METHOD_ID,
                # 保存基线方法。
                "baseline_interval_method_id": BASELINE_METHOD_ID,
                # 保存版本。
                "c2c_version": C2C_VERSION,
            })
    # 返回8条折行。
    return rows


# 将四折中位数汇总成10日和30日两条候选摘要。
def build_summary_metrics(fold_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 初始化两条摘要。
    rows: list[dict[str, str]] = []
    # 遍历两期限。
    for horizon in AVERAGE_HORIZONS:
        # 取得当前期限四折行。
        members = [row for row in fold_rows if int(row["average_horizon_business_days"]) == horizon]
        # 要求完整四折。
        if len(members) != 4:
            # 阻断缺折。
            raise C2CLockedLstmError("锁定LSTM区间摘要缺少完整四折")
        # 写出总体摘要。
        rows.append({
            # 保存期限。
            "average_horizon_business_days": str(horizon),
            # 保存折数。
            "fold_count": "4",
            # 保存每折种子数。
            "seed_count_per_fold": "3",
            # 保存候选覆盖率。
            "median_candidate_enterprise_macro_empirical_coverage": ratio(decimal_median([Decimal(row["median_candidate_enterprise_macro_empirical_coverage"]) for row in members])),
            # 保存基线覆盖率。
            "median_baseline_enterprise_macro_empirical_coverage": ratio(decimal_median([Decimal(row["median_baseline_enterprise_macro_empirical_coverage"]) for row in members])),
            # 保存候选宽度。
            "median_candidate_enterprise_macro_normalized_interval_width": ratio(decimal_median([Decimal(row["median_candidate_enterprise_macro_normalized_interval_width"]) for row in members])),
            # 保存基线宽度。
            "median_baseline_enterprise_macro_normalized_interval_width": ratio(decimal_median([Decimal(row["median_baseline_enterprise_macro_normalized_interval_width"]) for row in members])),
            # 保存候选评分。
            "median_candidate_enterprise_macro_normalized_interval_score": ratio(decimal_median([Decimal(row["median_candidate_enterprise_macro_normalized_interval_score"]) for row in members])),
            # 保存基线评分。
            "median_baseline_enterprise_macro_normalized_interval_score": ratio(decimal_median([Decimal(row["median_baseline_enterprise_macro_normalized_interval_score"]) for row in members])),
            # 保存总体区间Skill。
            "interval_skill_score_vs_baseline": ratio(decimal_median([Decimal(row["interval_skill_score_vs_baseline"]) for row in members])),
            # 保存正向折数。
            "positive_fold_count": str(sum(Decimal(row["interval_skill_score_vs_baseline"]) > 0 for row in members)),
            # 保存最弱折Skill。
            "minimum_fold_interval_skill_score": ratio(min(Decimal(row["interval_skill_score_vs_baseline"]) for row in members)),
            # 保存改善企业比例中位数。
            "median_improved_enterprise_rate": ratio(decimal_median([Decimal(row["median_improved_enterprise_rate"]) for row in members])),
            # 保存候选方法。
            "candidate_interval_method_id": LSTM_METHOD_ID,
            # 保存基线方法。
            "baseline_interval_method_id": BASELINE_METHOD_ID,
            # 保存版本。
            "c2c_version": C2C_VERSION,
        })
    # 返回两条摘要。
    return rows
