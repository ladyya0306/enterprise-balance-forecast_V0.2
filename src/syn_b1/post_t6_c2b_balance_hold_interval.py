"""POST-T6-C2B：用余额保持点预测建立名义80%的分割保形区间基线。"""

# 导入CSV标准库；它读取C2A共同配对表并写出残差、区间和指标文件。
import csv
# 从collections导入defaultdict；它按种子、滚动折和期限组织校准残差与验证记录。
from collections import defaultdict
# 从decimal导入Decimal；金额和分位数半径保持精确十进制口径。
from decimal import Decimal
# 从pathlib导入Path；它让函数以明确路径读取和写入文件。
from pathlib import Path

# 导入YAML；它读取冻结C1合同中的80%概率和基线方法规则。
import yaml

# 复用C2A公共区间公式和企业等权汇总，保证以后候选方法使用同一评分尺子。
from syn_b1.post_t6_c2a_average_interval_foundation import AVERAGE_HORIZONS, C2AFoundationError, FOLDS, SELECTED_CONFIGURATION, TRAINING_SEEDS, enterprise_macro_interval_summary, interval_measurement
# 复用既有比例显示函数，所有阶段统一保留八位小数比例格式。
from syn_b1.post_t6_a3_xgboost_windows import ratio

# 固定C2B实现版本；改变校准分位数公式或输出字段必须建立新版本。
C2B_VERSION = "syn_b1_post_t6_c2b_balance_hold_interval_baseline_1_0"
# 固定名义覆盖率80%，与C1的P10至P90约定一致。
NOMINAL_COVERAGE = Decimal("0.80")
# 固定余额保持基线方法编号，供C2C/C2D/C2E引用但不得改写。
BASELINE_METHOD_ID = "balance_hold_scaled_split_conformal"


# 声明C2B专用异常，区分数据或隔离缺陷与区间效果好坏。
class C2BBaselineError(ValueError):
    """当校准角色、保形分位数或基线区间文件不满足合同条件时抛出。"""


# 计算名义80%分割保形校准所需的一侧绝对残差分位数位置。
def finite_sample_conformal_rank(sample_count: int) -> int:
    # 拒绝空校准集。
    if sample_count <= 0:
        # 抛出明确异常。
        raise C2BBaselineError("分割保形校准至少需要一条calibration残差")
    # 将80%写成4/5，并对(样本数+1)乘积向上取整得到一基排名。
    rank = (4 * (sample_count + 1) + 4) // 5
    # 分位数排名不能超过样本数本身。
    return min(rank, sample_count)


# 从同一折calibration角色的归一化绝对残差计算对称区间半径。
def conformal_radius(residuals: list[Decimal]) -> tuple[int, Decimal]:
    # 拒绝空残差列表。
    if not residuals:
        # 抛出明确异常。
        raise C2BBaselineError("无法从空calibration残差计算保形半径")
    # 计算冻结的有限样本分位数排名。
    rank = finite_sample_conformal_rank(len(residuals))
    # 从小到大排序残差副本，避免改写调用者数据。
    ordered = sorted(residuals)
    # 用一基排名减一定位到Python零基下标。
    radius = ordered[rank - 1]
    # 返回排名和归一化半径。
    return rank, radius


# 读取C2A共同配对CSV，并只从calibration角色提取余额保持法归一化绝对残差。
def read_calibration_residuals(pair_path: Path) -> dict[tuple[int, str, int], list[dict[str, str | Decimal]]]:
    # 初始化种子、折、期限到残差记录列表的映射。
    grouped: dict[tuple[int, str, int], list[dict[str, str | Decimal]]] = defaultdict(list)
    # 以UTF-8和CSV标准换行读取C2A配对文件。
    with pair_path.open("r", encoding="utf-8", newline="") as handle:
        # 创建按表头取值的读取器。
        reader = csv.DictReader(handle)
        # 逐条读取共同配对记录。
        for row in reader:
            # 只允许calibration角色用于确定区间半径。
            if row["fold_role"] != "calibration":
                # 跳过validation角色，避免偷看评价答案。
                continue
            # 将训练种子转换为整数。
            seed = int(row["training_seed"])
            # 读取滚动折号。
            fold_id = row["fold_id"]
            # 将日均期限转换为整数。
            horizon = int(row["average_horizon_business_days"])
            # 拒绝不在冻结矩阵内的记录。
            if seed not in TRAINING_SEEDS or fold_id not in FOLDS or horizon not in AVERAGE_HORIZONS:
                # 阻断配对表漂移。
                raise C2BBaselineError("C2A calibration记录含未登记种子、折或期限")
            # 读取真实日均余额。
            actual = Decimal(row["actual_average_balance_cny"])
            # 读取余额保持点预测；本卡唯一使用的预测中心。
            point = Decimal(row["balance_hold_point_average_balance_cny"])
            # 读取预测起点可见账户尺度。
            scale = Decimal(row["balance_scale_cny"])
            # 尺度必须严格为正。
            if scale <= 0:
                # 阻断无定义归一化残差。
                raise C2BBaselineError("C2A calibration记录的账户尺度必须大于0")
            # 计算账户尺度归一化绝对残差。
            normalized_residual = abs(actual - point) / scale
            # 保存当前校准记录，只保留C2B必要字段。
            grouped[(seed, fold_id, horizon)].append({
                # 保存样本连接键用于审计。
                "sample_key": row["sample_key"],
                # 保存企业连接键用于审计。
                "enterprise_id": row["enterprise_id"],
                # 保存账户尺度。
                "balance_scale_cny": scale,
                # 保存真实日均余额。
                "actual_average_balance_cny": actual,
                # 保存余额保持点预测。
                "balance_hold_point_average_balance_cny": point,
                # 保存归一化绝对残差。
                "normalized_absolute_residual": normalized_residual,
            })
    # 遍历完整的三种子、四折、两期限组合。
    for seed in TRAINING_SEEDS:
        # 遍历滚动折。
        for fold_id in FOLDS:
            # 遍历日均期限。
            for horizon in AVERAGE_HORIZONS:
                # 每个组合都必须拥有校准残差。
                if not grouped[(seed, fold_id, horizon)]:
                    # 阻断缺失校准组。
                    raise C2BBaselineError("C2A共同配对表缺少完整calibration种子、折或期限")
    # 返回完整的校准残差分组。
    return grouped


# 为24个校准组合计算固定的保形半径，并生成可审计量化表。
def build_quantile_rows(grouped: dict[tuple[int, str, int], list[dict[str, str | Decimal]]]) -> tuple[dict[tuple[int, str, int], Decimal], list[dict[str, str]]]:
    # 初始化组合到归一化半径的映射。
    radii: dict[tuple[int, str, int], Decimal] = {}
    # 初始化24条分位数审计行。
    rows: list[dict[str, str]] = []
    # 按冻结种子顺序处理。
    for seed in TRAINING_SEEDS:
        # 按冻结折顺序处理。
        for fold_id in FOLDS:
            # 按冻结期限顺序处理。
            for horizon in AVERAGE_HORIZONS:
                # 取得当前组合的完整校准记录。
                records = grouped[(seed, fold_id, horizon)]
                # 提取归一化绝对残差。
                residuals = [Decimal(str(record["normalized_absolute_residual"])) for record in records]
                # 计算保形一基排名和半径。
                rank, radius = conformal_radius(residuals)
                # 保存当前组合半径。
                radii[(seed, fold_id, horizon)] = radius
                # 写出当前组合审计行。
                rows.append({
                    # 保存种子。
                    "training_seed": str(seed),
                    # 保存滚动折。
                    "fold_id": fold_id,
                    # 保存日均期限。
                    "average_horizon_business_days": str(horizon),
                    # 保存名义覆盖率。
                    "nominal_coverage": ratio(NOMINAL_COVERAGE),
                    # 保存校准记录数。
                    "calibration_record_count": str(len(records)),
                    # 保存不同校准企业数。
                    "calibration_enterprise_count": str(len({str(record['enterprise_id']) for record in records})),
                    # 保存有限样本一基排名。
                    "finite_sample_conformal_rank": str(rank),
                    # 保存归一化半径。
                    "normalized_conformal_radius": ratio(radius),
                    # 保存基线方法编号。
                    "interval_method_id": BASELINE_METHOD_ID,
                    # 保存版本。
                    "c2b_version": C2B_VERSION,
                })
    # 要求完整24行。
    if len(rows) != 24:
        # 阻断缺组合输出。
        raise C2BBaselineError("保形分位数审计行不是完整24行")
    # 返回半径映射和审计行。
    return radii, rows


# 将共同配对数据应用同折半径，写出calibration参考区间和validation评价区间。
def write_interval_pairs(pair_path: Path, output_path: Path, radii: dict[tuple[int, str, int], Decimal]) -> dict[tuple[int, str, int], list[dict[str, Decimal | str]]]:
    # 固定区间配对CSV列顺序。
    fields = ("fold_id", "fold_role", "training_seed", "sample_key", "enterprise_id", "average_horizon_business_days", "balance_scale_cny", "actual_average_balance_cny", "balance_hold_point_average_balance_cny", "normalized_conformal_radius", "p10_lower_bound_cny", "p50_point_estimate_cny", "p90_upper_bound_cny", "covered_by_interval", "normalized_interval_width", "normalized_interval_score", "interval_method_id", "c2b_version")
    # 初始化仅用于validation评价的组合记录。
    validation_records: dict[tuple[int, str, int], list[dict[str, Decimal | str]]] = defaultdict(list)
    # 以UTF-8和CSV标准换行写入新区间表。
    with output_path.open("w", encoding="utf-8", newline="") as output_handle:
        # 创建固定表头写入器。
        writer = csv.DictWriter(output_handle, fieldnames=fields)
        # 写入表头。
        writer.writeheader()
        # 以UTF-8和CSV标准换行读取共同配对文件。
        with pair_path.open("r", encoding="utf-8", newline="") as pair_handle:
            # 创建按表头取值的读取器。
            reader = csv.DictReader(pair_handle)
            # 逐条形成基线区间。
            for row in reader:
                # 读取种子。
                seed = int(row["training_seed"])
                # 读取滚动折。
                fold_id = row["fold_id"]
                # 读取日均期限。
                horizon = int(row["average_horizon_business_days"])
                # 读取角色。
                role = row["fold_role"]
                # 只允许两个C2A角色。
                if role not in ("calibration", "validation"):
                    # 阻断角色泄漏。
                    raise C2BBaselineError("C2A共同配对表含非法角色")
                # 读取冻结半径。
                radius = radii[(seed, fold_id, horizon)]
                # 读取账户尺度。
                scale = Decimal(row["balance_scale_cny"])
                # 读取真实日均余额。
                actual = Decimal(row["actual_average_balance_cny"])
                # 读取余额保持点预测。
                point = Decimal(row["balance_hold_point_average_balance_cny"])
                # 将归一化半径转换回当前账户的人民币半径。
                amount_radius = radius * scale
                # 构造对称P10下限。
                lower = point - amount_radius
                # 构造对称P90上限。
                upper = point + amount_radius
                # 计算覆盖、归一化宽度和80%区间评分。
                covered, normalized_width, normalized_score = interval_measurement(actual, lower, upper, scale)
                # 写出当前校准参考或验证评价区间。
                writer.writerow({
                    # 保存折号。
                    "fold_id": fold_id,
                    # 保存角色。
                    "fold_role": role,
                    # 保存种子。
                    "training_seed": str(seed),
                    # 保存样本连接键。
                    "sample_key": row["sample_key"],
                    # 保存企业连接键。
                    "enterprise_id": row["enterprise_id"],
                    # 保存期限。
                    "average_horizon_business_days": str(horizon),
                    # 保存账户尺度。
                    "balance_scale_cny": f"{scale:.2f}",
                    # 保存真实日均余额。
                    "actual_average_balance_cny": f"{actual:.2f}",
                    # 保存余额保持点预测。
                    "balance_hold_point_average_balance_cny": f"{point:.2f}",
                    # 保存归一化保形半径。
                    "normalized_conformal_radius": ratio(radius),
                    # 保存P10下限。
                    "p10_lower_bound_cny": f"{lower:.2f}",
                    # 保存P50点预测。
                    "p50_point_estimate_cny": f"{point:.2f}",
                    # 保存P90上限。
                    "p90_upper_bound_cny": f"{upper:.2f}",
                    # 保存是否覆盖。
                    "covered_by_interval": str(covered).lower(),
                    # 保存归一化宽度。
                    "normalized_interval_width": ratio(normalized_width),
                    # 保存归一化评分。
                    "normalized_interval_score": ratio(normalized_score),
                    # 保存方法编号。
                    "interval_method_id": BASELINE_METHOD_ID,
                    # 保存版本。
                    "c2b_version": C2B_VERSION,
                })
                # 只有validation记录进入真实评价桶。
                if role == "validation":
                    # 保存成候选与基线相同的通用记录，复用企业等权汇总公式。
                    validation_records[(seed, fold_id, horizon)].append({
                        # 保存企业编号。
                        "enterprise_id": row["enterprise_id"],
                        # 保存真实日均。
                        "actual_average_balance_cny": actual,
                        # 保存账户尺度。
                        "balance_scale_cny": scale,
                        # 余额保持基线既是当前候选也是比较基线。
                        "candidate_lower_cny": lower,
                        # 保存候选上限。
                        "candidate_upper_cny": upper,
                        # 保存比较基线下限。
                        "baseline_lower_cny": lower,
                        # 保存比较基线上限。
                        "baseline_upper_cny": upper,
                    })
    # 要求每个种子、折、期限都有validation记录。
    for seed in TRAINING_SEEDS:
        # 遍历滚动折。
        for fold_id in FOLDS:
            # 遍历期限。
            for horizon in AVERAGE_HORIZONS:
                # 缺失验证记录即阻断。
                if not validation_records[(seed, fold_id, horizon)]:
                    # 抛出明确异常。
                    raise C2BBaselineError("基线区间缺少完整validation种子、折或期限")
    # 返回validation评价记录。
    return validation_records


# 将24个种子折validation结果汇总为基线指标行。
def build_seed_fold_metrics(validation_records: dict[tuple[int, str, int], list[dict[str, Decimal | str]]], quantile_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 建立组合到分位数审计行的映射。
    quantile_by_key = {(int(row["training_seed"]), row["fold_id"], int(row["average_horizon_business_days"])): row for row in quantile_rows}
    # 初始化24条种子折指标。
    rows: list[dict[str, str]] = []
    # 按固定种子顺序处理。
    for seed in TRAINING_SEEDS:
        # 按固定折顺序处理。
        for fold_id in FOLDS:
            # 按固定期限顺序处理。
            for horizon in AVERAGE_HORIZONS:
                # 读取当前组合验证记录。
                records = validation_records[(seed, fold_id, horizon)]
                # 用候选等于基线的方式取得企业等权覆盖、宽度和评分。
                summary = enterprise_macro_interval_summary(records)
                # 读取对应校准分位数审计行。
                quantile = quantile_by_key[(seed, fold_id, horizon)]
                # 写出当前种子折指标。
                rows.append({
                    # 保存种子。
                    "training_seed": str(seed),
                    # 保存滚动折。
                    "fold_id": fold_id,
                    # 保存期限。
                    "average_horizon_business_days": str(horizon),
                    # 保存验证企业数。
                    "validation_enterprise_count": str(summary["enterprise_count"]),
                    # 保存验证预测起点数。
                    "validation_forecast_origin_count": str(len(records)),
                    # 保存校准记录数。
                    "calibration_record_count": quantile["calibration_record_count"],
                    # 保存保形排名。
                    "finite_sample_conformal_rank": quantile["finite_sample_conformal_rank"],
                    # 保存半径。
                    "normalized_conformal_radius": quantile["normalized_conformal_radius"],
                    # 保存企业等权覆盖率。
                    "enterprise_macro_empirical_coverage": ratio(Decimal(str(summary["candidate_empirical_coverage"]))),
                    # 保存企业等权宽度。
                    "enterprise_macro_normalized_interval_width": ratio(Decimal(str(summary["candidate_normalized_width"]))),
                    # 保存企业等权区间评分。
                    "enterprise_macro_normalized_interval_score": ratio(Decimal(str(summary["candidate_normalized_interval_score"]))),
                    # 基线与自身比较的Skill定义为0，仅用于说明非候选选择。
                    "interval_skill_vs_self": "0.00000000",
                    # 保存方法编号。
                    "interval_method_id": BASELINE_METHOD_ID,
                    # 保存版本。
                    "c2b_version": C2B_VERSION,
                })
    # 要求完整24条。
    if len(rows) != 24:
        # 阻断缺结果。
        raise C2BBaselineError("余额保持基线种子折指标不是完整24行")
    # 返回指标行。
    return rows


# 计算奇数或偶数个十进制数的稳定中位数。
def decimal_median(values: list[Decimal]) -> Decimal:
    # 拒绝空列表。
    if not values:
        # 抛出明确异常。
        raise C2BBaselineError("不能计算空列表的中位数")
    # 排序副本。
    ordered = sorted(values)
    # 读取长度。
    size = len(ordered)
    # 奇数长度直接取中间项。
    if size % 2 == 1:
        # 返回中间项。
        return ordered[size // 2]
    # 偶数长度取中间两项平均。
    return (ordered[size // 2 - 1] + ordered[size // 2]) / Decimal("2")


# 把三种子合成为每折日均基线指标。
def build_fold_metrics(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 初始化8条折指标。
    rows: list[dict[str, str]] = []
    # 遍历四折。
    for fold_id in FOLDS:
        # 遍历两个期限。
        for horizon in AVERAGE_HORIZONS:
            # 取得当前折期限的三种子行。
            members = [row for row in seed_rows if row["fold_id"] == fold_id and int(row["average_horizon_business_days"]) == horizon]
            # 要求完整三种子。
            if len(members) != 3 or {int(row["training_seed"]) for row in members} != set(TRAINING_SEEDS):
                # 阻断缺种子。
                raise C2BBaselineError("余额保持基线折指标缺少完整三种子")
            # 写出种子中位折结果。
            rows.append({
                # 保存折号。
                "fold_id": fold_id,
                # 保存期限。
                "average_horizon_business_days": str(horizon),
                # 保存种子数量。
                "seed_count": "3",
                # 保存覆盖率中位数。
                "median_enterprise_macro_empirical_coverage": ratio(decimal_median([Decimal(row["enterprise_macro_empirical_coverage"]) for row in members])),
                # 保存宽度中位数。
                "median_enterprise_macro_normalized_interval_width": ratio(decimal_median([Decimal(row["enterprise_macro_normalized_interval_width"]) for row in members])),
                # 保存评分中位数。
                "median_enterprise_macro_normalized_interval_score": ratio(decimal_median([Decimal(row["enterprise_macro_normalized_interval_score"]) for row in members])),
                # 保存半径中位数。
                "median_normalized_conformal_radius": ratio(decimal_median([Decimal(row["normalized_conformal_radius"]) for row in members])),
                # 保存方法编号。
                "interval_method_id": BASELINE_METHOD_ID,
                # 保存版本。
                "c2b_version": C2B_VERSION,
            })
    # 返回8条折级行。
    return rows


# 把四折中位数合成为10日和30日总体基线摘要。
def build_summary_metrics(fold_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 初始化两条期限摘要。
    rows: list[dict[str, str]] = []
    # 遍历两个期限。
    for horizon in AVERAGE_HORIZONS:
        # 取得当前期限四折行。
        members = [row for row in fold_rows if int(row["average_horizon_business_days"]) == horizon]
        # 要求完整四折。
        if len(members) != 4:
            # 阻断缺折。
            raise C2BBaselineError("余额保持基线总体摘要缺少完整四折")
        # 写出四折中位摘要。
        rows.append({
            # 保存期限。
            "average_horizon_business_days": str(horizon),
            # 保存折数。
            "fold_count": "4",
            # 保存每折种子数。
            "seed_count_per_fold": "3",
            # 保存总体覆盖率。
            "median_enterprise_macro_empirical_coverage": ratio(decimal_median([Decimal(row["median_enterprise_macro_empirical_coverage"]) for row in members])),
            # 保存总体宽度。
            "median_enterprise_macro_normalized_interval_width": ratio(decimal_median([Decimal(row["median_enterprise_macro_normalized_interval_width"]) for row in members])),
            # 保存总体评分。
            "median_enterprise_macro_normalized_interval_score": ratio(decimal_median([Decimal(row["median_enterprise_macro_normalized_interval_score"]) for row in members])),
            # 保存最弱折覆盖率，供后续候选公平比较。
            "minimum_fold_empirical_coverage": ratio(min(Decimal(row["median_enterprise_macro_empirical_coverage"]) for row in members)),
            # 保存最强折覆盖率。
            "maximum_fold_empirical_coverage": ratio(max(Decimal(row["median_enterprise_macro_empirical_coverage"]) for row in members)),
            # 保存方法编号。
            "interval_method_id": BASELINE_METHOD_ID,
            # 保存版本。
            "c2b_version": C2B_VERSION,
        })
    # 返回两条摘要。
    return rows
