# 模块字符串说明本文件只比较既有C2结果，不训练或修改任何预测模型。
"""POST-T6-C2E-R1日均概率区间统一判定核心。"""

# 导入CSV标准库；只读读取既有验证结果并按表头对齐。
import csv
# 导入随机数标准库；企业整块自助抽样需要固定可复现随机源。
import random
# 导入默认字典；分组汇总时避免手工初始化大量桶。
from collections import defaultdict
# 导入十进制定点数；金额和评分避免浮点数累计误差。
from decimal import Decimal
# 从统计模块导入中位数；折和种子需要按冻结规则取中位数。
from statistics import median
# 从类型模块导入Any；输入CSV字段同时包含字符串和十进制数。
from typing import Any

# 复用C2A的80%区间评分；禁止复制已冻结公式形成第二套算法。
from syn_b1.post_t6_c2a_average_interval_foundation import interval_measurement

# 固定两种待选方法的内部编号，避免文字拼写差异改变判定。
CANDIDATE_METHODS = ("locked_lstm_scaled_split_conformal", "fixed_lstm_conformalized_quantiles")
# 固定余额保持法编号；它只作公开参照而非R1资格否决。
BASELINE_METHOD = "balance_hold_scaled_split_conformal"
# 固定四个滚动折；缺任一折即不能形成完整判定。
FOLDS = ("A", "B", "C", "D")
# 固定三种训练种子；不得用最好种子替代家族结论。
SEEDS = (2026083102, 2026083103, 2026083104)
# 固定两个日均余额期限。
HORIZONS = (10, 30)
# 固定标准WIS中位数绝对误差权重。
WIS_MEDIAN_WEIGHT = Decimal("0.50")
# 固定标准WIS单个80%区间权重。
WIS_INTERVAL_WEIGHT = Decimal("0.10")
# 固定标准单区间WIS归一分母。
WIS_DENOMINATOR = Decimal("1.50")


# 定义C2E-R1专用异常；输入血缘问题不会被误判成模型优劣。
class C2ER1DecisionError(ValueError):  # 定义统一错误类型，方便入口和测试精确处理。
    # 模块级异常不增加额外行为，错误文字由发现问题的函数提供。
    pass


# 将普通数值安全转换为十进制，保证CSV读取和手算测试共用同一精度口径。
def as_decimal(value: Any) -> Decimal:
    # 通过字符串构造十进制，避免二进制浮点表示误差。
    return Decimal(str(value))


# 将十进制固定为八位小数文本，便于CSV稳定写出和指纹复算。
def decimal_text(value: Decimal) -> str:
    # 量化到八位小数，满足评分输出精度且避免科学计数法。
    return f"{value:.8f}"


# 计算一个十进制列表的中位数；空列表必须显式失败。
def decimal_median(values: list[Decimal]) -> Decimal:
    # 空集合没有中位数，阻断不完整折或种子汇总。
    if not values:
        # 抛出明确错误而不是返回随意默认值。
        raise C2ER1DecisionError("无法对空指标集合计算中位数")
    # statistics.median支持Decimal并保留偶数个值的平均定义。
    return as_decimal(median(values))


# 固定最近秩百分位数；自助抽样边界使用与C2A一致的保守向下索引。
def percentile_floor(values: list[Decimal], probability: Decimal) -> Decimal:
    # 空列表没有百分位数。
    if not values:
        # 阻断空的自助抽样或企业宽度分布。
        raise C2ER1DecisionError("无法对空指标集合计算百分位数")
    # 概率必须位于闭区间内。
    if probability < 0 or probability > 1:
        # 阻断错误概率设置。
        raise C2ER1DecisionError("百分位数概率必须在0至1之间")
    # 固定排序，保证同输入输出稳定。
    ordered = sorted(values)
    # 计算零基向下取整索引。
    index = int(Decimal(len(ordered) - 1) * probability)
    # 返回指定位置的十进制值。
    return ordered[index]


# 从统一记录取得用于不重复配对的六字段键。
def common_key(record: dict[str, Any]) -> tuple[str, str, int, str, str, int]:
    # 按冻结字段顺序返回不可变元组。
    return (str(record["fold_id"]), str(record["fold_role"]), int(record["training_seed"]), str(record["sample_key"]), str(record["enterprise_id"]), int(record["average_horizon_business_days"]))


# 从一条P10/P50/P90日均预测计算归一化点误差、区间评分和标准WIS80。
def calculate_row_metrics(actual: Decimal, balance_scale: Decimal, lower: Decimal, point: Decimal, upper: Decimal) -> dict[str, Decimal | bool]:
    # 账户尺度必须为正，否则所有归一化指标无定义。
    if balance_scale <= 0:
        # 阻断坏尺度，避免后续误导性评分。
        raise C2ER1DecisionError("账户尺度必须大于0")
    # P10、P50、P90必须有序，否则不能当作概率区间。
    if lower > point or point > upper:
        # 阻断分位数交叉。
        raise C2ER1DecisionError("P10、P50、P90必须满足P10不大于P50不大于P90")
    # 复用C2A计算覆盖、归一宽度和归一80%区间评分。
    covered, normalized_width, normalized_interval_score = interval_measurement(actual, lower, upper, balance_scale)
    # 计算归一化P50绝对误差。
    normalized_p50_absolute_error = abs(actual - point) / balance_scale
    # 按公开标准单区间公式计算归一化WIS80。
    normalized_wis80 = (WIS_MEDIAN_WEIGHT * normalized_p50_absolute_error + WIS_INTERVAL_WEIGHT * normalized_interval_score) / WIS_DENOMINATOR
    # 返回所有单行指标供后续企业等权汇总。
    return {"covered": covered, "normalized_width": normalized_width, "normalized_interval_score": normalized_interval_score, "normalized_p50_absolute_error": normalized_p50_absolute_error, "normalized_wis80": normalized_wis80}


# 核验三种方法的共同验证键、真实日均余额和账户尺度完全一致。
def validate_common_records(records: list[dict[str, Any]]) -> None:
    # 空记录不能形成任何C2E-R1判定。
    if not records:
        # 明确报告输入缺失。
        raise C2ER1DecisionError("C2E-R1共同验证记录为空")
    # 初始化已见共同键，防止重复预测行悄悄影响权重。
    seen_keys: set[tuple[str, str, int, str, str, int]] = set()
    # 初始化每个折种子期限组合的行数。
    combination_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    # 逐条检查合并后的共同记录。
    for record in records:
        # 生成当前共同键。
        key = common_key(record)
        # 重复键会让某些预测起点被多次计分。
        if key in seen_keys:
            # 阻断重复键。
            raise C2ER1DecisionError(f"共同验证键重复：{key}")
        # 保存已见键。
        seen_keys.add(key)
        # 当前卡只允许validation角色。
        if key[1] != "validation":
            # 阻断calibration或其它角色混入。
            raise C2ER1DecisionError(f"发现非validation角色：{key}")
        # 折号、种子和期限必须属于冻结集合。
        if key[0] not in FOLDS or key[2] not in SEEDS or key[5] not in HORIZONS:
            # 阻断未登记组合。
            raise C2ER1DecisionError(f"发现未冻结折、种子或期限：{key}")
        # 账户尺度必须为正。
        if as_decimal(record["balance_scale_cny"]) <= 0:
            # 阻断错误尺度。
            raise C2ER1DecisionError(f"账户尺度非正：{key}")
        # 两种候选和基线必须同时存在。
        if set(record["methods"]) != {BASELINE_METHOD, *CANDIDATE_METHODS}:
            # 阻断方法缺失或额外方法混入。
            raise C2ER1DecisionError(f"共同记录方法集合错误：{key}")
        # 统计当前折种子期限已有记录。
        combination_counts[(key[0], key[2], key[5])] += 1
    # 逐个要求24个折种子期限组合都有至少一条验证记录。
    for fold_id in FOLDS:
        # 遍历固定三个种子。
        for seed in SEEDS:
            # 遍历固定两个期限。
            for horizon in HORIZONS:
                # 缺少任一组合时阻断公平比较。
                if combination_counts[(fold_id, seed, horizon)] == 0:
                    # 说明缺少的组合。
                    raise C2ER1DecisionError(f"共同验证记录缺少组合：{fold_id}/{seed}/{horizon}")


# 计算一个候选方法相对余额保持法的企业级、种子折级、折级和期限级指标。
def build_method_evaluation(records: list[dict[str, Any]], method_id: str) -> dict[str, Any]:
    # 只允许冻结的两种候选方法进入资格比较。
    if method_id not in CANDIDATE_METHODS:
        # 阻断基线或未知方法被误作候选。
        raise C2ER1DecisionError(f"未知候选方法：{method_id}")
    # 初始化企业、种子、折和期限分组的逐行指标列表。
    by_enterprise_seed_fold_horizon: dict[tuple[str, int, str, int], list[dict[str, Decimal | bool]]] = defaultdict(list)
    # 初始化跨种子企业宽度与WIS分布。
    by_enterprise_horizon: dict[tuple[str, int], list[dict[str, Decimal | bool]]] = defaultdict(list)
    # 初始化可写出的共同配对行。
    common_rows: list[dict[str, str]] = []
    # 逐条生成候选与基线指标。
    for record in records:
        # 读取真实日均余额。
        actual = as_decimal(record["actual_average_balance_cny"])
        # 读取预测起点账户尺度。
        scale = as_decimal(record["balance_scale_cny"])
        # 读取候选P10/P50/P90。
        candidate = record["methods"][method_id]
        # 读取余额保持法P10/P50/P90。
        baseline = record["methods"][BASELINE_METHOD]
        # 计算候选单行指标。
        candidate_metrics = calculate_row_metrics(actual, scale, candidate["p10"], candidate["p50"], candidate["p90"])
        # 计算基线单行指标。
        baseline_metrics = calculate_row_metrics(actual, scale, baseline["p10"], baseline["p50"], baseline["p90"])
        # 将候选和基线指标放在同一条内部记录，后续汇总不再读取CSV。
        paired_metrics: dict[str, Decimal | bool] = {"candidate_covered": candidate_metrics["covered"], "baseline_covered": baseline_metrics["covered"], "candidate_width": candidate_metrics["normalized_width"], "baseline_width": baseline_metrics["normalized_width"], "candidate_interval_score": candidate_metrics["normalized_interval_score"], "baseline_interval_score": baseline_metrics["normalized_interval_score"], "candidate_p50_error": candidate_metrics["normalized_p50_absolute_error"], "baseline_p50_error": baseline_metrics["normalized_p50_absolute_error"], "candidate_wis80": candidate_metrics["normalized_wis80"], "baseline_wis80": baseline_metrics["normalized_wis80"]}
        # 构造企业、种子、折和期限分组键。
        enterprise_group_key = (str(record["enterprise_id"]), int(record["training_seed"]), str(record["fold_id"]), int(record["average_horizon_business_days"]))
        # 保存当前预测起点的配对指标。
        by_enterprise_seed_fold_horizon[enterprise_group_key].append(paired_metrics)
        # 构造跨种子企业分布键。
        enterprise_horizon_key = (str(record["enterprise_id"]), int(record["average_horizon_business_days"]))
        # 保存当前预测起点的配对指标。
        by_enterprise_horizon[enterprise_horizon_key].append(paired_metrics)
        # 写入可复核的候选共同配对行。
        common_rows.append({"fold_id": str(record["fold_id"]), "fold_role": str(record["fold_role"]), "training_seed": str(record["training_seed"]), "sample_key": str(record["sample_key"]), "enterprise_id": str(record["enterprise_id"]), "average_horizon_business_days": str(record["average_horizon_business_days"]), "balance_scale_cny": f"{scale:.2f}", "actual_average_balance_cny": f"{actual:.2f}", "interval_method_id": method_id, "p10_lower_bound_cny": f"{candidate['p10']:.2f}", "p50_point_estimate_cny": f"{candidate['p50']:.2f}", "p90_upper_bound_cny": f"{candidate['p90']:.2f}", "normalized_p50_absolute_error": decimal_text(candidate_metrics["normalized_p50_absolute_error"]), "normalized_interval_score": decimal_text(candidate_metrics["normalized_interval_score"]), "normalized_wis80": decimal_text(candidate_metrics["normalized_wis80"]), "covered_by_interval": str(candidate_metrics["covered"]).lower(), "normalized_interval_width": decimal_text(candidate_metrics["normalized_width"])})
    # 初始化企业级输出行。
    enterprise_rows: list[dict[str, str]] = []
    # 初始化种子折指标使用的企业级数值记录。
    seed_fold_inputs: dict[tuple[int, str, int], list[dict[str, Decimal]]] = defaultdict(list)
    # 按稳定顺序逐企业种子折期限汇总全部预测起点。
    for group_key in sorted(by_enterprise_seed_fold_horizon):
        # 展开企业、种子、折和期限。
        enterprise_id, seed, fold_id, horizon = group_key
        # 读取同一企业的全部预测起点指标。
        metric_rows = by_enterprise_seed_fold_horizon[group_key]
        # 计算当前企业预测起点数。
        count = Decimal(len(metric_rows))
        # 逐项计算同企业内部等权均值。
        aggregate = {name: sum((as_decimal(row[name]) for row in metric_rows), Decimal("0")) / count for name in ("candidate_width", "baseline_width", "candidate_interval_score", "baseline_interval_score", "candidate_p50_error", "baseline_p50_error", "candidate_wis80", "baseline_wis80")}
        # 覆盖率按布尔值转1或0后平均。
        candidate_coverage = sum((Decimal(int(bool(row["candidate_covered"]))) for row in metric_rows), Decimal("0")) / count
        # 基线覆盖率按同样口径计算。
        baseline_coverage = sum((Decimal(int(bool(row["baseline_covered"]))) for row in metric_rows), Decimal("0")) / count
        # 基线点误差为零时Skill无定义，必须阻断而不是随意写0。
        if aggregate["baseline_p50_error"] == 0 or aggregate["baseline_interval_score"] == 0:
            # 抛出可审计错误。
            raise C2ER1DecisionError("基线误差或区间评分为0，无法计算Skill Score")
        # 计算候选相对基线的点预测Skill。
        point_skill = Decimal("1") - aggregate["candidate_p50_error"] / aggregate["baseline_p50_error"]
        # 计算候选相对基线的区间Skill。
        interval_skill = Decimal("1") - aggregate["candidate_interval_score"] / aggregate["baseline_interval_score"]
        # 保存企业级机器可读行。
        enterprise_rows.append({"enterprise_id": enterprise_id, "training_seed": str(seed), "fold_id": fold_id, "average_horizon_business_days": str(horizon), "forecast_origin_count": str(int(count)), "candidate_enterprise_normalized_p50_absolute_error": decimal_text(aggregate["candidate_p50_error"]), "baseline_enterprise_normalized_p50_absolute_error": decimal_text(aggregate["baseline_p50_error"]), "candidate_enterprise_normalized_wis80": decimal_text(aggregate["candidate_wis80"]), "baseline_enterprise_normalized_wis80": decimal_text(aggregate["baseline_wis80"]), "candidate_enterprise_empirical_coverage": decimal_text(candidate_coverage), "baseline_enterprise_empirical_coverage": decimal_text(baseline_coverage), "candidate_enterprise_normalized_interval_width": decimal_text(aggregate["candidate_width"]), "baseline_enterprise_normalized_interval_width": decimal_text(aggregate["baseline_width"]), "candidate_enterprise_normalized_interval_score": decimal_text(aggregate["candidate_interval_score"]), "baseline_enterprise_normalized_interval_score": decimal_text(aggregate["baseline_interval_score"]), "point_skill_score_vs_baseline": decimal_text(point_skill), "interval_skill_score_vs_baseline": decimal_text(interval_skill), "interval_method_id": method_id})
        # 保存种子折企业等权汇总需要的数值。
        seed_fold_inputs[(seed, fold_id, horizon)].append({"candidate_p50_error": aggregate["candidate_p50_error"], "baseline_p50_error": aggregate["baseline_p50_error"], "candidate_wis80": aggregate["candidate_wis80"], "baseline_wis80": aggregate["baseline_wis80"], "candidate_coverage": candidate_coverage, "baseline_coverage": baseline_coverage, "candidate_width": aggregate["candidate_width"], "baseline_width": aggregate["baseline_width"], "candidate_interval_score": aggregate["candidate_interval_score"], "baseline_interval_score": aggregate["baseline_interval_score"], "interval_skill": interval_skill})
    # 初始化种子折输出行。
    seed_fold_rows: list[dict[str, str]] = []
    # 逐个固定组合计算企业等权指标。
    for seed in SEEDS:
        # 遍历固定四折。
        for fold_id in FOLDS:
            # 遍历固定双期限。
            for horizon in HORIZONS:
                # 读取当前组合全部企业。
                enterprise_values = seed_fold_inputs[(seed, fold_id, horizon)]
                # 缺企业时说明共同验证不完整。
                if not enterprise_values:
                    # 阻断缺失组合。
                    raise C2ER1DecisionError(f"候选方法缺少种子折期限组合：{method_id}/{seed}/{fold_id}/{horizon}")
                # 计算企业数量精确分母。
                enterprise_count = Decimal(len(enterprise_values))
                # 计算每项企业等权平均。
                macro = {name: sum((row[name] for row in enterprise_values), Decimal("0")) / enterprise_count for name in ("candidate_p50_error", "baseline_p50_error", "candidate_wis80", "baseline_wis80", "candidate_coverage", "baseline_coverage", "candidate_width", "baseline_width", "candidate_interval_score", "baseline_interval_score")}
                # 计算点预测Skill。
                point_skill = Decimal("1") - macro["candidate_p50_error"] / macro["baseline_p50_error"]
                # 计算区间Skill。
                interval_skill = Decimal("1") - macro["candidate_interval_score"] / macro["baseline_interval_score"]
                # 计算点预测改善企业比例，供原C1点预测门单独使用。
                point_improved_rate = Decimal(sum(1 for row in enterprise_values if row["candidate_p50_error"] < row["baseline_p50_error"])) / enterprise_count
                # 计算区间评分改善企业比例，供原C1概率区间门单独使用。
                interval_improved_rate = Decimal(sum(1 for row in enterprise_values if row["interval_skill"] > 0)) / enterprise_count
                # 写种子折指标行。
                seed_fold_rows.append({"training_seed": str(seed), "fold_id": fold_id, "average_horizon_business_days": str(horizon), "validation_enterprise_count": str(int(enterprise_count)), "candidate_enterprise_macro_normalized_p50_absolute_error": decimal_text(macro["candidate_p50_error"]), "baseline_enterprise_macro_normalized_p50_absolute_error": decimal_text(macro["baseline_p50_error"]), "candidate_enterprise_macro_normalized_wis80": decimal_text(macro["candidate_wis80"]), "baseline_enterprise_macro_normalized_wis80": decimal_text(macro["baseline_wis80"]), "candidate_enterprise_macro_empirical_coverage": decimal_text(macro["candidate_coverage"]), "baseline_enterprise_macro_empirical_coverage": decimal_text(macro["baseline_coverage"]), "candidate_enterprise_macro_normalized_interval_width": decimal_text(macro["candidate_width"]), "baseline_enterprise_macro_normalized_interval_width": decimal_text(macro["baseline_width"]), "candidate_enterprise_macro_normalized_interval_score": decimal_text(macro["candidate_interval_score"]), "baseline_enterprise_macro_normalized_interval_score": decimal_text(macro["baseline_interval_score"]), "point_skill_score_vs_baseline": decimal_text(point_skill), "interval_skill_score_vs_baseline": decimal_text(interval_skill), "point_improved_enterprise_rate": decimal_text(point_improved_rate), "interval_improved_enterprise_rate": decimal_text(interval_improved_rate), "interval_method_id": method_id})
    # 初始化折级输出行。
    fold_rows: list[dict[str, str]] = []
    # 按折和期限汇总三个种子中位数。
    for fold_id in FOLDS:
        # 遍历双期限。
        for horizon in HORIZONS:
            # 选取当前折期限的三个种子行。
            rows = [row for row in seed_fold_rows if row["fold_id"] == fold_id and int(row["average_horizon_business_days"]) == horizon]
            # 必须恰有三个种子行。
            if len(rows) != len(SEEDS):
                # 阻断种子缺失或重复。
                raise C2ER1DecisionError(f"折级汇总种子数错误：{method_id}/{fold_id}/{horizon}")
            # 对所有数值指标取三种子中位数。
            medians = {name: decimal_median([as_decimal(row[name]) for row in rows]) for name in ("candidate_enterprise_macro_normalized_p50_absolute_error", "baseline_enterprise_macro_normalized_p50_absolute_error", "candidate_enterprise_macro_normalized_wis80", "baseline_enterprise_macro_normalized_wis80", "candidate_enterprise_macro_empirical_coverage", "baseline_enterprise_macro_empirical_coverage", "candidate_enterprise_macro_normalized_interval_width", "baseline_enterprise_macro_normalized_interval_width", "candidate_enterprise_macro_normalized_interval_score", "baseline_enterprise_macro_normalized_interval_score", "point_skill_score_vs_baseline", "interval_skill_score_vs_baseline", "point_improved_enterprise_rate", "interval_improved_enterprise_rate")}
            # 写折级中位指标。
            fold_rows.append({"fold_id": fold_id, "average_horizon_business_days": str(horizon), "seed_count": str(len(SEEDS)), **{name: decimal_text(value) for name, value in medians.items()}, "interval_method_id": method_id})
    # 初始化期限级摘要行。
    summary_rows: list[dict[str, str]] = []
    # 初始化用于旧C1自助抽样的企业级跨种子记录。
    enterprise_overall: dict[tuple[str, int], dict[str, Decimal]] = {}
    # 按企业和期限把三个种子与全部起点等权汇总。
    for enterprise_horizon_key in sorted(by_enterprise_horizon):
        # 读取企业和期限。
        enterprise_id, horizon = enterprise_horizon_key
        # 读取该企业期限的全部种子起点指标。
        metric_rows = by_enterprise_horizon[enterprise_horizon_key]
        # 计算行数。
        count = Decimal(len(metric_rows))
        # 计算候选和基线的核心均值。
        enterprise_overall[enterprise_horizon_key] = {name: sum((as_decimal(row[name]) for row in metric_rows), Decimal("0")) / count for name in ("candidate_p50_error", "baseline_p50_error", "candidate_wis80", "baseline_wis80", "candidate_width", "baseline_width", "candidate_interval_score", "baseline_interval_score")}
        # 覆盖率同样保存供R1宽度外的企业级审计。
        enterprise_overall[enterprise_horizon_key]["candidate_coverage"] = sum((Decimal(int(bool(row["candidate_covered"]))) for row in metric_rows), Decimal("0")) / count
    # 逐期限汇总四折中位数和208户宽度分布。
    for horizon in HORIZONS:
        # 选取当前期限四条折级行。
        rows = [row for row in fold_rows if int(row["average_horizon_business_days"]) == horizon]
        # 四折必须完整。
        if len(rows) != len(FOLDS):
            # 阻断折缺失。
            raise C2ER1DecisionError(f"期限级汇总折数错误：{method_id}/{horizon}")
        # 对主要折级指标取四折中位数。
        medians = {name: decimal_median([as_decimal(row[name]) for row in rows]) for name in ("candidate_enterprise_macro_normalized_p50_absolute_error", "baseline_enterprise_macro_normalized_p50_absolute_error", "candidate_enterprise_macro_normalized_wis80", "baseline_enterprise_macro_normalized_wis80", "candidate_enterprise_macro_empirical_coverage", "baseline_enterprise_macro_empirical_coverage", "candidate_enterprise_macro_normalized_interval_width", "baseline_enterprise_macro_normalized_interval_width", "candidate_enterprise_macro_normalized_interval_score", "baseline_enterprise_macro_normalized_interval_score", "point_skill_score_vs_baseline", "interval_skill_score_vs_baseline", "point_improved_enterprise_rate", "interval_improved_enterprise_rate")}
        # 读取当前期限全部企业的候选宽度。
        widths = [value["candidate_width"] for (enterprise_id, item_horizon), value in enterprise_overall.items() if item_horizon == horizon]
        # 计算典型企业宽度中位数。
        width_median = decimal_median(widths)
        # 计算企业宽度第90百分位数。
        width_p90 = percentile_floor(widths, Decimal("0.90"))
        # 计算折级正点预测Skill数。
        positive_point_fold_count = sum(1 for row in rows if as_decimal(row["point_skill_score_vs_baseline"]) > 0)
        # 计算种子折正点预测Skill数。
        positive_point_seed_fold_count = sum(1 for row in seed_fold_rows if int(row["average_horizon_business_days"]) == horizon and as_decimal(row["point_skill_score_vs_baseline"]) > 0)
        # 计算折级正区间Skill数。
        positive_interval_fold_count = sum(1 for row in rows if as_decimal(row["interval_skill_score_vs_baseline"]) > 0)
        # 计算种子折正区间Skill数。
        positive_interval_seed_fold_count = sum(1 for row in seed_fold_rows if int(row["average_horizon_business_days"]) == horizon and as_decimal(row["interval_skill_score_vs_baseline"]) > 0)
        # 写期限级摘要。
        summary_rows.append({"average_horizon_business_days": str(horizon), "fold_count": str(len(FOLDS)), "seed_count_per_fold": str(len(SEEDS)), **{name: decimal_text(value) for name, value in medians.items()}, "candidate_enterprise_normalized_interval_width_median": decimal_text(width_median), "candidate_enterprise_normalized_interval_width_p90": decimal_text(width_p90), "positive_point_fold_count": str(positive_point_fold_count), "positive_point_seed_fold_count": str(positive_point_seed_fold_count), "positive_interval_fold_count": str(positive_interval_fold_count), "positive_interval_seed_fold_count": str(positive_interval_seed_fold_count), "interval_method_id": method_id})
    # 返回完整评价结构供工具写文件和判定。
    return {"common_rows": common_rows, "enterprise_rows": enterprise_rows, "seed_fold_rows": seed_fold_rows, "fold_rows": fold_rows, "summary_rows": summary_rows, "enterprise_overall": enterprise_overall}


# 对一个候选在一个期限内按企业整块计算任一已冻结Skill的95%自助抽样区间。
def bootstrap_skill(enterprise_overall: dict[tuple[str, int], dict[str, Decimal]], horizon: int, candidate_key: str, baseline_key: str, replicates: int, seed: int) -> tuple[Decimal, Decimal]:
    # 收集当前期限企业及其候选与基线的同口径评分。
    rows = [(enterprise_id, values) for (enterprise_id, item_horizon), values in enterprise_overall.items() if item_horizon == horizon]
    # 至少两户才有企业间不确定性含义。
    if len(rows) < 2:
        # 阻断伪置信区间。
        raise C2ER1DecisionError("企业整块自助抽样至少需要两户")
    # 固定企业排序，防止字典顺序改变结果。
    ordered = sorted(rows, key=lambda item: item[0])
    # 固定随机源，保证可复算。
    generator = random.Random(seed)
    # 初始化全部重复样本的Skill。
    skills: list[Decimal] = []
    # 执行冻结次数。
    for _ in range(replicates):
        # 有放回抽取与原企业数相同的整户。
        drawn = [generator.choice(ordered)[1] for _ in ordered]
        # 计算当前重复样本候选评分均值。
        candidate_score = sum((row[candidate_key] for row in drawn), Decimal("0")) / Decimal(len(drawn))
        # 计算当前重复样本基线评分均值。
        baseline_score = sum((row[baseline_key] for row in drawn), Decimal("0")) / Decimal(len(drawn))
        # 基线评分为零时Skill无定义，不能掩盖成零分。
        if baseline_score == 0:
            # 阻断不适用的抽样结果。
            raise C2ER1DecisionError("自助抽样基线评分为0，无法计算Skill Score")
        # 保存候选相对基线的Skill。
        skills.append(Decimal("1") - candidate_score / baseline_score)
    # 返回2.5%和97.5%边界。
    return percentile_floor(skills, Decimal("0.025")), percentile_floor(skills, Decimal("0.975"))


# 保留旧名称作为原C1区间评分调用的明确别名，避免阅读时误把它当成WIS80。
def bootstrap_interval_skill(enterprise_overall: dict[tuple[str, int], dict[str, Decimal]], horizon: int, replicates: int, seed: int) -> tuple[Decimal, Decimal]:
    # 使用原C1明确登记的区间评分字段调用通用抽样器。
    return bootstrap_skill(enterprise_overall, horizon, "candidate_interval_score", "baseline_interval_score", replicates, seed)


# 计算两个候选双期限综合WIS80差异的企业整块成对自助抽样记录。
def paired_wis80_bootstrap(left: dict[tuple[str, int], dict[str, Decimal]], right: dict[tuple[str, int], dict[str, Decimal]], replicates: int, seed: int) -> tuple[list[dict[str, str]], Decimal, Decimal]:
    # 取得左侧同时拥有10日和30日的企业集合。
    left_enterprises = {enterprise_id for enterprise_id, horizon in left if horizon in HORIZONS}
    # 取得右侧同时拥有10日和30日的企业集合。
    right_enterprises = {enterprise_id for enterprise_id, horizon in right if horizon in HORIZONS}
    # 取共同企业并固定排序。
    enterprise_ids = tuple(sorted(left_enterprises & right_enterprises))
    # 每户必须同时拥有两个期限，否则不能形成双期限成对比较。
    valid_ids = tuple(enterprise_id for enterprise_id in enterprise_ids if all((enterprise_id, horizon) in left and (enterprise_id, horizon) in right for horizon in HORIZONS))
    # 至少两户才允许企业整块抽样。
    if len(valid_ids) < 2:
        # 阻断企业交集不完整。
        raise C2ER1DecisionError("成对WIS80自助抽样缺少足够共同企业")
    # 预先计算每户左减右的双期限等权综合WIS80差。
    differences = {enterprise_id: (left[(enterprise_id, 10)]["candidate_wis80"] + left[(enterprise_id, 30)]["candidate_wis80"]) / Decimal("2") - (right[(enterprise_id, 10)]["candidate_wis80"] + right[(enterprise_id, 30)]["candidate_wis80"]) / Decimal("2") for enterprise_id in valid_ids}
    # 固定随机源。
    generator = random.Random(seed)
    # 初始化CSV行。
    rows: list[dict[str, str]] = []
    # 初始化重复样本差值。
    sampled_differences: list[Decimal] = []
    # 逐次有放回抽企业并计算均差。
    for replicate_index in range(1, replicates + 1):
        # 抽取与企业数相同数量的企业编号。
        drawn = [generator.choice(valid_ids) for _ in valid_ids]
        # 计算本次抽样差值均值。
        difference = sum((differences[enterprise_id] for enterprise_id in drawn), Decimal("0")) / Decimal(len(drawn))
        # 保存数值供百分位边界计算。
        sampled_differences.append(difference)
        # 写机器可读重复行。
        rows.append({"replicate_index": str(replicate_index), "c2c_minus_c2d_composite_wis80": decimal_text(difference), "bootstrap_seed": str(seed)})
    # 返回全部行与95%边界。
    return rows, percentile_floor(sampled_differences, Decimal("0.025")), percentile_floor(sampled_differences, Decimal("0.975"))


# 按原C1合同对一个候选形成不可覆盖的历史判定。
def decide_original_c1(evaluation: dict[str, Any], replicates: int, seed: int) -> dict[str, Any]:
    # 将期限摘要按整数期限索引。
    summary_by_horizon = {int(row["average_horizon_business_days"]): row for row in evaluation["summary_rows"]}
    # 初始化两个期限的门槛结果。
    horizon_results: list[dict[str, Any]] = []
    # 原C1逐期限检查10日和30日。
    for horizon in HORIZONS:
        # 读取当前期限摘要。
        row = summary_by_horizon[horizon]
        # 计算原C1点预测企业整块Skill区间。
        point_bootstrap_lower, point_bootstrap_upper = bootstrap_skill(evaluation["enterprise_overall"], horizon, "candidate_p50_error", "baseline_p50_error", replicates, seed + horizon)
        # 计算原C1区间评分企业整块Skill区间。
        interval_bootstrap_lower, interval_bootstrap_upper = bootstrap_interval_skill(evaluation["enterprise_overall"], horizon, replicates, seed + 100 + horizon)
        # 读取点预测Skill。
        point_skill = as_decimal(row["point_skill_score_vs_baseline"])
        # 读取区间Skill。
        interval_skill = as_decimal(row["interval_skill_score_vs_baseline"])
        # 读取原C1点预测改善企业比例。
        point_improved_rate = as_decimal(row["point_improved_enterprise_rate"])
        # 读取原C1区间评分改善企业比例。
        interval_improved_rate = as_decimal(row["interval_improved_enterprise_rate"])
        # 依原C1逐项形成通过布尔值。
        point_pass = point_skill >= Decimal("0.10") and int(row["positive_point_fold_count"]) == 4 and int(row["positive_point_seed_fold_count"]) >= 10 and point_improved_rate >= Decimal("0.60") and point_bootstrap_lower > 0
        # 区间还额外要求宽度不超过余额保持法。
        interval_pass = interval_skill >= Decimal("0.05") and int(row["positive_interval_fold_count"]) == 4 and int(row["positive_interval_seed_fold_count"]) >= 10 and interval_improved_rate >= Decimal("0.60") and interval_bootstrap_lower > 0 and as_decimal(row["candidate_enterprise_macro_normalized_interval_width"]) <= as_decimal(row["baseline_enterprise_macro_normalized_interval_width"])
        # 保存当前期限历史结果。
        horizon_results.append({"horizon": horizon, "point_skill": point_skill, "interval_skill": interval_skill, "point_bootstrap_lower": point_bootstrap_lower, "point_bootstrap_upper": point_bootstrap_upper, "interval_bootstrap_lower": interval_bootstrap_lower, "interval_bootstrap_upper": interval_bootstrap_upper, "point_pass": point_pass, "interval_pass": interval_pass})
    # 原C1方法合格要求两个期限的点和区间门均通过。
    method_pass = all(item["point_pass"] and item["interval_pass"] for item in horizon_results)
    # 返回历史判定和可写出细节。
    return {"method_pass": method_pass, "horizons": horizon_results}


# 按R1绝对真实值门对一个候选形成事后探索性资格判定。
def decide_r1(evaluation: dict[str, Any]) -> dict[str, Any]:
    # 将期限摘要按整数期限索引。
    summary_by_horizon = {int(row["average_horizon_business_days"]): row for row in evaluation["summary_rows"]}
    # 将折级行按期限分组。
    fold_rows_by_horizon: dict[int, list[dict[str, str]]] = defaultdict(list)
    # 保存折级行。
    for row in evaluation["fold_rows"]:
        # 按期限归组。
        fold_rows_by_horizon[int(row["average_horizon_business_days"])].append(row)
    # 初始化期限门结果。
    horizon_results: list[dict[str, Any]] = []
    # 分别检查10日和30日。
    for horizon in HORIZONS:
        # 读取期限摘要。
        row = summary_by_horizon[horizon]
        # 读取全部四折覆盖率。
        fold_coverages = [as_decimal(item["candidate_enterprise_macro_empirical_coverage"]) for item in fold_rows_by_horizon[horizon]]
        # 检查P50绝对误差门。
        p50_pass = as_decimal(row["candidate_enterprise_macro_normalized_p50_absolute_error"]) <= Decimal("0.50")
        # 检查WIS80绝对门。
        wis_pass = as_decimal(row["candidate_enterprise_macro_normalized_wis80"]) <= Decimal("1.00")
        # 检查总体覆盖率门。
        coverage_pass = Decimal("0.75") <= as_decimal(row["candidate_enterprise_macro_empirical_coverage"]) <= Decimal("0.85")
        # 检查每个滚动折的覆盖率门。
        fold_coverage_pass = len(fold_coverages) == 4 and all(Decimal("0.70") <= value <= Decimal("0.90") for value in fold_coverages)
        # 检查典型企业宽度门。
        width_median_pass = as_decimal(row["candidate_enterprise_normalized_interval_width_median"]) <= Decimal("1.00")
        # 检查90%企业宽度门。
        width_p90_pass = as_decimal(row["candidate_enterprise_normalized_interval_width_p90"]) <= Decimal("2.00")
        # 汇总当前期限是否全部通过。
        horizon_pass = all((p50_pass, wis_pass, coverage_pass, fold_coverage_pass, width_median_pass, width_p90_pass))
        # 保存当前期限各门。
        horizon_results.append({"horizon": horizon, "p50_pass": p50_pass, "wis_pass": wis_pass, "coverage_pass": coverage_pass, "fold_coverage_pass": fold_coverage_pass, "width_median_pass": width_median_pass, "width_p90_pass": width_p90_pass, "horizon_pass": horizon_pass})
    # 双期限都通过才是R1探索性合格候选。
    return {"method_pass": all(item["horizon_pass"] for item in horizon_results), "horizons": horizon_results}


# 根据R1候选资格和成对自助抽样边界形成唯一候选或停止决定。
def decide_unique_candidate(r1_by_method: dict[str, dict[str, Any]], paired_lower: Decimal | None, paired_upper: Decimal | None) -> dict[str, str | bool | None]:
    # 找出两个方法中已经通过R1全部绝对门的候选。
    qualified = [method_id for method_id in CANDIDATE_METHODS if r1_by_method[method_id]["method_pass"]]
    # 没有候选通过时停止概率主线。
    if not qualified:
        # 返回无合格候选决定。
        return {"decision": "no_qualified_candidate", "unique_candidate_selected": False, "selected_candidate": None, "reason_cn": "无合格概率候选；停止当前概率主线并保持最终52户密封"}
    # 恰有一个候选通过时无需比较另一个候选。
    if len(qualified) == 1:
        # 返回唯一候选决定。
        return {"decision": "unique_qualified_candidate", "unique_candidate_selected": True, "selected_candidate": qualified[0], "reason_cn": "只有一个候选通过R1双期限绝对真实值门"}
    # 两候选都通过时必须已有成对自助抽样边界。
    if paired_lower is None or paired_upper is None:
        # 阻断缺少统计区分证据。
        raise C2ER1DecisionError("两候选都通过时缺少成对WIS80自助抽样区间")
    # C2C减C2D上界小于0代表C2C综合WIS80明确更低。
    if paired_upper < 0:
        # 返回C2C唯一胜者。
        return {"decision": "unique_qualified_candidate", "unique_candidate_selected": True, "selected_candidate": CANDIDATE_METHODS[0], "reason_cn": "C2C减C2D的双期限综合WIS80成对95%区间完全小于0"}
    # C2C减C2D下界大于0代表C2D综合WIS80明确更低。
    if paired_lower > 0:
        # 返回C2D唯一胜者。
        return {"decision": "unique_qualified_candidate", "unique_candidate_selected": True, "selected_candidate": CANDIDATE_METHODS[1], "reason_cn": "C2C减C2D的双期限综合WIS80成对95%区间完全大于0"}
    # 区间包含0时统计上没有明确胜者，不能强选任一方法。
    return {"decision": "no_clear_winner", "unique_candidate_selected": False, "selected_candidate": None, "reason_cn": "两候选都通过但成对企业整块95%差异区间包含0；无明确胜者并保持最终52户密封"}


# 从既有CSV读取所有数据行；字段名由CSV首行固定，空文件必须阻断。
def read_csv_rows(path: str) -> list[dict[str, str]]:
    # 以UTF-8编码打开历史结果，不写入或修改任何既有文件。
    with open(path, "r", encoding="utf-8", newline="") as stream:
        # 用字典阅读器保留明确字段名，方便后续逐字段对齐。
        reader = csv.DictReader(stream)
        # 无表头不能证明字段口径，直接拒绝。
        if reader.fieldnames is None:
            # 抛出明确的输入格式错误。
            raise C2ER1DecisionError(f"CSV缺少表头：{path}")
        # 将生成器一次性收集为普通字典列表，之后不再持有文件句柄。
        rows = [dict(row) for row in reader]
    # 空结果不具备比较价值。
    if not rows:
        # 阻断空输入。
        raise C2ER1DecisionError(f"CSV没有数据行：{path}")
    # 返回原始字段文本，金额转换留给统一解析函数。
    return rows


# 从一条既有结果行读取共同键，并明确拒绝缺失字段。
def source_key(row: dict[str, str]) -> tuple[str, str, int, str, str, int]:
    # 固定六字段顺序与冻结合同完全一致。
    required = ("fold_id", "fold_role", "training_seed", "sample_key", "enterprise_id", "average_horizon_business_days")
    # 任意字段为空都会让跨方法对齐失去含义。
    if any(not row.get(name) for name in required):
        # 显示缺失的关键字段，便于定位历史文件问题。
        raise C2ER1DecisionError(f"源CSV共同键字段缺失：{required}")
    # 将数字字段转为整数，避免字符串前导零造成假不一致。
    return (row["fold_id"], row["fold_role"], int(row["training_seed"]), row["sample_key"], row["enterprise_id"], int(row["average_horizon_business_days"]))


# 把一个来源CSV按validation共同键建立索引，并阻断同键重复。
def validation_index(rows: list[dict[str, str]], source_name: str) -> dict[tuple[str, str, int, str, str, int], dict[str, str]]:
    # 初始化不可重复的验证行索引。
    indexed: dict[tuple[str, str, int, str, str, int], dict[str, str]] = {}
    # 逐行筛选合同允许的validation角色。
    for row in rows:
        # 取得标准化共同键。
        key = source_key(row)
        # calibration行不参与C2E-R1的公平比较。
        if key[1] != "validation":
            # 跳过已在历史阶段合法使用的校准行。
            continue
        # 验证行同键重复会重复计权，必须停止。
        if key in indexed:
            # 指出具体来源和键。
            raise C2ER1DecisionError(f"{source_name}存在重复validation共同键：{key}")
        # 保存这一条有效验证行。
        indexed[key] = row
    # 没有任何验证行说明阶段输出不完整。
    if not indexed:
        # 阻断错误来源。
        raise C2ER1DecisionError(f"{source_name}没有validation记录")
    # 返回按键查询的历史结果。
    return indexed


# 从来源行读取指定三分位数，并立即检查其概率顺序。
def quantiles_from_row(row: dict[str, str], lower_field: str, point_field: str, upper_field: str, source_name: str) -> dict[str, Decimal]:
    # 三个字段都必须存在且不为空。
    if any(not row.get(name) for name in (lower_field, point_field, upper_field)):
        # 阻断不完整概率输出。
        raise C2ER1DecisionError(f"{source_name}缺少P10/P50/P90字段")
    # 逐字段转换为十进制金额。
    result = {"p10": as_decimal(row[lower_field]), "p50": as_decimal(row[point_field]), "p90": as_decimal(row[upper_field])}
    # 复用行指标函数对顺序和正尺度以外的概率条件执行校验。
    if result["p10"] > result["p50"] or result["p50"] > result["p90"]:
        # 阻断交叉分位数。
        raise C2ER1DecisionError(f"{source_name}的P10/P50/P90顺序错误")
    # 返回可供统一评分使用的数值预测。
    return result


# 核验同一验证键在三份既有结果中的真实日均余额和账户尺度精确一致。
def assert_common_actual_and_scale(key: tuple[str, str, int, str, str, int], baseline_row: dict[str, str], candidate_row: dict[str, str], source_name: str) -> None:
    # 逐一检查冻结的真实标签和账户尺度字段。
    for field_name in ("actual_average_balance_cny", "balance_scale_cny"):
        # 两来源金额必须以十进制数值精确相等，禁止只比较格式。
        if as_decimal(baseline_row[field_name]) != as_decimal(candidate_row[field_name]):
            # 明示出现问题的来源、字段和共同键。
            raise C2ER1DecisionError(f"{source_name}与C2B的{field_name}不一致：{key}")


# 将C2B、C2C、C2D三份既有文件合成完全同键的只读共同验证记录。
def build_common_records(baseline_rows: list[dict[str, str]], c2c_rows: list[dict[str, str]], c2d_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    # 建立C2B验证索引。
    baseline_index = validation_index(baseline_rows, "C2B")
    # 建立C2C验证索引。
    c2c_index = validation_index(c2c_rows, "C2C")
    # 建立C2D验证索引。
    c2d_index = validation_index(c2d_rows, "C2D")
    # 三份输入必须有完全一致的共同键集合。
    if set(baseline_index) != set(c2c_index) or set(baseline_index) != set(c2d_index):
        # 阻断任何少行、多行或角色错位，不尝试取交集掩盖问题。
        raise C2ER1DecisionError("C2B、C2C、C2D的validation共同键集合不完全一致")
    # 初始化统一后的内部记录。
    combined: list[dict[str, Any]] = []
    # 稳定排序逐键构造记录，保证输出哈希可复现。
    for key in sorted(baseline_index):
        # 读取余额保持法来源行。
        baseline_row = baseline_index[key]
        # 读取锁定LSTM来源行。
        c2c_row = c2c_index[key]
        # 读取分位数LSTM来源行。
        c2d_row = c2d_index[key]
        # 核验C2C标签与尺度没有漂移。
        assert_common_actual_and_scale(key, baseline_row, c2c_row, "C2C")
        # 核验C2D标签与尺度没有漂移。
        assert_common_actual_and_scale(key, baseline_row, c2d_row, "C2D")
        # 读取C2B的基线概率输出。
        baseline_quantiles = quantiles_from_row(baseline_row, "p10_lower_bound_cny", "p50_point_estimate_cny", "p90_upper_bound_cny", "C2B")
        # 读取C2C候选概率输出。
        c2c_quantiles = quantiles_from_row(c2c_row, "candidate_p10_lower_bound_cny", "candidate_p50_point_estimate_cny", "candidate_p90_upper_bound_cny", "C2C")
        # 读取C2D候选概率输出。
        c2d_quantiles = quantiles_from_row(c2d_row, "candidate_p10_lower_bound_cny", "candidate_p50_point_estimate_cny", "candidate_p90_upper_bound_cny", "C2D")
        # C2C中内嵌的基线必须仍等于独立C2B，防止“同名基线”被替换。
        if baseline_quantiles != quantiles_from_row(c2c_row, "baseline_p10_lower_bound_cny", "baseline_p50_point_estimate_cny", "baseline_p90_upper_bound_cny", "C2C内嵌基线"):
            # 阻断基线预测不一致。
            raise C2ER1DecisionError(f"C2C内嵌基线与C2B不一致：{key}")
        # C2D中内嵌的基线也必须等于独立C2B。
        if baseline_quantiles != quantiles_from_row(c2d_row, "baseline_p10_lower_bound_cny", "baseline_p50_point_estimate_cny", "baseline_p90_upper_bound_cny", "C2D内嵌基线"):
            # 阻断基线预测不一致。
            raise C2ER1DecisionError(f"C2D内嵌基线与C2B不一致：{key}")
        # 写入只含已存在标签、尺度和三种预测的统一记录。
        combined.append({"fold_id": key[0], "fold_role": key[1], "training_seed": key[2], "sample_key": key[3], "enterprise_id": key[4], "average_horizon_business_days": key[5], "balance_scale_cny": as_decimal(baseline_row["balance_scale_cny"]), "actual_average_balance_cny": as_decimal(baseline_row["actual_average_balance_cny"]), "methods": {BASELINE_METHOD: baseline_quantiles, CANDIDATE_METHODS[0]: c2c_quantiles, CANDIDATE_METHODS[1]: c2d_quantiles}})
    # 进行与来源无关的完整性复核。
    validate_common_records(combined)
    # 返回供评分和输出使用的统一记录。
    return combined


# 把余额保持法也写成共同验证行，方便公开报告完整展示三个方法。
def build_baseline_common_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    # 初始化余额保持法公开行列表。
    rows: list[dict[str, str]] = []
    # 逐条生成与候选同字段结构的基线行。
    for record in records:
        # 读取真实日均余额和账户尺度。
        actual = as_decimal(record["actual_average_balance_cny"])
        # 读取账户尺度。
        scale = as_decimal(record["balance_scale_cny"])
        # 读取余额保持法三分位数。
        baseline = record["methods"][BASELINE_METHOD]
        # 重新按统一公式计算基线行指标。
        metrics = calculate_row_metrics(actual, scale, baseline["p10"], baseline["p50"], baseline["p90"])
        # 写出可与候选逐行对照的基线记录。
        rows.append({"fold_id": str(record["fold_id"]), "fold_role": str(record["fold_role"]), "training_seed": str(record["training_seed"]), "sample_key": str(record["sample_key"]), "enterprise_id": str(record["enterprise_id"]), "average_horizon_business_days": str(record["average_horizon_business_days"]), "balance_scale_cny": f"{scale:.2f}", "actual_average_balance_cny": f"{actual:.2f}", "interval_method_id": BASELINE_METHOD, "p10_lower_bound_cny": f"{baseline['p10']:.2f}", "p50_point_estimate_cny": f"{baseline['p50']:.2f}", "p90_upper_bound_cny": f"{baseline['p90']:.2f}", "normalized_p50_absolute_error": decimal_text(metrics["normalized_p50_absolute_error"]), "normalized_interval_score": decimal_text(metrics["normalized_interval_score"]), "normalized_wis80": decimal_text(metrics["normalized_wis80"]), "covered_by_interval": str(metrics["covered"]).lower(), "normalized_interval_width": decimal_text(metrics["normalized_width"])})
    # 返回全部基线公开行。
    return rows
