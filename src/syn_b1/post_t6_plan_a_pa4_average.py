# 模块说明：本文件只整理已经保存的日均余额结果，不训练、不预测也不重新计算概率范围。
"""Plan A PA4日均余额证据的只读整理共同实现。"""

# 导入默认字典；它把相同企业和方法的既有记录放在同一个汇总桶中。
from collections import defaultdict
# 导入中位数；固定汇总顺序要求先取三次训练的中间值和四个时间检查的中间值。
from statistics import median
# 导入类型标注；让输入和输出表格字段更容易复核。
from typing import Any

# 导入Plan A专用错误；来源不一致必须停止，而不是被解释成模型成绩。
from syn_b1.post_t6_plan_a_common_records import PlanAInputError

# 固定三种已经保存的日均预测方法编号；PA4不得增加新方法。
METHODS = (
    # 第一种是余额保持法的已保存概率范围编号。
    "balance_hold_scaled_split_conformal",
    # 第二种是已保存逐日路线模型加范围编号。
    "locked_lstm_scaled_split_conformal",
    # 第三种是已保存直接日均概率模型编号。
    "fixed_lstm_conformalized_quantiles",
    # 结束三方法固定元组。
)
# 固定两个已有日均期限；PA4不处理其它期限。
HORIZONS = (10, 30)
# 固定四个历史时间检查；缺少任何一个都不能形成公平总体数。
FOLDS = ("A", "B", "C", "D")
# 固定三次独立训练起始编号；不能挑选成绩最好的一次。
SEEDS = (2026083102, 2026083103, 2026083104)
# 说明日均表无法回答哪些逐日问题；此文字必须原样保留到结果事实单。
DAILY_NOT_APPLICABLE = "不适用：现有证据只给出未来10日或30日的平均余额，没有每天的预测路线或每天的概率范围。"


# 将CSV文本转成浮点数；输入为空或不合法时立即停止。
def as_number(value: Any, field_name: str) -> float:
    # 尝试转换为数值，避免把字符串直接参与计算。
    try:
        # 使用浮点数保存已写入CSV的日均金额或评分。
        number = float(value)
    # 转换失败说明既有证据无法可靠汇总。
    except (TypeError, ValueError) as error:
        # 抛出带字段名的统一输入错误。
        raise PlanAInputError(f"日均证据字段不是数值：{field_name}={value}") from error
    # 非有限数会让平均和比较失去含义。
    if number != number or number in (float("inf"), float("-inf")):
        # 阻断非有限数值。
        raise PlanAInputError(f"日均证据字段不是有限数：{field_name}={value}")
    # 返回已确认可用的数值。
    return number


# 把各种CSV布尔写法转换为明确真或假；其它文字一律拒绝。
def as_bool(value: Any, field_name: str) -> bool:
    # 统一成无空格的小写文字。
    text = str(value).strip().lower()
    # 识别CSV常见的真值写法。
    if text in {"true", "1"}:
        # 返回真值。
        return True
    # 识别CSV常见的假值写法。
    if text in {"false", "0"}:
        # 返回假值。
        return False
    # 其它写法不能默认为真或假。
    raise PlanAInputError(f"日均证据字段不是布尔值：{field_name}={value}")


# 形成跨阶段共同连接键；同一键必须代表同一企业、预测起点和日均期限。
def common_key(row: dict[str, str]) -> tuple[str, str, int, str, str, int]:
    # 按冻结字段顺序返回不可变键。
    return (str(row["fold_id"]), str(row["fold_role"]), int(row["training_seed"]), str(row["sample_key"]), str(row["enterprise_id"]), int(row["average_horizon_business_days"]))


# 建立一张阶段表的共同事实映射，并检查键不重复、角色和期限不越界。
def build_fact_map(rows: list[dict[str, str]], source_name: str) -> dict[tuple[str, str, int, str, str, int], tuple[float, float]]:
    # 初始化键到真实日均与账户参考大小的映射。
    result: dict[tuple[str, str, int, str, str, int], tuple[float, float]] = {}
    # 逐行核对已保存表。
    for row in rows:
        # 生成当前唯一连接键。
        key = common_key(row)
        # 同一阶段重复键会悄悄增加某个起点的权重。
        if key in result:
            # 阻断重复记录。
            raise PlanAInputError(f"{source_name}共同连接键重复：{key}")
        # 只允许日均任务已登记的两个期限。
        if key[5] not in HORIZONS:
            # 阻断其它期限混入。
            raise PlanAInputError(f"{source_name}出现未登记日均期限：{key}")
        # 读取真实答案；PA4只核对既有答案，不产生新答案。
        actual = as_number(row["actual_average_balance_cny"], f"{source_name}.actual_average_balance_cny")
        # 读取当时已保存的账户参考大小。
        scale = as_number(row["balance_scale_cny"], f"{source_name}.balance_scale_cny")
        # 账户参考大小必须为正。
        if scale <= 0:
            # 阻断不能归一化的记录。
            raise PlanAInputError(f"{source_name}账户参考大小非正：{key}")
        # 保存这一键的两项共同事实。
        result[key] = (actual, scale)
    # 空表没有可整理的证据。
    if not result:
        # 阻断空输入。
        raise PlanAInputError(f"{source_name}没有日均记录")
    # 返回已核对映射。
    return result


# 核对一个阶段的每一条共同事实都同C2A原始配对表一致。
def assert_same_facts(reference: dict[tuple[str, str, int, str, str, int], tuple[float, float]], candidate: dict[tuple[str, str, int, str, str, int], tuple[float, float]], source_name: str) -> None:
    # 键集合不一致意味着某些企业、起点或期限被替换、遗漏或新增。
    if set(reference) != set(candidate):
        # 阻断来源范围不一致。
        raise PlanAInputError(f"{source_name}与C2A的企业、起点或期限集合不一致")
    # 逐键比较真实答案和账户参考大小。
    for key in reference:
        # 读取C2A事实。
        expected_actual, expected_scale = reference[key]
        # 读取当前阶段事实。
        actual, scale = candidate[key]
        # 金额写入CSV以分为单位，必须精确相同。
        if actual != expected_actual or scale != expected_scale:
            # 立即停止，不能把错接记录汇总成分数。
            raise PlanAInputError(f"{source_name}与C2A共同事实不一致：{key}")


# 从C2E统一比较表整理三种方法的企业级日均指标。
def build_enterprise_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    # 初始化方法、折、训练、企业和期限的逐起点指标桶。
    grouped: dict[tuple[str, str, int, str, int], list[dict[str, float | bool]]] = defaultdict(list)
    # 初始化方法、期限和连接键集合，用于阻断重复或缺少方法。
    methods_per_key: dict[tuple[str, str, int, str, str, int], set[str]] = defaultdict(set)
    # 逐条读取统一比较表。
    for row in rows:
        # 取得当前方法编号。
        method_id = str(row["interval_method_id"])
        # 未登记方法不能混入已有三方法比较。
        if method_id not in METHODS:
            # 阻断新增或拼写错误方法。
            raise PlanAInputError(f"C2E出现未登记方法：{method_id}")
        # 形成共同键。
        key = common_key(row)
        # C2E正式比较只能使用validation角色。
        if key[1] != "validation":
            # 阻断校准或学习记录进入开发评价。
            raise PlanAInputError(f"C2E出现非validation记录：{key}")
        # 记录当前连接键已出现的方法。
        methods_per_key[key].add(method_id)
        # 读取账户尺度。
        scale = as_number(row["balance_scale_cny"], "C2E.balance_scale_cny")
        # 读取真实日均余额。
        actual = as_number(row["actual_average_balance_cny"], "C2E.actual_average_balance_cny")
        # 读取中间预测。
        point = as_number(row["p50_point_estimate_cny"], "C2E.p50_point_estimate_cny")
        # 读取表中保存的点预测距离。
        point_error = as_number(row["normalized_p50_absolute_error"], "C2E.normalized_p50_absolute_error")
        # 读取表中保存的概率综合扣分。
        wis80 = as_number(row["normalized_wis80"], "C2E.normalized_wis80")
        # 读取表中保存的覆盖标记。
        covered = as_bool(row["covered_by_interval"], "C2E.covered_by_interval")
        # 读取表中保存的范围宽度。
        width = as_number(row["normalized_interval_width"], "C2E.normalized_interval_width")
        # 读取表中保存的区间扣分。
        interval_score = as_number(row["normalized_interval_score"], "C2E.normalized_interval_score")
        # 以企业为单位建立汇总键。
        group_key = (method_id, key[0], key[2], key[4], key[5])
        # 保存这一预测起点的已保存指标和偏高偏低事实。
        grouped[group_key].append({"point_error": point_error, "wis80": wis80, "covered": covered, "width": width, "interval_score": interval_score, "signed_bias": (point - actual) / scale, "high": point > actual, "low": point < actual, "equal": point == actual})
    # 每个验证连接键必须刚好有三种已登记方法。
    for key, method_set in methods_per_key.items():
        # 方法缺少或多出都会使比较失去公平性。
        if method_set != set(METHODS):
            # 阻断不完整三方比较。
            raise PlanAInputError(f"C2E同一记录的方法集合不完整：{key}")
    # 初始化可写出的企业级汇总行。
    output: list[dict[str, object]] = []
    # 按稳定顺序逐企业汇总。
    for group_key in sorted(grouped):
        # 展开当前企业汇总键。
        method_id, fold_id, seed, enterprise_id, horizon = group_key
        # 取出该企业的全部预测起点。
        members = grouped[group_key]
        # 计算预测起点数。
        count = len(members)
        # 定义一个小函数计算同企业内的平均值。
        def average(name: str) -> float:
            # 返回当前字段在全部起点的普通平均。
            return sum(float(item[name]) for item in members) / count
        # 写出企业等权输入行。
        output.append({"interval_method_id": method_id, "fold_id": fold_id, "training_seed": seed, "enterprise_id": enterprise_id, "average_horizon_business_days": horizon, "forecast_origin_count": count, "enterprise_mean_normalized_p50_absolute_error": average("point_error"), "enterprise_mean_normalized_wis80": average("wis80"), "enterprise_empirical_coverage": average("covered"), "enterprise_mean_normalized_interval_width": average("width"), "enterprise_mean_normalized_interval_score": average("interval_score"), "enterprise_mean_normalized_signed_point_bias": average("signed_bias"), "enterprise_point_high_rate": average("high"), "enterprise_point_low_rate": average("low"), "enterprise_point_equal_rate": average("equal")})
    # 没有企业级行说明统一比较表不完整。
    if not output:
        # 阻断空输出。
        raise PlanAInputError("C2E没有可整理的企业级日均记录")
    # 返回稳定排序后的企业级表。
    return output


# 按固定“企业平均、三次训练中间值、四个时间检查中间值”顺序汇总方法结果。
def build_method_summary(enterprise_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # 固定要汇总的日均指标字段。
    metric_names = ("enterprise_mean_normalized_p50_absolute_error", "enterprise_mean_normalized_wis80", "enterprise_empirical_coverage", "enterprise_mean_normalized_interval_width", "enterprise_mean_normalized_interval_score", "enterprise_mean_normalized_signed_point_bias", "enterprise_point_high_rate", "enterprise_point_low_rate", "enterprise_point_equal_rate")
    # 按方法、期限、时间检查和训练起始编号保存企业平均值。
    by_seed_fold: dict[tuple[str, int, str, int], list[dict[str, object]]] = defaultdict(list)
    # 逐企业行加入对应组合。
    for row in enterprise_rows:
        # 形成当前组合键。
        key = (str(row["interval_method_id"]), int(row["average_horizon_business_days"]), str(row["fold_id"]), int(row["training_seed"]))
        # 保存当前企业结果。
        by_seed_fold[key].append(row)
    # 初始化方法、期限和时间检查的三次训练中间值。
    by_fold: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # 逐个冻结方法、期限、时间检查和训练起始编号取企业等权平均。
    for method_id in METHODS:
        # 固定按10日、30日处理。
        for horizon in HORIZONS:
            # 固定按四个时间检查处理。
            for fold_id in FOLDS:
                # 固定按三次训练处理。
                for seed in SEEDS:
                    # 取得当前组合企业行。
                    members = by_seed_fold.get((method_id, horizon, fold_id, seed), [])
                    # 缺少组合时不能改用其它组合填补。
                    if not members:
                        # 阻断不完整的公平汇总。
                        raise PlanAInputError(f"C2E缺少方法、期限、时间检查或训练组合：{method_id}/{horizon}/{fold_id}/{seed}")
                    # 逐项先在企业之间等权平均。
                    for metric_name in metric_names:
                        # 保存当前训练的企业等权平均。
                        by_fold[(method_id, horizon, fold_id)][metric_name].append(sum(float(item[metric_name]) for item in members) / len(members))
    # 初始化最终方法期限汇总行。
    output: list[dict[str, object]] = []
    # 逐方法、期限固定写出结果。
    for method_id in METHODS:
        # 逐期限处理。
        for horizon in HORIZONS:
            # 初始化每个指标的四个时间检查中间值。
            fold_medians: dict[str, list[float]] = defaultdict(list)
            # 逐时间检查取三次训练的中间值。
            for fold_id in FOLDS:
                # 读取当前时间检查的指标列表。
                values_by_metric = by_fold[(method_id, horizon, fold_id)]
                # 各项均必须正好有三次训练结果。
                if any(len(values_by_metric[name]) != len(SEEDS) for name in metric_names):
                    # 阻断训练结果数量异常。
                    raise PlanAInputError(f"C2E训练结果数量不是三次：{method_id}/{horizon}/{fold_id}")
                # 对每项取三次训练中间值。
                for metric_name in metric_names:
                    # 保存当前时间检查的中间值。
                    fold_medians[metric_name].append(float(median(values_by_metric[metric_name])))
            # 形成当前方法期限的基础行。
            row: dict[str, object] = {"interval_method_id": method_id, "average_horizon_business_days": horizon, "enterprise_equal_aggregation_order_cn": "先同一企业全部预测起点平均；再同一时间检查的三次训练取中间值；最后四个时间检查取中间值", "fold_count": len(FOLDS), "training_seed_count_per_fold": len(SEEDS)}
            # 对每项取四个时间检查的中间值。
            for metric_name in metric_names:
                # 保存最终固定顺序的总体值。
                row[metric_name.replace("enterprise_", "overall_")] = float(median(fold_medians[metric_name]))
            # 加入方法中文名称。
            row["method_name_cn"] = {METHODS[0]: "余额保持法的已保存概率范围", METHODS[1]: "已保存的逐日路线模型加范围", METHODS[2]: "已保存的直接日均概率模型"}[method_id]
            # 加入偏高偏低解释。
            row["bias_interpretation_cn"] = "偏高比例表示中间预测高于真实日均余额的起点占比；偏低比例相反；带方向偏差为预测减真实值后再除以账户参考大小。"
            # 保存当前方法期限行。
            output.append(row)
    # 返回稳定的六行方法摘要。
    return output


# 将C2E中并列保存的原历史判定和后开发判定展开为普通读者可查的表格行。
def build_decision_rows(decision: dict[str, Any]) -> list[dict[str, object]]:
    # 初始化输出行。
    rows: list[dict[str, object]] = []
    # 读取原历史判定；它只记录历史事实，不能被后来的判定覆盖。
    historical = decision.get("original_c1_historical", {})
    # 逐候选方法展开。
    for method_id, method_decision in sorted(historical.items()):
        # 逐期限展开历史结果。
        for item in method_decision.get("horizons", []):
            # 写出原历史规则下的两项通过信息。
            rows.append({"decision_layer_cn": "原历史判定", "interval_method_id": method_id, "average_horizon_business_days": int(item["horizon"]), "point_pass": bool(item["point_pass"]), "interval_pass": bool(item["interval_pass"]), "horizon_pass": bool(item["point_pass"]) and bool(item["interval_pass"]), "method_pass": bool(method_decision["method_pass"]), "decision_explanation_cn": "保留历史判定，不因本次整理而改写。"})
    # 读取结果后开发判定；它是单独一栏，不替代历史栏。
    revised = decision.get("r1_post_result_exploratory", {})
    # 逐候选方法展开。
    for method_id, method_decision in sorted(revised.items()):
        # 逐期限展开后开发结果。
        for item in method_decision.get("horizons", []):
            # 写出后开发规则下的通过信息。
            rows.append({"decision_layer_cn": "结果后开发判定", "interval_method_id": method_id, "average_horizon_business_days": int(item["horizon"]), "point_pass": bool(item["p50_pass"]), "interval_pass": bool(item["wis_pass"]), "horizon_pass": bool(item["horizon_pass"]), "method_pass": bool(method_decision["method_pass"]), "decision_explanation_cn": "这是既有结果后的开发检查，与原历史判定并列保存，不能倒写为历史决定。"})
    # 读取C2E已经保存的唯一候选决定。
    unique = decision.get("unique_candidate_decision", {})
    # 以一条总行保留其选择结果与原因。
    rows.append({"decision_layer_cn": "既有唯一候选记录", "interval_method_id": str(unique.get("selected_candidate", "未选出")), "average_horizon_business_days": "全部", "point_pass": "不适用", "interval_pass": "不适用", "horizon_pass": "不适用", "method_pass": bool(unique.get("unique_candidate_selected", False)), "decision_explanation_cn": str(unique.get("reason_cn", "既有文件未写明原因"))})
    # 返回两栏和一条候选记录。
    return rows
