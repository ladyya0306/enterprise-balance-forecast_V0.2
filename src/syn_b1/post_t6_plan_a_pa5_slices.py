# 模块说明：本文件按真实未来余额给既有预测起点分组，并固定公平案例排序。
"""Plan A PA5不同情形统计和固定案例选择共同实现。"""

# 导入哈希标准库；固定案例顺序只使用公开且事前固定的文字计算指纹。
import hashlib
# 导入默认字典；它按情形和企业收集已保存指标。
from collections import defaultdict
# 导入中位数；总体统计须按三次训练和四个时间检查的固定顺序汇总。
from statistics import median
# 导入类型标注；让表格行的键和值更清楚。
from typing import Any

# 导入Plan A专用输入错误；分组事实不完整必须停止。
from syn_b1.post_t6_plan_a_common_records import PlanAInputError
# 导入PA2已经冻结的真实路线转折规则；PA5不另写一套转折定义。
from syn_b1.post_t6_plan_a_pa2_one_shot import turning_points

# 固定三种路线期限。
HORIZONS = (5, 10, 30)
# 固定案例盲选文字；任何改动都会改变排序，故不得在运行后变更。
CASE_SALT = "PLAN_A_CASES_20260902_V1"
# 固定四个主要真实余额情形。
BASE_SCENARIOS = ("small_change", "significant_up", "significant_down", "large_change")
# 固定10日和30日才有定义的转折情形。
TURNING_SCENARIOS = ("turning", "no_turning")
# 少于该企业数即标为证据不足。
MIN_ENTERPRISES = 20
# 少于该预测起点数即标为证据不足。
MIN_ORIGINS = 500
# 固定可从PA2已保存结果中整理的路线指标。
METRIC_NAMES = ("path_mae_normalized", "velocity_difference_normalized", "velocity_absolute_difference_normalized", "volatility_difference_normalized", "volatility_absolute_difference_normalized", "high_value_error_normalized", "high_date_error_business_days", "low_value_error_normalized", "low_date_error_business_days", "maximum_drawdown_error_normalized", "maximum_drawdown_start_date_error_business_days", "maximum_drawdown_end_date_error_business_days", "turning_discovery_rate", "turning_false_positive_rate", "turning_matched_date_error_business_days")


# 对一条真实未来余额路线按冻结边界给出可重叠的情形标签。
def classify_actual_route(values: list[float], cutoff: float, scale: float, horizon: int) -> set[str]:
    # 路线长度必须与当前期限完全相同。
    if len(values) != horizon:
        # 阻断不完整或多日路线。
        raise PlanAInputError(f"PA5真实未来路线长度不是{horizon}日")
    # 账户参考大小必须为正，才能按百分比边界分组。
    if scale <= 0:
        # 阻断无定义的相对变化。
        raise PlanAInputError("PA5账户参考大小非正")
    # 计算整段内离截止日余额最远的绝对变化。
    maximum_absolute_change = max(abs(value - cutoff) for value in values)
    # 计算期限最后一日相对截止日的变化。
    end_change = values[-1] - cutoff
    # 初始化当前路线的可重叠标签。
    labels: set[str] = set()
    # 全程最大变化小于2%参考大小时标为变化很小。
    if maximum_absolute_change < 0.02 * scale:
        # 加入变化很小标签。
        labels.add("small_change")
    # 最后一天上涨达到或超过25%参考大小时标为明显上涨。
    if end_change >= 0.25 * scale:
        # 加入明显上涨标签。
        labels.add("significant_up")
    # 最后一天下降达到或超过25%参考大小时标为明显下降。
    if end_change <= -0.25 * scale:
        # 加入明显下降标签。
        labels.add("significant_down")
    # 任意一天绝对变化达到或超过50%参考大小时标为大幅变化。
    if maximum_absolute_change >= 0.50 * scale:
        # 加入大幅变化标签。
        labels.add("large_change")
    # 五日路线没有居中五日平滑转折的定义。
    if horizon == 5:
        # 直接返回四个基础情形。
        return labels
    # 复用PA2的真实路线转折规则；它只读取真实余额，不看模型预测。
    actual_turns = turning_points(values, scale)
    # 有至少一个合格转折时标为转折。
    if actual_turns:
        # 加入转折标签。
        labels.add("turning")
    # 没有合格转折时标为无转折。
    else:
        # 加入无转折标签。
        labels.add("no_turning")
    # 返回可重叠标签集合。
    return labels


# 返回情形的普通中文定义；定义写入结果表，避免读者猜测边界。
def scenario_definition_cn(scenario_id: str, horizon: int) -> str:
    # 基础情形使用统一边界定义。
    definitions = {"small_change": "真实未来每一天相对预测起点余额的绝对变化都小于账户参考大小的2%", "significant_up": "真实未来第%d日余额比预测起点余额高至少账户参考大小的25%%" % horizon, "significant_down": "真实未来第%d日余额比预测起点余额低至少账户参考大小的25%%" % horizon, "large_change": "真实未来任意一天相对预测起点余额的绝对变化达到或超过账户参考大小的50%", "turning": "真实未来余额按PA2既有五日平滑和25%显著性规则存在至少一个转折", "no_turning": "真实未来余额按PA2既有五日平滑和25%显著性规则没有合格转折"}
    # 五日转折固定不适用，不误写成没有转折。
    if horizon == 5 and scenario_id in TURNING_SCENARIOS:
        # 返回固定不适用说明。
        return "不适用：5日路线没有按PA2规则判断转折的条件"
    # 未登记情形不能输出随意文字。
    if scenario_id not in definitions:
        # 阻断错误情形编号。
        raise PlanAInputError(f"PA5未知情形：{scenario_id}")
    # 返回固定中文定义。
    return definitions[scenario_id]


# 计算案例盲选指纹；严格使用UTF-8编码且末尾不添加换行。
def case_hash(enterprise_id: str, cutoff_date: str, fold_id: str) -> str:
    # 按计划书固定顺序拼接唯一案例文字。
    payload = f"{CASE_SALT}|{enterprise_id}|{cutoff_date}|{fold_id}"
    # 使用UTF-8无换行字节计算64位十六进制指纹。
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# 形成单一真实案例键；三次训练的预测不同，但真实案例和展示顺序必须只保留一份。
def case_key(origin: dict[str, Any]) -> tuple[str, str, str, str]:
    # 返回时间检查、企业、样本和预测起点日期。
    return (str(origin["fold_id"]), str(origin["enterprise_id"]), str(origin["sample_key"]), str(origin["cutoff_date"]))


# 统一将CSV数值转换为有限浮点数；坏值必须停止。
def as_number(value: Any, field_name: str) -> float:
    # 尝试转换已保存CSV字段。
    try:
        # 返回浮点数以便计算均值和边界。
        number = float(value)
    # 不能转换时提供字段名。
    except (TypeError, ValueError) as error:
        # 抛出输入证据错误。
        raise PlanAInputError(f"PA5字段不是数值：{field_name}={value}") from error
    # NaN或无穷大都不能用于评价。
    if number != number or number in (float("inf"), float("-inf")):
        # 阻断非有限数。
        raise PlanAInputError(f"PA5字段不是有限数：{field_name}={value}")
    # 返回已核对数值。
    return number


# 按企业、训练和时间检查先平均，再按冻结顺序汇总每个情形的两种方法指标。
def aggregate_slice_metrics(origin_metric_rows: list[dict[str, str]], members: dict[tuple[int, str], set[tuple[str, int, str, str]]]) -> list[dict[str, object]]:
    # 初始化每个情形、方法、企业、时间检查、训练和指标的起点值桶。
    values: dict[tuple[int, str, str, str, int, str, str], list[float]] = defaultdict(list)
    # 逐条扫描PA2已保存起点指标。
    for row in origin_metric_rows:
        # 读取当前期限。
        horizon = int(row["horizon_business_days"])
        # 建立当前预测起点身份。
        origin_identity = (str(row["fold_id"]), int(row["training_seed"]), str(row["enterprise_id"]), str(row["sample_key"]))
        # 每个期限逐情形检查成员资格。
        for scenario_id in BASE_SCENARIOS + TURNING_SCENARIOS:
            # 五日转折类固定不适用，不能收集指标。
            if horizon == 5 and scenario_id in TURNING_SCENARIOS:
                # 跳过不适用情形。
                continue
            # 当前起点不在该真实情形时不进入本组分数。
            if origin_identity not in members.get((horizon, scenario_id), set()):
                # 跳过非成员记录。
                continue
            # 逐项保存可用数值。
            for metric_name in METRIC_NAMES:
                # 读取CSV中对应指标文字。
                raw_value = row[metric_name]
                # 不适用或NaN的项目不强行转换成零。
                if raw_value in {"not_applicable", "nan", ""}:
                    # 跳过当前无定义指标。
                    continue
                # 形成企业等权桶键。
                key = (horizon, scenario_id, str(row["method_id"]), str(row["fold_id"]), int(row["training_seed"]), str(row["enterprise_id"]), metric_name)
                # 加入当前起点的已保存指标。
                values[key].append(as_number(raw_value, metric_name))
    # 初始化已在企业内部平均的快速查询，避免每个总体指标重复扫描全部起点。
    enterprise_means_by_seed: dict[tuple[int, str, str, str, int, str], list[float]] = defaultdict(list)
    # 逐企业指标桶先完成企业内部平均。
    for (horizon, scenario_id, method_id, fold_id, seed, _enterprise, metric_name), items in values.items():
        # 按方法、时间检查、训练和指标保存该企业唯一平均值。
        enterprise_means_by_seed[(horizon, scenario_id, method_id, fold_id, seed, metric_name)].append(sum(items) / len(items))
    # 初始化最终一行同时列出模型、余额保持法和相对变化的表。
    output: list[dict[str, object]] = []
    # 逐期限和情形固定输出。
    for horizon in HORIZONS:
        # 逐基础与转折情形输出。
        for scenario_id in BASE_SCENARIOS + TURNING_SCENARIOS:
            # 五日转折项只写不适用行。
            if horizon == 5 and scenario_id in TURNING_SCENARIOS:
                # 逐指标写明不适用。
                for metric_name in METRIC_NAMES:
                    # 保存不适用行。
                    output.append({"horizon_business_days": horizon, "scenario_id": scenario_id, "metric_name": metric_name, "lstm_value": "not_applicable", "balance_hold_value": "not_applicable", "relative_improvement_vs_balance_hold": "not_applicable", "aggregation_order_cn": "不适用：5日路线不判断转折", "interpretation_cn": "不适用：5日路线不判断转折"})
                # 处理下一情形。
                continue
            # 初始化方法到总体值的映射。
            overall_values: dict[str, float | None] = {}
            # 逐指标分别计算两种方法。
            for metric_name in METRIC_NAMES:
                # 逐方法按固定顺序汇总。
                for method_id in ("lstm", "balance_hold"):
                    # 初始化四个时间检查值。
                    fold_values: list[float] = []
                    # 固定按四折处理。
                    for fold_id in ("A", "B", "C", "D"):
                        # 初始化三次训练的企业等权结果。
                        seed_values: list[float] = []
                        # 固定按三次训练处理。
                        for seed in (2026083102, 2026083103, 2026083104):
                            # 收集每户先平均后的值。
                            enterprise_values = enterprise_means_by_seed[(horizon, scenario_id, method_id, fold_id, seed, metric_name)]
                            # 当前组合有企业时才形成该训练结果。
                            if enterprise_values:
                                # 加入企业等权平均。
                                seed_values.append(sum(enterprise_values) / len(enterprise_values))
                        # 三次训练必须完整才形成当前时间检查结果。
                        if len(seed_values) == 3:
                            # 保存三次训练的中间值。
                            fold_values.append(float(median(seed_values)))
                    # 四个时间检查完整时保存其两个中间值平均。
                    overall_values[f"{method_id}:{metric_name}"] = (sorted(fold_values)[1] + sorted(fold_values)[2]) / 2 if len(fold_values) == 4 else None
                # 读取当前指标两种方法值。
                model = overall_values[f"lstm:{metric_name}"]
                # 读取当前指标余额保持法值。
                baseline = overall_values[f"balance_hold:{metric_name}"]
                # 有方向的差值和转折发现率不适合使用“越小越好”的改善公式。
                no_relative_formula = metric_name in {"velocity_difference_normalized", "volatility_difference_normalized", "turning_discovery_rate"}
                # 其它可比较指标在基线非零时最后计算改善。
                improvement: float | str = "not_applicable" if no_relative_formula or model is None or baseline in (None, 0.0) else 1 - float(model) / float(baseline)
                # 保存当前情形、指标的两方法汇总行。
                output.append({"horizon_business_days": horizon, "scenario_id": scenario_id, "metric_name": metric_name, "lstm_value": "not_applicable" if model is None else model, "balance_hold_value": "not_applicable" if baseline is None else baseline, "relative_improvement_vs_balance_hold": improvement, "aggregation_order_cn": "先同一企业内预测起点平均；再三次训练取中间值；最后四个时间检查取中间值", "interpretation_cn": "带方向差值只并列比较方向和偏差大小，不计算改善百分比" if no_relative_formula else "越小越好，最后计算模型相对余额保持法的改善"})
    # 返回全部情形指标。
    return output
