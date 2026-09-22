# 模块说明：本文件集中实现PA3连续窗口、方向、连续误差和企业等权汇总规则。
"""Plan A PA3每天更新后预测下一日的可手算评价规则。"""

# 从collections导入defaultdict；连续窗口和企业汇总需要按固定身份分组。
from collections import defaultdict
# 从statistics导入mean与median；固定规则先企业平均、再训练中间值、最后时间检查中间值。
from statistics import mean, median

# 导入固定四个时间检查和三次训练起始编号。
from syn_b1.post_t6_plan_a_common_records import PlanAInputError, SELECTED_B2_FOLDS, SELECTED_B2_TRAINING_SEEDS
# 导入四档方向边界和包含边界的方向判断函数。
from syn_b1.post_t6_plan_a_pa2_one_shot import DIRECTION_THRESHOLDS, direction


# 检查滚动评价输入只能是每个连续预测起点的第一日预测。
def require_first_day_records(records: list[dict[str, object]]) -> None:
    # 逐条检查保留的未来日编号。
    for row in records:
        # 任何第二日至第三十日进入滚动记录都必须停止。
        if int(row["forecast_step"]) != 1:
            # 抛出明确错误，禁止把一次性30日路线冒充逐日滚动。
            raise PlanAInputError("PA3滚动记录混入未来第2日至第30日")


# 按企业、时间检查、训练起始状态和连续段分组并稳定排序。
def group_continuous_records(records: list[dict[str, object]]) -> dict[tuple[str, str, int, str], list[dict[str, object]]]:
    # 先确认输入全部是未来第一日。
    require_first_day_records(records)
    # 初始化不会跨身份或日期中断的分组。
    grouped: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    # 逐条放入唯一连续段。
    for row in records:
        # 组合企业、时间检查、训练起始状态和既有连续段编号。
        key = (str(row["enterprise_id"]), str(row["fold_id"]), int(row["training_seed"]), str(row["continuous_segment_id"]))
        # 当前记录只能加入这一组。
        grouped[key].append(row)
    # 逐组按预测截止日排序并检查身份没有混入。
    for key, members in grouped.items():
        # 按截止日排序，保证窗口每天向前移动一次。
        members.sort(key=lambda row: str(row["cutoff_date"]))
        # 组内每条记录必须与分组身份完全相同。
        if any((str(row["enterprise_id"]), str(row["fold_id"]), int(row["training_seed"]), str(row["continuous_segment_id"])) != key for row in members):
            # 防止跨企业、时间检查、训练或连续段拼接。
            raise PlanAInputError("PA3连续段内部身份不一致")
    # 返回严格隔离的连续分组。
    return grouped


# 对一个连续段生成每天向前移动一次的完整固定长度窗口。
def make_windows(grouped: dict[tuple[str, str, int, str], list[dict[str, object]]], width: int) -> list[dict[str, object]]:
    # 窗口长度必须为正数。
    if width <= 0:
        # 拒绝没有实际含义的窗口。
        raise ValueError("连续窗口长度必须为正数")
    # 初始化窗口结果。
    windows: list[dict[str, object]] = []
    # 每个连续段独立处理，绝不跨段补足天数。
    for (_enterprise_id, _fold_id, _seed, segment_id), members in grouped.items():
        # 只有完整达到长度的段才产生窗口。
        for start in range(0, len(members) - width + 1):
            # 截取当前完整连续窗口。
            window = members[start:start + width]
            # 保存可复核的起止日期和两种方法平均距离。
            windows.append({"continuous_segment_id": segment_id, "enterprise_id": window[0]["enterprise_id"], "fold_id": window[0]["fold_id"], "training_seed": window[0]["training_seed"], "window_start_cutoff_date": window[0]["cutoff_date"], "window_end_cutoff_date": window[-1]["cutoff_date"], "window_start_target_date": window[0]["target_date"], "window_end_target_date": window[-1]["target_date"], "window_business_day_count": width, "lstm_mean_normalized_absolute_error": mean(float(row["model_normalized_absolute_error"]) for row in window), "balance_hold_mean_normalized_absolute_error": mean(float(row["balance_hold_normalized_absolute_error"]) for row in window)})
    # 返回全部不跨段的窗口。
    return windows


# 先在企业内部平均，再保留时间检查和训练起始状态层级。
def enterprise_means(records: list[dict[str, object]], model_name: str, baseline_name: str, count_name: str) -> list[dict[str, object]]:
    # 初始化企业分组。
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    # 逐条按时间检查、训练起始状态和企业分组。
    for row in records:
        # 同一企业的全部合格日期或窗口一起平均。
        grouped[(str(row["fold_id"]), int(row["training_seed"]), str(row["enterprise_id"]))].append(row)
    # 初始化企业级结果。
    rows: list[dict[str, object]] = []
    # 逐企业计算两种方法的内部平均距离。
    for (fold_id, seed, enterprise_id), members in sorted(grouped.items()):
        # 每户企业只形成一行，所以记录多不会增加总体权重。
        rows.append({"fold_id": fold_id, "training_seed": seed, "enterprise_id": enterprise_id, count_name: len(members), "lstm_mean_normalized_absolute_error": mean(float(row[model_name]) for row in members), "balance_hold_mean_normalized_absolute_error": mean(float(row[baseline_name]) for row in members)})
    # 返回企业等权汇总前的企业级结果。
    return rows


# 按“企业平均、三次训练中间值、四个时间检查中间值”取得总体数值。
def fixed_overall(enterprise_rows: list[dict[str, object]], value_name: str) -> float | None:
    # 初始化每个时间检查和训练起始状态的企业值。
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    # 逐企业放入对应分组。
    for row in enterprise_rows:
        # 当前企业只贡献一个已经内部平均的数值。
        groups[(str(row["fold_id"]), int(row["training_seed"]))].append(float(row[value_name]))
    # 十二组必须全部有企业才能形成总体结果。
    if any(not groups[(fold_id, seed)] for fold_id in SELECTED_B2_FOLDS for seed in SELECTED_B2_TRAINING_SEEDS):
        # 资料不足时明确不适用。
        return None
    # 初始化四个时间检查结果。
    fold_values: list[float] = []
    # 固定按A到D处理。
    for fold_id in SELECTED_B2_FOLDS:
        # 每次训练先让企业同等重要，再取三次训练的中间值。
        fold_values.append(median(mean(groups[(fold_id, seed)]) for seed in SELECTED_B2_TRAINING_SEEDS))
    # 四个时间检查取排序后中间两个的平均。
    return (sorted(fold_values)[1] + sorted(fold_values)[2]) / 2


# 计算一个连续段内偏高、偏低和四档大误差的最长连续天数。
def segment_longest_runs(members: list[dict[str, object]], prediction_name: str) -> dict[str, int]:
    # 初始化当前连续偏高和偏低天数。
    above = below = 0
    # 初始化最长连续偏高和偏低天数。
    longest_above = longest_below = 0
    # 初始化四档当前连续大误差天数。
    current_errors = {threshold: 0 for threshold in DIRECTION_THRESHOLDS}
    # 初始化四档最长连续大误差天数。
    longest_errors = {threshold: 0 for threshold in DIRECTION_THRESHOLDS}
    # 按连续日期逐日检查。
    for row in members:
        # 读取当前方法预测余额。
        prediction = float(row[prediction_name])
        # 读取真实下一日余额。
        actual = float(row["actual_target_balance_cny"])
        # 读取账户参考大小。
        scale = float(row["balance_scale_cny"])
        # 预测恰好等于真实时偏高和偏低都中断。
        above = above + 1 if prediction > actual else 0
        # 预测低于真实才延续偏低。
        below = below + 1 if prediction < actual else 0
        # 更新最长偏高天数。
        longest_above = max(longest_above, above)
        # 更新最长偏低天数。
        longest_below = max(longest_below, below)
        # 四档边界分别判断连续大误差。
        for threshold in DIRECTION_THRESHOLDS:
            # 达到或超过边界才延续，否则中断。
            current_errors[threshold] = current_errors[threshold] + 1 if abs(prediction - actual) / scale >= threshold else 0
            # 更新当前边界最长天数。
            longest_errors[threshold] = max(longest_errors[threshold], current_errors[threshold])
    # 组合固定文字键返回结果。
    result = {"above_actual": longest_above, "below_actual": longest_below}
    # 加入四档大误差最长天数。
    result.update({f"error_at_least_{int(threshold * 100)}pct": longest_errors[threshold] for threshold in DIRECTION_THRESHOLDS})
    # 返回全部最长连续天数。
    return result


# 累计每日方向三行三列数量，并保存企业等权所需身份。
def daily_direction_counts(records: list[dict[str, object]]) -> dict[tuple[object, ...], int]:
    # 初始化方向格累计数。
    counts: dict[tuple[object, ...], int] = defaultdict(int)
    # 逐日处理两种方法。
    for row in records:
        # 固定LSTM和余额保持法的预测字段。
        for method_id, prediction_name in (("lstm", "predicted_target_balance_cny"), ("balance_hold", "balance_hold_predicted_balance_cny")):
            # 固定处理四档方向边界。
            for threshold in DIRECTION_THRESHOLDS:
                # 真实方向与更新当天最新真实余额比较。
                actual_direction = direction(float(row["actual_target_balance_cny"]), float(row["cutoff_balance_cny"]), float(row["balance_scale_cny"]), threshold)
                # 预测方向使用相同起点和相同边界。
                predicted_direction = direction(float(row[prediction_name]), float(row["cutoff_balance_cny"]), float(row["balance_scale_cny"]), threshold)
                # 按企业、时间检查、训练、方法和方向格累计。
                counts[(str(row["fold_id"]), int(row["training_seed"]), str(row["enterprise_id"]), method_id, threshold, actual_direction, predicted_direction)] += 1
    # 返回方向累计数。
    return counts


# 计算一组企业比例的固定总体结果。
def fixed_rate(values: dict[tuple[str, int, str], float]) -> float | None:
    # 转成与距离汇总相同结构的企业行。
    rows = [{"fold_id": fold_id, "training_seed": seed, "enterprise_id": enterprise_id, "value": value} for (fold_id, seed, enterprise_id), value in values.items()]
    # 没有记录时直接不适用。
    if not rows:
        # 返回空值。
        return None
    # 复用固定企业等权顺序。
    return fixed_overall(rows, "value")


# 形成PA3方向完整九格表、识别比例和两种方法百分点差。
def rolling_direction_summary(counts: dict[tuple[object, ...], int]) -> list[dict[str, object]]:
    # 固定三种方向顺序。
    categories = ("up", "stable", "down")
    # 初始化输出行。
    rows: list[dict[str, object]] = []
    # 保存识别比例供方法比较。
    rates: dict[tuple[float, str, str], float | None] = {}
    # 逐边界和方法形成九格对照。
    for threshold in DIRECTION_THRESHOLDS:
        # 分别处理两种方法。
        for method_id in ("lstm", "balance_hold"):
            # 取得当前方法出现的全部企业身份。
            enterprise_keys = sorted({(str(key[0]), int(key[1]), str(key[2])) for key in counts if key[3] == method_id and float(key[4]) == threshold})
            # 逐真实和预测方向格输出。
            for actual_value in categories:
                # 固定预测方向顺序。
                for predicted_value in categories:
                    # 初始化每户企业当前格占比。
                    shares: dict[tuple[str, int, str], float] = {}
                    # 初始化原始记录数。
                    raw_count = 0
                    # 逐企业计算内部比例。
                    for fold_id, seed, enterprise_id in enterprise_keys:
                        # 计算当前企业全部九格数量。
                        total = sum(counts.get((fold_id, seed, enterprise_id, method_id, threshold, actual_direction, predicted_direction), 0) for actual_direction in categories for predicted_direction in categories)
                        # 读取当前格数量。
                        count = counts.get((fold_id, seed, enterprise_id, method_id, threshold, actual_value, predicted_value), 0)
                        # 累加原始数量。
                        raw_count += count
                        # 企业有记录时保存内部比例。
                        if total:
                            # 当前企业只贡献一个比例。
                            shares[(fold_id, seed, enterprise_id)] = count / total
                    # 汇总企业等权比例。
                    value = fixed_rate(shares)
                    # 保存九格中的当前一格。
                    rows.append({"record_type": "confusion_matrix_cell", "method_id": method_id, "threshold": threshold, "actual_direction": actual_value, "predicted_direction": predicted_value, "raw_record_count": raw_count, "rate_name": "enterprise_equal_share", "rate_value": "not_applicable" if value is None else value, "balance_hold_rate": "", "difference_percentage_points": "", "interpretation_cn": "原始数量说明资料多少；比例让每家企业同等重要"})
            # 计算整体判断相同和三类识别比例。
            for rate_name, actual_filter in (("overall_accuracy", None), ("recall_up", "up"), ("recall_stable", "stable"), ("recall_down", "down")):
                # 初始化企业内部识别比例。
                enterprise_rates: dict[tuple[str, int, str], float] = {}
                # 逐企业计算。
                for fold_id, seed, enterprise_id in enterprise_keys:
                    # 整体使用全部真实方向，分类只使用指定真实方向。
                    actual_values = categories if actual_filter is None else (actual_filter,)
                    # 计算分母。
                    denominator = sum(counts.get((fold_id, seed, enterprise_id, method_id, threshold, actual_direction, predicted_direction), 0) for actual_direction in actual_values for predicted_direction in categories)
                    # 计算判断相同的分子。
                    numerator = sum(counts.get((fold_id, seed, enterprise_id, method_id, threshold, actual_direction, actual_direction), 0) for actual_direction in actual_values)
                    # 没有该类真实记录的企业不进入该类比例。
                    if denominator:
                        # 保存企业内部识别比例。
                        enterprise_rates[(fold_id, seed, enterprise_id)] = numerator / denominator
                # 汇总当前识别比例。
                value = fixed_rate(enterprise_rates)
                # 保存供方法比较。
                rates[(threshold, method_id, rate_name)] = value
                # 写出当前方法比例。
                rows.append({"record_type": "direction_rate", "method_id": method_id, "threshold": threshold, "actual_direction": "", "predicted_direction": "", "raw_record_count": "", "rate_name": rate_name, "rate_value": "not_applicable" if value is None else value, "balance_hold_rate": "", "difference_percentage_points": "", "interpretation_cn": "比例越高越好，不计算相对误差改善"})
        # 当前边界增加两种方法百分点差。
        for rate_name in ("overall_accuracy", "recall_up", "recall_stable", "recall_down"):
            # 读取LSTM比例。
            model_rate = rates[(threshold, "lstm", rate_name)]
            # 读取余额保持法比例。
            baseline_rate = rates[(threshold, "balance_hold", rate_name)]
            # 任一不适用时差值不适用。
            difference = None if model_rate is None or baseline_rate is None else (model_rate - baseline_rate) * 100
            # 写出百分点比较。
            rows.append({"record_type": "percentage_point_comparison", "method_id": "lstm_vs_balance_hold", "threshold": threshold, "actual_direction": "", "predicted_direction": "", "raw_record_count": "", "rate_name": rate_name, "rate_value": "not_applicable" if model_rate is None else model_rate, "balance_hold_rate": "not_applicable" if baseline_rate is None else baseline_rate, "difference_percentage_points": "not_applicable" if difference is None else difference, "interpretation_cn": "正数表示LSTM方向判断比例更高，负数表示余额保持法更高"})
    # 返回完整方向汇总。
    return rows
