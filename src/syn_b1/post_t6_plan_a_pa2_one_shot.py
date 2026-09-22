# 模块说明：本文件计算Plan A PA2的一次性未来路线指标，不训练或产生新预测。
"""Plan A PA2一次性未来余额路线评价的可复核指标函数。"""

# 导入math标准库；本模块用它表达“不适用”的非数值结果。
import math
# 从statistics导入mean；本模块按文字规则计算平均距离和平均起伏。
from statistics import mean

# 固定三种一次性未来路线期限。
HORIZONS = (5, 10, 30)
# 固定四档方向观察边界。
DIRECTION_THRESHOLDS = (0.02, 0.05, 0.10, 0.25)


# 将相对截止日变化按固定边界分成上涨、稳定或下降。
def direction(value: float, cutoff: float, scale: float, threshold: float) -> str:
    # 计算相对截止日余额的金额变化。
    change = value - cutoff
    # 变化达到正边界时归为上涨，边界本身包含在上涨侧。
    if change >= threshold * scale:
        # 返回固定中文类别键。
        return "up"
    # 变化达到负边界时归为下降，边界本身包含在下降侧。
    if change <= -threshold * scale:
        # 返回固定中文类别键。
        return "down"
    # 两个边界之间归为稳定。
    return "stable"


# 计算路线首尾平均每日变化速度；短于两日的路线没有定义。
def velocity(values: list[float]) -> float:
    # 少于两日不能形成首尾变化速度。
    if len(values) < 2:
        # 返回非数值代表不适用。
        return math.nan
    # 用最后日减第一日并除以中间经过的银行工作日数。
    return (values[-1] - values[0]) / (len(values) - 1)


# 计算相邻两日余额变化绝对值的平均，表示路线每天起伏大小。
def volatility(values: list[float]) -> float:
    # 少于两日不能形成相邻变化。
    if len(values) < 2:
        # 返回非数值代表不适用。
        return math.nan
    # 逐相邻日期计算绝对变化后求平均。
    return mean(abs(current - previous) for previous, current in zip(values, values[1:]))


# 取得最高或最低余额及其最早日期位置；相同金额固定取最早位置。
def extreme(values: list[float], highest: bool) -> tuple[float, int]:
    # 选择最高或最低的目标金额。
    target = max(values) if highest else min(values)
    # index总是返回第一个同值位置，满足最早日期规则。
    return target, values.index(target)


# 计算路线内最大下降幅度以及开始和结束位置；同值按最早开始、最早结束处理。
def maximum_drawdown(values: list[float]) -> tuple[float, int, int]:
    # 初始化到第一日余额和第一日位置。
    peak_value = values[0]
    # 初始化最高点位置。
    peak_index = 0
    # 初始化尚未出现下降的结果。
    best_drop = 0.0
    # 初始化开始位置。
    best_start = 0
    # 初始化结束位置。
    best_end = 0
    # 逐日扫描，每天只与此前最高余额比较。
    for index, value in enumerate(values):
        # 当前从此前最高点到今日的下降金额。
        drop = peak_value - value
        # 更大下降直接取代旧记录。
        if drop > best_drop:
            # 保存新的最大下降金额。
            best_drop = drop
            # 保存产生下降的最高点位置。
            best_start = peak_index
            # 保存下降终点位置。
            best_end = index
        # 相同下降时优先更早开始，再优先更早结束。
        elif drop == best_drop and (peak_index, index) < (best_start, best_end):
            # 保存按固定同值规则更早的一对日期。
            best_start, best_end = peak_index, index
        # 只有更高余额才更新最高点；相同余额保留更早日期。
        if value > peak_value:
            # 更新此前最高余额。
            peak_value = value
            # 更新最高余额位置。
            peak_index = index
    # 返回下降金额和固定日期位置。
    return best_drop, best_start, best_end


# 计算居中五银行工作日平均余额；最前和最后两日不产生平滑值。
def smooth_five(values: list[float]) -> dict[int, float]:
    # 初始化按原路线位置保存的平滑余额。
    smoothed: dict[int, float] = {}
    # 只遍历前后各至少两日的位置。
    for index in range(2, len(values) - 2):
        # 保存当前位置前后五日的平均余额。
        smoothed[index] = mean(values[index - 2:index + 3])
    # 返回可用于转折判断的平滑余额。
    return smoothed


# 从平滑路线找出满足25%账户尺度显著性要求的上转下或下转上转折。
def turning_points(values: list[float], scale: float) -> list[tuple[int, str]]:
    # 先计算可用的五日平滑余额。
    smoothed = smooth_five(values)
    # 平滑点少于两点无法观察方向变化。
    if len(smoothed) < 2:
        # 返回空列表表示不适用或没有合格转折。
        return []
    # 初始化最近一个非零变化方向。
    previous_direction = 0
    # 初始化最近非零变化所连接的平滑位置。
    previous_index = -1
    # 初始化候选转折列表。
    candidates: list[tuple[int, str]] = []
    # 按平滑日期位置递增观察相邻非零变化。
    for index in sorted(smoothed)[1:]:
        # 取得当前平滑值前一个可用平滑位置。
        prior = max(position for position in smoothed if position < index)
        # 计算当前平滑余额变化。
        change = smoothed[index] - smoothed[prior]
        # 精确为零时不更新方向，等待之后的非零变化。
        if change == 0:
            # 继续处理下一个平滑位置。
            continue
        # 将金额变化转换为正负方向。
        current_direction = 1 if change > 0 else -1
        # 首个非零变化只建立方向基准。
        if previous_direction == 0:
            # 保存当前方向。
            previous_direction = current_direction
            # 保存当前位置。
            previous_index = index
            # 继续下一变化。
            continue
        # 方向改变时，转折日期是两个变化共同连接的此前平滑位置。
        if current_direction != previous_direction:
            # 保存候选转折位置。
            turning_index = prior
            # 上转下和下转上使用不同的显著性检查。
            if previous_direction > 0:
                # 计算转折前平滑最低余额。
                before_low = min(value for position, value in smoothed.items() if position <= turning_index)
                # 计算转折后平滑最低余额。
                after_low = min(value for position, value in smoothed.items() if position >= turning_index)
                # 两侧变化均达到25%账户尺度才登记。
                if smoothed[turning_index] - before_low >= 0.25 * scale and smoothed[turning_index] - after_low >= 0.25 * scale:
                    # 记录上升转下降。
                    candidates.append((turning_index, "up_to_down"))
            # 下转上使用两侧最高余额。
            else:
                # 计算转折前平滑最高余额。
                before_high = max(value for position, value in smoothed.items() if position <= turning_index)
                # 计算转折后平滑最高余额。
                after_high = max(value for position, value in smoothed.items() if position >= turning_index)
                # 两侧变化均达到25%账户尺度才登记。
                if before_high - smoothed[turning_index] >= 0.25 * scale and after_high - smoothed[turning_index] >= 0.25 * scale:
                    # 记录下降转上升。
                    candidates.append((turning_index, "down_to_up"))
        # 当前非零变化成为下一次比较的方向基准。
        previous_direction = current_direction
        # 保存当前平滑位置。
        previous_index = index
    # 返回符合显著性条件的转折。
    return candidates


# 按固定日期差、真实日期、预测日期顺序一一配对预测和真实转折。
def match_turning_points(predicted: list[tuple[int, str]], actual: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    # 初始化所有允许配对候选。
    candidates: list[tuple[int, int, int, str]] = []
    # 逐个预测转折和真实转折寻找同类型且相差不超过3日的候选。
    for predicted_index, predicted_kind in predicted:
        # 遍历真实转折。
        for actual_index, actual_kind in actual:
            # 类型必须相同才可称为同一种转折。
            if predicted_kind == actual_kind and abs(predicted_index - actual_index) <= 3:
                # 按固定排序键保存候选。
                candidates.append((abs(predicted_index - actual_index), actual_index, predicted_index, predicted_kind))
    # 按日期差、真实日期、预测日期稳定排序。
    candidates.sort()
    # 初始化已使用预测转折集合。
    used_predicted: set[int] = set()
    # 初始化已使用真实转折集合。
    used_actual: set[int] = set()
    # 初始化最终一对一配对。
    matches: list[tuple[int, int, str]] = []
    # 依排序顺序贪心选择未使用的候选。
    for _, actual_index, predicted_index, kind in candidates:
        # 预测和真实任一已使用则跳过。
        if predicted_index in used_predicted or actual_index in used_actual:
            # 继续下一个候选。
            continue
        # 记录已使用预测转折。
        used_predicted.add(predicted_index)
        # 记录已使用真实转折。
        used_actual.add(actual_index)
        # 保存一对一匹配。
        matches.append((predicted_index, actual_index, kind))
    # 返回最终配对。
    return matches


# 计算一个预测起点、一种方法和一个期限的路线形状、方向和转折指标。
def origin_metrics(actual: list[float], predicted: list[float], cutoff: float, scale: float) -> dict[str, float | str]:
    # 计算逐日归一化绝对距离并取平均。
    path_mae = mean(abs(prediction - truth) / scale for prediction, truth in zip(predicted, actual))
    # 计算预测和真实路线的平均每日速度。
    predicted_velocity = velocity(predicted)
    # 计算真实路线的平均每日速度。
    actual_velocity = velocity(actual)
    # 计算预测和真实路线的每日起伏大小。
    predicted_volatility = volatility(predicted)
    # 保存真实路线起伏大小。
    actual_volatility = volatility(actual)
    # 取得预测和真实最高余额及位置。
    predicted_high, predicted_high_index = extreme(predicted, True)
    # 取得真实最高余额及位置。
    actual_high, actual_high_index = extreme(actual, True)
    # 取得预测和真实最低余额及位置。
    predicted_low, predicted_low_index = extreme(predicted, False)
    # 取得真实最低余额及位置。
    actual_low, actual_low_index = extreme(actual, False)
    # 取得预测最大下降和两端位置。
    predicted_drawdown, predicted_drawdown_start, predicted_drawdown_end = maximum_drawdown(predicted)
    # 取得真实最大下降和两端位置。
    actual_drawdown, actual_drawdown_start, actual_drawdown_end = maximum_drawdown(actual)
    # 找出预测与真实的显著转折。
    predicted_turns = turning_points(predicted, scale)
    # 找出真实转折。
    actual_turns = turning_points(actual, scale)
    # 按冻结规则一对一配对。
    matches = match_turning_points(predicted_turns, actual_turns)
    # 真实没有合格转折时发现率无定义。
    discovery_rate = math.nan if not actual_turns else len(matches) / len(actual_turns)
    # 预测没有合格转折时误报率固定为0。
    false_positive_rate = 0.0 if not predicted_turns else (len(predicted_turns) - len(matches)) / len(predicted_turns)
    # 没有配对时日期误差无定义。
    matched_date_error = math.nan if not matches else mean(abs(predicted_index - actual_index) for predicted_index, actual_index, _ in matches)
    # 汇总所有可直接按文字手算的路线指标。
    return {"path_mae_normalized": path_mae, "velocity_difference_normalized": (predicted_velocity - actual_velocity) / scale, "velocity_absolute_difference_normalized": abs(predicted_velocity - actual_velocity) / scale, "volatility_difference_normalized": (predicted_volatility - actual_volatility) / scale, "volatility_absolute_difference_normalized": abs(predicted_volatility - actual_volatility) / scale, "high_value_error_normalized": abs(predicted_high - actual_high) / scale, "high_date_error_business_days": abs(predicted_high_index - actual_high_index), "low_value_error_normalized": abs(predicted_low - actual_low) / scale, "low_date_error_business_days": abs(predicted_low_index - actual_low_index), "maximum_drawdown_error_normalized": abs(predicted_drawdown - actual_drawdown) / scale, "maximum_drawdown_start_date_error_business_days": abs(predicted_drawdown_start - actual_drawdown_start), "maximum_drawdown_end_date_error_business_days": abs(predicted_drawdown_end - actual_drawdown_end), "actual_turn_count": len(actual_turns), "predicted_turn_count": len(predicted_turns), "matched_turn_count": len(matches), "turning_discovery_rate": discovery_rate, "turning_false_positive_rate": false_positive_rate, "turning_matched_date_error_business_days": matched_date_error}
