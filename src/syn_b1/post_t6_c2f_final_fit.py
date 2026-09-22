# 模块字符串说明本模块只构造208户开发池最终定型所需的时间角色和保形校准，不接触最终52户。
"""POST-T6-C2F最终分位数LSTM定型的时间角色、三种子组合与校准核心。"""

# 从collections导入defaultdict；它按日期角色累计样本键，避免手工初始化字典。
from collections import defaultdict
# 从dataclasses导入dataclass；它用不可变对象保存标签的时间边界证据。
from dataclasses import dataclass
# 从datetime导入date；它以日期对象而非字符串比较合同冻结的季度边界。
from datetime import date
# 从decimal导入Decimal；它精确计算校准残差和人民币区间边界。
from decimal import Decimal
# 从typing导入Iterable；它标注可流式读取的标签行集合。
from typing import Iterable

# 导入numpy；它保存三种子预测矩阵并逐坐标计算中位数。
import numpy as np

# 复用C2B有限样本保形排名和半径，防止最终校准私自另写一套统计规则。
from syn_b1.post_t6_c2b_balance_hold_interval import conformal_radius
# 复用C2D已冻结的两个日均期限常量，确保最终模型不偷换产品目标。
from syn_b1.post_t6_c2d_quantile_lstm import AVERAGE_HORIZONS

# 固定最终参数训练所能使用的最后一个完整目标日期。
FIT_TARGET_END = date(2026, 3, 31)
# 固定早停季度的第一个目标路径日期。
EARLY_STOPPING_TARGET_START = date(2026, 4, 1)
# 固定早停季度的最后一个完整目标路径日期。
EARLY_STOPPING_TARGET_END = date(2026, 6, 30)
# 固定最终保形校准季度的第一个目标路径日期。
CALIBRATION_TARGET_START = date(2026, 7, 1)
# 固定最终保形校准季度的最后一个完整目标路径日期。
CALIBRATION_TARGET_END = date(2026, 9, 30)
# 固定训练、早停和校准三种内部角色名称。
FINAL_FIT_ROLES = ("fit", "early_stopping", "calibration")


# 声明最终定型专用异常，防止时间泄漏被误当作普通训练问题。
class C2FFinalFitError(ValueError):
    # 异常类别只承载清晰错误文字，不额外改变程序状态。
    pass


# 保存一个开发池标签行中与C2F时间隔离有关的最小信息。
@dataclass(frozen=True)
# 定义不可变时间证据类；它把每个开发池样本的完整未来路径日期绑定到唯一键。
class FinalFitTiming:
    # 保存样本键，只用于连接A1标签、序列和模型输出。
    sample_key: str
    # 保存企业编号，只用于证明三段都覆盖208户，不进入模型特征。
    enterprise_id: str
    # 保存未来路径的第一天，验证完整30日路径不会跨入错误季度。
    future_start_date: date
    # 保存未来30日路径的最后一天，作为合同中的目标日期边界。
    target_end_date: date


# 将一条A1开发池标签行解析为最小时间证据，拒绝缺字段或非法日期。
def parse_timing(row: dict[str, str]) -> FinalFitTiming:
    # 逐项要求连接键、企业编号和两项日期存在。
    if any(not row.get(field_name) for field_name in ("sample_key", "enterprise_id", "future_start_date", "target_date_t30")):
        # 阻断无法审计完整路径的标签行。
        raise C2FFinalFitError("C2F标签缺少样本键、企业编号或完整路径日期")
    # 将ISO日期文本转换为可比较的日期对象。
    future_start = date.fromisoformat(row["future_start_date"])
    # 将30日路径终点转换为可比较的日期对象。
    target_end = date.fromisoformat(row["target_date_t30"])
    # 路径终点早于起点代表标签日期自身矛盾。
    if target_end < future_start:
        # 阻断不可能的未来路径。
        raise C2FFinalFitError("C2F标签的30日终点早于未来路径起点")
    # 返回不可变时间证据对象。
    return FinalFitTiming(row["sample_key"], row["enterprise_id"], future_start, target_end)


# 根据完整路径时间边界把开发池样本划为训练、早停、校准或不使用。
def classify_timing(timing: FinalFitTiming) -> str | None:
    # 所有路径终点在第一季度末前的样本可参与参数训练。
    if timing.target_end_date <= FIT_TARGET_END:
        # 返回参数训练角色。
        return "fit"
    # 早停样本必须整条未来路径都落在第二季度内。
    if EARLY_STOPPING_TARGET_START <= timing.future_start_date and timing.target_end_date <= EARLY_STOPPING_TARGET_END:
        # 返回早停角色。
        return "early_stopping"
    # 校准样本必须整条未来路径都落在第三季度内。
    if CALIBRATION_TARGET_START <= timing.future_start_date and timing.target_end_date <= CALIBRATION_TARGET_END:
        # 返回保形校准角色。
        return "calibration"
    # 其它日期例如第四季度不参与C2F任何拟合或校准。
    return None


# 从A1开发池标签行建立C2F三种时间角色，并核验每段均覆盖全部208户。
def build_final_fit_roles(rows: Iterable[dict[str, str]], expected_sample_keys: set[str], expected_enterprise_count: int) -> tuple[dict[str, np.ndarray], dict[str, FinalFitTiming], dict[str, int]]:
    # 初始化样本键到时间证据的映射。
    timings: dict[str, FinalFitTiming] = {}
    # 初始化角色到样本键列表的桶。
    keys_by_role: dict[str, list[str]] = defaultdict(list)
    # 初始化角色到企业编号集合的桶。
    enterprises_by_role: dict[str, set[str]] = defaultdict(set)
    # 初始化未使用日期样本计数。
    excluded_count = 0
    # 流式逐行读取已经由A1隔离的开发池标签表。
    for row in rows:
        # 解析当前行最小日期证据。
        timing = parse_timing(row)
        # 同一开发池标签键不能重复。
        if timing.sample_key in timings:
            # 阻断重复样本被重复计权。
            raise C2FFinalFitError(f"C2F开发池标签样本键重复：{timing.sample_key}")
        # 保存当前时间证据。
        timings[timing.sample_key] = timing
        # 按冻结边界给当前完整路径分类。
        role = classify_timing(timing)
        # 不属于前三季度的开发池样本不参与C2F。
        if role is None:
            # 增加排除样本计数，供报告解释。
            excluded_count += 1
            # 继续读取下一行。
            continue
        # 保存角色样本键。
        keys_by_role[role].append(timing.sample_key)
        # 保存角色中出现的企业编号。
        enterprises_by_role[role].add(timing.enterprise_id)
    # A1标签键必须与先前读取的开发池标签对象完全一致。
    if set(timings) != expected_sample_keys:
        # 阻断少行、额外行或错版本标签表。
        raise C2FFinalFitError("C2F时间标签键与A1开发池标签键不完全一致")
    # 三种角色都必须非空且覆盖全部208户开发企业。
    for role in FINAL_FIT_ROLES:
        # 空角色无法训练、早停或校准。
        if not keys_by_role[role]:
            # 指出缺失角色。
            raise C2FFinalFitError(f"C2F时间角色为空：{role}")
        # 每个角色都要覆盖合同规定的全部开发企业。
        if len(enterprises_by_role[role]) != expected_enterprise_count:
            # 指出企业覆盖数异常的角色。
            raise C2FFinalFitError(f"C2F时间角色企业数不是{expected_enterprise_count}：{role}")
    # 按A1标签原始稳定顺序创建键到数组位置映射。
    positions = {sample_key: index for index, sample_key in enumerate(timings)}
    # 将角色样本键转换为按标签顺序排列的numpy下标数组。
    indices = {role: np.asarray(sorted((positions[key] for key in keys_by_role[role])), dtype=np.int64) for role in FINAL_FIT_ROLES}
    # 三段样本下标必须彼此不重叠，防止早停或校准泄漏进训练。
    if set(indices["fit"]) & set(indices["early_stopping"]) or set(indices["fit"]) & set(indices["calibration"]) or set(indices["early_stopping"]) & set(indices["calibration"]):
        # 阻断角色重叠。
        raise C2FFinalFitError("C2F训练、早停和校准时间角色发生重叠")
    # 汇总各角色样本数、企业数与排除数供报告写入。
    counts = {f"{role}_sample_count": int(len(indices[role])) for role in FINAL_FIT_ROLES}
    # 汇总各角色企业数供报告写入。
    counts.update({f"{role}_enterprise_count": len(enterprises_by_role[role]) for role in FINAL_FIT_ROLES})
    # 保存不参与C2F的后续日期样本数。
    counts["excluded_outside_c2f_time_roles_sample_count"] = excluded_count
    # 返回稳定下标、时间证据和报告计数。
    return indices, timings, counts


# 对三个独立种子同坐标的两期限三分位数预测取中位数。
def coordinatewise_median_quantiles(predictions_by_seed: list[np.ndarray]) -> np.ndarray:
    # C2F固定只允许三个种子预测，不许挑最好或补第四种子。
    if len(predictions_by_seed) != 3:
        # 阻断种子数量漂移。
        raise C2FFinalFitError("C2F必须恰有三个种子预测")
    # 第一个预测定义冻结的输出形状。
    expected_shape = predictions_by_seed[0].shape
    # 每一个种子输出形状必须完全一致。
    if any(prediction.shape != expected_shape for prediction in predictions_by_seed):
        # 阻断样本或输出维度错位。
        raise C2FFinalFitError("C2F三种子预测形状不一致")
    # 输出必须是样本、双期限、三分位数的三维矩阵。
    if len(expected_shape) != 3 or expected_shape[1:] != (len(AVERAGE_HORIZONS), 3):
        # 阻断不是10日和30日P10/P50/P90的输出。
        raise C2FFinalFitError("C2F三种子预测不是双期限三分位数矩阵")
    # 在新增种子轴逐坐标取中位数，不以任何验证表现加权。
    median_predictions = np.median(np.stack(predictions_by_seed, axis=0), axis=0)
    # 中位数后仍必须保持每期限P10不大于P50不大于P90。
    if not np.all(median_predictions[:, :, 0] <= median_predictions[:, :, 1]) or not np.all(median_predictions[:, :, 1] <= median_predictions[:, :, 2]):
        # 阻断三种子组合后出现的量化顺序错误。
        raise C2FFinalFitError("C2F三种子中位数组合出现分位数交叉")
    # 返回稳定的最终原始分位数余额变化。
    return median_predictions


# 从第三季度真实日均余额与三种子中位数原始分位数变化计算两个保形调整量。
def calibration_adjustments(predicted_changes: np.ndarray, cutoff_balances: list[Decimal], actual_average_balances: np.ndarray, account_scales: list[Decimal]) -> tuple[dict[int, tuple[int, Decimal]], list[list[Decimal]]]:
    # 样本数必须在预测、截止余额、真实日均和账户尺度之间完全相同。
    if predicted_changes.shape[0] != len(cutoff_balances) or predicted_changes.shape[0] != actual_average_balances.shape[0] or predicted_changes.shape[0] != len(account_scales):
        # 阻断校准样本错位。
        raise C2FFinalFitError("C2F校准预测、余额、真实值或尺度行数不一致")
    # 初始化期限到有限样本排名和调整量的映射。
    adjustments: dict[int, tuple[int, Decimal]] = {}
    # 初始化期限到每条校准记录归一化非一致性分数的审计表。
    scores_by_horizon: list[list[Decimal]] = []
    # 逐个冻结产品期限计算非一致性分数。
    for horizon_index, horizon in enumerate(AVERAGE_HORIZONS):
        # 初始化当前期限的全部校准分数。
        scores: list[Decimal] = []
        # 逐样本计算“真实值跑到P10/P90外”的账户尺度归一化距离。
        for sample_index in range(predicted_changes.shape[0]):
            # 读取当前预测起点的可见余额。
            cutoff = cutoff_balances[sample_index]
            # 读取当前账户尺度。
            scale = account_scales[sample_index]
            # 尺度必须为正，否则保形距离无定义。
            if scale <= 0:
                # 阻断坏尺度。
                raise C2FFinalFitError("C2F校准账户尺度必须大于0")
            # 读取中位数组合的原始P10余额变化。
            p10_balance = cutoff + Decimal(str(float(predicted_changes[sample_index, horizon_index, 0])))
            # 读取中位数组合的原始P90余额变化。
            p90_balance = cutoff + Decimal(str(float(predicted_changes[sample_index, horizon_index, 2])))
            # 读取同一日均期限的真实余额。
            actual = Decimal(str(float(actual_average_balances[sample_index, horizon_index])))
            # 计算保形非一致性，命中原始区间时为零。
            scores.append(max(p10_balance - actual, actual - p90_balance, Decimal("0")) / scale)
        # 复用C2B相同的有限样本80%排名与半径函数。
        rank, adjustment = conformal_radius(scores)
        # 保存当前期限的固定排名和调整量。
        adjustments[horizon] = (rank, adjustment)
        # 保存全部校准分数供逐行审计输出。
        scores_by_horizon.append(scores)
    # 返回双期限调整与逐样本分数。
    return adjustments, scores_by_horizon


# 将中位数原始余额变化套用第三季度保形调整，转换为最终可交付的日均余额区间。
def apply_calibration_adjustments(predicted_changes: np.ndarray, cutoff_balances: list[Decimal], account_scales: list[Decimal], adjustments: dict[int, tuple[int, Decimal]]) -> np.ndarray:
    # 检查预测与账户信息的样本数完全一致。
    if predicted_changes.shape[0] != len(cutoff_balances) or predicted_changes.shape[0] != len(account_scales):
        # 阻断预测与账户信息错位。
        raise C2FFinalFitError("C2F最终区间预测与账户余额或尺度行数不一致")
    # 创建和预测相同形状的float64最终余额区间矩阵。
    final_balances = np.empty(predicted_changes.shape, dtype=np.float64)
    # 逐样本还原余额并应用每期限固定半径。
    for sample_index in range(predicted_changes.shape[0]):
        # 读取当前样本截止日余额。
        cutoff = cutoff_balances[sample_index]
        # 读取当前样本账户尺度。
        scale = account_scales[sample_index]
        # 逐10日和30日还原三分位数余额。
        for horizon_index, horizon in enumerate(AVERAGE_HORIZONS):
            # 调整映射必须包含当前冻结期限。
            if horizon not in adjustments:
                # 阻断缺少保形参数。
                raise C2FFinalFitError("C2F缺少期限保形调整")
            # 读取当前期限的归一化半径。
            adjustment = adjustments[horizon][1]
            # 还原原始P10余额并向下扩张区间。
            lower = cutoff + Decimal(str(float(predicted_changes[sample_index, horizon_index, 0]))) - scale * adjustment
            # 还原原始P50余额，中心不因保形调整移动。
            point = cutoff + Decimal(str(float(predicted_changes[sample_index, horizon_index, 1])))
            # 还原原始P90余额并向上扩张区间。
            upper = cutoff + Decimal(str(float(predicted_changes[sample_index, horizon_index, 2]))) + scale * adjustment
            # 保形扩张后仍必须保持正确顺序。
            if lower > point or point > upper:
                # 阻断数值异常。
                raise C2FFinalFitError("C2F最终保形区间发生分位数交叉")
            # 写入最终P10余额。
            final_balances[sample_index, horizon_index, 0] = float(lower)
            # 写入最终P50余额。
            final_balances[sample_index, horizon_index, 1] = float(point)
            # 写入最终P90余额。
            final_balances[sample_index, horizon_index, 2] = float(upper)
    # 返回最终模型包使用的双期限余额区间。
    return final_balances
