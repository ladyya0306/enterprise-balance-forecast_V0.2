# 使用文档字符串说明当前模块或对象的职责，便于教学阅读。
"""v4 冻结评分工具：只接收内存数据，不读文件、不训练。"""

# 导入本文件后续计算所需的模块、类型或既有评分工具。
from __future__ import annotations

# 导入本文件后续计算所需的模块、类型或既有评分工具。
from collections import defaultdict
# 导入本文件后续计算所需的模块、类型或既有评分工具。
from decimal import Decimal
# 导入本文件后续计算所需的模块、类型或既有评分工具。
from typing import Any, Literal

# 导入本文件后续计算所需的模块、类型或既有评分工具。
import numpy as np

# 导入本文件后续计算所需的模块、类型或既有评分工具。
from syn_b1.post_t6_c2b_balance_hold_interval import conformal_radius


# 计算并保存 Kind，供当前评分步骤的后续校验或汇总使用。
Kind = Literal["point", "quantile"]
# 计算并保存 _HORIZONS，供当前评分步骤的后续校验或汇总使用。
_HORIZONS = (10, 30)
# 计算并保存 _REQUIRED_DATA，供当前评分步骤的后续校验或汇总使用。
_REQUIRED_DATA = ("Y_cny", "base", "scale", "enterprise_id", "origin_date", "bank_index", "shape", "future_cny")


# 定义评分相关的异常或测试类，集中组织同类行为。
class ScoringInputError(ValueError):
    # 使用文档字符串说明当前模块或对象的职责，便于教学阅读。
    """Raised when input data would make a score incomplete or ambiguous."""


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    # 计算并保存 result，供当前评分步骤的后续校验或汇总使用。
    result = np.asarray(value)
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if ndim is not None and result.ndim != ndim:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError(f"{name} must have {ndim} dimensions; got {result.ndim}")
    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return result


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def _numeric(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    # 尝试执行可能因输入结构失败的转换，并在异常分支给出明确错误。
    try:
        # 计算并保存 result，供当前评分步骤的后续校验或汇总使用。
        result = np.asarray(value, dtype=np.float64)
    # 捕获预期的输入异常，并转成评分模块统一的报错类型。
    except (TypeError, ValueError) as exc:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError(f"{name} must be numeric") from exc
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if shape is not None and result.shape != shape:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError(f"{name} must have shape {shape}; got {result.shape}")
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if not np.isfinite(result).all():
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError(f"{name} contains NaN or infinity")
    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return result


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def _validated_data(data: dict[str, Any]) -> dict[str, np.ndarray]:
    # 计算并保存 missing，供当前评分步骤的后续校验或汇总使用。
    missing = [key for key in _REQUIRED_DATA if key not in data]
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if missing:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError(f"data missing required keys: {missing}")
    # 计算并保存 y，供当前评分步骤的后续校验或汇总使用。
    y = _numeric(data["Y_cny"], "Y_cny")
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if y.ndim != 2 or y.shape[1] != 5 or y.shape[0] == 0:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("Y_cny must be non-empty n×5 in e1/e10/e30/mean10/mean30 order")
    # 计算并保存 n，供当前评分步骤的后续校验或汇总使用。
    n = y.shape[0]
    # 计算并保存 base，供当前评分步骤的后续校验或汇总使用。
    base = _numeric(data["base"], "base", (n,))
    # 计算并保存 scale，供当前评分步骤的后续校验或汇总使用。
    scale = _numeric(data["scale"], "scale", (n,))
    # 计算并保存 expected_scale，供当前评分步骤的后续校验或汇总使用。
    expected_scale = np.maximum(np.abs(base), 1_000_000.0)
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if not np.allclose(scale, expected_scale, rtol=0.0, atol=0.0):
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("scale must equal max(abs(base), 1e6) for every row")
    # 计算并保存 future，供当前评分步骤的后续校验或汇总使用。
    future = _numeric(data["future_cny"], "future_cny", (n, 30))
    # 计算并保存 expected_y，供当前评分步骤的后续校验或汇总使用。
    expected_y = np.column_stack((future[:, 0], future[:, 9], future[:, 29], future[:, :10].mean(axis=1), future.mean(axis=1)))
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if not np.allclose(y, expected_y, rtol=0.0, atol=1e-6):
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("Y_cny does not agree with complete future_cny route")

    # 执行当前表达式以完成本行评分计算或测试准备。
    text_fields: dict[str, np.ndarray] = {}
    # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
    for name in ("enterprise_id", "origin_date", "shape"):
        # 计算并保存 values，供当前评分步骤的后续校验或汇总使用。
        values = _array(data[name], name, 1)
        # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
        if len(values) != n or any(not str(item).strip() for item in values):
            # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
            raise ScoringInputError(f"{name} must contain one non-empty value per row")
        # 计算并保存 text_fields[name]，供当前评分步骤的后续校验或汇总使用。
        text_fields[name] = values.astype(str)
    # 计算并保存 bank_index，供当前评分步骤的后续校验或汇总使用。
    bank_index = _numeric(data["bank_index"], "bank_index", (n,))
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if not np.all(np.equal(bank_index, np.floor(bank_index))):
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("bank_index must be integral")
    # 计算并保存 keys，供当前评分步骤的后续校验或汇总使用。
    keys = list(zip(text_fields["enterprise_id"], text_fields["origin_date"], bank_index.astype(np.int64), strict=True))
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if len(set(keys)) != n:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("enterprise_id/origin_date/bank_index contains duplicate score keys")
    # 执行当前表达式以完成本行评分计算或测试准备。
    shapes_by_enterprise: dict[str, set[str]] = defaultdict(set)
    # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
    for enterprise, shape in zip(text_fields["enterprise_id"], text_fields["shape"], strict=True):
        # 执行当前表达式以完成本行评分计算或测试准备。
        shapes_by_enterprise[enterprise].add(shape)
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if any(len(shapes) != 1 for shapes in shapes_by_enterprise.values()):
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("each enterprise must have exactly one shape")
    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return {"Y_cny": y, "base": base, "scale": scale, "future_cny": future, "bank_index": bank_index, **text_fields}


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def _prediction_scale(data: dict[str, Any]) -> tuple[int, np.ndarray]:
    # 使用文档字符串说明当前模块或对象的职责，便于教学阅读。
    """构造预测区间时只核公开起点余额和尺度，绝不要求未来标签。"""
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if "base" not in data or "scale" not in data:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("interval construction requires base and scale")
    # 计算并保存 base，供当前评分步骤的后续校验或汇总使用。
    base = _numeric(data["base"], "base")
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if base.ndim != 1 or len(base) == 0:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("base must be a non-empty vector")
    # 计算并保存 scale，供当前评分步骤的后续校验或汇总使用。
    scale = _numeric(data["scale"], "scale", (len(base),))
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if not np.allclose(scale, np.maximum(np.abs(base), 1_000_000.0), rtol=0.0, atol=0.0):
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("scale must equal max(abs(base), 1e6) for every row")
    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return len(base), scale


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def _validate_kind(kind: str) -> Kind:
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if kind not in ("point", "quantile"):
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("kind must be 'point' or 'quantile'")
    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return kind  # type: ignore[return-value]


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def calibrate(pred_cal: np.ndarray, cal: dict[str, Any], kind: Kind) -> dict[str, Any]:
    # 使用文档字符串说明当前模块或对象的职责，便于教学阅读。
    """从仅含70户校准集的标签计算双期限80%有限样本半径。

    ``point`` accepts n×5 centers and calibrates columns 3:5.  ``quantile``
    accepts n×2×3 absolute-CNY P10/P50/P90 values.  The returned radii are
    normalized by each row's supplied account scale.
    """
    # 计算并保存 kind，供当前评分步骤的后续校验或汇总使用。
    kind = _validate_kind(kind)
    # 计算并保存 values，供当前评分步骤的后续校验或汇总使用。
    values = _validated_data(cal)
    # 计算并保存 n，供当前评分步骤的后续校验或汇总使用。
    n = values["Y_cny"].shape[0]
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if kind == "point":
        # 计算并保存 predictions，供当前评分步骤的后续校验或汇总使用。
        predictions = _numeric(pred_cal, "pred_cal", (n, 5))[:, 3:5]
        # 计算并保存 residual_matrix，供当前评分步骤的后续校验或汇总使用。
        residual_matrix = np.abs(predictions - values["Y_cny"][:, 3:5]) / values["scale"][:, None]
    # 处理前面条件不成立的其余情况，保持两类评分路线互斥。
    else:
        # 计算并保存 predictions，供当前评分步骤的后续校验或汇总使用。
        predictions = _numeric(pred_cal, "pred_cal", (n, 2, 3))
        # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
        if np.any(predictions[:, :, 0] > predictions[:, :, 1]) or np.any(predictions[:, :, 1] > predictions[:, :, 2]):
            # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
            raise ScoringInputError("quantile pred_cal must satisfy P10 <= P50 <= P90")
        # 计算并保存 actual，供当前评分步骤的后续校验或汇总使用。
        actual = values["Y_cny"][:, 3:5]
        # 计算并保存 residual_matrix，供当前评分步骤的后续校验或汇总使用。
        residual_matrix = np.maximum.reduce((predictions[:, :, 0] - actual, actual - predictions[:, :, 2], np.zeros((n, 2)))) / values["scale"][:, None]

    # 执行当前表达式以完成本行评分计算或测试准备。
    radii: dict[str, float] = {}
    # 执行当前表达式以完成本行评分计算或测试准备。
    ranks: dict[str, int] = {}
    # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
    for column, horizon in enumerate(_HORIZONS):
        # 执行当前表达式以完成本行评分计算或测试准备。
        rank, radius = conformal_radius([Decimal(str(item)) for item in residual_matrix[:, column]])
        # 计算并保存 radii[str(horizon)]，供当前评分步骤的后续校验或汇总使用。
        radii[str(horizon)] = float(radius)
        # 计算并保存 ranks[str(horizon)]，供当前评分步骤的后续校验或汇总使用。
        ranks[str(horizon)] = rank
    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return {"kind": kind, "coverage_target": 0.80, "finite_sample_rank": ranks, "normalized_radius": radii, "calibration_row_count": n}


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def intervalize(pred: np.ndarray, data: dict[str, Any], calibration: dict[str, Any], kind: Kind) -> np.ndarray:
    # 使用文档字符串说明当前模块或对象的职责，便于教学阅读。
    """构造绝对CNY的 n×2×3 区间；不读取或核验未来标签。"""
    # 计算并保存 kind，供当前评分步骤的后续校验或汇总使用。
    kind = _validate_kind(kind)
    # 执行当前表达式以完成本行评分计算或测试准备。
    n, scale = _prediction_scale(data)
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if calibration.get("kind") != kind:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("calibration kind does not match intervalize kind")
    # 尝试执行可能因输入结构失败的转换，并在异常分支给出明确错误。
    try:
        # 计算并保存 radii，供当前评分步骤的后续校验或汇总使用。
        radii = np.asarray([float(calibration["normalized_radius"][str(horizon)]) for horizon in _HORIZONS], dtype=np.float64)
    # 捕获预期的输入异常，并转成评分模块统一的报错类型。
    except (KeyError, TypeError, ValueError) as exc:
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("calibration lacks normalized radii for 10 and 30") from exc
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if not np.isfinite(radii).all() or np.any(radii < 0):
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("calibration radii must be finite and non-negative")
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if kind == "point":
        # 计算并保存 centers，供当前评分步骤的后续校验或汇总使用。
        centers = _numeric(pred, "pred", (n, 5))[:, 3:5]
        # 计算并保存 extension，供当前评分步骤的后续校验或汇总使用。
        extension = scale[:, None] * radii[None, :]
        # 计算并保存 result，供当前评分步骤的后续校验或汇总使用。
        result = np.stack((centers - extension, centers, centers + extension), axis=2)
    # 处理前面条件不成立的其余情况，保持两类评分路线互斥。
    else:
        # 计算并保存 quantiles，供当前评分步骤的后续校验或汇总使用。
        quantiles = _numeric(pred, "pred", (n, 2, 3))
        # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
        if np.any(quantiles[:, :, 0] > quantiles[:, :, 1]) or np.any(quantiles[:, :, 1] > quantiles[:, :, 2]):
            # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
            raise ScoringInputError("quantile pred must satisfy P10 <= P50 <= P90")
        # 计算并保存 result，供当前评分步骤的后续校验或汇总使用。
        result = quantiles.copy()
        # 计算并保存 extension，供当前评分步骤的后续校验或汇总使用。
        extension = scale[:, None] * radii[None, :]
        # 计算并保存 result[:, :, 0]，供当前评分步骤的后续校验或汇总使用。
        result[:, :, 0] -= extension
        # 计算并保存 result[:, :, 2]，供当前评分步骤的后续校验或汇总使用。
        result[:, :, 2] += extension
    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return result


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def _metric_rows(intervals: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # 计算并保存 actual，供当前评分步骤的后续校验或汇总使用。
    actual = values["Y_cny"][:, 3:5]
    # 执行当前表达式以完成本行评分计算或测试准备。
    lower, center, upper = intervals[:, :, 0], intervals[:, :, 1], intervals[:, :, 2]
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if np.any(lower > center) or np.any(center > upper):
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("intervals must satisfy P10 <= P50 <= P90")
    # 计算并保存 scale，供当前评分步骤的后续校验或汇总使用。
    scale = values["scale"][:, None]
    # 计算并保存 abs_error，供当前评分步骤的后续校验或汇总使用。
    abs_error = np.abs(actual - center)
    # 计算并保存 norm_error，供当前评分步骤的后续校验或汇总使用。
    norm_error = abs_error / scale
    # 计算并保存 width，供当前评分步骤的后续校验或汇总使用。
    width = (upper - lower) / scale
    # 计算并保存 low_miss，供当前评分步骤的后续校验或汇总使用。
    low_miss = np.maximum(lower - actual, 0.0) / scale
    # 计算并保存 high_miss，供当前评分步骤的后续校验或汇总使用。
    high_miss = np.maximum(actual - upper, 0.0) / scale
    # 计算并保存 interval_score，供当前评分步骤的后续校验或汇总使用。
    interval_score = width + 10.0 * low_miss + 10.0 * high_miss
    # 计算并保存 wis80，供当前评分步骤的后续校验或汇总使用。
    wis80 = (0.5 * norm_error + 0.1 * interval_score) / 1.5
    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return {"coverage": ((lower <= actual) & (actual <= upper)).astype(np.float64), "mae_cny": abs_error, "norm_mae": norm_error, "norm_width": width, "wis": wis80, "wis80": wis80}


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def _summary(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    # 计算并保存 keys，供当前评分步骤的后续校验或汇总使用。
    keys = ("coverage", "mae_cny", "norm_mae", "norm_width", "wis", "wis80")
    # 计算并保存 result，供当前评分步骤的后续校验或汇总使用。
    result = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    # 计算并保存 widths，供当前评分步骤的后续校验或汇总使用。
    widths = sorted(float(row["norm_width"]) for row in rows)
    # Retain the historic discrete order-statistic convention: lower median and
    # ceil(0.9*n)-1, rather than an interpolated NumPy quantile.
    # 计算并保存 lower_median，供当前评分步骤的后续校验或汇总使用。
    lower_median = widths[(len(widths) - 1) // 2]
    # 计算并保存 p90，供当前评分步骤的后续校验或汇总使用。
    p90 = widths[max(0, int(np.ceil(len(widths) * 0.90)) - 1)]
    # 执行当前表达式以完成本行评分计算或测试准备。
    result.update({"horizon_business_days": horizon, "enterprise_count": len(rows), "forecast_origin_count": int(sum(row["forecast_origin_count"] for row in rows)), "enterprise_width_median": lower_median, "enterprise_width_p90": p90})
    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return result


# 定义辅助函数；把这一项评分规则封装为可重复调用的步骤。
def evaluate(model_id: str, point_or_none: np.ndarray | None, intervals: np.ndarray, data: dict[str, Any]) -> dict[str, Any]:
    # 使用文档字符串说明当前模块或对象的职责，便于教学阅读。
    """评分完整双期限区间和可选五输出点路线。

    不取交集、不静默删行；坏键、坏形状、非有限值或中心不一致均直接报错。
    """
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if not str(model_id).strip():
        # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
        raise ScoringInputError("model_id must be non-empty")
    # 计算并保存 values，供当前评分步骤的后续校验或汇总使用。
    values = _validated_data(data)
    # 计算并保存 n，供当前评分步骤的后续校验或汇总使用。
    n = values["Y_cny"].shape[0]
    # 计算并保存 interval_values，供当前评分步骤的后续校验或汇总使用。
    interval_values = _numeric(intervals, "intervals", (n, 2, 3))
    # 计算并保存 metrics，供当前评分步骤的后续校验或汇总使用。
    metrics = _metric_rows(interval_values, values)
    # 执行当前表达式以完成本行评分计算或测试准备。
    point: np.ndarray | None = None
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if point_or_none is not None:
        # 计算并保存 point，供当前评分步骤的后续校验或汇总使用。
        point = _numeric(point_or_none, "point_or_none", (n, 5))
        # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
        if not np.allclose(point[:, 3:5], interval_values[:, :, 1], rtol=0.0, atol=1e-6):
            # 主动抛出明确错误，阻止不完整或歧义数据被静默计分。
            raise ScoringInputError("point mean10/mean30 centers do not match interval P50")

    # 执行当前表达式以完成本行评分计算或测试准备。
    per_enterprise: dict[tuple[str, int], list[int]] = defaultdict(list)
    # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
    for index, enterprise in enumerate(values["enterprise_id"]):
        # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
        for horizon_index, horizon in enumerate(_HORIZONS):
            # 执行当前表达式以完成本行评分计算或测试准备。
            per_enterprise[(enterprise, horizon)].append(index)
    # 执行当前表达式以完成本行评分计算或测试准备。
    enterprise_rows: list[dict[str, Any]] = []
    # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
    for (enterprise, horizon) in sorted(per_enterprise):
        # 计算并保存 indices，供当前评分步骤的后续校验或汇总使用。
        indices = np.asarray(per_enterprise[(enterprise, horizon)], dtype=np.int64)
        # 计算并保存 horizon_index，供当前评分步骤的后续校验或汇总使用。
        horizon_index = _HORIZONS.index(horizon)
        # 执行当前表达式以完成本行评分计算或测试准备。
        enterprise_rows.append({
            # 继续组装上一行开始的数组、容器或函数参数，保持表达式结构完整。
            "enterprise_id": enterprise,
            # 继续组装上一行开始的数组、容器或函数参数，保持表达式结构完整。
            "shape": values["shape"][indices[0]],
            # 继续组装上一行开始的数组、容器或函数参数，保持表达式结构完整。
            "horizon_business_days": horizon,
            # 继续组装上一行开始的数组、容器或函数参数，保持表达式结构完整。
            "forecast_origin_count": int(len(indices)),
            # 继续组装上一行开始的数组、容器或函数参数，保持表达式结构完整。
            **{key: float(np.mean(metric[indices, horizon_index])) for key, metric in metrics.items()},
        # 执行当前表达式以完成本行评分计算或测试准备。
        })

    # 计算并保存 summary，供当前评分步骤的后续校验或汇总使用。
    summary = {str(horizon): _summary([row for row in enterprise_rows if row["horizon_business_days"] == horizon], horizon) for horizon in _HORIZONS}
    # 执行当前表达式以完成本行评分计算或测试准备。
    shape_rows: list[dict[str, Any]] = []
    # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
    for shape in sorted(set(values["shape"])):
        # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
        for horizon in _HORIZONS:
            # 计算并保存 rows，供当前评分步骤的后续校验或汇总使用。
            rows = [row for row in enterprise_rows if row["shape"] == shape and row["horizon_business_days"] == horizon]
            # 执行当前表达式以完成本行评分计算或测试准备。
            shape_rows.append({"shape": shape, **_summary(rows, horizon)})

    # 执行当前表达式以完成本行评分计算或测试准备。
    endpoint_metrics: dict[str, dict[str, float | int]] = {}
    # 进入条件分支；只有满足该输入状态时才执行下面的保护或处理。
    if point is not None:
        # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
        for column, label in enumerate(("endpoint1", "endpoint10", "endpoint30")):
            # 计算并保存 abs_error，供当前评分步骤的后续校验或汇总使用。
            abs_error = np.abs(values["Y_cny"][:, column] - point[:, column])
            # 执行当前表达式以完成本行评分计算或测试准备。
            by_enterprise: dict[str, list[int]] = defaultdict(list)
            # 遍历当前集合，让同一规则逐项应用到每个期限、企业或字段。
            for index, enterprise in enumerate(values["enterprise_id"]):
                # 执行当前表达式以完成本行评分计算或测试准备。
                by_enterprise[enterprise].append(index)
            # 计算并保存 enterprise_abs，供当前评分步骤的后续校验或汇总使用。
            enterprise_abs = [float(abs_error[indices].mean()) for indices in by_enterprise.values()]
            # 计算并保存 enterprise_norm，供当前评分步骤的后续校验或汇总使用。
            enterprise_norm = [float((abs_error[indices] / values["scale"][indices]).mean()) for indices in by_enterprise.values()]
            # 计算并保存 endpoint_metrics[label]，供当前评分步骤的后续校验或汇总使用。
            endpoint_metrics[label] = {"enterprise_count": len(by_enterprise), "forecast_origin_count": n, "mae_cny": float(np.mean(enterprise_abs)), "norm_mae": float(np.mean(enterprise_norm))}

    # 返回本步骤计算出的结果，交给调用方继续构造区间或汇总。
    return {"model_id": str(model_id), "horizons": list(_HORIZONS), "summary": summary, "enterprise_rows": enterprise_rows, "shape_rows": shape_rows, "endpoint_metrics": endpoint_metrics}
