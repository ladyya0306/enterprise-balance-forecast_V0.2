# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
"""v4模型适配层：只定义可恢复的拟合与推理，不自行读取真实训练数据。"""

# 导入标准库以保存模型回执、固定随机状态和计量每个底层单位的耗时。
import json
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
import random
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
from dataclasses import asdict
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
from pathlib import Path
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
from time import perf_counter
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
from typing import Callable

# NumPy承载公开输入、监督标签和最终人民币预测。
import numpy as np
# PyTorch提供R0旧式分位数网络以及N5/N6的联合五输出网络。
import torch
# joblib保存sklearn表格模型和输入标准化器。
import joblib
# 导入四个已登记的表格估计器；它们只在调用fit_model时才实例化。
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
from sklearn.linear_model import Ridge
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
from sklearn.pipeline import make_pipeline
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
from sklearn.preprocessing import StandardScaler
# 导入成熟XGBoost接口，保留旧诊断的400轮hist训练方式。
import xgboost as xgb
# 复用第一轮的有序分位数网络与反标准化函数，不改变R0的输出定义。
try:
    # 当入口已把项目src加入模块路径时，使用发布形态的顶层包导入。
    from syn_b1.post_t6_c2d_quantile_lstm import FixedQuantileLSTM, restore_average_quantiles
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
except ModuleNotFoundError:
    # 当pytest从项目根以src命名空间导入时，保留等价的测试兼容路径。
    from src.syn_b1.post_t6_c2d_quantile_lstm import FixedQuantileLSTM, restore_average_quantiles


# 固定v4联合标签的列顺序；所有保存的模型和调用方都以此顺序交流。
TARGET_NAMES = ("endpoint_1", "endpoint_10", "endpoint_30", "mean_10", "mean_30")
# 固定R0和L0两项日均目标的位置，避免把端点标签误传给旧式分位数网络。
AVERAGE_TARGET_INDICES = (3, 4)
# 固定第一轮兼容的三个网络种子；R0保留它们而不继承v3种子。
R0_SEEDS = (2026083102, 2026083103, 2026083104)
# 固定新增模型的唯一版本种子；不根据开发成绩换种子。
NEW_MODEL_SEED = 2026091401
# 固定本轮CPU上限；它是资源约束，不是可搜索的模型参数。
CPU_THREADS = 4
# 旧12字段序列中四个金额字段的位置；N5/N6沿原诊断先按每行账户尺度归一化它们。
SEQUENCE_MONEY_INDICES = (2, 3, 4, 5)


# 将已审树表的货币字段按预测起点scale归一化；非货币字段保持原值。
def _normalized_tabular_inputs(data: dict) -> np.ndarray:
    # 复制调用方二维树输入，确保模型层不修改共享特征缓存。
    values = np.array(data["X_tab"], dtype=np.float64, copy=True)
    # 从成熟特征模块读取固定字段顺序，以免根据数值大小猜测货币列。
    try:
        # 生产入口通常以顶层syn_b1包导入。
        from syn_b1.pretrain_feature_engine import TREE_INPUT_COLUMNS
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    except ModuleNotFoundError:
        # pytest从项目根导入时使用对应命名空间。
        from src.syn_b1.pretrain_feature_engine import TREE_INPUT_COLUMNS
    # 拒绝未知树表宽度，避免错列归一化。
    if values.shape[1] != len(TREE_INPUT_COLUMNS):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("X_tab列数与成熟TREE_INPUT_COLUMNS不一致")
    # 只对名字以_cny结尾的金额字段按本行scale归一化。
    for index, name in enumerate(TREE_INPUT_COLUMNS):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        if name.endswith("_cny"):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            values[:, index] /= data["scale"]
    # 返回旧诊断相同的树模型输入尺度。
    return values


# 将旧12字段序列的四个金额列按本行账户尺度归一化，供N5/N6训练和推理共同使用。
def _normalized_sequence_inputs(data: dict) -> np.ndarray:
    # 复制float64公开序列，避免改变调用方原始X_raw或R0/L0所需金额输入。
    values = np.array(data["X_raw"], dtype=np.float64, copy=True)
    # 仅对旧字段顺序中的余额、流入、流出和净流量四列除以本预测起点尺度。
    values[:, :, SEQUENCE_MONEY_INDICES] /= data["scale"][:, None, None]
    # 返回与旧诊断一致、再供训练集标准化的输入。
    return values


# 声明清晰的契约异常，使数据/恢复问题不会被误解为正常模型表现。
class Round2V4ModelError(ValueError):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    """当v4模型输入、恢复文件或固定配置不满足时抛出。"""


# 定义五输出循环网络；N5与N6只由recurrent_kind区分，避免两份训练逻辑漂移。
class JointSequenceRegressor(torch.nn.Module):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    """读取28日公开序列并共同输出三个端点与两个直接日均人民币变化。"""

    # 初始化固定单层16隐藏单元的LSTM或GRU网络。
    def __init__(self, input_size: int, recurrent_kind: str) -> None:
        # 注册父类所需的参数容器。
        super().__init__()
        # 记住网络类别，恢复时可拒绝把GRU权重加载进LSTM。
        self.recurrent_kind = recurrent_kind
        # 根据冻结类别创建单层批优先循环层。
        self.recurrent = torch.nn.LSTM(input_size, 16, 1, batch_first=True) if recurrent_kind == "lstm" else torch.nn.GRU(input_size, 16, 1, batch_first=True)
        # 用线性头一次给出五个联合标签，避免为主赛额外创建30日独立头。
        self.output = torch.nn.Linear(16, len(TARGET_NAMES))

    # 定义公开序列到五个标准化变化预测的前向传播。
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        # 让循环层按时间顺序汇总全部28日输入。
        outputs, _ = self.recurrent(values)
        # 只取最后一个已知历史日的隐藏状态，保证不读取未来。
        return self.output(outputs[:, -1, :])


# 返回JSON可保存的全部配置；其中每一项都是本模块真正会使用的参数。
def model_specs() -> dict:
    # 返回普通Python对象，调用方可直接JSON序列化而不含numpy或Path实例。
    return {
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "B0": {"kind": "deterministic_balance_hold", "training": "forbidden", "outputs": "full_30_day_path"},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "B1": {"kind": "deterministic_recent_drift", "training": "forbidden", "outputs": "full_30_day_path", "lookback": 5},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "R0": {"kind": "fixed_quantile_lstm", "seeds": list(R0_SEEDS), "lookback": 28, "features": 12, "hidden_size": 16, "layers": 1, "batch_size": 256, "learning_rate": 0.001, "weight_decay": 0.0001, "maximum_epochs": 30, "early_stopping_patience": 5, "early_stopping_min_delta": 0.0001, "gradient_clip_norm": 1.0, "loss": "per_sample_account_scaled_pinball", "outputs": "mean10_mean30_p10_p50_p90"},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "N1": {"kind": "ridge", "seed": NEW_MODEL_SEED, "targets": list(TARGET_NAMES), "alpha": 1.0, "standardize": True},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "N2": {"kind": "xgboost", "seed": NEW_MODEL_SEED, "targets": list(TARGET_NAMES), "objective": "reg:squarederror", "tree_method": "hist", "max_depth": 4, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5.0, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0, "num_boost_round": 400, "nthread": CPU_THREADS},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "N3": {"kind": "extra_trees", "seed": NEW_MODEL_SEED, "targets": list(TARGET_NAMES), "n_estimators": 96, "max_depth": 12, "min_samples_leaf": 20, "min_samples_split": 2, "max_features": 0.8, "criterion": "squared_error", "bootstrap": False, "n_jobs": CPU_THREADS},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "N4": {"kind": "hist_gradient_boosting", "seed": NEW_MODEL_SEED, "targets": list(TARGET_NAMES), "loss": "squared_error", "learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 15, "max_depth": None, "min_samples_leaf": 20, "l2_regularization": 1.0, "max_bins": 255, "early_stopping": False},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "N5": {"kind": "joint_lstm", "seed": NEW_MODEL_SEED, "lookback": 28, "features": 12, "hidden_size": 16, "layers": 1, "outputs": list(TARGET_NAMES), "batch_size": 512, "epochs": 20, "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip_norm": 1.0, "loss": "enterprise_weighted_normalized_l1"},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "N6": {"kind": "joint_gru", "seed": NEW_MODEL_SEED, "lookback": 28, "features": 12, "hidden_size": 16, "layers": 1, "outputs": list(TARGET_NAMES), "batch_size": 512, "epochs": 20, "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip_norm": 1.0, "loss": "enterprise_weighted_normalized_l1"},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        "L0": {"kind": "saved_old_quantile_lstm", "outputs": "mean10_mean30_p10_p50_p90", "training": "forbidden"},
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    }


# 将进度事件交给编排层；没有回调时保持模型层可独立使用。
def _progress(callback: Callable[[dict], None] | None, payload: dict) -> None:
    # 只在调用方明确提供回调时发送可JSON化事件。
    if callback is not None:
        # 复制字典，防止调用方修改模型层后续仍会使用的对象。
        callback(dict(payload))


# 固定Python、NumPy和CPU PyTorch随机源，保证每个独立拟合单位可复跑。
def _set_seed(seed: int) -> None:
    # 固定标准库随机状态。
    random.seed(seed)
    # 固定NumPy随机状态。
    np.random.seed(seed)
    # 固定PyTorch权重初始化和DataLoader打乱顺序。
    torch.manual_seed(seed)
    # 限制本模块每个活动拟合最多使用四个CPU线程。
    torch.set_num_threads(CPU_THREADS)
    # 要求CPU使用确定性算法；若环境不支持会明确失败。
    torch.use_deterministic_algorithms(True)


# 核验一个角色数据字典的公共字段、维度和有限性，且绝不原地修改调用方数组。
def _validate_data(data: dict, require_labels: bool) -> dict:
    # 检查调用方没有遗漏v4公共数据契约字段。
    required = {"X_raw", "X_tab", "base", "scale", "enterprise_id", "origin_date", "bank_index", "shape"}
    # 训练还必须带五个人民币监督目标。
    if require_labels:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        required.add("Y_cny")
        # 完整未来30日余额只在训练角色输入中核对日均标签来源；预测接口绝不读取它。
        required.add("future_cny")
    # 计算实际缺失字段并在首次使用前阻断。
    missing = sorted(required - set(data))
    # 给出易定位的字段错误而非在网络内部报索引异常。
    if missing:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError(f"v4模型数据缺少字段：{missing}")
    # 读取原始12字段；旧模型金额输入必须由调用方显式保留float64，不能静默降精度或转换。
    if not isinstance(data["X_raw"], np.ndarray) or data["X_raw"].dtype != np.float64:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("X_raw必须由调用方显式提供float64数组")
    # 以float64视图读取原始字段，旧模型的金额精度不降为float32。
    raw = np.asarray(data["X_raw"], dtype=np.float64)
    # 读取树输入；它允许不同于序列的已审扁平特征数量。
    tab = np.asarray(data["X_tab"], dtype=np.float64)
    # 读取人民币起点余额和账户尺度。
    base = np.asarray(data["base"], dtype=np.float64)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    scale = np.asarray(data["scale"], dtype=np.float64)
    # 检查原始序列严格为样本×28×12。
    if raw.ndim != 3 or raw.shape[1:] != (28, 12):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError(f"X_raw必须是n×28×12，实际为{raw.shape}")
    # 检查树输入是一行一个预测起点且样本数一致。
    if tab.ndim != 2 or tab.shape[0] != raw.shape[0]:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("X_tab必须是与X_raw样本数相同的二维数组")
    # 检查起点余额和尺度是一维且每项都有对应样本。
    if base.shape != (raw.shape[0],) or scale.shape != (raw.shape[0],):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("base或scale长度与X_raw不一致")
    # 拒绝空角色、NaN/Inf和非正尺度，避免损失或还原悄悄产生无效值。
    if raw.shape[0] == 0 or not np.isfinite(raw).all() or not np.isfinite(tab).all() or not np.isfinite(base).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("模型输入为空、非有限或scale非正")
    # 创建返回字典，保留调用方附加的审计键但用已核验数组替换核心数组。
    checked = dict(data)
    # 写入统一的数值数组供后续函数使用。
    checked.update({"X_raw": raw, "X_tab": tab, "base": base, "scale": scale})
    # 训练角色额外检查五项人民币绝对标签。
    if require_labels:
        # 读取完整未来30日人民币余额；训练时用形状检查守住日均标签来源，绝不将其接入特征。
        future = np.asarray(data["future_cny"], dtype=np.float64)
        # 要求每条训练样本都有完整未来30个银行日余额，避免以端点代替未来日均。
        if future.shape != (raw.shape[0], 30) or not np.isfinite(future).all():
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            raise Round2V4ModelError("future_cny必须是有限的n×30人民币余额数组")
        # 保存已验证的训练标签来源数组，但预测接口不会到达这条分支。
        checked["future_cny"] = future
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        y = np.asarray(data["Y_cny"], dtype=np.float64)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        if y.shape != (raw.shape[0], len(TARGET_NAMES)) or not np.isfinite(y).all():
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            raise Round2V4ModelError("Y_cny必须是有限的n×5人民币绝对标签")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        checked["Y_cny"] = y
    # 返回已验证的角色数据，不改变传入data或其中数组。
    return checked


# 计算企业等权样本权重，使每家企业全部训练起点的总权重相同且平均为1。
def _enterprise_weights(enterprise_ids: np.ndarray) -> np.ndarray:
    # 将企业编号变为一维数组以便统计每户起点数。
    values = np.asarray(enterprise_ids)
    # 拒绝缺少或长度为零的企业键，因为无法定义企业等权。
    if values.ndim != 1 or values.size == 0:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("enterprise_id必须是非空一维数组")
    # 取得每个企业出现次数及每行对应的反向索引。
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    # 令每行权重与本企业样本数成反比。
    weights = 1.0 / counts[inverse].astype(np.float64)
    # 将平均值缩放为1，匹配v4训练权重约定。
    weights *= float(weights.size / weights.sum())
    # 返回连续float64权重以兼容sklearn、XGBoost和torch。
    return weights


# 只用训练角色拟合序列输入的均值和尺度；零方差字段按成熟规则置1。
def _fit_input_standardizer(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 合并样本和时间轴，只按训练公开历史计算每个字段的统计量。
    flat = raw.reshape(-1, raw.shape[-1])
    # 计算训练角色字段均值。
    means = flat.mean(axis=0)
    # 计算训练角色字段标准差。
    scales = flat.std(axis=0)
    # 将零标准差替换成1，避免常量字段产生NaN。
    scales = np.where(scales == 0.0, 1.0, scales)
    # 返回可JSON保存前再转换的数值数组。
    return means.astype(np.float64), scales.astype(np.float64)


# 保存JSON时拒绝NaN，避免回执看似成功却无法复现。
def _write_json(path: Path, payload: dict) -> None:
    # 创建指定单元目录的父目录，不写入其他输出根。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用稳定缩进和键排序，便于人工检查与指纹计算。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


# 返回N1到N4的固定表格估计器，且参数均来自model_specs。
def _tabular_estimator(model_id: str):
    # 读取当前模型的普通Python配置。
    spec = model_specs()[model_id]
    # N1沿用成熟的训练集标准化后Ridge。
    if model_id == "N1":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        return make_pipeline(StandardScaler(), Ridge(alpha=spec["alpha"], solver="svd", fit_intercept=True))
    # N2使用旧诊断同参数的XGBoost原生Booster，故在外层单独处理。
    if model_id == "N2":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        return None
    # N3显式传入所有登记的ExtraTrees参数。
    if model_id == "N3":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        return ExtraTreesRegressor(n_estimators=96, max_depth=12, min_samples_leaf=20, min_samples_split=2, max_features=0.8, criterion="squared_error", bootstrap=False, n_jobs=CPU_THREADS, random_state=NEW_MODEL_SEED)
    # N4显式传入所有登记的HGB参数，并关闭早停。
    if model_id == "N4":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        return HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.1, max_iter=100, max_leaf_nodes=15, max_depth=None, min_samples_leaf=20, l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=NEW_MODEL_SEED)
    # 其他编号不属于表格模型。
    raise Round2V4ModelError(f"不是表格模型：{model_id}")


# 计算R0按企业权重、账户尺度归一化且两期限三分位数等权的pinball损失。
def _r0_loss(quantiles: torch.Tensor, target_changes: torch.Tensor, account_scales: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    # 将真值增加分位数轴，得到批×2×1的可广播标签。
    errors = target_changes.unsqueeze(2) - quantiles
    # 创建P10/P50/P90概率张量。
    probabilities = torch.tensor((0.10, 0.50, 0.90), dtype=quantiles.dtype, device=quantiles.device).view(1, 1, 3)
    # 计算标准pinball项。
    pinball = torch.maximum(probabilities * errors, (probabilities - 1.0) * errors)
    # 先按两期限和三分位数平均，再按账户尺度归一化。
    per_row = pinball.mean(dim=(1, 2)) / account_scales
    # R0严格沿用旧C2D逐样本平均账户尺度pinball；weights参数仅为兼容测试调用，绝不改变科学损失。
    return per_row.mean()


# 训练并保存一个R0种子；early只服务旧规则的checkpoint选择，绝不参与梯度更新。
def _fit_r0_seed(train: dict, early: dict, seed: int, directory: Path, callback: Callable[[dict], None] | None) -> dict:
    # 记录本底层拟合单位的真实开始时间。
    started = perf_counter()
    # 先写运行中回执，使中断也留下实际成本证据。
    receipt = {"model_id": "R0", "seed": seed, "unit": "joint_mean10_mean30", "status": "RUNNING", "elapsed_seconds": None, "epochs_completed": 0, "error": None}
    # 保存运行中状态到该种子唯一目录。
    _write_json(directory / "receipt.json", receipt)
    # 向根编排器报告一个真正的底层拟合开始。
    _progress(callback, {"event": "fit_start", **receipt})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    try:
        # 固定当前种子和CPU线程。
        _set_seed(seed)
        # 只用train拟合输入标准化器。
        input_mean, input_scale = _fit_input_standardizer(train["X_raw"])
        # 将训练和早停输入按同一训练统计量标准化。
        train_x = ((train["X_raw"] - input_mean) / input_scale).astype(np.float32)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        early_x = ((early["X_raw"] - input_mean) / input_scale).astype(np.float32)
        # 从人民币绝对日均标签减起点余额，取得第一轮定义的人民币变化目标。
        train_changes = (train["Y_cny"][:, AVERAGE_TARGET_INDICES] - train["base"][:, None]).astype(np.float32)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        early_changes = (early["Y_cny"][:, AVERAGE_TARGET_INDICES] - early["base"][:, None]).astype(np.float32)
        # 只用训练角色计算旧目标标准化器。
        target_mean = train_changes.mean(axis=0).astype(np.float32)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        target_scale = train_changes.std(axis=0).astype(np.float32)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        target_scale = np.where(target_scale == 0.0, 1.0, target_scale).astype(np.float32)
        # 将R0网络输出的标准化分位数还原为人民币变化所需的对象。
        standardizer = type("TargetStandardizer", (), {"means": tuple(float(v) for v in target_mean), "scales": tuple(float(v) for v in target_scale)})()
        # 创建固定单层16维有序分位数LSTM。
        model = FixedQuantileLSTM(12, 16, 1).to(torch.device("cpu"))
        # 创建冻结AdamW优化器。
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        # 组织训练张量并用确定性随机顺序读取；R0旧合同不使用N系列的企业等权权重。
        generator = torch.Generator().manual_seed(seed)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_changes), torch.from_numpy(train["scale"].astype(np.float32)))
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, generator=generator)
        # 组织早停张量；其标签只在此处用于选择同一固定配置的checkpoint。
        early_dataset = torch.utils.data.TensorDataset(torch.from_numpy(early_x), torch.from_numpy(early_changes), torch.from_numpy(early["scale"].astype(np.float32)))
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        early_loader = torch.utils.data.DataLoader(early_dataset, batch_size=256, shuffle=False)
        # 初始化旧早停规则所需状态。
        best_loss, best_epoch, no_improvement, best_state = float("inf"), 0, 0, None
        # 最多训练30轮，轮数不按开发成绩增加。
        for epoch in range(1, 31):
            # 切换到训练状态。
            model.train()
            # 逐批更新参数。
            for x_batch, y_batch, scale_batch in loader:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                optimizer.zero_grad()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                q = restore_average_quantiles(model(x_batch), standardizer)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                loss = _r0_loss(q, y_batch, scale_batch)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                loss.backward()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                optimizer.step()
            # 在early角色计算旧式损失，以选择checkpoint而不更新参数。
            model.eval()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            total, row_total = 0.0, 0
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            with torch.no_grad():
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                for x_batch, y_batch, scale_batch in early_loader:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                    q = restore_average_quantiles(model(x_batch), standardizer)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                    batch_loss = _r0_loss(q, y_batch, scale_batch)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                    total += float(batch_loss) * int(x_batch.shape[0])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                    row_total += int(x_batch.shape[0])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            current = total / row_total
            # 改善时保存CPU权重副本。
            if current < best_loss - 0.0001:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                best_loss, best_epoch, no_improvement = current, epoch, 0
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            else:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                no_improvement += 1
            # 每轮把损失和耗时交给根的6小时预算记录。
            _progress(callback, {"event": "epoch", "model_id": "R0", "seed": seed, "epoch": epoch, "loss": float(current), "elapsed_seconds": perf_counter() - started})
            # 更新中断时可见的轮数。
            receipt["epochs_completed"] = epoch
            # 依照旧patience规则停止。
            if no_improvement >= 5:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                break
        # 没有任何有效checkpoint属于工程失败。
        if best_state is None:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            raise Round2V4ModelError("R0未产生可恢复checkpoint")
        # 恢复最佳early checkpoint，而不是最后一轮权重。
        model.load_state_dict(best_state)
        # 保存本种子的全部恢复材料。
        torch.save({"model_id": "R0", "seed": seed, "input_mean": input_mean.tolist(), "input_scale": input_scale.tolist(), "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(), "best_epoch": best_epoch, "state_dict": model.state_dict()}, directory / "model.pt")
        # 回执只在模型、统计量和权重都已写完后标记完成。
        receipt.update({"status": "COMPLETE", "best_epoch": best_epoch, "early_loss": float(best_loss)})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    except Exception as error:
        # 保留失败状态和真实异常，不删除半成品以伪装没有成本。
        receipt.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}"})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    finally:
        # 无论成功失败都记录累计耗时并写回同一单位回执。
        receipt["elapsed_seconds"] = perf_counter() - started
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        _write_json(directory / "receipt.json", receipt)
        # 向根编排器报告结束事件，失败时它也能计入预算。
        _progress(callback, {"event": "fit_end", **receipt})
    # 返回可聚合的成功回执。
    return receipt


# 训练N1到N4的五个标量头，并为每头保存模型、完整参数和失败回执。
def _fit_tabular(model_id: str, train: dict, directory: Path, callback: Callable[[dict], None] | None) -> list[dict]:
    # 创建公共企业权重；同一模型所有目标头使用相同训练行权重。
    weights = _enterprise_weights(train["enterprise_id"])
    # 沿成熟诊断先按账户尺度归一化树金额字段；输入绝不直接使用原始人民币量级。
    inputs = _normalized_tabular_inputs(train)
    # 沿成熟诊断学习账户尺度归一化的余额变化，而不是直接拟合绝对余额。
    targets = (train["Y_cny"] - train["base"][:, None]) / train["scale"][:, None]
    # 保存五个底层单位的回执列表。
    receipts: list[dict] = []
    # 逐目标训练，保持S2登记的五次标量拟合计数。
    for index, target_name in enumerate(TARGET_NAMES):
        # 为当前目标创建唯一子目录，避免一个失败覆盖另一个头。
        unit_dir = directory / target_name
        # 记录单位开始。
        started = perf_counter()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        receipt = {"model_id": model_id, "seed": NEW_MODEL_SEED, "unit": target_name, "status": "RUNNING", "elapsed_seconds": None, "epochs_completed": None, "error": None}
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        _write_json(unit_dir / "receipt.json", receipt)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        _progress(callback, {"event": "fit_start", **receipt})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        try:
            # 固定每个独立估计器的随机源。
            _set_seed(NEW_MODEL_SEED)
            # N2使用Booster并单独保存json模型；其余用joblib保存sklearn对象。
            if model_id == "N2":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                params = dict(model_specs()["N2"])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                rounds = int(params.pop("num_boost_round"))
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                params.pop("kind"); params.pop("targets")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                booster = xgb.train(params, xgb.DMatrix(inputs, label=targets[:, index], weight=weights), num_boost_round=rounds)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                booster.save_model(unit_dir / "model.json")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                get_params = params | {"num_boost_round": rounds}
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            else:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                estimator = _tabular_estimator(model_id)
                # Pipeline必须把sample_weight送进Ridge步骤；其他估计器直接接收。
                if model_id == "N1":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                    estimator.fit(inputs, targets[:, index], ridge__sample_weight=weights)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                else:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                    estimator.fit(inputs, targets[:, index], sample_weight=weights)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                joblib.dump(estimator, unit_dir / "model.joblib")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                get_params = estimator.get_params(deep=True)
            # 保存完整实际参数，确保库默认值不会只凭记忆复现。
            _write_json(unit_dir / "parameters.json", {key: value for key, value in get_params.items() if isinstance(value, (str, int, float, bool, type(None)))})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            receipt["status"] = "COMPLETE"
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        except Exception as error:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            receipt.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}"})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            raise
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        finally:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            receipt["elapsed_seconds"] = perf_counter() - started
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            _write_json(unit_dir / "receipt.json", receipt)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            _progress(callback, {"event": "fit_end", **receipt})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            receipts.append(dict(receipt))
    # 返回每头记录，供顶层模型回执准确计算实际完成数。
    return receipts


# 训练N5或N6的单一联合五输出网络，并保存仅由train拟合的标准化器。
def _fit_joint(model_id: str, train: dict, directory: Path, callback: Callable[[dict], None] | None) -> dict:
    # 记录联合单位开始时间。
    started = perf_counter()
    # 定义一份覆盖五输出的单一回执。
    receipt = {"model_id": model_id, "seed": NEW_MODEL_SEED, "unit": "joint_five_outputs", "status": "RUNNING", "elapsed_seconds": None, "epochs_completed": 0, "error": None}
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    _write_json(directory / "receipt.json", receipt)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    _progress(callback, {"event": "fit_start", **receipt})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    try:
        # 固定CPU训练行为。
        _set_seed(NEW_MODEL_SEED)
        # 先按每行账户尺度归一化四个金额序列字段，再只用训练角色拟合输入标准化。
        normalized_raw = _normalized_sequence_inputs(train)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        mean, std = _fit_input_standardizer(normalized_raw)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        x = ((normalized_raw - mean) / std).astype(np.float32)
        # 先将五个绝对余额标签变为账户尺度归一化变化，再计算训练期目标标准化。
        normalized_changes = (train["Y_cny"] - train["base"][:, None]) / train["scale"][:, None]
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        target_mean = normalized_changes.mean(axis=0)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        target_std = np.where(normalized_changes.std(axis=0) == 0.0, 1.0, normalized_changes.std(axis=0))
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        y = ((normalized_changes - target_mean) / target_std).astype(np.float32)
        # 创建联合网络；N6只替换循环层类型。
        model = JointSequenceRegressor(12, "lstm" if model_id == "N5" else "gru")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        # 按企业权重组成训练集并保持种子确定的批次顺序。
        generator = torch.Generator().manual_seed(NEW_MODEL_SEED)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(_enterprise_weights(train["enterprise_id"]).astype(np.float32)))
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, generator=generator)
        # 固定训练20轮，不使用early或开发标签早停。
        for epoch in range(1, 21):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            loss_total, row_total = 0.0, 0
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            model.train()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            for x_batch, y_batch, weight_batch in loader:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                optimizer.zero_grad()
                # 先对五输出分别计算标准化L1，再按企业权重汇总。
                per_row = torch.abs(model(x_batch) - y_batch).mean(dim=1)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                loss = (per_row * weight_batch).sum() / weight_batch.sum()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                loss.backward()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                optimizer.step()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                loss_total += float(loss) * int(x_batch.shape[0])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
                row_total += int(x_batch.shape[0])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            receipt["epochs_completed"] = epoch
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            _progress(callback, {"event": "epoch", "model_id": model_id, "seed": NEW_MODEL_SEED, "epoch": epoch, "loss": loss_total / row_total, "elapsed_seconds": perf_counter() - started})
        # 保存恢复所需权重、输入标准化和目标标准化。
        torch.save({"model_id": model_id, "input_mean": mean.tolist(), "input_scale": std.tolist(), "target_mean": target_mean.tolist(), "target_scale": target_std.tolist(), "state_dict": model.state_dict()}, directory / "model.pt")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        receipt["status"] = "COMPLETE"
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    except Exception as error:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        receipt.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}"})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    finally:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        receipt["elapsed_seconds"] = perf_counter() - started
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        _write_json(directory / "receipt.json", receipt)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        _progress(callback, {"event": "fit_end", **receipt})
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    return receipt


# 公开训练接口；它只使用调用方显式传入的数据，绝不扫描项目真实人口或输出目录。
def fit_model(model_id: str, train: dict, early: dict | None, output_dir: Path, progress_callback: Callable[[dict], None] | None = None) -> dict:
    # 拒绝L0训练，避免旧模型被错误重训或覆盖。
    if model_id == "L0":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("L0只允许predict_model加载旧manifest，不允许fit_model")
    # 拒绝未知模型编号。
    if model_id not in {"R0", "N1", "N2", "N3", "N4", "N5", "N6"}:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError(f"未知v4模型：{model_id}")
    # 验证训练数据；此过程不拟合真实数据。
    train_checked = _validate_data(train, require_labels=True)
    # 规定所有输出只能写进调用方指定的一个模型目录。
    directory = Path(output_dir)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    directory.mkdir(parents=True, exist_ok=True)
    # 先保存固定规格和训练样本计数，方便中断后检查范围。
    _write_json(directory / "metadata.json", {"model_id": model_id, "spec": model_specs()[model_id], "train_rows": int(train_checked["X_raw"].shape[0]), "target_names": list(TARGET_NAMES), "real_training_executed_by_this_call": True})
    # R0必须有明确early角色，避免以任何替代角色偷偷改变旧早停规则。
    if model_id == "R0":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        if early is None:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            raise Round2V4ModelError("R0需要显式early角色以执行旧patience早停规则")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        early_checked = _validate_data(early, require_labels=True)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        receipts = [_fit_r0_seed(train_checked, early_checked, seed, directory / f"seed_{seed}", progress_callback) for seed in R0_SEEDS]
    # 表格模型不使用early角色，按S2登记逐目标训练五个独立单位。
    elif model_id in {"N1", "N2", "N3", "N4"}:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        receipts = _fit_tabular(model_id, train_checked, directory, progress_callback)
    # 联合循环模型固定20轮，early不参与其无早停合同。
    else:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        receipts = [_fit_joint(model_id, train_checked, directory, progress_callback)]
    # 顶层回执汇总每一个底层单位，而不把计划数伪装为完成数。
    result = {"model_id": model_id, "artifact_dir": str(directory), "status": "COMPLETE", "planned_fit_units": len(receipts), "completed_fit_units": sum(item["status"] == "COMPLETE" for item in receipts), "receipts": receipts}
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    _write_json(directory / "fit_receipt.json", result)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    return result


# 从R0三份恢复文件加载模型并输出未校准的人民币绝对日均分位数。
def _predict_r0(directory: Path, data: dict) -> np.ndarray:
    # 收集各种子预测，之后沿种子维逐坐标取中位数。
    predictions = []
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    for seed in R0_SEEDS:
        # 读取该种子完整恢复包。
        package = torch.load(directory / f"seed_{seed}" / "model.pt", map_location="cpu", weights_only=False)
        # 用保存时的结构重建有序分位数网络。
        model = FixedQuantileLSTM(12, 16, 1)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        model.load_state_dict(package["state_dict"])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        model.eval()
        # 只按对应种子训练统计量变换公开输入。
        x = ((data["X_raw"] - np.asarray(package["input_mean"])) / np.asarray(package["input_scale"]))
        # 创建兼容旧反标准化函数的轻量标准化对象。
        standardizer = type("TargetStandardizer", (), {"means": tuple(package["target_mean"]), "scales": tuple(package["target_scale"])})()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        with torch.no_grad():
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            changes = restore_average_quantiles(model(torch.from_numpy(x.astype(np.float32))), standardizer).cpu().numpy()
        # 将预测变化还原成两个期限的人民币绝对余额。
        predictions.append(changes + data["base"][:, None, None])
    # 按合同先对同一坐标三种子取中位数，校准由根层另行完成。
    return np.median(np.stack(predictions, axis=0), axis=0).astype(np.float64)


# 从旧C2F manifest安全加载三份checkpoint并应用manifest中已有的旧范围半径。
def _predict_l0(directory: Path, data: dict) -> np.ndarray:
    # 允许调用方传入manifest文件或包含manifest的旧模型目录。
    manifest_path = directory if directory.name.endswith(".json") else directory / "final_quantile_lstm_model_manifest.json"
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 旧模型合同固定恰好三个checkpoint；缺失或额外版本都不能静默改写中位数组合。
    if len(manifest.get("models", [])) != 3:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("L0 manifest必须恰有三个checkpoint")
    # 旧C3B恢复入口以manifest的输入统计量为唯一来源，不假定checkpoint重复保存它。
    input_standardizer = manifest.get("input_standardizer")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    if not isinstance(input_standardizer, dict) or "feature_means" not in input_standardizer or "feature_scales" not in input_standardizer:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("L0 manifest缺少输入标准化器")
    # 旧范围调整必须明确登记两个日均期限，不能将缺失误当作零半径。
    adjustments = manifest.get("conformal_adjustments")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    if not isinstance(adjustments, dict) or any(str(horizon) not in adjustments for horizon in (10, 30)):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("L0 manifest缺少10或30日旧校准半径")
    # 按manifest相对路径读取各个checkpoint，不扫描任何旧校准答案文件。
    predictions = []
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    for record in manifest["models"]:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        package = torch.load(manifest_path.parent / record["checkpoint_path"], map_location="cpu", weights_only=False)
        # checkpoint的种子和方法编号必须与manifest记录一致，防止混入另一轮旧模型。
        if int(package.get("training_seed", -1)) != int(record["training_seed"]) or package.get("selected_method_id") != manifest.get("selected_method_id"):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            raise Round2V4ModelError("L0 checkpoint与manifest的种子或方法编号不一致")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        model = FixedQuantileLSTM(12, 16, 1)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        model.load_state_dict(package["model_state_dict"])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        model.eval()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        standardizer = package["average_target_standardizer"]
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        # 严格复现旧加载顺序：原始公开输入、均值和尺度均先为float32，不能先用float64相减后再降精度。
        x = (data["X_raw"].astype(np.float32) - np.asarray(input_standardizer["feature_means"], dtype=np.float32)) / np.asarray(input_standardizer["feature_scales"], dtype=np.float32)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        target = type("TargetStandardizer", (), {"means": tuple(standardizer["means"]), "scales": tuple(standardizer["scales"])})()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        with torch.no_grad():
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            changes = restore_average_quantiles(model(torch.from_numpy(x.astype(np.float32))), target).cpu().numpy()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        predictions.append(changes + data["base"][:, None, None])
    # 先按旧规则对三种子逐坐标取中位数。
    output = np.median(np.stack(predictions, axis=0), axis=0)
    # 读取旧半径；中心不动，仅向外扩P10和P90。
    for average_index, horizon in enumerate((10, 30)):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        q = float(adjustments[str(horizon)]["normalized_adjustment"])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        output[:, average_index, 0] -= q * data["scale"]
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        output[:, average_index, 2] += q * data["scale"]
    # 检查分位数顺序和有限性，拒绝异常旧文件而不静默重排。
    if not np.isfinite(output).all() or not np.all(output[:, :, 0] <= output[:, :, 1]) or not np.all(output[:, :, 1] <= output[:, :, 2]):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError("L0恢复后预测非有限或分位数交叉")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    return output.astype(np.float64)


# 从已保存的N1到N4五个标量工件恢复人民币绝对预测。
def _predict_tabular(model_id: str, directory: Path, data: dict) -> np.ndarray:
    # 初始化每个目标一列的人民币预测数组。
    output = np.empty((data["X_tab"].shape[0], len(TARGET_NAMES)), dtype=np.float64)
    # 应用同一账户尺度归一化树输入，不从预测角色重新拟合任何统计量。
    inputs = _normalized_tabular_inputs(data)
    # 逐目标按其对应恢复格式预测。
    for index, target_name in enumerate(TARGET_NAMES):
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        unit_dir = directory / target_name
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        if model_id == "N2":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            model = xgb.Booster()
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            model.load_model(unit_dir / "model.json")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            output[:, index] = model.predict(xgb.DMatrix(inputs))
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        else:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
            output[:, index] = joblib.load(unit_dir / "model.joblib").predict(inputs)
    # 返回人民币绝对标签空间的五列预测。
    return data["base"][:, None] + output * data["scale"][:, None]


# 从N5/N6单一工件恢复五个人民币绝对预测。
def _predict_joint(model_id: str, directory: Path, data: dict) -> np.ndarray:
    # 读取保存的结构、输入标准化和目标标准化。
    package = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    # 重建正确循环网络并载入权重。
    model = JointSequenceRegressor(12, "lstm" if model_id == "N5" else "gru")
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    model.load_state_dict(package["state_dict"])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    model.eval()
    # 先按每行账户尺度归一化四个金额序列字段，再使用训练角色保存的输入统计量。
    normalized_raw = _normalized_sequence_inputs(data)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    x = ((normalized_raw - np.asarray(package["input_mean"])) / np.asarray(package["input_scale"]))
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    with torch.no_grad():
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        normalized = model(torch.from_numpy(x.astype(np.float32))).cpu().numpy()
    # 先反标准化账户尺度变化，再按每行scale还原人民币变化并加回起点余额。
    changes = normalized * np.asarray(package["target_scale"]) + np.asarray(package["target_mean"])
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    return data["base"][:, None] + changes * data["scale"][:, None]


# 公开推理接口；它绝不拟合统计量或修改模型文件，并返回合同规定的float64人民币数组。
def predict_model(model_id: str, artifact_dir: Path, data: dict) -> np.ndarray:
    # 验证仅预测所需输入字段和维度。
    checked = _validate_data(data, require_labels=False)
    # 规范化工件路径，调用方必须显式指出模型目录或L0 manifest。
    directory = Path(artifact_dir)
    # L0沿旧manifest和checkpoint只读推理。
    if model_id == "L0":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        output = _predict_l0(directory, checked)
    # R0从三种子中位数组合输出未校准分位数。
    elif model_id == "R0":
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        output = _predict_r0(directory, checked)
    # N1到N4返回五列中心人民币预测。
    elif model_id in {"N1", "N2", "N3", "N4"}:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        output = _predict_tabular(model_id, directory, checked)
    # N5/N6返回五列联合中心人民币预测。
    elif model_id in {"N5", "N6"}:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        output = _predict_joint(model_id, directory, checked)
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
    else:
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError(f"未知v4模型：{model_id}")
    # 统一拒绝非有限输出；余额允许为负，模型层绝不截断。
    if not np.isfinite(output).all():
# 中文教学注释：本行执行当前模型的固定契约、训练、恢复或预测步骤。
        raise Round2V4ModelError(f"{model_id}预测出现NaN或Inf")
    # 所有推理结果用float64返回，供后续人民币评分不受float32截断影响。
    return np.asarray(output, dtype=np.float64)
