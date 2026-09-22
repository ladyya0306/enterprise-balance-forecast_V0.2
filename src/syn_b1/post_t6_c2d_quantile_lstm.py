"""POST-T6-C2D：固定小型分位数LSTM的训练、预测和保形校准基础。"""

# 导入random标准库；它固定Python层面的随机状态以支持可复现训练。
import random
# 从dataclasses导入dataclass；它保存不可变的两期限目标标准化参数。
from dataclasses import dataclass

# 导入numpy；它保存连续的序列、目标和预测数组。
import numpy as np
# 导入torch；它提供CPU上的确定性张量计算和神经网络训练。
import torch
# 从torch导入神经网络模块；它创建LSTM、线性层和Softplus排序变换。
from torch import nn
# 从torch导入批加载工具；它保证fit、calibration和预测按清晰角色处理。
from torch.utils.data import DataLoader, TensorDataset

# 固定C2D实现版本；网络、损失、目标或校准规则变化都必须升级版本。
C2D_VERSION = "syn_b1_post_t6_c2d_fixed_quantile_lstm_1_0"
# 固定C1登记的分位数候选方法编号；它不能与C2C简单候选混淆。
QUANTILE_METHOD_ID = "fixed_lstm_conformalized_quantiles"
# 固定两个产品期限；第一个位置对应10日日均，第二个位置对应30日日均。
AVERAGE_HORIZONS = (10, 30)
# 固定三个分位数；P10、P50、P90共同组成名义80%范围。
QUANTILE_LEVELS = (0.10, 0.50, 0.90)


# 声明本卡异常；它把合同/数据错误与模型预测表现明确区分。
class C2DQuantileLstmError(ValueError):
    """当C2D固定结构、目标形状或训练结果不满足合同要求时抛出。"""


# 用不可变对象保存每个期限的fit角色目标均值和尺度。
@dataclass(frozen=True)
class AverageTargetStandardizer:
    # 保存10日和30日日均余额变化的fit角色均值。
    means: tuple[float, float]
    # 保存10日和30日日均余额变化的fit角色标准差。
    scales: tuple[float, float]


# 定义固定单层16维LSTM；它一次输出两个期限各三个有序分位数的余额变化。
class FixedQuantileLSTM(nn.Module):
    # 初始化网络；输入字段数来自A1公开特征，隐藏维度和层数来自冻结训练配置。
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        # 调用父类以注册LSTM、线性层和Softplus模块。
        super().__init__()
        # 创建批优先LSTM；它从早到晚读取28个银行工作日的公开历史。
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # 创建六维原始输出头；维度为2个期限乘3个有序分位数的增量参数。
        self.output = nn.Linear(hidden_size, len(AVERAGE_HORIZONS) * len(QUANTILE_LEVELS))
        # 创建Softplus；它把相邻分位数的间隔限制为正，从结构上杜绝P10大于P50等交叉。
        self.positive_gap = nn.Softplus()

    # 定义前向传播；输入形状为批大小×28日×公开特征数。
    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        # 让LSTM按历史时间顺序处理全部公开输入。
        outputs, _ = self.lstm(sequence_tensor)
        # 读取最后一个历史日的隐藏状态；它代表预测截止日能看到的历史摘要。
        last_hidden = outputs[:, -1, :]
        # 将隐藏状态映射为每期限三个原始输出值。
        raw = self.output(last_hidden).reshape(-1, len(AVERAGE_HORIZONS), len(QUANTILE_LEVELS))
        # 将第一个输出定义为P10的标准化余额变化；它可为正或负。
        p10 = raw[:, :, 0]
        # 把第二个输出经Softplus变为正间隔后加到P10，得到不小于P10的P50。
        p50 = p10 + self.positive_gap(raw[:, :, 1])
        # 把第三个输出经Softplus变为正间隔后加到P50，得到不小于P50的P90。
        p90 = p50 + self.positive_gap(raw[:, :, 2])
        # 沿最后一维组合有序P10、P50、P90，供损失和预测共用。
        return torch.stack((p10, p50, p90), dim=2)


# 从fit角色两期限平均余额变化拟合目标标准化器；校准与验证绝不参与。
def fit_average_target_standardizer(target_changes: np.ndarray, fit_indices: np.ndarray) -> AverageTargetStandardizer:
    # 读取当前折fit角色的两期限余额变化标签。
    fit_targets = target_changes[fit_indices]
    # 要求二维两期限目标且至少有一个fit样本，避免无定义统计量。
    if fit_targets.ndim != 2 or fit_targets.shape[1] != len(AVERAGE_HORIZONS) or fit_targets.shape[0] == 0:
        # 阻断错误切分或目标形状。
        raise C2DQuantileLstmError("C2D fit角色两期限目标为空或形状错误")
    # 分别计算10日和30日目标均值。
    means = fit_targets.mean(axis=0)
    # 分别计算10日和30日目标标准差。
    scales = fit_targets.std(axis=0)
    # 常量目标的尺度置为1，避免之后标准化出现除零或NaN。
    scales = np.where(scales == 0.0, 1.0, scales)
    # 返回可保存且不含未来角色信息的普通浮点元组。
    return AverageTargetStandardizer(tuple(float(value) for value in means), tuple(float(value) for value in scales))


# 将人民币两期限余额变化转换成网络训练使用的标准化目标。
def transform_average_targets(target_changes: np.ndarray, standardizer: AverageTargetStandardizer) -> np.ndarray:
    # 将均值变为可沿样本维广播的float32数组。
    means = np.asarray(standardizer.means, dtype=np.float32)
    # 将尺度变为可沿样本维广播的float32数组。
    scales = np.asarray(standardizer.scales, dtype=np.float32)
    # 返回每期限使用同一均值和尺度的标准化余额变化。
    return (target_changes - means) / scales


# 将网络有序标准化分位数还原为人民币两期限余额变化。
def restore_average_quantiles(scaled_quantiles: torch.Tensor, standardizer: AverageTargetStandardizer) -> torch.Tensor:
    # 将两期限尺度放到与网络输出相同的设备和数值类型。
    scales = torch.tensor(standardizer.scales, dtype=scaled_quantiles.dtype, device=scaled_quantiles.device).view(1, len(AVERAGE_HORIZONS), 1)
    # 将两期限均值放到与网络输出相同的设备和数值类型。
    means = torch.tensor(standardizer.means, dtype=scaled_quantiles.dtype, device=scaled_quantiles.device).view(1, len(AVERAGE_HORIZONS), 1)
    # 用正尺度逐期限反标准化；同一期限的分位数顺序因而保持不变。
    return scaled_quantiles * scales + means


# 计算账户尺度加权的分位数损失；它使大账户的绝对金额不垄断训练目标。
def account_scaled_pinball_loss(scaled_quantiles: torch.Tensor, raw_targets: torch.Tensor, account_scales: torch.Tensor, standardizer: AverageTargetStandardizer) -> torch.Tensor:
    # 将网络输出还原为人民币两期限P10、P50、P90余额变化。
    raw_quantiles = restore_average_quantiles(scaled_quantiles, standardizer)
    # 将真实两期限余额变化扩展为最后一维，便于与三个分位数逐项比较。
    expanded_targets = raw_targets.unsqueeze(2)
    # 计算真实值减预测值；正数表示预测分位数偏低，负数表示偏高。
    errors = expanded_targets - raw_quantiles
    # 创建P10、P50、P90概率张量并让它自动广播到每个样本和期限。
    probabilities = torch.tensor(QUANTILE_LEVELS, dtype=scaled_quantiles.dtype, device=scaled_quantiles.device).view(1, 1, len(QUANTILE_LEVELS))
    # 按分位数定义计算pinball损失；它使P10更重罚预测过低、P90更重罚预测过高。
    pinball = torch.maximum(probabilities * errors, (probabilities - 1.0) * errors)
    # 将每个样本三个分位数和两期限误差除以该账户预测时可见尺度。
    normalized = pinball / account_scales.view(-1, 1, 1)
    # 返回全批平均损失，作为固定训练和早停的唯一目标。
    return normalized.mean()


# 固定Python、numpy、PyTorch和CPU线程状态，保证同一输入与种子可复现。
def set_training_seed(seed: int, torch_num_threads: int) -> None:
    # 固定Python随机源；它影响任何标准库随机调用。
    random.seed(seed)
    # 固定numpy随机源；它影响数组层面的随机操作。
    np.random.seed(seed)
    # 固定PyTorch网络初始化与数据装载器随机顺序。
    torch.manual_seed(seed)
    # 固定CPU线程数；它是资源一致性设置，不是模型调参。
    torch.set_num_threads(torch_num_threads)
    # 强制PyTorch选择确定性算法；不支持时会明确报错而不会悄悄漂移。
    torch.use_deterministic_algorithms(True)


# 在calibration角色上计算当前checkpoint的账户尺度pinball损失；它只用于早停。
def calibration_pinball_loss(model: FixedQuantileLSTM, loader: DataLoader, standardizer: AverageTargetStandardizer, device: torch.device) -> float:
    # 切换为评估模式，关闭训练时可能存在的随机行为。
    model.eval()
    # 初始化按样本数加权的损失合计。
    loss_sum = 0.0
    # 初始化样本数量。
    sample_count = 0
    # 关闭梯度，避免校准评价建立反向传播图。
    with torch.no_grad():
        # 稳定顺序读取校准批次。
        for sequence_tensor, raw_target_tensor, scale_tensor in loader:
            # 将公开历史序列移动到冻结CPU设备。
            sequence_tensor = sequence_tensor.to(device)
            # 将校准真实两期限标签移动到冻结CPU设备。
            raw_target_tensor = raw_target_tensor.to(device)
            # 将可见账户尺度移动到冻结CPU设备。
            scale_tensor = scale_tensor.to(device)
            # 计算当前模型的有序分位数输出。
            scaled_quantiles = model(sequence_tensor)
            # 计算当前批固定账户尺度pinball损失。
            loss = account_scaled_pinball_loss(scaled_quantiles, raw_target_tensor, scale_tensor, standardizer)
            # 按本批样本数累计，避免最后一个小批被放大权重。
            loss_sum += loss.item() * int(sequence_tensor.shape[0])
            # 累加当前批样本数。
            sample_count += int(sequence_tensor.shape[0])
    # calibration角色不能为空，否则不能定义早停。
    if sample_count == 0:
        # 阻断错误切分。
        raise C2DQuantileLstmError("C2D calibration角色为空")
    # 返回每个样本平均的固定校准损失。
    return loss_sum / sample_count


# 训练一个折和一个种子的固定分位数LSTM；fit训练、calibration早停、validation完全隔离。
def fit_one_quantile_model(fit_inputs: np.ndarray, fit_targets: np.ndarray, fit_scales: np.ndarray, calibration_inputs: np.ndarray, calibration_targets: np.ndarray, calibration_scales: np.ndarray, seed: int, hidden_size: int, num_layers: int, batch_size: int, learning_rate: float, weight_decay: float, maximum_epochs: int, early_stopping_patience: int, early_stopping_min_delta: float, gradient_clip_norm: float, torch_num_threads: int, standardizer: AverageTargetStandardizer) -> tuple[FixedQuantileLSTM, list[dict[str, str]], int]:
    # 固定当前独立模型的全部随机源与CPU资源。
    set_training_seed(seed, torch_num_threads)
    # 固定合同登记的CPU设备；本卡不因设备差异改变结果。
    device = torch.device("cpu")
    # 将fit角色的公开输入、真实标签和账户尺度组成训练数据集。
    fit_dataset = TensorDataset(torch.from_numpy(fit_inputs), torch.from_numpy(fit_targets), torch.from_numpy(fit_scales))
    # 将calibration角色的对应数组组成仅供早停和以后校准的数据集。
    calibration_dataset = TensorDataset(torch.from_numpy(calibration_inputs), torch.from_numpy(calibration_targets), torch.from_numpy(calibration_scales))
    # 创建只服务当前种子的批次随机生成器。
    loader_generator = torch.Generator()
    # 固定fit角色批次打乱顺序。
    loader_generator.manual_seed(seed)
    # 创建允许打乱的fit加载器；只有fit数据参与参数更新。
    fit_loader = DataLoader(fit_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)
    # 创建不打乱的calibration加载器；它不能作为参数训练数据。
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    # 创建冻结的单层16维有序分位数网络。
    model = FixedQuantileLSTM(fit_inputs.shape[2], hidden_size, num_layers).to(device)
    # 创建固定AdamW优化器；学习率和权重衰减来自A0已冻结公共参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 初始化尚未出现的最佳calibration损失。
    best_loss = float("inf")
    # 初始化最佳轮次。
    best_epoch = 0
    # 初始化连续无改善轮数。
    no_improvement = 0
    # 初始化最佳权重副本。
    best_state: dict[str, torch.Tensor] | None = None
    # 初始化供用户检查的逐轮训练历史。
    history: list[dict[str, str]] = []
    # 按冻结最大轮数训练；不得依据C2B/C2C成绩增减预算。
    for epoch in range(1, maximum_epochs + 1):
        # 切换训练模式。
        model.train()
        # 初始化本轮按样本加权训练损失。
        train_loss_sum = 0.0
        # 初始化本轮训练样本数。
        train_count = 0
        # 逐fit批次更新网络参数。
        for sequence_tensor, raw_target_tensor, scale_tensor in fit_loader:
            # 将公开历史送入冻结CPU设备。
            sequence_tensor = sequence_tensor.to(device)
            # 将fit角色真实标签送入冻结CPU设备。
            raw_target_tensor = raw_target_tensor.to(device)
            # 将fit角色账户尺度送入冻结CPU设备。
            scale_tensor = scale_tensor.to(device)
            # 清空上个批次遗留的梯度。
            optimizer.zero_grad()
            # 计算当前批有序P10/P50/P90输出。
            scaled_quantiles = model(sequence_tensor)
            # 用固定账户尺度pinball损失评价当前批。
            loss = account_scaled_pinball_loss(scaled_quantiles, raw_target_tensor, scale_tensor, standardizer)
            # 反向传播计算参数梯度。
            loss.backward()
            # 按A0冻结上限裁剪梯度，避免循环网络数值爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            # 使用AdamW更新当前模型参数。
            optimizer.step()
            # 按样本数累计当前批损失。
            train_loss_sum += loss.item() * int(sequence_tensor.shape[0])
            # 累计当前批样本数。
            train_count += int(sequence_tensor.shape[0])
        # fit角色必须非空，否则训练结果无定义。
        if train_count == 0:
            # 阻断空训练集。
            raise C2DQuantileLstmError("C2D fit角色为空")
        # 计算本轮fit角色平均损失。
        train_loss = train_loss_sum / train_count
        # 只在calibration角色评价checkpoint选择目标。
        current_calibration_loss = calibration_pinball_loss(model, calibration_loader, standardizer, device)
        # 判断是否达到冻结的最小改善幅度。
        improved = current_calibration_loss < best_loss - early_stopping_min_delta
        # 校准损失改善时保存不可变的CPU权重副本。
        if improved:
            # 更新最佳校准损失。
            best_loss = current_calibration_loss
            # 更新最佳轮次。
            best_epoch = epoch
            # 克隆状态字典，避免后续训练覆盖最佳checkpoint。
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            # 重置无改善轮数。
            no_improvement = 0
        # 未改善时只增加耐心计数，不读取validation。
        else:
            # 增加连续无改善轮数。
            no_improvement += 1
        # 保存每轮的fit与calibration固定目标，供训练过程审计。
        history.append({"epoch": str(epoch), "train_account_scaled_pinball_loss": f"{train_loss:.10f}", "calibration_account_scaled_pinball_loss": f"{current_calibration_loss:.10f}", "improved": str(improved).lower(), "training_seed": str(seed), "c2d_version": C2D_VERSION})
        # 达到冻结早停耐心时停止，不额外多训练。
        if no_improvement >= early_stopping_patience:
            # 退出当前模型的训练轮循环。
            break
    # 每个模型都必须产生至少一个最佳checkpoint。
    if best_state is None:
        # 阻断异常训练状态。
        raise C2DQuantileLstmError("C2D模型没有有效checkpoint")
    # 恢复最佳calibration checkpoint，之后只用它输出校准和验证预测。
    model.load_state_dict(best_state)
    # 返回锁定checkpoint、完整历史和最佳轮次。
    return model, history, best_epoch


# 对一个角色批量输出人民币两期限P10/P50/P90余额变化；顺序与输入样本顺序一致。
def predict_average_quantiles(model: FixedQuantileLSTM, inputs: np.ndarray, batch_size: int, standardizer: AverageTargetStandardizer) -> np.ndarray:
    # 创建不打乱的预测数据集。
    dataset = TensorDataset(torch.from_numpy(inputs))
    # 创建稳定顺序加载器。
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 初始化批结果列表。
    batches: list[np.ndarray] = []
    # 切换评估模式。
    model.eval()
    # 关闭梯度。
    with torch.no_grad():
        # 逐批读取公开历史序列。
        for (sequence_tensor,) in loader:
            # 将序列送到冻结CPU设备。
            sequence_tensor = sequence_tensor.to(torch.device("cpu"))
            # 计算并还原人民币两期限有序分位数变化。
            raw_quantiles = restore_average_quantiles(model(sequence_tensor), standardizer)
            # 保存当前批到numpy数组。
            batches.append(raw_quantiles.cpu().numpy())
    # 空输入没有可定义预测，必须明确阻断。
    if not batches:
        # 抛出清晰异常。
        raise C2DQuantileLstmError("C2D预测输入为空")
    # 合并所有批并保持原角色样本顺序。
    predictions = np.concatenate(batches, axis=0)
    # 每行必须是两个期限乘三个分位数。
    if predictions.ndim != 3 or predictions.shape[1:] != (len(AVERAGE_HORIZONS), len(QUANTILE_LEVELS)):
        # 阻断网络输出形状漂移。
        raise C2DQuantileLstmError("C2D分位数预测形状错误")
    # 检查每个期限均满足P10不大于P50且P50不大于P90。
    if not np.all(predictions[:, :, 0] <= predictions[:, :, 1]) or not np.all(predictions[:, :, 1] <= predictions[:, :, 2]):
        # 阻断任何分位数交叉。
        raise C2DQuantileLstmError("C2D分位数预测发生交叉")
    # 返回经结构约束验证的人民币余额变化预测。
    return predictions
