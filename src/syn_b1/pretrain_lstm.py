"""SYN-B1 T5 的小型 LSTM 三种子训练、验证与公平比较逻辑。"""

# 导入 csv 标准库；它流式读取T2长序列表与T3锁定基线，并输出可审计预测明细。
import csv
# 导入 json 标准库；它保存训练配置、标准化器、指标和报告。
import json
# 导入 random 标准库；它固定Python层面的随机状态，确保每个训练种子可复现。
import random
# 从 dataclasses 导入 dataclass；它把序列样本、标准化器和训练配置表达为字段明确的对象。
from dataclasses import asdict, dataclass
# 从 decimal 导入 Decimal 和舍入规则；验证金额指标使用十进制展示，避免浮点尾差影响报告。
from decimal import Decimal, ROUND_HALF_UP
# 从 pathlib 导入 Path；它定位冻结的T2/T3输入与未来输出目录。
from pathlib import Path
# 从 typing 导入 Iterable；它说明CSV读取函数逐条产生记录，不一次复制全部文件。
from typing import Iterable

# 导入 numpy；它高效保存14日序列并执行只在训练集拟合的标准化。
import numpy as np
# 导入 torch；它提供LSTM网络、张量、优化器、数据装载和模型保存能力。
import torch
# 从torch导入神经网络模块；它提供LSTM、线性层和L1损失。
from torch import nn
# 从torch导入DataLoader与TensorDataset；它把冻结样本分批送入训练循环。
from torch.utils.data import DataLoader, TensorDataset

# 复用T3样本与文件指纹工具，保证余额归一化误差口径和锁定基线完全一致。
from syn_b1.pretrain_baselines import BaselineContractError, TreeSample, read_tree_samples, sha256_file, validate_t2_report
# 复用T4的T3血缘校验，确保LSTM、XGBoost和基线都基于同一T2快照。
from syn_b1.pretrain_xgboost import validate_t3_contract

# 固定本阶段实现版本；它写入所有产物，防止未来不同LSTM结构混入当前结果。
LSTM_VERSION = "syn_b1_pretrain_t5_lstm_1_0"
# 固定候选模型名称；它区别于锁定基线与T4的XGBoost候选。
MODEL_NAME = "small_lstm_sequence_candidate"
# 固定14银行工作日窗口长度；任何非14行样本均属于T2契约破坏。
SEQUENCE_LENGTH = 14
# 固定合同预注册的三个独立训练种子；结果必须全部保留并以中位数代表LSTM家族。
TRAINING_SEEDS = (2026083002, 2026083003, 2026083004)
# 冻结序列输入字段；它们全部来自预测截止日及以前的账户日度历史和银行日历。
SEQUENCE_FEATURE_COLUMNS = (
    "closing_balance_cny", "daily_inflow_cny", "daily_outflow_cny", "daily_net_flow_cny",
    "transaction_count", "day_has_transaction", "weekday_monday_0", "is_month_end_business_day",
    "calendar_gap_days_from_prior_business_day", "calendar_gap_days_to_next_business_day",
)


# 声明T5专用异常类型；它使数据密封、序列结构或配对错误能够明确停止而不是产生误导性模型。
class LSTMContractError(BaselineContractError):
    """当LSTM训练前置契约、序列契约或公平比较不满足时抛出。"""


# 用不可变对象保存一个完整14日样本；审计列与模型输入列严格物理分开。
@dataclass(frozen=True)
class SequenceSample:
    # 保存样本唯一键；它只用于与T3基线和预测明细一一配对，绝不进入张量特征。
    sample_key: str
    # 保存企业编号；它只用于企业隔离核验和逐企业评价，绝不进入张量特征。
    enterprise_id: str
    # 保存训练或验证组；最终测试样本不允许被本类读取。
    split_group: str
    # 保存预测截止日；它只用于审计时间边界。
    cutoff_date: str
    # 保存目标银行工作日；它只用于审计配对，不作为输入。
    target_date: str
    # 保存截止日余额；它是预测余额换算和归一化误差分母的已知金额。
    closing_balance_cny: Decimal
    # 保存下一工作日真实余额；它只用于训练或验证标签，不输入网络。
    target_balance_cny: Decimal
    # 保存下一工作日真实余额变化；它是网络的唯一监督目标。
    target_change_cny: Decimal
    # 保存冻结的余额下限；它防止低余额企业支配归一化指标。
    balance_floor_cny: Decimal
    # 保存14行乘10列公开历史特征；数组不含身份、日期、标签、备注、画像或种子。
    sequence: tuple[tuple[float, ...], ...]


# 用不可变对象保存训练集拟合的均值与尺度；验证集只能复用，禁止重新拟合。
@dataclass(frozen=True)
class Standardizer:
    # 保存每项序列特征的训练集均值。
    feature_means: tuple[float, ...]
    # 保存每项序列特征的训练集标准差；零方差列安全替换为1。
    feature_scales: tuple[float, ...]
    # 保存训练集目标变化均值。
    target_mean: float
    # 保存训练集目标变化标准差；零方差时安全替换为1。
    target_scale: float


# 用不可变对象保存唯一小型LSTM训练配置；本卡不会在验证集上搜索其中任何值。
@dataclass(frozen=True)
class LSTMTrainingConfig:
    # 设置单层LSTM，保持网络容量与260户、14日窗口的首轮研究规模匹配。
    num_layers: int = 1
    # 设置隐藏状态维度为16，避免用过大网络吸收合成样本细节。
    hidden_size: int = 16
    # 设置批大小为256，兼顾CPU内存和训练稳定性。
    batch_size: int = 256
    # 设置AdamW学习率为0.001；它在训练前固定，不按验证结果搜索。
    learning_rate: float = 0.001
    # 设置轻微L2权重衰减，降低过拟合风险。
    weight_decay: float = 0.0001
    # 设置最长训练30轮，防止首轮研究无限训练。
    max_epochs: int = 30
    # 设置验证无改善5轮即早停；早停只选择同一固定配置中的checkpoint，不改变结构或超参数。
    early_stopping_patience: int = 5
    # 设置最小改善阈值为0.0001个标准化目标单位，防止浮点微小波动不断重置耐心。
    early_stopping_min_delta: float = 0.0001
    # 设置梯度范数上限为1，防止循环网络出现不稳定的大梯度。
    gradient_clip_norm: float = 1.0
    # 明确本卡只使用CPU，保证本机可复现且不依赖GPU环境。
    device: str = "cpu"
    # 控制PyTorch计算线程数；它是资源设置，不改变数据或实验规则。
    torch_num_threads: int = 4


# 将文本金额转为Decimal；失败时带上字段名，便于定位冻结CSV问题。
def decimal_value(value: str, field_name: str) -> Decimal:
    # 使用try捕获非法文本，避免空字符串或非金额静默进入评价。
    try:
        # 通过字符串直接构造十进制金额，保持CSV原有小数精度。
        return Decimal(value)
    # 捕获所有Decimal转换异常并统一改写为项目契约异常。
    except Exception as error:
        # 抛出带字段名的清晰错误，并保留原始异常链。
        raise LSTMContractError(f"{field_name}不是有效金额：{value!r}") from error


# 把金额格式化到分；它只影响输出CSV的展示，不改变训练内部浮点数。
def money(value: Decimal) -> str:
    # 使用普通财务四舍五入，使用户和既有T3/T4报告保持同一金额口径。
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# 把比例固定显示为八位小数；账户归一化MAE和Skill Score沿用T3/T4格式。
def ratio(value: Decimal) -> str:
    # 使用定点格式避免科学计数法影响Excel筛选和人工查看。
    return f"{value.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP):f}"


# 逐行读取UTF-8或UTF-8-BOM CSV；生成器和T2的Excel友好CSV均可无改写读取。
def read_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    # 以utf-8-sig打开文件会自动忽略BOM，也支持普通UTF-8。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # DictReader按表头返回字典，后续只访问冻结字段名。
        yield from csv.DictReader(handle)


# 将一组同键的14行长表记录转换为一个序列样本，并严格验证时间与标签一致性。
def build_sequence_sample(rows: list[dict[str, str]], expected_group: str) -> SequenceSample:
    # 要求样本正好14行，防止短窗口和长窗口混入当前固定模型。
    if len(rows) != SEQUENCE_LENGTH:
        # 抛出异常并标识样本键，便于定位T2输出问题。
        raise LSTMContractError(f"LSTM样本不是{SEQUENCE_LENGTH}行：{rows[0]['sample_key']}")
    # 读取第一行作为共同审计字段的基准。
    first = rows[0]
    # 要求所有行的切分组相同且等于调用方请求组。
    if any(row["split_group"] != expected_group for row in rows):
        # 阻止训练和验证行被混装进同一个样本。
        raise LSTMContractError(f"LSTM样本切分组不一致：{first['sample_key']}")
    # 将序列位置转为整数，后续要求恰好是1到14。
    positions = [int(row["sequence_position"]) for row in rows]
    # 要求记录已按从最早到最新的冻结顺序排列。
    if positions != list(range(1, SEQUENCE_LENGTH + 1)):
        # 阻止乱序行导致LSTM把时间倒置。
        raise LSTMContractError(f"LSTM序列位置不是1至{SEQUENCE_LENGTH}：{first['sample_key']}")
    # 要求每行声明的序列长度均为14。
    if any(int(row["sequence_length"]) != SEQUENCE_LENGTH for row in rows):
        # 阻止不一致元数据掩盖窗口错误。
        raise LSTMContractError(f"LSTM序列长度声明错误：{first['sample_key']}")
    # 声明必须在14行中保持不变的审计和标签字段。
    shared_fields = ("sample_key", "enterprise_id", "split_group", "target_date", "target_balance_cny", "target_change_cny")
    # 逐项检查每行与第一行一致。
    for field_name in shared_fields:
        # 发现任何不一致即停止，避免错误标签进入训练。
        if any(row[field_name] != first[field_name] for row in rows):
            # 抛出字段和样本键，方便定位源文件。
            raise LSTMContractError(f"LSTM样本字段不一致：{first['sample_key']} / {field_name}")
    # 依次抽取每行的10项冻结特征并转成float。
    sequence = tuple(tuple(float(row[field_name]) for field_name in SEQUENCE_FEATURE_COLUMNS) for row in rows)
    # 要求全部数值有限，拒绝NaN或无穷进入神经网络。
    if not np.isfinite(np.asarray(sequence, dtype=np.float64)).all():
        # 抛出异常，要求先修复T2特征而不是让模型掩盖问题。
        raise LSTMContractError(f"LSTM特征存在缺失或无穷值：{first['sample_key']}")
    # 以最后一个历史工作日作为预测截止日；它只用于审计和余额换算。
    last = rows[-1]
    # 构造不可变样本对象，明确把标签与网络输入分开保存。
    return SequenceSample(
        # 保存样本键。
        sample_key=first["sample_key"],
        # 保存企业编号。
        enterprise_id=first["enterprise_id"],
        # 保存已经核验的切分组。
        split_group=first["split_group"],
        # 保存最后一个历史日作为截止日。
        cutoff_date=last["calendar_date"],
        # 保存目标工作日。
        target_date=first["target_date"],
        # 保存截止日余额。
        closing_balance_cny=decimal_value(last["closing_balance_cny"], "closing_balance_cny"),
        # 保存真实目标余额。
        target_balance_cny=decimal_value(first["target_balance_cny"], "target_balance_cny"),
        # 保存真实余额变化。
        target_change_cny=decimal_value(first["target_change_cny"], "target_change_cny"),
        # 使用T2固定的100万元余额下限；此值来自树表，非验证集计算值。
        balance_floor_cny=Decimal("1000000.00"),
        # 保存14×10公开历史特征。
        sequence=sequence,
    )


# 读取T2长序列表中的一个切分组；函数遇到final_test行立即失败，并拒绝同键非连续出现。
def read_sequence_samples(path: Path, allowed_group: str) -> list[SequenceSample]:
    # 创建最终样本列表；其中只保存调用方指定的训练或验证组。
    samples: list[SequenceSample] = []
    # 保存当前连续样本键；T2输出应把同一键的14行连续写出。
    active_key: str | None = None
    # 保存当前样本的14行记录。
    active_rows: list[dict[str, str]] = []
    # 保存已经完成的样本键，检测同一键在文件中被拆成多段的错误。
    completed_keys: set[str] = set()
    # 流式扫描T2长表，不读取逐户流水、备注或最终测试标签。
    for row in read_csv_rows(path):
        # 读取切分组，后续严格执行最终测试密封。
        split_group = row.get("split_group", "")
        # 最终测试行不应存在于长表；若出现说明密封被破坏。
        if split_group == "final_test":
            # 立即停止，任何候选训练都不得接触最终测试序列。
            raise LSTMContractError("T2 LSTM序列表不应含最终测试行")
        # 跳过另一个允许组，避免训练过程保留不必要标签。
        if split_group != allowed_group:
            # 继续读取下一行。
            continue
        # 取得当前样本键，作为14行归组依据。
        sample_key = row["sample_key"]
        # 第一次遇到允许组时初始化当前分组。
        if active_key is None:
            # 保存当前键。
            active_key = sample_key
        # 遇到新键时先完成前一组14行。
        if sample_key != active_key:
            # 要求前一键此前未完成，防止非连续重复样本。
            if active_key in completed_keys:
                # 抛出异常，避免后半段行被错误拼接。
                raise LSTMContractError(f"LSTM样本键非连续重复：{active_key}")
            # 生成并保存前一序列样本。
            samples.append(build_sequence_sample(active_rows, allowed_group))
            # 标记前一键已完成。
            completed_keys.add(active_key)
            # 切换到新的样本键。
            active_key = sample_key
            # 清空行列表，以收集新键的14行。
            active_rows = []
        # 追加当前行到同键序列。
        active_rows.append(row)
    # 文件结束后若仍有一组行则完成最后样本。
    if active_key is not None:
        # 再次检查最后键没有此前完成，保证一致的非重复规则。
        if active_key in completed_keys:
            # 抛出异常，避免尾部重复键漏检。
            raise LSTMContractError(f"LSTM样本键非连续重复：{active_key}")
        # 保存最后一组样本。
        samples.append(build_sequence_sample(active_rows, allowed_group))
    # 空样本不能训练或评价，必须明确阻断。
    if not samples:
        # 抛出包含组名的错误，帮助定位输入路径或切分问题。
        raise LSTMContractError(f"LSTM序列表没有{allowed_group}样本")
    # 返回与T2文件顺序一致的样本列表。
    return samples


# 只用训练序列拟合输入和目标标准化器；验证序列绝不能参与均值或尺度计算。
def fit_standardizer(train_samples: list[SequenceSample]) -> Standardizer:
    # 将所有训练序列堆成三维数组，维度为样本、14日、10特征。
    sequences = np.asarray([sample.sequence for sample in train_samples], dtype=np.float64)
    # 将训练目标余额变化组成一维数组。
    targets = np.asarray([float(sample.target_change_cny) for sample in train_samples], dtype=np.float64)
    # 要求数组形状严格符合冻结窗口和特征数。
    if sequences.ndim != 3 or sequences.shape[1:] != (SEQUENCE_LENGTH, len(SEQUENCE_FEATURE_COLUMNS)):
        # 抛出异常，防止未来字段增减被静默接受。
        raise LSTMContractError("训练LSTM序列形状不等于14×10")
    # 将样本和时间维合并，只在训练历史值上计算每项输入均值。
    feature_means = sequences.reshape(-1, len(SEQUENCE_FEATURE_COLUMNS)).mean(axis=0)
    # 在同一训练历史值上计算标准差。
    feature_scales = sequences.reshape(-1, len(SEQUENCE_FEATURE_COLUMNS)).std(axis=0)
    # 将零标准差替换为1，保证常量字段标准化时不会除零。
    feature_scales = np.where(feature_scales == 0.0, 1.0, feature_scales)
    # 计算训练目标均值。
    target_mean = float(targets.mean())
    # 计算训练目标标准差，并在极端常量标签时安全替换为1。
    target_scale = float(targets.std()) or 1.0
    # 返回可JSON保存的不可变标准化器对象。
    return Standardizer(tuple(float(value) for value in feature_means), tuple(float(value) for value in feature_scales), target_mean, target_scale)


# 按训练期均值和尺度转换序列输入；本函数不会重新计算任何统计量。
def transform_sequences(samples: list[SequenceSample], standardizer: Standardizer) -> np.ndarray:
    # 取出原始14日序列并使用float32降低CPU训练内存。
    values = np.asarray([sample.sequence for sample in samples], dtype=np.float32)
    # 将标准化参数转成可广播的一维float32数组。
    means = np.asarray(standardizer.feature_means, dtype=np.float32)
    # 将标准化尺度转成可广播的一维float32数组。
    scales = np.asarray(standardizer.feature_scales, dtype=np.float32)
    # 返回逐特征标准化后的三维输入，不改变时间顺序。
    return (values - means) / scales


# 按训练期均值和尺度转换监督目标；验证目标只能使用相同变换。
def transform_targets(samples: list[SequenceSample], standardizer: Standardizer) -> np.ndarray:
    # 取出余额变化标签并用float32保存。
    values = np.asarray([float(sample.target_change_cny) for sample in samples], dtype=np.float32)
    # 返回训练期标准化后的目标。
    return (values - np.float32(standardizer.target_mean)) / np.float32(standardizer.target_scale)


# 定义单层小型LSTM；它只接收14×10历史序列并输出一个标准化余额变化预测。
class SmallLSTMRegressor(nn.Module):
    # 初始化网络结构；输入维度和隐藏维度来自冻结配置。
    def __init__(self, input_size: int, config: LSTMTrainingConfig) -> None:
        # 调用父类初始化，注册随后创建的子模块。
        super().__init__()
        # 创建单层批优先LSTM；dropout仅多层生效，因此此处不声明无效dropout参数。
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True)
        # 创建线性输出头，把最后时点的16维隐藏状态映射为一个余额变化标量。
        self.output = nn.Linear(config.hidden_size, 1)

    # 定义前向计算；输入形状是批大小×14×10，输出形状是批大小。
    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        # 让LSTM按时间顺序处理14天公开历史，输出每个时间点的隐藏状态。
        outputs, _ = self.lstm(sequence_tensor)
        # 选择最后一个时间点的隐藏状态，它对应预测截止日的全部历史信息。
        last_hidden = outputs[:, -1, :]
        # 经线性层产生单一标准化余额变化，并移除长度为1的末维。
        return self.output(last_hidden).squeeze(1)


# 固定Python、numpy和PyTorch随机状态；三个种子各自完全独立，不共享权重或优化器。
def set_training_seed(seed: int, config: LSTMTrainingConfig) -> None:
    # 设置Python随机种子，覆盖DataLoader之外的普通随机调用。
    random.seed(seed)
    # 设置numpy随机种子，覆盖数组层面的随机调用。
    np.random.seed(seed)
    # 设置PyTorch全局随机种子，决定网络初始化和其他张量随机过程。
    torch.manual_seed(seed)
    # 明确设置CPU线程数，减少本机环境变化对运行表现的影响。
    torch.set_num_threads(config.torch_num_threads)
    # 强制尽可能使用确定性算法；当前CPU LSTM支持此模式。
    torch.use_deterministic_algorithms(True)


# 计算标准化目标空间的验证MAE；它仅用于同一固定配置的早停checkpoint选择。
def validation_scaled_mae(model: SmallLSTMRegressor, loader: DataLoader, device: torch.device) -> float:
    # 切换到评估模式，关闭训练态行为。
    model.eval()
    # 初始化误差总和。
    absolute_error_sum = 0.0
    # 初始化样本计数。
    sample_count = 0
    # 关闭梯度计算，避免验证无意构建反向图。
    with torch.no_grad():
        # 逐批读取验证序列和标准化目标。
        for sequence_tensor, target_tensor in loader:
            # 将序列移动到固定CPU设备。
            sequence_tensor = sequence_tensor.to(device)
            # 将目标移动到固定CPU设备。
            target_tensor = target_tensor.to(device)
            # 获得标准化预测。
            predictions = model(sequence_tensor)
            # 累加本批L1误差。
            absolute_error_sum += torch.abs(predictions - target_tensor).sum().item()
            # 累加本批样本数。
            sample_count += int(target_tensor.numel())
    # 验证集不能为空；否则早停指标没有定义。
    if sample_count == 0:
        # 抛出契约错误，避免空验证集被解释为0误差。
        raise LSTMContractError("LSTM验证集为空")
    # 返回标准化目标空间的平均绝对误差。
    return absolute_error_sum / sample_count


# 训练一个种子的完整小型LSTM，并返回最佳checkpoint、每轮损失历史和最佳轮次。
def fit_single_seed(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    seed: int,
    config: LSTMTrainingConfig,
) -> tuple[SmallLSTMRegressor, list[dict[str, str]], int]:
    # 固定该种子的全部随机来源，不继承上一种子训练的状态。
    set_training_seed(seed, config)
    # 将配置中的设备字符串转换为torch设备对象。
    device = torch.device(config.device)
    # 将训练输入转成张量，不再从原始CSV读取。
    train_input_tensor = torch.from_numpy(train_inputs)
    # 将训练目标转成张量。
    train_target_tensor = torch.from_numpy(train_targets)
    # 将验证输入转成张量；它只用于早停与训练后的评价。
    validation_input_tensor = torch.from_numpy(validation_inputs)
    # 将验证目标转成张量。
    validation_target_tensor = torch.from_numpy(validation_targets)
    # 创建训练数据集，序列与标签按同一T2样本顺序绑定。
    train_dataset = TensorDataset(train_input_tensor, train_target_tensor)
    # 创建验证数据集，禁止打乱评价样本。
    validation_dataset = TensorDataset(validation_input_tensor, validation_target_tensor)
    # 为训练DataLoader创建该种子专属的随机生成器，保证批次顺序可复现。
    loader_generator = torch.Generator()
    # 设置DataLoader生成器种子，与网络初始化种子一致但不复用跨种子状态。
    loader_generator.manual_seed(seed)
    # 创建训练加载器；shuffle只作用于训练样本，绝不改变验证预测输出顺序。
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=loader_generator)
    # 创建验证加载器；shuffle=False保持验证样本与审计键顺序一致。
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    # 构建一套新的小型LSTM；每个种子都从独立随机初始化开始。
    model = SmallLSTMRegressor(len(SEQUENCE_FEATURE_COLUMNS), config).to(device)
    # 选择AdamW优化器；它只绑定当前种子的模型参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # 使用L1损失，使训练目标与报告MAE方向一致。
    loss_function = nn.L1Loss()
    # 初始化当前最佳验证误差为无穷大。
    best_validation_mae = float("inf")
    # 初始化最佳轮次为0，表示尚未保存checkpoint。
    best_epoch = 0
    # 初始化无改善计数，用于早停。
    no_improvement_epochs = 0
    # 初始化最佳权重副本；它会在第一轮验证后被真实权重替换。
    best_state: dict[str, torch.Tensor] | None = None
    # 初始化每轮损失历史，供用户查看训练是否稳定。
    history: list[dict[str, str]] = []
    # 从第1轮开始训练，最多达到预注册的30轮。
    for epoch in range(1, config.max_epochs + 1):
        # 切换到训练模式。
        model.train()
        # 初始化本轮训练损失加总。
        training_loss_sum = 0.0
        # 初始化本轮训练样本计数。
        training_count = 0
        # 按固定种子确定的打乱批次进行训练。
        for sequence_tensor, target_tensor in train_loader:
            # 将当前序列批移动到CPU设备。
            sequence_tensor = sequence_tensor.to(device)
            # 将当前目标批移动到CPU设备。
            target_tensor = target_tensor.to(device)
            # 清空上一批残留梯度，避免梯度跨批累积。
            optimizer.zero_grad()
            # 执行前向传播得到标准化余额变化预测。
            predictions = model(sequence_tensor)
            # 计算本批L1损失。
            loss = loss_function(predictions, target_tensor)
            # 反向传播计算每个参数梯度。
            loss.backward()
            # 对全部模型参数做梯度裁剪，限制循环网络梯度爆炸。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            # 按AdamW规则更新当前种子自己的模型参数。
            optimizer.step()
            # 累加按样本数加权的训练损失。
            training_loss_sum += loss.item() * int(target_tensor.numel())
            # 累加训练样本数。
            training_count += int(target_tensor.numel())
        # 计算当前轮的训练平均L1损失。
        training_mae = training_loss_sum / training_count
        # 在验证集计算标准化目标空间MAE，用于早停而非超参数搜索。
        current_validation_mae = validation_scaled_mae(model, validation_loader, device)
        # 判断当前轮是否较历史最佳至少改善固定阈值。
        improved = current_validation_mae < best_validation_mae - config.early_stopping_min_delta
        # 若改善则保存当前模型权重作为唯一配置下的最佳checkpoint。
        if improved:
            # 更新最佳验证误差。
            best_validation_mae = current_validation_mae
            # 记录最佳轮次。
            best_epoch = epoch
            # 复制每个张量到CPU并克隆，防止后续训练原地修改该checkpoint。
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            # 重置无改善计数。
            no_improvement_epochs = 0
        # 若未改善则增加无改善轮数。
        else:
            # 增加早停耐心计数。
            no_improvement_epochs += 1
        # 记录本轮损失与是否改善，便于公开研究复核。
        history.append({
            # 写入训练种子。
            "training_seed": str(seed),
            # 写入轮次。
            "epoch": str(epoch),
            # 写入训练标准化MAE。
            "train_scaled_mae": f"{training_mae:.10f}",
            # 写入验证标准化MAE。
            "validation_scaled_mae": f"{current_validation_mae:.10f}",
            # 写入当前轮是否保存为最佳checkpoint。
            "is_best_checkpoint": str(improved).lower(),
        })
        # 达到预注册耐心上限时停止当前种子训练。
        if no_improvement_epochs >= config.early_stopping_patience:
            # 跳出轮次循环，保留此前最佳checkpoint。
            break
    # 若从未保存checkpoint说明训练过程异常，不能继续生成预测。
    if best_state is None:
        # 抛出明确错误，避免返回随机初始模型。
        raise LSTMContractError(f"LSTM没有保存有效checkpoint：种子{seed}")
    # 将最佳权重加载回模型，确保输出预测不使用最后一轮可能退化的权重。
    model.load_state_dict(best_state)
    # 返回最佳模型、完整损失历史和最佳轮次。
    return model, history, best_epoch


# 从T3基线预测明细读取唯一锁定方法，并要求与当前验证样本逐条配对。
def read_locked_baseline_rows(path: Path, locked_method: str, validation_samples: list[SequenceSample]) -> dict[str, dict[str, str]]:
    # 只选取锁定方法的行，其他四种基线不参与Skill Score分母。
    rows = [row for row in read_csv_rows(path) if row.get("baseline_method") == locked_method]
    # 以样本键建立映射；重复键会在长度检查时被拒绝。
    by_key = {row["sample_key"]: row for row in rows}
    # 提取当前验证样本的唯一键集合。
    expected_keys = {sample.sample_key for sample in validation_samples}
    # 同时检查无重复、无缺失、无额外行。
    if len(rows) != len(by_key) or set(by_key) != expected_keys:
        # 拒绝不同企业或不同日期的比较，防止模型漏掉困难样本。
        raise LSTMContractError("锁定基线与LSTM验证样本没有逐条一一对应")
    # 返回按样本键索引的基线明细。
    return by_key


# 用已训练模型预测验证集，并恢复到人民币余额变化尺度后生成逐样本配对明细。
def build_validation_predictions(
    model: SmallLSTMRegressor,
    validation_inputs: np.ndarray,
    validation_samples: list[SequenceSample],
    baseline_by_key: dict[str, dict[str, str]],
    standardizer: Standardizer,
    seed: int,
    config: LSTMTrainingConfig,
) -> list[dict[str, str]]:
    # 将配置设备转换为torch设备对象。
    device = torch.device(config.device)
    # 切换模型到评估模式。
    model.eval()
    # 将标准化验证输入转为张量并放入固定CPU设备。
    input_tensor = torch.from_numpy(validation_inputs).to(device)
    # 关闭梯度计算，避免预测阶段保留训练图。
    with torch.no_grad():
        # 得到标准化目标空间预测并移回CPU numpy数组。
        scaled_predictions = model(input_tensor).cpu().numpy()
    # 恢复人民币余额变化尺度；均值和尺度只来自训练集。
    predicted_changes = scaled_predictions * standardizer.target_scale + standardizer.target_mean
    # 要求预测行数与验证样本数一致。
    if len(predicted_changes) != len(validation_samples):
        # 阻止底层批处理错误造成不完整评价。
        raise LSTMContractError("LSTM验证预测行数与样本数不一致")
    # 创建预测明细列表。
    records: list[dict[str, str]] = []
    # 按冻结验证样本顺序逐条计算误差。
    for sample, prediction in zip(validation_samples, predicted_changes):
        # 将float预测通过字符串转换为Decimal，避免直接带入二进制尾差。
        predicted_change = Decimal(str(float(prediction)))
        # 将预测变化加到截止日已知余额，得到预测目标日余额。
        predicted_balance = sample.closing_balance_cny + predicted_change
        # 计算余额变化绝对误差。
        change_error = abs(predicted_change - sample.target_change_cny)
        # 计算余额绝对误差。
        balance_error = abs(predicted_balance - sample.target_balance_cny)
        # 以截止日余额绝对值和冻结下限的较大者计算归一化误差。
        normalized_error = change_error / max(abs(sample.closing_balance_cny), sample.balance_floor_cny)
        # 读取同一键的锁定基线明细。
        baseline = baseline_by_key[sample.sample_key]
        # 以分为单位核对基线标签与T2验证标签一致。
        if Decimal(baseline["actual_target_change_cny"]) != sample.target_change_cny.quantize(Decimal("0.01")):
            # 任何标签不一致都阻断Skill Score，避免比较不同快照。
            raise LSTMContractError(f"锁定基线标签与T2不一致：{sample.sample_key}")
        # 追加一条可审计预测记录；身份和日期仅用于配对，不在张量里。
        records.append({
            # 写入候选名称。
            "model_name": MODEL_NAME,
            # 写入独立训练种子。
            "training_seed": str(seed),
            # 写入样本键。
            "sample_key": sample.sample_key,
            # 写入企业编号。
            "enterprise_id": sample.enterprise_id,
            # 写入切分组。
            "split_group": sample.split_group,
            # 写入预测截止日。
            "cutoff_date": sample.cutoff_date,
            # 写入目标日。
            "target_date": sample.target_date,
            # 写入真实目标余额。
            "actual_target_balance_cny": money(sample.target_balance_cny),
            # 写入真实目标变化。
            "actual_target_change_cny": money(sample.target_change_cny),
            # 写入预测目标余额。
            "predicted_target_balance_cny": money(predicted_balance),
            # 写入预测目标变化。
            "predicted_target_change_cny": money(predicted_change),
            # 写入变化绝对误差。
            "absolute_error_target_change_cny": money(change_error),
            # 写入余额绝对误差。
            "absolute_error_target_balance_cny": money(balance_error),
            # 写入账户归一化绝对误差。
            "account_normalized_absolute_error": ratio(normalized_error),
            # 写入同一样本锁定基线误差，便于逐行复核Skill Score。
            "locked_baseline_absolute_error_target_change_cny": baseline["absolute_error_target_change_cny"],
            # 写入T5实现版本。
            "feature_manifest_version": LSTM_VERSION,
        })
    # 返回完整3,904行验证预测。
    return records


# 汇总一个种子的验证指标，并按冻结公式计算相对锁定基线Skill Score。
def summarize_predictions(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    # 空结果不能形成模型结论。
    if not rows:
        # 抛出错误阻止空CSV或虚假0误差。
        raise LSTMContractError("LSTM验证预测为空")
    # 汇总模型变化绝对误差。
    model_error_sum = sum((Decimal(row["absolute_error_target_change_cny"]) for row in rows), Decimal("0"))
    # 汇总同一验证行锁定基线误差。
    baseline_error_sum = sum((Decimal(row["locked_baseline_absolute_error_target_change_cny"]) for row in rows), Decimal("0"))
    # 汇总余额绝对误差。
    balance_error_sum = sum((Decimal(row["absolute_error_target_balance_cny"]) for row in rows), Decimal("0"))
    # 汇总账户归一化误差。
    normalized_error_sum = sum((Decimal(row["account_normalized_absolute_error"]) for row in rows), Decimal("0"))
    # 要求基线总误差为正，保证Skill Score分母有效。
    if baseline_error_sum <= Decimal("0"):
        # 抛出异常，不用人为替代分母。
        raise LSTMContractError("锁定基线总绝对误差不是正数")
    # 将样本数转换为Decimal以精确计算均值。
    count = Decimal(len(rows))
    # 计算冻结定义的技能分数，正值才表示优于基线。
    skill_score = Decimal("1") - model_error_sum / baseline_error_sum
    # 返回一行完整种子指标。
    return {
        # 写入候选名称。
        "model_name": MODEL_NAME,
        # 写入训练种子。
        "training_seed": str(seed),
        # 写入验证样本数。
        "validation_sample_count": str(len(rows)),
        # 写入变化MAE。
        "mae_target_change_cny": money(model_error_sum / count),
        # 写入余额MAE。
        "mae_target_balance_cny": money(balance_error_sum / count),
        # 写入主指标。
        "account_normalized_mae": ratio(normalized_error_sum / count),
        # 写入模型误差合计。
        "sum_absolute_error_target_change_cny": money(model_error_sum),
        # 写入锁定基线误差合计。
        "locked_baseline_sum_absolute_error_target_change_cny": money(baseline_error_sum),
        # 写入Skill Score。
        "skill_score_vs_locked_baseline": ratio(skill_score),
        # 明确只在验证集评价。
        "evaluation_scope": "validation_only",
        # 写入实现版本。
        "feature_manifest_version": LSTM_VERSION,
    }


# 按企业汇总一个种子的验证误差；困难企业不会从总体或逐户报告中删除。
def summarize_by_enterprise(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    # 创建企业到预测行列表的映射。
    grouped: dict[str, list[dict[str, str]]] = {}
    # 逐条加入对应企业。
    for row in rows:
        # setdefault在首次出现时创建空列表，再追加当前行。
        grouped.setdefault(row["enterprise_id"], []).append(row)
    # 创建稳定排序的输出列表。
    results: list[dict[str, str]] = []
    # 按企业编号排序，保证重复运行文件顺序稳定。
    for enterprise_id in sorted(grouped):
        # 取出当前企业的所有验证日。
        enterprise_rows = grouped[enterprise_id]
        # 汇总模型变化误差。
        model_sum = sum((Decimal(row["absolute_error_target_change_cny"]) for row in enterprise_rows), Decimal("0"))
        # 汇总锁定基线变化误差。
        baseline_sum = sum((Decimal(row["locked_baseline_absolute_error_target_change_cny"]) for row in enterprise_rows), Decimal("0"))
        # 汇总归一化误差。
        normalized_sum = sum((Decimal(row["account_normalized_absolute_error"]) for row in enterprise_rows), Decimal("0"))
        # 计算当前企业样本数。
        count = Decimal(len(enterprise_rows))
        # 当基线误差为0时不伪造Skill Score。
        skill_score = "基线误差为0，无法计算" if baseline_sum == 0 else ratio(Decimal("1") - model_sum / baseline_sum)
        # 追加一行企业指标。
        results.append({
            # 写入训练种子。
            "training_seed": str(seed),
            # 写入企业编号。
            "enterprise_id": enterprise_id,
            # 写入验证样本数。
            "validation_sample_count": str(len(enterprise_rows)),
            # 写入变化MAE。
            "mae_target_change_cny": money(model_sum / count),
            # 写入账户归一化MAE。
            "account_normalized_mae": ratio(normalized_sum / count),
            # 写入相对该企业锁定基线的Skill Score。
            "skill_score_vs_locked_baseline": skill_score,
        })
    # 返回完整52户结果。
    return results


# 计算三个种子的中位数；合同要求以中位数而非最好种子代表LSTM家族。
def median_decimal(values: list[Decimal]) -> Decimal:
    # 要求恰有预注册的三次结果，缺一不可。
    if len(values) != len(TRAINING_SEEDS):
        # 抛出错误，禁止少跑一个种子后冒充家族指标。
        raise LSTMContractError("LSTM家族必须保留三个预注册种子")
    # 对三个数值排序后取中间值。
    return sorted(values)[1]


# 根据三个种子指标生成家族中位数报告；它不选择最佳单个checkpoint或种子。
def summarize_family(seed_metrics: list[dict[str, str]]) -> dict[str, object]:
    # 要求指标种子集合恰等于合同预注册集合。
    if {int(row["training_seed"]) for row in seed_metrics} != set(TRAINING_SEEDS):
        # 阻止临时增加、替换或遗漏种子。
        raise LSTMContractError("LSTM种子集合不等于冻结的2026083002—2026083004")
    # 计算主指标三种子中位数。
    median_primary = median_decimal([Decimal(row["account_normalized_mae"]) for row in seed_metrics])
    # 计算变化MAE三种子中位数。
    median_change_mae = median_decimal([Decimal(row["mae_target_change_cny"]) for row in seed_metrics])
    # 计算余额MAE三种子中位数。
    median_balance_mae = median_decimal([Decimal(row["mae_target_balance_cny"]) for row in seed_metrics])
    # 计算Skill Score三种子中位数。
    median_skill = median_decimal([Decimal(row["skill_score_vs_locked_baseline"]) for row in seed_metrics])
    # 返回完整家族记录。
    return {
        # 写入候选名称。
        "model_name": MODEL_NAME,
        # 写入所有必须保留的种子。
        "training_seeds": list(TRAINING_SEEDS),
        # 明确家族比较规则。
        "family_score_rule_cn": "三个独立训练种子均保留；使用三者验证集主指标中位数，不挑选最佳单个种子。",
        # 写入中位数主指标。
        "median_account_normalized_mae": ratio(median_primary),
        # 写入中位数变化MAE。
        "median_mae_target_change_cny": money(median_change_mae),
        # 写入中位数余额MAE。
        "median_mae_target_balance_cny": money(median_balance_mae),
        # 写入中位数Skill Score。
        "median_skill_score_vs_locked_baseline": ratio(median_skill),
        # 明确评价范围。
        "evaluation_scope": "validation_only",
        # 明确尚未选择唯一候选。
        "unique_candidate_selected": False,
        # 写入实现版本。
        "feature_manifest_version": LSTM_VERSION,
    }
