"""SYN-B1 T3 的五类共同基线与验证集锁定逻辑。

本模块只使用 T2 已生成的训练/验证派生快照；最终测试没有特征和标签，
因此任何调用本模块的代码都不具备读取最终测试余额的路径。
"""

# 导入 csv 标准库；它负责逐行读取 T2 的表格快照，避免引入额外数据库依赖。
import csv
# 导入 hashlib 标准库；它为输入和输出生成指纹，保证基线结果可追溯。
import hashlib
# 导入 json 标准库；它把锁定的基线配置写成可人工复核的结构化文件。
import json
# 从 dataclasses 导入 dataclass；它把每个监督样本表达为字段明确、不可随意拼错的对象。
from dataclasses import dataclass
# 从 decimal 导入 Decimal；它以十进制精确计算金额，避免二进制浮点误差影响金额指标。
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
# 从 math 导入 sqrt；它用于把训练集方差转换为标准差，供岭回归只在训练集标准化。
from math import sqrt
# 从 pathlib 导入 Path；它用跨平台路径对象定位冻结的 T2 文件。
from pathlib import Path
# 从 typing 导入 Iterable；它说明一个函数会逐条产出记录而不是一次性返回列表。
from typing import Iterable

# 固定本阶段的实现版本；它写入每条输出，避免未来实现混入当前基线结果。
BASELINE_VERSION = "syn_b1_pretrain_t3_baseline_1_0"
# 固定季节朴素法的周期为五个银行工作日；它表达“同一银行工作周位置”的可解释假设。
SEASONAL_PERIOD_BUSINESS_DAYS = 5
# 固定简单指数平滑的系数；0.20 让近期余额占20%、既有平滑水平占80%，不在验证集调参。
SES_ALPHA = Decimal("0.20")
# 固定岭回归的惩罚强度；特征先在训练集标准化后使用1.0，且不能根据验证结果改动。
RIDGE_ALPHA = 1.0
# 固定五类基线的比较顺序；完全相同的验证指标以该预注册顺序打破平局。
BASELINE_ORDER = ("last_balance_naive", "seasonal_naive", "window_average", "simple_exponential_smoothing", "ridge_regression")
# 固定 T2 特征清单中允许进入岭回归的字段；没有身份、日期、画像、种子或目标字段。
RIDGE_FEATURE_COLUMNS = (
    "closing_balance_cny", "daily_inflow_cny", "daily_outflow_cny", "daily_net_flow_cny", "transaction_count",
    "balance_lag_1bd_cny", "balance_change_1bd_cny", "balance_change_5bd_cny", "inflow_sum_5bd_cny",
    "outflow_sum_5bd_cny", "net_flow_sum_5bd_cny", "inflow_sum_14bd_cny", "outflow_sum_14bd_cny",
    "transaction_count_sum_14bd", "weekday_monday_0", "is_month_end_business_day", "calendar_gap_days_to_next_business_day",
)


# 声明专用异常类型；调用方看到它就知道是冻结输入或基线契约违规，而不是模型成绩问题。
class BaselineContractError(ValueError):
    """当T2输入、切分或最终测试密封规则不满足时抛出。"""


# 用不可变数据类描述一条树表样本；不可变可防止预测过程中被意外改写特征或标签。
@dataclass(frozen=True)
class TreeSample:
    # 保存样本唯一键；它只用来把树表、LSTM窗口和预测明细一一对应，绝不作为模型特征。
    sample_key: str
    # 保存企业编号；它只用于逐户归一化评价和审计，绝不作为模型特征。
    enterprise_id: str
    # 保存切分组；它确保训练只用于拟合、验证只用于选择，最终测试不会进入本类对象。
    split_group: str
    # 保存预测截止日；它只用于审计时间顺序，绝不作为模型特征。
    cutoff_date: str
    # 保存目标日；它只用于切分和配对审计，绝不作为模型特征。
    target_date: str
    # 保存截止日余额；它是预测时已经可见的金额，也是朴素基线的起点。
    closing_balance_cny: Decimal
    # 保存下一银行工作日实际余额；它只在训练和验证样本中存在，不能在最终测试阶段读取。
    target_balance_cny: Decimal
    # 保存下一银行工作日实际余额变化；它是本项目的主预测目标。
    target_change_cny: Decimal
    # 保存余额归一化下限；它避免低余额企业的相对误差被无限放大。
    balance_floor_cny: Decimal
    # 保存岭回归允许的数值特征；键必须严格等于冻结清单，不可自行增删。
    features: tuple[float, ...]


# 用不可变数据类保存一个LSTM窗口派生出的三种统计量；这让三个统计基线共享同一14日信息。
@dataclass(frozen=True)
class WindowStatistics:
    # 保存样本唯一键；它连接树表的标签与LSTM长表的14日历史。
    sample_key: str
    # 保存季节朴素法使用的余额；它是目标日前五个银行工作日的可见余额。
    seasonal_balance_cny: Decimal
    # 保存14日余额均值；它是窗口均值基线的预测余额。
    window_average_balance_cny: Decimal
    # 保存指数平滑后的水平值；它是简单指数平滑基线的预测余额。
    ses_level_balance_cny: Decimal


# 用不可变数据类保存已拟合的岭回归；均值和标准差都只能从训练集得到。
@dataclass(frozen=True)
class RidgeModel:
    # 保存训练集特征均值；验证集只能复用它，不能重新拟合。
    feature_means: tuple[float, ...]
    # 保存训练集特征标准差；零方差字段会安全地替换为1，避免除零。
    feature_scales: tuple[float, ...]
    # 保存包含截距的系数；第0个元素对应截距，之后依次对应冻结特征。
    coefficients: tuple[float, ...]


# 将文件字节计算为SHA-256；指纹让后续模型阶段能确认基线输入没有被覆盖。
def sha256_file(path: Path) -> str:
    # 读取整个冻结快照的字节并传给哈希函数；此处不解释或转换任何数据内容。
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 将文本金额转换为Decimal；发生无效金额时给出字段名，方便定位契约问题。
def _decimal(value: str, field: str) -> Decimal:
    # 用try语句捕获Decimal转换错误，防止错误值静默进入评价。
    try:
        # 返回十进制金额；Decimal保持分级精度而不是使用浮点近似。
        return Decimal(value)
    # 捕获非法十进制和空字符串等两类常见异常。
    except (InvalidOperation, ValueError) as exc:
        # 抛出项目专用异常并保留原始异常链，便于测试和用户理解失败原因。
        raise BaselineContractError(f"{field}不是有效金额：{value!r}") from exc


# 将金额统一格式化到分；预测明细和指标因此可稳定比较和复现。
def _money(value: Decimal) -> str:
    # quantize将金额四舍五入到0.01元，ROUND_HALF_UP符合普通金额展示习惯。
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# 将比例统一格式化到八位小数；账户归一化误差需要较高精度但不应无限展开。
def _ratio(value: Decimal) -> str:
    # 用固定格式f避免Decimal把0显示为科学计数法，方便Excel和人工阅读。
    return f"{value.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP):f}"


# 逐行读取UTF-8或UTF-8-BOM CSV；T2快照兼容Excel友好导出而不改写源文件。
def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    # 以utf-8-sig打开文件会自动忽略BOM，同时仍支持普通UTF-8。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # DictReader把首行字段名映射为字典键，保证后续按冻结字段取值。
        yield from csv.DictReader(handle)


# 读取T2阶段报告并确认最终测试仍密封；这是任何基线计算之前的硬门。
def validate_t2_report(t2_directory: Path) -> dict[str, object]:
    # 构造报告路径；报告由T2写入且不能由本阶段覆盖。
    report_path = t2_directory / "feature_construction_report.json"
    # 若报告不存在则立即阻断，避免把未知来源CSV当作训练数据。
    if not report_path.is_file():
        # 抛出明确错误，让调用方知道缺少T2前置证据而不是模型效果差。
        raise BaselineContractError("缺少T2特征构造报告")
    # 读取JSON文本并解析为字典；该报告不包含逐笔流水或人工备注。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 检查T2是否通过；失败的派生数据不得进入基线比较。
    if report.get("status") != "passed":
        # 抛出阻断异常，保证不会在失败输入上产生看似正常的基线成绩。
        raise BaselineContractError("T2特征构造未通过")
    # 检查最终测试标签是否仍未读取或组装；这是本任务最关键的不可逆边界。
    if report.get("final_test_labels_read_or_assembled") is not False:
        # 一旦该值不是False，说明密封状态已被破坏，本阶段必须停止。
        raise BaselineContractError("最终测试标签已被读取或组装，禁止运行基线")
    # 检查最终测试特征是否仍未生成；T3只允许训练和验证快照参与比较。
    if report.get("final_test_features_created") is not False:
        # 抛出异常，避免意外使用已开放的最终测试特征。
        raise BaselineContractError("最终测试特征已被生成，禁止运行基线")
    # 返回通过核验的报告，供调用方写入输入血缘指纹。
    return report


# 从T2树表读取训练或验证样本；最终测试行在T2树表中必须为零，因此遇到即视为契约错误。
def read_tree_samples(path: Path, allowed_group: str) -> list[TreeSample]:
    # 创建空列表收集同一切分组的样本；训练组用于拟合，验证组用于锁定基线。
    samples: list[TreeSample] = []
    # 逐行读取树表，避免把LSTM长表或逐笔流水混入当前计算。
    for row in _read_csv(path):
        # 读取当前行的切分组，后续只接受调用方显式指定的训练或验证组。
        split_group = row.get("split_group", "")
        # 若出现最终测试行，立即失败；这验证T2的密封边界没有被意外改变。
        if split_group == "final_test":
            # 抛出契约错误，因为最终测试特征在候选锁定前不得存在于本任务。
            raise BaselineContractError("T2树特征表不应含最终测试行")
        # 跳过另一组数据；同一文件可以顺序读取训练与验证，但每次函数只返回一个组。
        if split_group != allowed_group:
            # continue跳到下一行，避免验证标签参与岭回归拟合或训练标签参与验证评价。
            continue
        # 读取所有冻结岭回归特征并转换为float；这些字段来自T2公开日度历史。
        features = tuple(float(row[name]) for name in RIDGE_FEATURE_COLUMNS)
        # 构造不可变样本对象，明确区分可见特征、目标和仅审计字段。
        samples.append(TreeSample(
            # 记录样本键用于与LSTM窗口统计及预测明细配对。
            sample_key=row["sample_key"],
            # 记录企业编号用于逐户归一化误差，但不会作为岭回归输入。
            enterprise_id=row["enterprise_id"],
            # 记录切分组以供后续断言当前读取路径正确。
            split_group=split_group,
            # 记录截止日用于审计预测时点。
            cutoff_date=row["cutoff_date"],
            # 记录目标日用于验证集配对和时间边界复核。
            target_date=row["target_date"],
            # 读取当日已知余额，它是所有简单余额基线的共同起点。
            closing_balance_cny=_decimal(row["closing_balance_cny"], "closing_balance_cny"),
            # 读取下一日实际余额，它仅在训练/验证中作为标签存在。
            target_balance_cny=_decimal(row["target_balance_cny"], "target_balance_cny"),
            # 读取下一日实际余额变化，它是本项目的主评价目标。
            target_change_cny=_decimal(row["target_change_cny"], "target_change_cny"),
            # 读取余额归一化下限，保证各企业的相对误差计算口径一致。
            balance_floor_cny=_decimal(row["balance_floor_cny"], "balance_floor_cny"),
            # 保存已经转换的冻结数值特征元组，供岭回归训练和验证预测使用。
            features=features,
        ))
    # 若指定组没有样本就阻断，因为空验证集不能锁定任何基线。
    if not samples:
        # 抛出异常并包含组名，方便用户定位缺少训练还是缺少验证快照。
        raise BaselineContractError(f"树特征表没有{allowed_group}样本")
    # 返回按CSV原顺序排列的样本；该顺序来自T2冻结写出而非本阶段按结果重排。
    return samples


# 从T2 LSTM长表提取三种统计基线所需的14日余额；只允许读取验证组，避免无关数据扩散。
def read_validation_window_statistics(path: Path) -> dict[str, WindowStatistics]:
    # 用字典按sample_key临时聚集14行；每个键最终必须恰好有14个余额。
    balances_by_sample: dict[str, list[Decimal]] = {}
    # 逐行读取LSTM长表；该表包含历史日度余额但不含人工备注。
    for row in _read_csv(path):
        # 读取当前切分组，确保最终测试行从未进入此函数。
        split_group = row.get("split_group", "")
        # 若出现最终测试行则立即拒绝，保证T3不触碰最终测试历史或标签。
        if split_group == "final_test":
            # 抛出异常使任何密封破坏都不能被忽略。
            raise BaselineContractError("T2 LSTM序列表不应含最终测试行")
        # 只收集验证组，因为基线锁定依据验证集而非训练集成绩。
        if split_group != "validation":
            # continue跳过训练组，减少内存占用并避免误把训练成绩用于选择。
            continue
        # 取出样本键；相同键的14行按sequence_position从早到晚写出。
        sample_key = row["sample_key"]
        # 若该键首次出现则创建空列表，setdefault返回可追加的同一列表对象。
        values = balances_by_sample.setdefault(sample_key, [])
        # 追加当前历史余额；它是当前预测截止日之前可见的值。
        values.append(_decimal(row["closing_balance_cny"], "lstm.closing_balance_cny"))
    # 创建最终统计字典；键保持样本键，使树表标签能精确配对到同一14日窗口。
    statistics: dict[str, WindowStatistics] = {}
    # 遍历每个样本的14日余额；字典遍历顺序不影响单个样本的统计计算。
    for sample_key, balances in balances_by_sample.items():
        # 要求窗口长度恰为14，防止缺行或多行导致LSTM和统计基线使用不同信息。
        if len(balances) != 14:
            # 抛出异常并带上样本键，便于审计具体数据缺口。
            raise BaselineContractError(f"验证LSTM窗口不是14行：{sample_key}")
        # 取目标日前五个银行工作日的余额；14行窗口的第10项是截止日前4日，即目标日前5日。
        seasonal_balance = balances[-5]
        # 计算14日余额总和；sum从Decimal零值开始，保持金额精确性。
        window_total = sum(balances, Decimal("0"))
        # 计算14日平均余额；它是窗口均值基线的下一日余额预测。
        window_average = window_total / Decimal(len(balances))
        # 用最早一天余额初始化SES水平；这符合简单指数平滑的递推起点。
        level = balances[0]
        # 依次处理之后13天可见余额，使最近历史在SES中获得更高权重。
        for balance in balances[1:]:
            # 更新平滑水平：alpha乘当前观察，1-alpha乘此前水平；不读取未来目标。
            level = SES_ALPHA * balance + (Decimal("1") - SES_ALPHA) * level
        # 保存同一窗口得到的三项统计量；它们会与树表的同一sample_key配对。
        statistics[sample_key] = WindowStatistics(sample_key, seasonal_balance, window_average, level)
    # 若验证长表为空则阻断，因为缺少季节、均值和SES基线所需的共同历史。
    if not statistics:
        # 抛出异常，避免出现“只有岭回归被比较”的不完整基线集合。
        raise BaselineContractError("验证LSTM序列表没有可用窗口统计")
    # 返回按样本键索引的统计量，供预测循环O(1)查找。
    return statistics


# 计算训练集每个特征的均值和标准差；验证集只能使用这里得到的尺度。
def _training_standardizer(samples: list[TreeSample]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    # 读取冻结特征数量，后续所有向量必须保持同一长度。
    width = len(RIDGE_FEATURE_COLUMNS)
    # 创建全零累加器保存每个特征的和。
    sums = [0.0] * width
    # 创建全零累加器保存每个特征平方和，用于方差计算。
    squared_sums = [0.0] * width
    # 遍历训练样本；验证样本绝不会传入此函数。
    for sample in samples:
        # 同时遍历特征序号与值，便于把值累计到对应列。
        for index, value in enumerate(sample.features):
            # 累加原始值，稍后除以样本数得到均值。
            sums[index] += value
            # 累加平方值，稍后结合均值得到方差。
            squared_sums[index] += value * value
    # 把整数样本量转换为float，避免Python整数除法语义干扰后续表达。
    count = float(len(samples))
    # 用列表推导逐列计算训练均值；它不读取验证数据。
    means = tuple(total / count for total in sums)
    # 创建标准差列表；每项至少为1以处理常数特征。
    scales: list[float] = []
    # 同时遍历平方和与均值，以稳定方式计算每列方差。
    for squared_total, mean in zip(squared_sums, means):
        # 计算总体方差并用max抵消极小的浮点负误差。
        variance = max(0.0, squared_total / count - mean * mean)
        # 计算标准差；若为0则用1保留零标准化值并避免除零。
        scales.append(sqrt(variance) if variance > 0.0 else 1.0)
    # 返回不可变均值和尺度元组，后续岭回归对象不能被调用方修改。
    return means, tuple(scales)


# 用高斯消元求解小型线性方程组；岭回归只有18个参数，纯标准库即可稳定完成。
def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    # 读取方程维度；它等于截距加17个冻结特征。
    size = len(vector)
    # 创建增广矩阵副本，避免原始X'X被后续审计或测试意外改写。
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    # 逐列选择主元并消去该列；range遍历每一个未知数的位置。
    for pivot_index in range(size):
        # 在当前列及其下方寻找绝对值最大的行，提高数值稳定性。
        pivot_row = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        # 若最大主元极小，说明训练特征矩阵退化且岭惩罚仍无法使其可逆。
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            # 抛出契约错误，禁止把数值不可靠的岭系数写成基线结果。
            raise BaselineContractError("岭回归正规方程不可逆")
        # 若最大主元不在当前位置，就交换两行以便本列从稳定主元开始消元。
        if pivot_row != pivot_index:
            # Python的并列赋值语法在一条语句中安全交换两个列表引用。
            augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        # 读取当前主元，之后把这一行标准化为主元等于1。
        pivot = augmented[pivot_index][pivot_index]
        # 对增广行从当前列到末尾逐项除以主元。
        for column in range(pivot_index, size + 1):
            # 原地更新当前单元，使主元行成为标准化的基向量。
            augmented[pivot_index][column] /= pivot
        # 对其他所有行消去当前列，得到简化行阶梯形式。
        for row_index in range(size):
            # 主元行自身无需消去，continue跳过它减少无意义计算。
            if row_index == pivot_index:
                # 继续下一行的循环。
                continue
            # 读取该行在主元列的系数，作为本次消去的倍数。
            factor = augmented[row_index][pivot_index]
            # 若该系数已为0则无需更新整行。
            if factor == 0.0:
                # 继续下一行的循环，保持算法表达清晰。
                continue
            # 对增广行从当前列到末尾执行“本行减去倍数乘主元行”。
            for column in range(pivot_index, size + 1):
                # 更新当前元素，逐步将主元列变成0。
                augmented[row_index][column] -= factor * augmented[pivot_index][column]
    # 在简化后每行最后一列就是对应未知数，按顺序组成系数向量。
    return [augmented[index][size] for index in range(size)]


# 只用训练样本拟合岭回归；验证标签从不参与均值、尺度、正规方程或系数计算。
def fit_ridge(train_samples: list[TreeSample]) -> RidgeModel:
    # 再次断言所有样本确属训练组，防止调用方错误把验证样本传进拟合函数。
    if any(sample.split_group != "train" for sample in train_samples):
        # 发现切分污染立即失败，而不是悄悄计算一组泄漏系数。
        raise BaselineContractError("岭回归拟合只能使用训练样本")
    # 从训练样本计算均值和标准差；它们是训练集唯一拟合的预处理器。
    means, scales = _training_standardizer(train_samples)
    # 维度为截距1加特征数17。
    width = len(RIDGE_FEATURE_COLUMNS) + 1
    # 初始化X转置乘X的方阵；每个元素随后累加训练样本贡献。
    xtx = [[0.0] * width for _ in range(width)]
    # 初始化X转置乘y的向量；y是训练集实际余额变化。
    xty = [0.0] * width
    # 遍历每个训练样本，把其标准化特征和目标加入正规方程。
    for sample in train_samples:
        # 在标准化特征前添加1.0截距，列表拼接语法形成长度18的设计行。
        design = [1.0] + [(value - mean) / scale for value, mean, scale in zip(sample.features, means, scales)]
        # 把十进制目标变化转为float，仅用于线性代数；原始金额仍保存在样本与预测明细中。
        target = float(sample.target_change_cny)
        # 遍历设计向量的每一列，为X'X和X'y累积行贡献。
        for row_index, row_value in enumerate(design):
            # 累加当前列与目标的乘积，形成X'y的对应元素。
            xty[row_index] += row_value * target
            # 遍历设计向量的每一列，形成当前样本对X'X的外积贡献。
            for column_index, column_value in enumerate(design):
                # 累加行值乘列值，逐样本构建正规方程。
                xtx[row_index][column_index] += row_value * column_value
    # 仅对非截距列添加岭惩罚，避免把整体平均目标强行向0收缩。
    for index in range(1, width):
        # 给X'X的对角线增加固定alpha，稳定共线特征的求解。
        xtx[index][index] += RIDGE_ALPHA
    # 解出截距和特征系数；若不可逆会由求解器明确阻断。
    coefficients = _solve_linear_system(xtx, xty)
    # 返回不可变模型对象，后续验证预测只能调用它而不能重新拟合。
    return RidgeModel(means, scales, tuple(coefficients))


# 用训练期已拟合的岭模型预测余额变化；此函数不接受标签，避免把预测写成泄漏计算。
def ridge_predict_change(model: RidgeModel, sample: TreeSample) -> Decimal:
    # 构建与训练阶段同顺序、同尺度的设计向量，首项仍为截距1。
    design = [1.0] + [(value - mean) / scale for value, mean, scale in zip(sample.features, model.feature_means, model.feature_scales)]
    # 用生成器表达式逐项相乘并求和，得到浮点的预测余额变化。
    predicted = sum(coefficient * value for coefficient, value in zip(model.coefficients, design))
    # 用str包装float再转Decimal，避免Decimal直接接收二进制浮点尾差。
    return Decimal(str(predicted))


# 计算账户归一化绝对误差；分母只使用预测截止日已知余额和冻结下限。
def account_normalized_absolute_error(sample: TreeSample, predicted_change: Decimal) -> Decimal:
    # 计算预测与实际变化的绝对金额差，abs保证正负变化均按同一尺度评价。
    absolute_error = abs(predicted_change - sample.target_change_cny)
    # 取截止日余额绝对值与下限的较大者，防止低余额账户支配总体指标。
    denominator = max(abs(sample.closing_balance_cny), sample.balance_floor_cny)
    # 返回无量纲误差比例，后续对所有验证样本求平均就是主指标。
    return absolute_error / denominator


# 为一个验证样本生成五类基线预测；所有预测都只用该样本截止日前的T2历史。
def prediction_rows(sample: TreeSample, statistics: WindowStatistics, ridge_model: RidgeModel) -> Iterable[dict[str, str]]:
    # 计算“明日余额等于今日余额”的朴素预测变化，恒为0且不使用未来信息。
    last_change = Decimal("0")
    # 计算季节朴素预测变化，即目标日前五个银行工作日余额减去当前截止日余额。
    seasonal_change = statistics.seasonal_balance_cny - sample.closing_balance_cny
    # 计算窗口均值预测变化，即14日平均余额减去当前截止日余额。
    window_change = statistics.window_average_balance_cny - sample.closing_balance_cny
    # 计算简单指数平滑预测变化，即平滑水平减去当前截止日余额。
    ses_change = statistics.ses_level_balance_cny - sample.closing_balance_cny
    # 调用已拟合岭模型预测变化；模型仅由训练集拟合，当前调用不读取验证标签。
    ridge_change = ridge_predict_change(ridge_model, sample)
    # 按冻结候选顺序组织方法与预测变化，顺序也将用于稳定的同分平局处理。
    candidates = (("last_balance_naive", last_change), ("seasonal_naive", seasonal_change), ("window_average", window_change), ("simple_exponential_smoothing", ses_change), ("ridge_regression", ridge_change))
    # 逐个候选产出预测明细；yield使调用方可直接写CSV而无需保存全部行。
    for method, predicted_change in candidates:
        # 由截止日余额加预测变化得到预测期末余额，保持目标变化和余额两个报告口径一致。
        predicted_balance = sample.closing_balance_cny + predicted_change
        # 计算余额变化绝对误差，供主指标和Skill Score的后续配对使用。
        change_absolute_error = abs(predicted_change - sample.target_change_cny)
        # 计算余额绝对误差，供业务阅读“预测期末余额差多少元”。
        balance_absolute_error = abs(predicted_balance - sample.target_balance_cny)
        # 计算账户归一化误差，避免大额账户凭金额规模压过小额账户。
        normalized_error = account_normalized_absolute_error(sample, predicted_change)
        # 产出一条可审计预测记录；身份和日期列只供配对审计，模型输入清单不包含它们。
        yield {
            "baseline_method": method,
            "sample_key": sample.sample_key,
            "enterprise_id": sample.enterprise_id,
            "split_group": sample.split_group,
            "cutoff_date": sample.cutoff_date,
            "target_date": sample.target_date,
            "actual_target_balance_cny": _money(sample.target_balance_cny),
            "actual_target_change_cny": _money(sample.target_change_cny),
            "predicted_target_balance_cny": _money(predicted_balance),
            "predicted_target_change_cny": _money(predicted_change),
            "absolute_error_target_change_cny": _money(change_absolute_error),
            "absolute_error_target_balance_cny": _money(balance_absolute_error),
            "account_normalized_absolute_error": _ratio(normalized_error),
            "feature_manifest_version": BASELINE_VERSION,
        }


# 汇总每个基线在验证集的配对指标；所有方法必须覆盖同一批样本才能比较。
def summarize_validation_predictions(rows: list[dict[str, str]], expected_samples: int) -> list[dict[str, str]]:
    # 创建按方法聚集的容器；五个键来自冻结顺序而非运行结果。
    grouped: dict[str, list[dict[str, str]]] = {method: [] for method in BASELINE_ORDER}
    # 遍历预测明细，将每行放入其声明的方法组。
    for row in rows:
        # 读取方法名；未知方法说明实现越过冻结合同，应立即拒绝。
        method = row["baseline_method"]
        # 检查方法是否在预注册集合中。
        if method not in grouped:
            # 抛出错误，避免额外方法借机影响最佳基线选择。
            raise BaselineContractError(f"出现未冻结基线：{method}")
        # 把当前预测行加入对应集合，供后续逐方法求和。
        grouped[method].append(row)
    # 创建最终指标列表；顺序保持BASELINE_ORDER，确保可复现平局规则。
    metrics: list[dict[str, str]] = []
    # 按冻结顺序计算五个方法的聚合指标。
    for method in BASELINE_ORDER:
        # 读取该方法全部验证预测行。
        method_rows = grouped[method]
        # 要求每个方法都有且仅有每个验证样本一条预测，保证公平配对。
        if len(method_rows) != expected_samples:
            # 抛出异常并说明实际行数，防止缺预测的方法因少算困难样本而占优。
            raise BaselineContractError(f"{method}验证预测行数不是{expected_samples}：{len(method_rows)}")
        # 汇总变化绝对误差金额，用于辅助MAE与后续复杂模型Skill Score分母。
        change_error_sum = sum((_decimal(row["absolute_error_target_change_cny"], "change_error") for row in method_rows), Decimal("0"))
        # 汇总余额绝对误差金额，便于业务理解期末余额的平均偏差。
        balance_error_sum = sum((_decimal(row["absolute_error_target_balance_cny"], "balance_error") for row in method_rows), Decimal("0"))
        # 汇总账户归一化误差比例，它是冻结的主选择指标。
        normalized_sum = sum((_decimal(row["account_normalized_absolute_error"], "normalized_error") for row in method_rows), Decimal("0"))
        # 把样本数转为Decimal，保证金额/比例除法仍保持十进制精度。
        count = Decimal(len(method_rows))
        # 写入一条方法级指标；此处不写最终测试，也不做复杂模型比较。
        metrics.append({
            "baseline_method": method,
            "validation_sample_count": str(len(method_rows)),
            "mae_target_change_cny": _money(change_error_sum / count),
            "mae_target_balance_cny": _money(balance_error_sum / count),
            "account_normalized_mae": _ratio(normalized_sum / count),
            "sum_absolute_error_target_change_cny": _money(change_error_sum),
            "selection_scope": "validation_only",
            "feature_manifest_version": BASELINE_VERSION,
        })
    # 返回五条完整指标，调用方随后只按主指标和固定顺序锁定一条。
    return metrics


# 从验证指标中选出唯一最佳基线；若主指标完全相同，固定候选顺序优先，不查看最终测试。
def lock_best_baseline(metrics: list[dict[str, str]]) -> dict[str, str]:
    # 建立方法到指标的映射，后续可以按冻结顺序稳定地访问。
    by_method = {row["baseline_method"]: row for row in metrics}
    # 要求映射键刚好覆盖五个预注册方法，防止缺项或额外方法改变选择。
    if tuple(by_method) != BASELINE_ORDER:
        # 字典保留插入顺序，但这里更直接地比较键集合以避免CSV顺序影响判断。
        if set(by_method) != set(BASELINE_ORDER):
            # 抛出异常，强制调用方先修复完整的共同基线计算。
            raise BaselineContractError("验证指标没有完整覆盖五类冻结基线")
    # 按主指标升序比较；第二排序键是冻结候选的索引，从而稳定处理精确同分。
    winner = min(BASELINE_ORDER, key=lambda method: (Decimal(by_method[method]["account_normalized_mae"]), BASELINE_ORDER.index(method)))
    # 读取胜出方法的完整指标行，后续写入锁定JSON和模型阶段的基线分母。
    winner_metric = by_method[winner]
    # 返回明确的锁定记录；它不包含任何最终测试读取行为。
    return {
        "locked_baseline_method": winner,
        "selection_metric": "account_normalized_mae",
        "selection_metric_value": winner_metric["account_normalized_mae"],
        "tie_breaker_cn": "主指标完全相同则按预注册候选顺序选择更靠前的方法；未读取最终测试。",
        "selection_scope": "validation_only",
        "feature_manifest_version": BASELINE_VERSION,
    }
