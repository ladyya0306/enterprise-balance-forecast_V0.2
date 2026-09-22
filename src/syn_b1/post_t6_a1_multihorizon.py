"""POST-T6-A1：只为208户开发池构造多期限公开特征、标签和四折索引。"""

# 导入csv标准库；它读取冻结的公开CSV并写出可由Excel查看的派生表。
import csv
# 导入hashlib标准库；它为输入和输出文件生成SHA-256指纹以便复核。
import hashlib
# 导入json标准库；它读取T1报告并写出A1机器可读报告。
import json
# 从collections导入defaultdict；它把逐笔交易金额按日期归入公开工作日。
from collections import defaultdict
# 从dataclasses导入dataclass；它把工作日和样本描述成字段清楚的不可变对象。
from dataclasses import dataclass
# 从datetime导入date；它安全比较切分日期而不是比较文本外观。
from datetime import date
# 从decimal导入金额类型和舍入规则；人民币金额计算不依赖二进制浮点近似。
from decimal import Decimal, ROUND_HALF_UP
# 从math导入ceil和sqrt；前者划分按日期的80%拟合集，后者计算公开净流量标准差。
from math import ceil, sqrt
# 从pathlib导入Path；它用跨平台路径对象定位冻结输入和新输出。
from pathlib import Path
# 从typing导入Iterable；它标注会逐行产生CSV记录的函数。
from typing import Iterable

# 导入yaml；它读取A0冻结合同中的期限、窗口和四折日期，避免在代码里另写一套规则。
import yaml

# 固定A1实现版本；以后变更字段、切分或计算口径必须新建版本，不能覆盖本次结果。
A1_VERSION = "syn_b1_post_t6_a1_multihorizon_derivative_1_0"
# 固定最长公开历史窗口；A0要求每个样本至少拥有60个银行工作日历史。
MAX_LOOKBACK_BUSINESS_DAYS = 60
# 固定三个已登记观察窗口；树模型和LSTM之后只能从这三种历史长度中选择。
LOOKBACK_BUSINESS_DAYS = (14, 28, 60)
# 固定完整未来路径长度；A0要求一次准备从T+1到T+30的标签。
FORECAST_PATH_BUSINESS_DAYS = 30
# 固定需要报告的四个期限节点；它们与完整30日路径共用同一批截止点。
REGISTERED_HORIZONS = (1, 5, 10, 30)
# 固定人民币归一化下限；沿用首轮T2的一百万元下限以保持历史可比性。
BALANCE_FLOOR_CNY = Decimal("1000000.00")
# 固定四折企业组名称；排序后按名次轮流分组时使用此顺序。
FOLD_GROUPS = ("A", "B", "C", "D")


# 声明A1专用异常；调用者可以区分合同、输入或切分边界失败。
class A1ContractError(ValueError):
    """当A1遇到冻结合同、公开数据或密封边界违规时抛出。"""


# 用不可变对象保存一个已公开的银行工作日日度记录。
@dataclass(frozen=True)
class BusinessDay:
    # 保存自然日期；它只用于时间顺序、日历特征和目标对齐。
    calendar_date: date
    # 保存公开日流入金额；它属于预测截止日及以前可见的账户历史。
    inflow_cny: Decimal
    # 保存公开日流出金额；它属于预测截止日及以前可见的账户历史。
    outflow_cny: Decimal
    # 保存公开日交易笔数；它反映可见的活跃程度。
    transaction_count: int
    # 保存工作日日终余额；它是余额预测的可见起点。
    ending_balance_cny: Decimal
    # 保存连续银行工作日序号；它保证窗口按银行工作日而不是自然日取数。
    business_step: int

    # 定义净流量属性；它统一由流入减流出得到，避免CSV中存在多份不一致定义。
    @property
    def net_flow_cny(self) -> Decimal:
        # 返回当日净流量；正数表示净流入、负数表示净流出。
        return self.inflow_cny - self.outflow_cny


# 用不可变对象保存一个可建模截止点的元信息；大体量特征会立即写盘而不长期占用内存。
@dataclass(frozen=True)
class SampleDescriptor:
    # 保存稳定样本键；它只用于把输入、标签和折索引一一连接，绝不作为模型特征。
    sample_key: str
    # 保存企业编号；它只用于企业隔离和评价，不进入树模型或LSTM。
    enterprise_id: str
    # 保存企业的四折组；它只用于切分，不输入模型。
    enterprise_fold_group: str
    # 保存预测截止日；特征只能来自此日及以前。
    cutoff_date: date
    # 保存未来第一天；验证窗口从它开始判断。
    future_start_date: date
    # 保存未来第5天；它是T+5主路径的期末节点。
    target_date_t5: date
    # 保存未来第10天；它是T+10辅助节点。
    target_date_t10: date
    # 保存未来第30天；它也是完整路径是否落在切分区间的硬边界。
    target_date_t30: date


# 固定目标表的基础列；后续30个余额和30个变化列由函数按期限展开。
TARGET_BASE_COLUMNS = (
    "sample_key", "enterprise_id", "cutoff_date", "future_start_date", "target_date_t5", "target_date_t10", "target_date_t30",
    "cutoff_balance_cny", "balance_scale_cny", "balance_floor_cny", "minimum_balance_t30_cny", "liquidity_threshold_cny", "feature_manifest_version",
)
# 固定树表的审计和公开特征列；不含任何target字段，避免训练器误把答案当输入。
TREE_BASE_COLUMNS = (
    "sample_key", "enterprise_id", "cutoff_date", "lookback_start_date", "lookback_business_days", "business_step",
    "closing_balance_cny", "daily_inflow_cny", "daily_outflow_cny", "daily_net_flow_cny", "transaction_count",
    "weekday_monday_0", "business_day_position_in_month", "business_days_to_month_end", "is_month_end_business_day",
    "calendar_gap_days_from_prior_business_day", "calendar_gap_days_to_next_business_day",
)
# 固定树表每个窗口都计算的公开历史统计列；名称中不包含企业画像、种子或未来真值。
TREE_SUMMARY_COLUMNS = (
    "inflow_sum_cny", "outflow_sum_cny", "net_flow_sum_cny", "absolute_net_flow_sum_cny", "transaction_count_sum",
    "active_day_ratio", "net_flow_mean_cny", "net_flow_standard_deviation_cny", "transaction_amount_p50_cny",
    "transaction_amount_p90_cny", "days_since_last_transaction", "days_since_last_inflow", "days_since_last_outflow",
    "last_nonzero_inflow_amount_cny", "last_nonzero_outflow_amount_cny", "feature_manifest_version",
)
# 固定树模型真正允许读取的数值和公开日历特征；样本键、企业编号、日期和版本只用于审计与连接。
TREE_MODEL_INPUT_COLUMNS = (
    "closing_balance_cny", "daily_inflow_cny", "daily_outflow_cny", "daily_net_flow_cny", "transaction_count",
    "weekday_monday_0", "business_day_position_in_month", "business_days_to_month_end", "is_month_end_business_day",
    "calendar_gap_days_from_prior_business_day", "calendar_gap_days_to_next_business_day", "inflow_sum_cny",
    "outflow_sum_cny", "net_flow_sum_cny", "absolute_net_flow_sum_cny", "transaction_count_sum", "active_day_ratio",
    "net_flow_mean_cny", "net_flow_standard_deviation_cny", "transaction_amount_p50_cny", "transaction_amount_p90_cny",
    "days_since_last_transaction", "days_since_last_inflow", "days_since_last_outflow", "last_nonzero_inflow_amount_cny",
    "last_nonzero_outflow_amount_cny",
)
# 固定LSTM长表的列；每一行只表示历史窗口中的一天，不重复写入未来标签。
SEQUENCE_COLUMNS = (
    "sample_key", "enterprise_id", "cutoff_date", "sequence_position", "sequence_length", "calendar_date", "business_step",
    "calendar_gap_days_from_prior_business_day", "calendar_gap_days_to_next_business_day", "closing_balance_cny",
    "daily_inflow_cny", "daily_outflow_cny", "daily_net_flow_cny", "transaction_count", "day_has_transaction",
    "weekday_monday_0", "business_day_position_in_month", "business_days_to_month_end", "is_month_end_business_day", "feature_manifest_version",
)
# 固定LSTM每个历史日真正允许读取的输入；样本键、企业编号、截止日和位置只帮助重建序列，绝不进入网络张量。
SEQUENCE_MODEL_INPUT_COLUMNS = (
    "calendar_gap_days_from_prior_business_day", "calendar_gap_days_to_next_business_day", "closing_balance_cny", "daily_inflow_cny",
    "daily_outflow_cny", "daily_net_flow_cny", "transaction_count", "day_has_transaction", "weekday_monday_0",
    "business_day_position_in_month", "business_days_to_month_end", "is_month_end_business_day",
)
# 固定四折索引列；角色只标示fit、calibration或validation，绝不含特征或标签金额。
FOLD_INDEX_COLUMNS = (
    "sample_key", "enterprise_id", "enterprise_fold_group", "fold_id", "fold_role", "cutoff_date", "future_start_date",
    "target_date_t5", "target_date_t10", "target_date_t30", "feature_manifest_version",
)


# 返回30个未来余额和30个未来变化的动态表头；编号补零保证Excel和文本排序一致。
def target_columns() -> tuple[str, ...]:
    # 生成未来期末余额列；每列对应截止日之后一个明确的银行工作日。
    balance_columns = tuple(f"target_balance_t_plus_{horizon:02d}_cny" for horizon in range(1, FORECAST_PATH_BUSINESS_DAYS + 1))
    # 生成未来余额变化列；每列均相对同一个截止日余额计算。
    change_columns = tuple(f"target_change_t_plus_{horizon:02d}_cny" for horizon in range(1, FORECAST_PATH_BUSINESS_DAYS + 1))
    # 返回基础列和两段动态列；所有输入和标签使用这份唯一列清单。
    return (*TARGET_BASE_COLUMNS, *balance_columns, *change_columns)


# 计算文件的SHA-256；输入指纹可证明A1没有在变更后的流水上运行。
def sha256_file(path: Path) -> str:
    # 读取文件原始字节并计算哈希；不改写、不解析文件内容。
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 将Decimal金额按人民币分展示；格式化只影响CSV文本，不改变内部精确金额。
def money(value: Decimal) -> str:
    # 使用普通财务四舍五入保留两位小数。
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# 将Decimal比例以八位小数展示；它足以用于误差和覆盖率审计。
def ratio(value: Decimal) -> str:
    # 使用定点格式避免Excel把小比例自动转成科学计数法。
    return f"{value.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP):f}"


# 读取UTF-8或带BOM的CSV；只返回文本字段，金额转换由调用方显式完成。
def read_csv(path: Path) -> list[dict[str, str]]:
    # 以utf-8-sig打开可同时兼容普通UTF-8和Excel写出的BOM。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # 将全部行转成字典列表；A1单户文件规模可控，便于严格校验字段。
        return list(csv.DictReader(handle))


# 读取A0冻结合同；本函数只读配置文件，不读取任何企业运行目录。
def load_a0_contract(contract_path: Path) -> dict[str, object]:
    # 安全解析YAML文本；合同不允许包含可执行对象。
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    # 要求阶段编号匹配，防止误把其他YAML作为A0边界。
    if contract.get("stage_id") != "POST-T6-A0":
        # 抛出清楚错误，使错误合同不会静默进入A1。
        raise A1ContractError("A0合同阶段编号不正确")
    # 要求合同已冻结，草稿不能作为派生数据依据。
    if contract.get("status") != "frozen":
        # 阻断未冻结规则下的实际数据加工。
        raise A1ContractError("A0合同尚未冻结")
    # 要求完整未来路径固定为30日，避免A1和后续模型使用不同路径长度。
    if contract["horizon_contract"]["full_development_path_business_days"] != FORECAST_PATH_BUSINESS_DAYS:
        # 阻断期限长度冲突。
        raise A1ContractError("A0完整路径长度与A1不一致")
    # 要求观察窗口和本模块常量完全一致。
    if tuple(contract["feature_contract"]["registered_lookback_business_days"]) != LOOKBACK_BUSINESS_DAYS:
        # 阻断窗口清单冲突。
        raise A1ContractError("A0观察窗口与A1不一致")
    # 返回已核验合同供调用方读取四折日期。
    return contract


# 只从T1人口清单取得208户开发企业；最终52户只计数，绝不定位其运行目录。
def read_development_population(qualification_directory: Path) -> tuple[list[dict[str, str]], int, dict[str, object]]:
    # 定位T1资格报告；它是A1开始前的密封证据。
    report_path = qualification_directory / "training_input_qualification_report.json"
    # 定位T1合并人口清单；它保存企业、原切分和冻结输入指纹。
    manifest_path = qualification_directory / "combined_training_population_manifest.csv"
    # 缺少任一文件则不能证明输入人口和最终测试边界。
    if not report_path.is_file() or not manifest_path.is_file():
        # 阻断不完整前置证据。
        raise A1ContractError("缺少T1资格报告或合并人口清单")
    # 读取T1报告；报告不包含最终测试的余额标签。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 要求T1已经通过260户资格检查。
    if report.get("status") != "passed" or report.get("enterprise_count") != 260:
        # 阻断未知来源或未通过资格的样本。
        raise A1ContractError("T1资格审计不是260户通过状态")
    # 要求T1明确记录最终测试标签没有被读取或组装。
    if report.get("final_test_labels_read_or_assembled") is not False:
        # 一旦历史密封已破坏，本阶段不得继续。
        raise A1ContractError("T1报告显示最终测试标签已被读取")
    # 读取人口清单文本；此处尚不访问任何run_directory。
    rows = read_csv(manifest_path)
    # 要求260个唯一企业，避免重复或漏户被掩盖。
    if len(rows) != 260 or len({row.get("sample_id") for row in rows}) != 260:
        # 阻断人口数量或唯一性错误。
        raise A1ContractError("T1人口清单不是260户唯一企业")
    # 选择原训练和原验证企业组成的新开发池。
    development_rows = [row for row in rows if row.get("split_group") in {"train", "validation"}]
    # 仅按清单计数原最终测试企业；后续不会迭代这些行的路径。
    final_test_count = sum(row.get("split_group") == "final_test" for row in rows)
    # 要求开发池恰为208户、最终测试恰为52户。
    if len(development_rows) != 208 or final_test_count != 52:
        # 阻断A0所绑定的人口结构被改变。
        raise A1ContractError("T1人口不是208户开发池加52户最终测试")
    # 要求开发池原归属仍为156户训练和52户验证，便于追溯旧实验事实。
    if sum(row.get("split_group") == "train" for row in development_rows) != 156 or sum(row.get("split_group") == "validation" for row in development_rows) != 52:
        # 阻断开发池组成与历史记录不一致。
        raise A1ContractError("开发池原训练/验证数量不正确")
    # 按企业编号末段整数排序，得到A0规定的确定性“轮流发牌”顺序。
    development_rows.sort(key=lambda row: int(row["sample_id"].rsplit("_", 1)[1]))
    # 返回开发清单、最终测试计数和T1报告；调用方只会遍历第一个返回值。
    return development_rows, final_test_count, report


# 将208户排序后的名次按四组轮流分配；它不读取标签或模型成绩。
def assign_enterprise_groups(development_rows: list[dict[str, str]]) -> dict[str, str]:
    # 初始化企业到组名的映射。
    assignments: dict[str, str] = {}
    # 从1开始枚举以符合合同中的“第1、5、9名进入A组”表述。
    for rank, row in enumerate(development_rows, start=1):
        # 用名次减1对4取模得到A、B、C、D的循环位置。
        assignments[row["sample_id"]] = FOLD_GROUPS[(rank - 1) % len(FOLD_GROUPS)]
    # 要求四组均为52户，防止清单长度或循环实现错误。
    if {group: list(assignments.values()).count(group) for group in FOLD_GROUPS} != {group: 52 for group in FOLD_GROUPS}:
        # 阻断不平衡的四折企业分配。
        raise A1ContractError("A0轮流分组不是四组各52户")
    # 返回稳定可复现的企业组映射。
    return assignments


# 读取单户公开日度主表；人工备注、画像、种子和受限目录都不在本函数的读取路径上。
def read_business_days(daily_path: Path) -> list[BusinessDay]:
    # 读取日度CSV行；文件已经由T1资格审计指纹保护。
    raw_rows = read_csv(daily_path)
    # 声明A1需要的公开字段；缺字段不能用猜测值补齐。
    required_fields = {"calendar_date", "inflow_cny", "outflow_cny", "transaction_count", "ending_balance_cny", "reconciliation_status", "is_bank_workday", "business_step"}
    # 初始化工作日记录列表。
    records: list[BusinessDay] = []
    # 逐行检查并转换日度公开字段。
    for raw in raw_rows:
        # 要求日度主表具有全部必需字段。
        if not required_fields.issubset(raw):
            # 阻断结构不完整的日度文件。
            raise A1ContractError(f"日度主表字段不足：{daily_path}")
        # 要求账务勾稽状态可信；不可信日度数据不得入模型开发池。
        if raw["reconciliation_status"] != "trusted":
            # 阻断非可信日度记录。
            raise A1ContractError(f"日度主表存在非可信勾稽状态：{daily_path}")
        # 只把银行工作日纳入序列；自然日不应改变14/28/60日窗口长度。
        if raw["is_bank_workday"] != "True":
            # 跳过非银行工作日。
            continue
        # 将当前公开行转换成强类型工作日对象。
        records.append(BusinessDay(
            # 解析ISO日期。
            calendar_date=date.fromisoformat(raw["calendar_date"]),
            # 解析精确流入金额。
            inflow_cny=Decimal(raw["inflow_cny"]),
            # 解析精确流出金额。
            outflow_cny=Decimal(raw["outflow_cny"]),
            # 解析整数交易笔数。
            transaction_count=int(raw["transaction_count"]),
            # 解析精确日终余额。
            ending_balance_cny=Decimal(raw["ending_balance_cny"]),
            # 解析连续工作日序号。
            business_step=int(raw["business_step"]),
        ))
    # 要求至少具备60日历史和30日前瞻，才可能和D0使用同一资格标准。
    if len(records) < MAX_LOOKBACK_BUSINESS_DAYS + FORECAST_PATH_BUSINESS_DAYS:
        # 阻断期间不足的企业。
        raise A1ContractError(f"银行工作日不足60日历史加30日前瞻：{daily_path}")
    # 要求序号恰为1至N，保证切片不会跨越缺失工作日。
    if [record.business_step for record in records] != list(range(1, len(records) + 1)):
        # 阻断工作日序号缺失或重复。
        raise A1ContractError(f"银行工作日序号不连续：{daily_path}")
    # 返回严格连续的公开工作日序列。
    return records


# 读取开发企业的公开逐笔主表并按日期保存金额；本函数不读取人工备注CSV。
def read_transaction_amounts_by_date(transaction_path: Path) -> dict[date, list[Decimal]]:
    # 读取逐笔主表；它是可见流水而不是受限生成器真值。
    raw_rows = read_csv(transaction_path)
    # 声明逐笔金额分位数需要的公开字段。
    required_fields = {"booking_datetime", "debit_cny", "credit_cny"}
    # 创建默认空列表字典，方便同日多笔金额追加。
    amounts_by_date: dict[date, list[Decimal]] = defaultdict(list)
    # 逐笔转换公开交易金额。
    for raw in raw_rows:
        # 要求字段齐全，防止从备注或摘要猜测金额。
        if not required_fields.issubset(raw):
            # 阻断逐笔主表结构错误。
            raise A1ContractError(f"逐笔主表字段不足：{transaction_path}")
        # 解析借方金额。
        debit = Decimal(raw["debit_cny"])
        # 解析贷方金额。
        credit = Decimal(raw["credit_cny"])
        # 要求一笔交易不能同时有借方和贷方金额，避免金额重复计算。
        if debit > 0 and credit > 0:
            # 阻断不符合单账户借贷表示的记录。
            raise A1ContractError(f"逐笔主表同笔借贷同时为正：{transaction_path}")
        # 取非零方向的绝对金额；零金额记录保留在流水但不应扭曲金额分位数。
        amount = max(debit, credit)
        # 仅将正金额放入分位数样本。
        if amount > 0:
            # 从ISO时间文本的前10位取得记账日期。
            booking_date = date.fromisoformat(raw["booking_datetime"][:10])
            # 追加到对应自然日。
            amounts_by_date[booking_date].append(amount)
    # 返回按日期组织的公开金额；调用方只会访问截止日前的窗口日期。
    return amounts_by_date


# 计算有序Decimal列表的线性插值分位数；空列表采用0表示窗口内没有公开逐笔金额。
def percentile(values: list[Decimal], probability: Decimal) -> Decimal:
    # 空金额窗口的P50/P90定义为0，避免把不存在的交易伪造为缺失值。
    if not values:
        # 返回精确0元。
        return Decimal("0")
    # 将金额由小到大排列，保证位置计算可复现。
    ordered = sorted(values)
    # 单值窗口的任意分位数就是该值自身。
    if len(ordered) == 1:
        # 返回唯一金额。
        return ordered[0]
    # 计算0开始的线性插值位置。
    position = probability * Decimal(len(ordered) - 1)
    # 取得左侧整数索引。
    lower_index = int(position)
    # 取得不超过末尾的右侧整数索引。
    upper_index = min(lower_index + 1, len(ordered) - 1)
    # 计算左右两点的插值权重。
    weight = position - Decimal(lower_index)
    # 返回线性插值金额。
    return ordered[lower_index] * (Decimal("1") - weight) + ordered[upper_index] * weight


# 计算一个窗口内最近一次满足条件的距离；找不到时返回窗口长度，表示“至少这么久未发生”。
def days_since(window: tuple[BusinessDay, ...], predicate) -> int:
    # 从截止日向前枚举窗口位置，距离0代表截止日当天。
    for distance, row in enumerate(reversed(window)):
        # 一旦命中公开事件，立即返回已过去的银行工作日数。
        if predicate(row):
            # 返回该距离。
            return distance
    # 窗口内没有命中时按合同将距离截断为窗口长度。
    return len(window)


# 在窗口中取得最近一次非零金额；没有事件时返回0元而不借用未来金额。
def last_nonzero_amount(window: tuple[BusinessDay, ...], attribute_name: str) -> Decimal:
    # 从截止日向前遍历，保证取到的是最后一次已发生金额。
    for row in reversed(window):
        # 按调用方指定字段取公开流入或流出金额。
        value = getattr(row, attribute_name)
        # 找到正金额后直接返回。
        if value > 0:
            # 返回最近一次金额。
            return value
    # 窗口没有该方向收支时返回0元。
    return Decimal("0")


# 预先计算每个日期在当月的工作日位置与至月末工作日距离；日历是预测时已知信息。
def build_month_positions(records: list[BusinessDay]) -> dict[date, tuple[int, int]]:
    # 将所有工作日按年月收集，键为(year, month)。
    by_month: dict[tuple[int, int], list[BusinessDay]] = defaultdict(list)
    # 逐日放入对应月份。
    for record in records:
        # 使用年月元组作为分组键。
        by_month[(record.calendar_date.year, record.calendar_date.month)].append(record)
    # 初始化日期到两个位置数的映射。
    positions: dict[date, tuple[int, int]] = {}
    # 逐月计算位置和剩余工作日。
    for month_rows in by_month.values():
        # 逐个工作日从1开始编号。
        for position, record in enumerate(month_rows, start=1):
            # 保存月内第几个工作日和距月末还有几个工作日。
            positions[record.calendar_date] = (position, len(month_rows) - position)
    # 返回完整日历位置映射。
    return positions


# 将一个60日历史窗口和30日未来路径写成完全物理分离的标签行。
def target_row(enterprise_id: str, cutoff: BusinessDay, history: tuple[BusinessDay, ...], future: tuple[BusinessDay, ...]) -> dict[str, str]:
    # 用企业编号和截止工作日序号构成全局唯一键。
    sample_key = f"{enterprise_id}__bd{cutoff.business_step:04d}"
    # 计算与主指标相同的账户余额尺度。
    balance_scale = max(abs(cutoff.ending_balance_cny), BALANCE_FLOOR_CNY)
    # 计算最近60个可见余额的第10百分位并与0元取较大值。
    liquidity_threshold = max(percentile([row.ending_balance_cny for row in history], Decimal("0.10")), Decimal("0"))
    # 初始化基础标签和审计字段。
    row = {
        "sample_key": sample_key,
        "enterprise_id": enterprise_id,
        "cutoff_date": cutoff.calendar_date.isoformat(),
        "future_start_date": future[0].calendar_date.isoformat(),
        "target_date_t5": future[4].calendar_date.isoformat(),
        "target_date_t10": future[9].calendar_date.isoformat(),
        "target_date_t30": future[29].calendar_date.isoformat(),
        "cutoff_balance_cny": money(cutoff.ending_balance_cny),
        "balance_scale_cny": money(balance_scale),
        "balance_floor_cny": money(BALANCE_FLOOR_CNY),
        "minimum_balance_t30_cny": money(min(item.ending_balance_cny for item in future)),
        "liquidity_threshold_cny": money(liquidity_threshold),
        "feature_manifest_version": A1_VERSION,
    }
    # 逐个未来工作日写出实际期末余额与相对截止日余额变化。
    for horizon, target in enumerate(future, start=1):
        # 写入该期限的实际期末余额标签。
        row[f"target_balance_t_plus_{horizon:02d}_cny"] = money(target.ending_balance_cny)
        # 写入该期限相对同一截止日的余额变化标签。
        row[f"target_change_t_plus_{horizon:02d}_cny"] = money(target.ending_balance_cny - cutoff.ending_balance_cny)
    # 返回只含标签和审计字段的一行。
    return row


# 从一个已知历史窗口构造树模型单行公开特征；未来路径不传入本函数。
def tree_feature_row(enterprise_id: str, history: tuple[BusinessDay, ...], lookback: int, amounts_by_date: dict[date, list[Decimal]], month_positions: dict[date, tuple[int, int]]) -> dict[str, str]:
    # 取得本配置允许看到的最后lookback个工作日。
    window = history[-lookback:]
    # 取得预测截止日记录。
    cutoff = window[-1]
    # 取得截止日前一个银行工作日，因窗口最短14日所以一定存在。
    prior = window[-2]
    # 汇总窗口内公开逐笔金额；只查历史日期，不查未来日期。
    transaction_amounts = [amount for item in window for amount in amounts_by_date.get(item.calendar_date, [])]
    # 取得截止日在当月的位置和至月末距离。
    position_in_month, days_to_month_end = month_positions[cutoff.calendar_date]
    # 计算窗口净流量均值。
    net_mean = sum((item.net_flow_cny for item in window), Decimal("0")) / Decimal(len(window))
    # 用浮点平方根计算标准差展示值；它只对公开历史金额做描述，不改变财务账本。
    net_standard_deviation = Decimal(str(sqrt(sum((float(item.net_flow_cny - net_mean) ** 2 for item in window)) / len(window))))
    # 取得最近一个方向事件的公开金额。
    last_inflow = last_nonzero_amount(window, "inflow_cny")
    # 取得最近一个方向事件的公开金额。
    last_outflow = last_nonzero_amount(window, "outflow_cny")
    # 取得下一个银行工作日以计算公开日历间隔；该日期不是余额或交易真值。
    next_business_day = history[-1].calendar_date if len(history) == lookback else None
    # 使用历史60日中紧随截止日的未来一天计算间隔；调用方会在下面覆写为样本可见日历。
    next_gap_days = 0 if next_business_day is None else 0
    # 返回固定字段名的公开特征；标签、画像、种子和企业组均不在此表中。
    return {
        "sample_key": f"{enterprise_id}__bd{cutoff.business_step:04d}",
        "enterprise_id": enterprise_id,
        "cutoff_date": cutoff.calendar_date.isoformat(),
        "lookback_start_date": window[0].calendar_date.isoformat(),
        "lookback_business_days": str(lookback),
        "business_step": str(cutoff.business_step),
        "closing_balance_cny": money(cutoff.ending_balance_cny),
        "daily_inflow_cny": money(cutoff.inflow_cny),
        "daily_outflow_cny": money(cutoff.outflow_cny),
        "daily_net_flow_cny": money(cutoff.net_flow_cny),
        "transaction_count": str(cutoff.transaction_count),
        "weekday_monday_0": str(cutoff.calendar_date.weekday()),
        "business_day_position_in_month": str(position_in_month),
        "business_days_to_month_end": str(days_to_month_end),
        "is_month_end_business_day": str(int(days_to_month_end == 0)),
        "calendar_gap_days_from_prior_business_day": str((cutoff.calendar_date - prior.calendar_date).days),
        "calendar_gap_days_to_next_business_day": str(next_gap_days),
        "inflow_sum_cny": money(sum((item.inflow_cny for item in window), Decimal("0"))),
        "outflow_sum_cny": money(sum((item.outflow_cny for item in window), Decimal("0"))),
        "net_flow_sum_cny": money(sum((item.net_flow_cny for item in window), Decimal("0"))),
        "absolute_net_flow_sum_cny": money(sum((abs(item.net_flow_cny) for item in window), Decimal("0"))),
        "transaction_count_sum": str(sum(item.transaction_count for item in window)),
        "active_day_ratio": ratio(Decimal(sum(item.transaction_count > 0 for item in window)) / Decimal(len(window))),
        "net_flow_mean_cny": money(net_mean),
        "net_flow_standard_deviation_cny": money(net_standard_deviation),
        "transaction_amount_p50_cny": money(percentile(transaction_amounts, Decimal("0.50"))),
        "transaction_amount_p90_cny": money(percentile(transaction_amounts, Decimal("0.90"))),
        "days_since_last_transaction": str(days_since(window, lambda item: item.transaction_count > 0)),
        "days_since_last_inflow": str(days_since(window, lambda item: item.inflow_cny > 0)),
        "days_since_last_outflow": str(days_since(window, lambda item: item.outflow_cny > 0)),
        "last_nonzero_inflow_amount_cny": money(last_inflow),
        "last_nonzero_outflow_amount_cny": money(last_outflow),
        "feature_manifest_version": A1_VERSION,
    }


# 为避免在树特征函数中传入未来日度真值，本函数只以公开银行日历补写已知的下一个工作日间隔。
def set_tree_calendar_gap(feature_row: dict[str, str], cutoff: BusinessDay, future_first: BusinessDay) -> dict[str, str]:
    # 复制字典而非原地改写，使调用方可以安全复用原始特征行。
    updated = dict(feature_row)
    # 写入截止日至下一银行工作日的自然日间隔；周末和节假日信息属于公开日历。
    updated["calendar_gap_days_to_next_business_day"] = str((future_first.calendar_date - cutoff.calendar_date).days)
    # 返回补写后仍不含标签的特征行。
    return updated


# 将一个历史窗口展开为LSTM长表；每行只携带一个可见工作日特征。
def sequence_feature_rows(enterprise_id: str, history: tuple[BusinessDay, ...], lookback: int, month_positions: dict[date, tuple[int, int]], future_first: BusinessDay) -> Iterable[dict[str, str]]:
    # 取得本配置允许的最后lookback个工作日。
    window = history[-lookback:]
    # 保存完整样本键，便于后续与独立标签表连接。
    sample_key = f"{enterprise_id}__bd{window[-1].business_step:04d}"
    # 逐个历史工作日写出序列位置和公开特征。
    for position, row in enumerate(window, start=1):
        # 取得窗口内前一个工作日；首行没有前一行时间隔写0。
        prior = window[position - 2] if position > 1 else None
        # 对最后一个历史日，下一工作日来自公开日历对应的future_first日期；其他行来自窗口内下一行。
        next_row = future_first if position == len(window) else window[position]
        # 取得当前日的月内位置和至月末距离。
        position_in_month, days_to_month_end = month_positions[row.calendar_date]
        # 产生一行不含未来余额标签的序列特征。
        yield {
            "sample_key": sample_key,
            "enterprise_id": enterprise_id,
            "cutoff_date": window[-1].calendar_date.isoformat(),
            "sequence_position": str(position),
            "sequence_length": str(lookback),
            "calendar_date": row.calendar_date.isoformat(),
            "business_step": str(row.business_step),
            "calendar_gap_days_from_prior_business_day": str((row.calendar_date - prior.calendar_date).days) if prior else "0",
            "calendar_gap_days_to_next_business_day": str((next_row.calendar_date - row.calendar_date).days),
            "closing_balance_cny": money(row.ending_balance_cny),
            "daily_inflow_cny": money(row.inflow_cny),
            "daily_outflow_cny": money(row.outflow_cny),
            "daily_net_flow_cny": money(row.net_flow_cny),
            "transaction_count": str(row.transaction_count),
            "day_has_transaction": str(int(row.transaction_count > 0)),
            "weekday_monday_0": str(row.calendar_date.weekday()),
            "business_day_position_in_month": str(position_in_month),
            "business_days_to_month_end": str(days_to_month_end),
            "is_month_end_business_day": str(int(days_to_month_end == 0)),
            "feature_manifest_version": A1_VERSION,
        }


# 计算样本描述并通过回调立即写出大型特征；返回值只保留轻量索引信息供四折切分使用。
def build_enterprise_derivatives(enterprise_id: str, enterprise_group: str, records: list[BusinessDay], amounts_by_date: dict[date, list[Decimal]], writers: dict[str, csv.DictWriter]) -> list[SampleDescriptor]:
    # 预先计算本企业所有公开月内日历位置。
    month_positions = build_month_positions(records)
    # 初始化轻量样本描述列表。
    descriptors: list[SampleDescriptor] = []
    # 最早截止点必须包含60个工作日历史，因此索引从59开始。
    first_cutoff_index = MAX_LOOKBACK_BUSINESS_DAYS - 1
    # 最晚截止点之后必须完整拥有30个工作日，因此为总行数减31。
    last_cutoff_index = len(records) - FORECAST_PATH_BUSINESS_DAYS - 1
    # 遍历全部共同合格截止点；四个期限因此天然使用相同样本集合。
    for cutoff_index in range(first_cutoff_index, last_cutoff_index + 1):
        # 取得当前预测截止日。
        cutoff = records[cutoff_index]
        # 取得含截止日的连续60日历史窗口。
        history = tuple(records[cutoff_index - MAX_LOOKBACK_BUSINESS_DAYS + 1:cutoff_index + 1])
        # 取得不含截止日的连续30日未来路径；它只写入独立标签表。
        future = tuple(records[cutoff_index + 1:cutoff_index + FORECAST_PATH_BUSINESS_DAYS + 1])
        # 写出一行未来标签表。
        writers["targets"].writerow(target_row(enterprise_id, cutoff, history, future))
        # 逐个窗口写出树模型公开特征表。
        for lookback in LOOKBACK_BUSINESS_DAYS:
            # 构造该窗口的树表特征。
            raw_tree_row = tree_feature_row(enterprise_id, history, lookback, amounts_by_date, month_positions)
            # 补写截止日至下一银行工作日的公开日历间隔，不写任何未来金额。
            writers[f"tree_{lookback}"].writerow(set_tree_calendar_gap(raw_tree_row, cutoff, future[0]))
            # 逐行写出同一窗口的LSTM公开历史序列。
            for sequence_row in sequence_feature_rows(enterprise_id, history, lookback, month_positions, future[0]):
                # 将当前历史日写入对应窗口的长表。
                writers[f"sequence_{lookback}"].writerow(sequence_row)
        # 保存供四折索引使用的轻量描述，不复制60日历史或30日未来金额。
        descriptors.append(SampleDescriptor(
            # 保存样本键。
            sample_key=f"{enterprise_id}__bd{cutoff.business_step:04d}",
            # 保存企业编号。
            enterprise_id=enterprise_id,
            # 保存预先分配的企业组。
            enterprise_fold_group=enterprise_group,
            # 保存截止日。
            cutoff_date=cutoff.calendar_date,
            # 保存未来第一日。
            future_start_date=future[0].calendar_date,
            # 保存T+5期末日。
            target_date_t5=future[4].calendar_date,
            # 保存T+10期末日。
            target_date_t10=future[9].calendar_date,
            # 保存T+30期末日。
            target_date_t30=future[29].calendar_date,
        ))
    # 要求每户至少有一个共同截止点，避免静默漏掉企业。
    if not descriptors:
        # 阻断没有可用多期限样本的企业。
        raise A1ContractError(f"没有可用60日历史加30日前瞻样本：{enterprise_id}")
    # 返回轻量描述供四折索引构造。
    return descriptors


# 将日期文本转成date；合同日期必须是ISO格式以便精确比较。
def contract_date(value: str) -> date:
    # 解析ISO日期文本。
    return date.fromisoformat(value)


# 为四个滚动折构造fit、calibration和validation索引；最终测试企业不在参数中。
def build_fold_index_rows(descriptors: list[SampleDescriptor], contract: dict[str, object]) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    # 取得合同中的四折定义。
    folds = contract["development_split_contract"]["folds"]
    # 初始化可写入CSV的索引行列表。
    rows: list[dict[str, str]] = []
    # 初始化每折、每角色的样本计数。
    counts: dict[str, dict[str, int]] = {}
    # 逐个处理预先登记的滚动折。
    for fold in folds:
        # 取得折编号。
        fold_id = fold["fold_id"]
        # 取得该折验证企业组。
        validation_group = fold["validation_enterprise_group"]
        # 将训练企业组转成集合以便成员判断。
        training_groups = set(fold["training_enterprise_groups"])
        # 解析训练标签允许的最晚T+30期末日期。
        train_end = contract_date(fold["train_latest_target_endpoint"])
        # 解析验证路径允许的第一天。
        validation_start = contract_date(fold["validation_first_target_date"])
        # 解析验证路径允许的最晚T+30期末日期。
        validation_end = contract_date(fold["validation_latest_target_endpoint"])
        # 选择该折训练企业且完整30日路径未越过训练时间边界的样本。
        training_samples = [item for item in descriptors if item.enterprise_fold_group in training_groups and item.target_date_t30 <= train_end]
        # 选择该折验证企业且整条30日路径完全位于验证季度内的样本。
        validation_samples = [item for item in descriptors if item.enterprise_fold_group == validation_group and item.future_start_date >= validation_start and item.target_date_t30 <= validation_end]
        # 要求训练和验证都非空，避免生成表面成功却不能实验的折。
        if not training_samples or not validation_samples:
            # 阻断空折。
            raise A1ContractError(f"滚动折{fold_id}缺少训练或验证样本")
        # 取得训练样本实际出现过的T+30期末日期并排序；按日期而非行数切80/20。
        training_dates = sorted({item.target_date_t30 for item in training_samples})
        # 取前80%的日期作为拟合期；ceil保证小样本时拟合期不会被截短为0。
        fit_date_count = max(1, ceil(len(training_dates) * 0.80))
        # 将前80%日期放入拟合集。
        fit_dates = set(training_dates[:fit_date_count])
        # 将后20%日期放入时间靠后的校准集。
        calibration_dates = set(training_dates[fit_date_count:])
        # 要求校准集非空；A1的人口和两年日期应满足此条件。
        if not calibration_dates:
            # 阻断无法形成T+30区间校准的折。
            raise A1ContractError(f"滚动折{fold_id}没有时间留出校准日期")
        # 初始化该折角色计数。
        counts[fold_id] = {"fit": 0, "calibration": 0, "validation": 0}
        # 逐个训练企业样本写入拟合或校准角色。
        for item in training_samples:
            # 根据T+30期末日所属时间段确定角色。
            role = "fit" if item.target_date_t30 in fit_dates else "calibration"
            # 写入不含金额的折索引行。
            rows.append(fold_index_row(item, fold_id, role))
            # 对应角色计数加1。
            counts[fold_id][role] += 1
        # 逐个验证样本写入验证角色。
        for item in validation_samples:
            # 写入不含金额的验证索引行。
            rows.append(fold_index_row(item, fold_id, "validation"))
            # 验证角色计数加1。
            counts[fold_id]["validation"] += 1
        # 要求本折验证企业不会混进本折训练或校准企业。
        training_enterprises = {item.enterprise_id for item in training_samples}
        # 取得验证企业集合。
        validation_enterprises = {item.enterprise_id for item in validation_samples}
        # 集合交集非空说明企业隔离被破坏。
        if training_enterprises.intersection(validation_enterprises):
            # 阻断企业泄漏。
            raise A1ContractError(f"滚动折{fold_id}发生企业隔离泄漏")
    # 返回全部索引行和紧凑计数。
    return rows, counts


# 将一个轻量样本描述变成CSV折索引行；它有日期但没有任何余额、金额或标签。
def fold_index_row(item: SampleDescriptor, fold_id: str, role: str) -> dict[str, str]:
    # 返回字段顺序与FOLD_INDEX_COLUMNS一致的字典。
    return {
        "sample_key": item.sample_key,
        "enterprise_id": item.enterprise_id,
        "enterprise_fold_group": item.enterprise_fold_group,
        "fold_id": fold_id,
        "fold_role": role,
        "cutoff_date": item.cutoff_date.isoformat(),
        "future_start_date": item.future_start_date.isoformat(),
        "target_date_t5": item.target_date_t5.isoformat(),
        "target_date_t10": item.target_date_t10.isoformat(),
        "target_date_t30": item.target_date_t30.isoformat(),
        "feature_manifest_version": A1_VERSION,
    }


# 写入CSV表并保证字段顺序固定；调用方只传入已经验证的字典行。
def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, str]]) -> None:
    # 以UTF-8新建文件；A1输出目录必须全新，因此不会覆盖历史结果。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 建立严格字段清单的字典写入器。
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        # 写入首行表头。
        writer.writeheader()
        # 按调用方顺序逐行写入。
        writer.writerows(rows)
