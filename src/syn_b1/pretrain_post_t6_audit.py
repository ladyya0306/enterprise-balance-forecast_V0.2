"""POST-T6-D0的纯计算逻辑：审计开发池频率、多期限目标和可见历史先兆。"""

# 导入dataclass；它把自然日日度记录表达成字段明确的不可变对象。
from dataclasses import dataclass
# 导入Decimal；余额和流量计算因此保持人民币金额精度。
from decimal import Decimal

# 固定审计版本；未来改变期限、门槛或计算口径必须升级版本。
AUDIT_VERSION = "syn_b1_post_t6_d0_frequency_horizon_audit_1_0"
# 固定预先登记的四个预测期限；单位均为银行工作日。
FORECAST_HORIZONS = (1, 5, 10, 30)
# 固定最长观察窗口；所有期限使用同一批至少60日历史的截止点。
MAX_LOOKBACK_BUSINESS_DAYS = 60
# 固定先兆观察窗口；它与当前首轮14银行工作日历史保持可比。
PRECURSOR_LOOKBACK_BUSINESS_DAYS = 14
# 固定描述性活动分箱；它们在读取结果前声明，避免按结果临时切档。
ACTIVITY_BINS = (("0笔", 0, 0), ("1—3笔", 1, 3), ("4—7笔", 4, 7), ("8—14笔", 8, 14), ("15笔以上", 15, None))


# 声明D0专用合同错误；它区分数据边界失败与“没有发现强先兆”。
class PostT6AuditContractError(ValueError):
    """当开发池、日度结构或期限对齐不满足只读审计合同时抛出。"""


# 使用不可变数据类保存一个银行工作日的公开日度字段。
@dataclass(frozen=True)
class BusinessDayRecord:
    # 保存自然日期文本，供截止日和目标日审计。
    calendar_date: str
    # 保存当日公开流入金额。
    inflow_cny: Decimal
    # 保存当日公开流出金额。
    outflow_cny: Decimal
    # 保存当日公开交易笔数。
    transaction_count: int
    # 保存日终余额。
    ending_balance_cny: Decimal
    # 保存连续银行工作日序号。
    business_step: int


# 使用不可变数据类保存一个截止点和一个期限的描述性观察。
@dataclass(frozen=True)
class HorizonObservation:
    # 保存企业编号；只用于分组审计，不进入未来模型。
    enterprise_id: str
    # 保存预测截止日。
    cutoff_date: str
    # 保存期限长度。
    horizon_business_days: int
    # 保存目标期限末日。
    target_date: str
    # 保存截止日余额。
    cutoff_balance_cny: Decimal
    # 保存期限末余额。
    target_balance_cny: Decimal
    # 保存期限末余额相对截止日的变化。
    target_change_cny: Decimal
    # 保存未来窗口内最低余额。
    minimum_future_balance_cny: Decimal
    # 保存未来窗口内公开交易总笔数。
    future_transaction_count: int
    # 保存截止日及此前14工作日的公开交易总笔数。
    recent_transaction_count_14bd: int
    # 保存截止日及此前14工作日的公开收支周转额。
    recent_turnover_cny_14bd: Decimal
    # 保存该企业冻结的归一化余额下限。
    balance_floor_cny: Decimal


# 把字符串布尔值解析为真正布尔值；日度CSV使用True或False文本。
def parse_bool(value: str) -> bool:
    # 去除空格并转为小写，以兼容True、true等稳定写法。
    normalized = value.strip().lower()
    # 识别真值。
    if normalized == "true":
        # 返回真，表示该日属于银行工作日。
        return True
    # 识别假值。
    if normalized == "false":
        # 返回假，表示自然日不进入工作日序列。
        return False
    # 其他文本说明CSV结构被改写。
    raise PostT6AuditContractError(f"无法解析银行工作日布尔值：{value}")


# 将原始CSV字典转换成连续银行工作日日度记录。
def build_business_day_records(rows: list[dict[str, str]]) -> list[BusinessDayRecord]:
    # 只保留明确标记为银行工作日的公开日度行。
    workday_rows = [row for row in rows if parse_bool(row["is_bank_workday"])]
    # 要求至少有最长观察窗口加最长预测期限，才能形成共同截止点。
    if len(workday_rows) < MAX_LOOKBACK_BUSINESS_DAYS + max(FORECAST_HORIZONS):
        # 阻断期间不足的企业，避免不同期限偷偷使用不同资格标准。
        raise PostT6AuditContractError("银行工作日数量不足以形成60日历史和30日前瞻")
    # 创建转换后的记录列表。
    records: list[BusinessDayRecord] = []
    # 按CSV原顺序逐行转换；正式日度文件已经按日期排序。
    for row in workday_rows:
        # 构造一条精确金额和整数笔数记录。
        record = BusinessDayRecord(
            # 保存日期。
            calendar_date=row["calendar_date"],
            # 解析流入金额。
            inflow_cny=Decimal(row["inflow_cny"]),
            # 解析流出金额。
            outflow_cny=Decimal(row["outflow_cny"]),
            # 解析交易笔数。
            transaction_count=int(row["transaction_count"]),
            # 解析日终余额。
            ending_balance_cny=Decimal(row["ending_balance_cny"]),
            # 解析银行工作日序号。
            business_step=int(row["business_step"]),
        )
        # 将当前记录追加到稳定序列。
        records.append(record)
    # 逐对检查日期和工作日序号严格递增。
    for previous, current in zip(records, records[1:]):
        # 工作日序号必须每次增加1且日期文本按ISO顺序递增。
        if current.business_step != previous.business_step + 1 or current.calendar_date <= previous.calendar_date:
            # 阻断断号、倒序或重复日期。
            raise PostT6AuditContractError("银行工作日序列不连续或日期未递增")
    # 返回可用于频率和期限审计的公开日度序列。
    return records


# 计算一个有序Decimal列表的线性插值分位数。
def percentile(values: list[Decimal], probability: Decimal) -> Decimal:
    # 空列表没有分位数定义。
    if not values:
        # 抛出清晰异常，防止输出伪造的0。
        raise PostT6AuditContractError("空列表不能计算分位数")
    # 将金额从小到大排序。
    ordered = sorted(values)
    # 单个值的任意分位数都是它自身。
    if len(ordered) == 1:
        # 返回唯一值。
        return ordered[0]
    # 计算从0开始的理论位置。
    position = probability * Decimal(len(ordered) - 1)
    # 取得左侧整数位置。
    lower_index = int(position)
    # 取得不超过末尾的右侧位置。
    upper_index = min(lower_index + 1, len(ordered) - 1)
    # 计算小数插值权重。
    weight = position - Decimal(lower_index)
    # 在线性相邻值之间插值得到分位数。
    return ordered[lower_index] * (Decimal("1") - weight) + ordered[upper_index] * weight


# 根据活跃工作日的典型笔数划分与Plan B一致的频率覆盖档。
def classify_frequency(median_transactions_per_active_day: Decimal) -> str:
    # 小于3笔属于当前计划贸易频率下限以下。
    if median_transactions_per_active_day < Decimal("3"):
        # 返回覆盖不足档。
        return "低于计划贸易频率"
    # 3至7笔属于计划低档。
    if median_transactions_per_active_day <= Decimal("7"):
        # 返回低档名称。
        return "计划低档3—7笔"
    # 8至15笔属于计划中档。
    if median_transactions_per_active_day <= Decimal("15"):
        # 返回中档名称。
        return "计划中档8—15笔"
    # 16至30笔属于计划高档。
    if median_transactions_per_active_day <= Decimal("30"):
        # 返回高档名称。
        return "计划高档16—30笔"
    # 超过30笔单独标记，避免把极高频静默压入高档。
    return "高于计划高档"


# 汇总单户开发企业的公开工作日交易频率。
def summarize_enterprise_frequency(enterprise_id: str, split_group: str, records: list[BusinessDayRecord]) -> dict[str, str]:
    # 提取每个工作日的交易笔数。
    daily_counts = [record.transaction_count for record in records]
    # 提取至少有一笔交易的活跃工作日笔数。
    active_counts = [count for count in daily_counts if count > 0]
    # 要求企业至少存在一个活跃工作日。
    if not active_counts:
        # 阻断完全没有公开交易的企业。
        raise PostT6AuditContractError(f"开发企业没有活跃工作日：{enterprise_id}")
    # 转为Decimal以精确计算均值和比例。
    workday_count = Decimal(len(daily_counts))
    # 计算活跃日中位数。
    active_median = percentile([Decimal(count) for count in active_counts], Decimal("0.5"))
    # 计算工作日90分位笔数。
    workday_p90 = percentile([Decimal(count) for count in daily_counts], Decimal("0.9"))
    # 计算总交易笔数。
    total_transactions = sum(daily_counts)
    # 返回稳定列顺序的可读频率画像。
    return {
        # 保存企业编号。
        "enterprise_id": enterprise_id,
        # 保存开发池原归属。
        "development_source_group": split_group,
        # 保存工作日数。
        "business_day_count": str(len(daily_counts)),
        # 保存总交易笔数。
        "total_transaction_count": str(total_transactions),
        # 保存活跃工作日数。
        "active_business_day_count": str(len(active_counts)),
        # 保存活跃日比例。
        "active_business_day_ratio": f"{Decimal(len(active_counts)) / workday_count:.8f}",
        # 保存每工作日平均笔数。
        "mean_transactions_per_business_day": f"{Decimal(total_transactions) / workday_count:.8f}",
        # 保存活跃日中位笔数。
        "median_transactions_per_active_day": f"{active_median:.2f}",
        # 保存工作日90分位笔数。
        "p90_transactions_per_business_day": f"{workday_p90:.2f}",
        # 保存单日最大笔数。
        "max_transactions_per_business_day": str(max(daily_counts)),
        # 保存与Plan B对齐的频率覆盖档。
        "frequency_coverage_band": classify_frequency(active_median),
    }


# 为同一企业构造共享截止点上的T+1、T+5、T+10和T+30观察。
def build_horizon_observations(enterprise_id: str, records: list[BusinessDayRecord], balance_floor_cny: Decimal) -> list[HorizonObservation]:
    # 创建全部期限观察列表。
    observations: list[HorizonObservation] = []
    # 最早截止点必须已有完整60工作日历史。
    first_cutoff_index = MAX_LOOKBACK_BUSINESS_DAYS - 1
    # 最晚截止点必须仍有完整30工作日前瞻。
    last_cutoff_index = len(records) - max(FORECAST_HORIZONS) - 1
    # 遍历所有共同合格截止点。
    for cutoff_index in range(first_cutoff_index, last_cutoff_index + 1):
        # 取得截止日前含当日的14工作日公开历史。
        recent_records = records[cutoff_index - PRECURSOR_LOOKBACK_BUSINESS_DAYS + 1:cutoff_index + 1]
        # 汇总近期交易笔数。
        recent_transaction_count = sum(record.transaction_count for record in recent_records)
        # 汇总近期公开收支周转额。
        recent_turnover = sum((record.inflow_cny + record.outflow_cny for record in recent_records), Decimal("0"))
        # 逐个构造预注册期限。
        for horizon in FORECAST_HORIZONS:
            # 取得该期限未来路径，不包含截止日、包含目标日。
            future_records = records[cutoff_index + 1:cutoff_index + horizon + 1]
            # 取得期限末记录。
            target_record = future_records[-1]
            # 取得截止日记录。
            cutoff_record = records[cutoff_index]
            # 构造一条描述性观察。
            observation = HorizonObservation(
                # 保存企业编号。
                enterprise_id=enterprise_id,
                # 保存截止日。
                cutoff_date=cutoff_record.calendar_date,
                # 保存期限。
                horizon_business_days=horizon,
                # 保存目标日。
                target_date=target_record.calendar_date,
                # 保存截止余额。
                cutoff_balance_cny=cutoff_record.ending_balance_cny,
                # 保存目标余额。
                target_balance_cny=target_record.ending_balance_cny,
                # 计算目标变化。
                target_change_cny=target_record.ending_balance_cny - cutoff_record.ending_balance_cny,
                # 取得未来路径最低余额。
                minimum_future_balance_cny=min(record.ending_balance_cny for record in future_records),
                # 汇总未来窗口交易笔数。
                future_transaction_count=sum(record.transaction_count for record in future_records),
                # 保存近期交易笔数。
                recent_transaction_count_14bd=recent_transaction_count,
                # 保存近期周转额。
                recent_turnover_cny_14bd=recent_turnover,
                # 保存归一化余额下限。
                balance_floor_cny=balance_floor_cny,
            )
            # 追加当前期限观察。
            observations.append(observation)
    # 返回四个期限共享截止点的完整观察。
    return observations


# 汇总一个期限的目标稀疏度、尾部和余额不变基线。
def summarize_horizon(observations: list[HorizonObservation], horizon: int) -> dict[str, str]:
    # 选择当前期限观察。
    rows = [row for row in observations if row.horizon_business_days == horizon]
    # 要求当前期限非空。
    if not rows:
        # 阻断缺期限输出。
        raise PostT6AuditContractError(f"期限T+{horizon}没有观察")
    # 提取绝对余额变化；它等于余额不变基线的绝对误差。
    absolute_changes = [abs(row.target_change_cny) for row in rows]
    # 计算余额不变基线的账户归一化误差。
    normalized_errors = [abs(row.target_change_cny) / max(abs(row.cutoff_balance_cny), row.balance_floor_cny) for row in rows]
    # 统计期限末余额变化为零的观察。
    zero_count = sum(row.target_change_cny == Decimal("0") for row in rows)
    # 统计未来路径内至少有一笔交易的观察。
    future_event_count = sum(row.future_transaction_count > 0 for row in rows)
    # 计算绝对变化总额供尾部贡献率使用。
    total_absolute_change = sum(absolute_changes, Decimal("0"))
    # 计算至少一行的顶部1%行数。
    top_count = max(1, (len(rows) + 99) // 100)
    # 汇总最大1%绝对变化。
    top_absolute_change = sum(sorted(absolute_changes, reverse=True)[:top_count], Decimal("0"))
    # 将样本数转成Decimal。
    count = Decimal(len(rows))
    # 返回期限级描述结果。
    return {
        # 保存期限。
        "horizon_business_days": str(horizon),
        # 保存共同截止点样本数。
        "sample_count": str(len(rows)),
        # 保存企业数。
        "enterprise_count": str(len({row.enterprise_id for row in rows})),
        # 保存期限末零变化比例。
        "zero_endpoint_change_ratio": f"{Decimal(zero_count) / count:.8f}",
        # 保存未来窗口有交易比例。
        "any_future_transaction_ratio": f"{Decimal(future_event_count) / count:.8f}",
        # 保存余额不变基线人民币MAE。
        "last_balance_naive_mae_cny": f"{sum(absolute_changes, Decimal('0')) / count:.2f}",
        # 保存余额不变基线账户归一化MAE。
        "last_balance_naive_account_normalized_mae": f"{sum(normalized_errors, Decimal('0')) / count:.8f}",
        # 保存绝对变化中位数。
        "absolute_change_p50_cny": f"{percentile(absolute_changes, Decimal('0.5')):.2f}",
        # 保存绝对变化90分位数。
        "absolute_change_p90_cny": f"{percentile(absolute_changes, Decimal('0.9')):.2f}",
        # 保存绝对变化99分位数。
        "absolute_change_p99_cny": f"{percentile(absolute_changes, Decimal('0.99')):.2f}",
        # 保存最大1%对绝对变化总额的贡献；全零时显示0。
        "top_1pct_absolute_change_share": f"{top_absolute_change / total_absolute_change:.8f}" if total_absolute_change > 0 else "0.00000000",
    }


# 将最近14日交易笔数映射到预注册活动分箱。
def activity_bin_name(transaction_count: int) -> str:
    # 顺序检查每个固定上下界。
    for name, lower, upper in ACTIVITY_BINS:
        # 无上界代表达到下限即可。
        if transaction_count >= lower and (upper is None or transaction_count <= upper):
            # 返回命中的中文分箱。
            return name
    # 理论上非负整数总能命中；未命中说明输入异常。
    raise PostT6AuditContractError(f"近期交易笔数无法分箱：{transaction_count}")


# 按期限和历史活动档汇总未来事件率，形成不含模型训练的先兆描述。
def summarize_precursor_bins(observations: list[HorizonObservation]) -> list[dict[str, str]]:
    # 创建输出列表。
    results: list[dict[str, str]] = []
    # 逐个处理预注册期限。
    for horizon in FORECAST_HORIZONS:
        # 取得当前期限全部观察。
        horizon_rows = [row for row in observations if row.horizon_business_days == horizon]
        # 按活动档顺序输出，避免结果排序影响解释。
        for bin_name, _, _ in ACTIVITY_BINS:
            # 选择近期交易笔数属于当前档的观察。
            rows = [row for row in horizon_rows if activity_bin_name(row.recent_transaction_count_14bd) == bin_name]
            # 空档仍输出0行证据，明确当前样本没有覆盖。
            if not rows:
                # 追加无覆盖行。
                results.append({"horizon_business_days": str(horizon), "recent_14bd_transaction_count_band": bin_name, "sample_count": "0", "enterprise_count": "0", "future_transaction_event_ratio": "", "nonzero_endpoint_change_ratio": "", "mean_absolute_endpoint_change_cny": "", "median_absolute_endpoint_change_cny": ""})
                # 继续下一档。
                continue
            # 将当前档行数转成Decimal。
            count = Decimal(len(rows))
            # 提取绝对期限末变化。
            absolute_changes = [abs(row.target_change_cny) for row in rows]
            # 统计未来窗口内有交易的观察。
            future_event_count = sum(row.future_transaction_count > 0 for row in rows)
            # 统计期限末余额非零变化的观察。
            endpoint_event_count = sum(row.target_change_cny != Decimal("0") for row in rows)
            # 追加当前档描述结果。
            results.append({
                # 保存期限。
                "horizon_business_days": str(horizon),
                # 保存活动档。
                "recent_14bd_transaction_count_band": bin_name,
                # 保存样本数。
                "sample_count": str(len(rows)),
                # 保存企业数。
                "enterprise_count": str(len({row.enterprise_id for row in rows})),
                # 保存未来交易事件率。
                "future_transaction_event_ratio": f"{Decimal(future_event_count) / count:.8f}",
                # 保存期限末非零变化率。
                "nonzero_endpoint_change_ratio": f"{Decimal(endpoint_event_count) / count:.8f}",
                # 保存平均绝对期限末变化。
                "mean_absolute_endpoint_change_cny": f"{sum(absolute_changes, Decimal('0')) / count:.2f}",
                # 保存中位绝对期限末变化。
                "median_absolute_endpoint_change_cny": f"{percentile(absolute_changes, Decimal('0.5')):.2f}",
            })
    # 返回全部期限和活动档组合。
    return results
