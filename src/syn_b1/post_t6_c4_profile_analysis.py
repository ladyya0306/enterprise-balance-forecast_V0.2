# 模块说明：本模块只把已冻结的C3B最终测试指标按预登记企业类型汇总；它不训练或调用模型。
"""POST-T6-C4 的只读企业类型辅助分析计算。"""

# 导入CSV标准库，用于读取和写出可由Excel打开的审计表。
import csv
# 导入哈希标准库，用于核验C3B已冻结的源文件没有漂移。
import hashlib
# 导入JSON标准库，用于读取画像并写出机器可读报告。
import json
# 导入确定性伪随机数工具，用企业整块自助抽样计算95%范围。
import random
# 导入日期类型，用于判断大额、转折和融资事件是否落入预测窗口。
from datetime import date
# 导入路径类型，用于安全拼接项目内的中文目录。
from pathlib import Path
# 导入类型提示，使教学代码的输入输出边界更清楚。
from typing import Any

# 固定C4阶段编号，供报告和测试核对。
STAGE_ID = "POST-T6-C4"
# 固定最终测试企业总数，防止读取到被替换的群体。
EXPECTED_FINAL_ENTERPRISES = 52
# 固定C3B最终指标行数，10日和30日各一行。
EXPECTED_METRIC_ROWS = 4568
# 固定预登记的两个产品期限，禁止本卡擅自增加期限。
HORIZONS = (10, 30)
# 固定证据门要求的最少企业数。
MIN_ENTERPRISES = 10
# 固定证据门要求的最少预测起点数。
MIN_ORIGINS = 500
# 固定企业整块自助抽样次数，和合同保持一致。
BOOTSTRAP_REPETITIONS = 2000
# 固定自助抽样根种子，保证相同冻结输入得到相同范围。
BOOTSTRAP_SEED = 20260902


# 定义C4专用异常，让数据边界或冻结条件失败时给出清晰原因。
class C4ProfileAnalysisError(ValueError):
    # 此异常类不增加字段，只区分普通输入错误与C4合同错误。
    pass


# 计算任意文件的SHA-256指纹，供只读输入核验使用。
def sha256_file(path: Path) -> str:
    # 读取文件原始字节并返回稳定十六进制哈希。
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 读取UTF-8或UTF-8-BOM编码CSV为字典行列表。
def read_csv(path: Path) -> list[dict[str, str]]:
    # 缺少冻结输入不能用其他文件静默替代。
    if not path.is_file():
        # 抛出带路径的合同错误，便于人工定位。
        raise C4ProfileAnalysisError(f"C4缺少输入CSV：{path}")
    # 以newline空字符串交给csv模块统一处理Windows换行。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # 完整读取小型清单或C3B结果，字段名保留为字符串。
        return list(csv.DictReader(handle))


# 读取必须为JSON对象的元数据文件。
def read_json(path: Path) -> dict[str, Any]:
    # 缺文件时立即停止，不能把空对象当成冻结证据。
    if not path.is_file():
        # 抛出可读错误文本。
        raise C4ProfileAnalysisError(f"C4缺少输入JSON：{path}")
    # 读取UTF-8文本并解析JSON结构。
    value = json.loads(path.read_text(encoding="utf-8"))
    # 报告和画像都必须是键值对象而不是数组或标量。
    if not isinstance(value, dict):
        # 阻断错误结构。
        raise C4ProfileAnalysisError(f"C4 JSON不是对象：{path.name}")
    # 返回已验证对象。
    return value


# 写出非空CSV并固定由首行决定的列顺序。
def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    # 空表代表没有形成可复核证据，不能写成貌似成功的文件。
    if not rows:
        # 阻断空输出。
        raise C4ProfileAnalysisError(f"C4拒绝写出空CSV：{path.name}")
    # 以UTF-8写入，确保中文标签被Excel和文本编辑器读取。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 使用第一行键顺序固定表头顺序。
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        # 先写表头。
        writer.writeheader()
        # 再写全部数据行。
        writer.writerows(rows)


# 写出带缩进的稳定JSON，便于人工复核和后续哈希。
def write_json(path: Path, payload: dict[str, Any]) -> None:
    # 排序键并保留中文，补结尾换行使文本比较稳定。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# 把字符串金额转换为浮点数；C4只做汇总显示，不承担记账金额计算。
def number(value: str) -> float:
    # 直接转换非空数值文本。
    return float(value)


# 计算数值列表均值；空列表说明调用方分组逻辑错误。
def mean(values: list[float]) -> float:
    # 空分组不能产生看似合理的零结果。
    if not values:
        # 抛出明确异常。
        raise C4ProfileAnalysisError("C4不能计算空数值列表的均值")
    # 使用Python求和并除以长度，C4指标已经是冻结的小数结果。
    return sum(values) / len(values)


# 计算线性插值百分位，供95%自助抽样边界使用。
def percentile(values: list[float], probability: float) -> float:
    # 空值不能计算百分位。
    if not values:
        # 阻断异常。
        raise C4ProfileAnalysisError("C4不能计算空列表百分位")
    # 先排序，避免调用方输入顺序影响边界。
    ordered = sorted(values)
    # 只有一个值时上下界都等于它本身。
    if len(ordered) == 1:
        # 返回唯一值。
        return ordered[0]
    # 计算位于0到n-1之间的连续位置。
    position = probability * (len(ordered) - 1)
    # 取得左侧整数下标。
    lower = int(position)
    # 取得不超过最后一个元素的右侧下标。
    upper = min(lower + 1, len(ordered) - 1)
    # 计算线性插值比例。
    weight = position - lower
    # 返回相邻两个有序值的线性插值。
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# 按企业整块重抽样，避免把同一企业内高度相关的多个预测起点误当独立样本。
def bootstrap_mean_interval(values: list[float], seed_offset: int) -> tuple[float, float]:
    # 创建由合同根种子和当前分组文字共同确定的随机数发生器。
    generator = random.Random(BOOTSTRAP_SEED + seed_offset)
    # 保存每一次重抽样后的企业等权均值。
    samples: list[float] = []
    # 重复固定次数，避免不同分组使用不同精度。
    for _ in range(BOOTSTRAP_REPETITIONS):
        # 有放回抽取与原企业数相同的企业均值。
        resample = [values[generator.randrange(len(values))] for _ in values]
        # 保存该次企业等权均值。
        samples.append(mean(resample))
    # 返回中间95%范围的下界与上界。
    return percentile(samples, 0.025), percentile(samples, 0.975)


# 从既有分配清单读取每户声明的转折月份；它只是报告标签来源。
def read_allocation_rows(root: Path) -> dict[str, dict[str, str]]:
    # 固定200户批次的画像分配清单路径。
    formal200 = root / "data" / "synthetic_generated" / "syn_b1_v2" / "_pretrain_formal_200x24m_v1" / "profile_allocation_manifest.csv"
    # 固定高营收60户批次的画像分配清单路径。
    high60 = root / "data" / "synthetic_generated" / "syn_b1_v2" / "_pretrain_formal_high_revenue_60x24m_v1" / "profile_allocation_manifest.csv"
    # 先读取两个不相互覆盖的历史分配表。
    rows = read_csv(formal200) + read_csv(high60)
    # 创建企业编号到唯一画像分配行的映射。
    mapped: dict[str, dict[str, str]] = {}
    # 遍历所有候选画像行。
    for row in rows:
        # 两批清单都以sample_id作为企业唯一编号。
        enterprise_id = row.get("sample_id", "")
        # 缺编号或重复编号都说明血缘清单异常。
        if not enterprise_id or enterprise_id in mapped:
            # 阻断不唯一的画像来源。
            raise C4ProfileAnalysisError(f"C4画像分配清单企业编号缺失或重复：{enterprise_id}")
        # 保存该企业唯一分配行。
        mapped[enterprise_id] = row
    # 返回可供最终52户查找的完整映射。
    return mapped


# 以份额加权计算一个收款或付款计划的平均账期。
def weighted_delay_days(schedule: list[dict[str, Any]]) -> float:
    # 空计划没有可解释的账期。
    if not schedule:
        # 阻断画像缺失。
        raise C4ProfileAnalysisError("C4画像的收付款计划为空")
    # 累加分子：每段天数乘份额。
    numerator = sum(float(item["delay_days"]) * float(item["share_percent"]) for item in schedule)
    # 累加分母：计划份额总和。
    denominator = sum(float(item["share_percent"]) for item in schedule)
    # 份额总和为零会使加权平均没有意义。
    if denominator <= 0:
        # 阻断错误画像。
        raise C4ProfileAnalysisError("C4画像的收付款计划份额总和不是正数")
    # 返回以天为单位的加权平均。
    return numerator / denominator


# 从交易拆分参数得到销售回款与采购付款的计划笔数。
def planned_operating_transaction_counts(transactions: dict[str, Any]) -> tuple[float, float]:
    # 读取全部类别覆盖项；不存在时按空对象处理。
    overrides = transactions.get("category_overrides") or {}
    # 读取默认交易规则；缺少它无法解释普通企业的笔数。
    default = transactions.get("default_policy") or {}
    # 取得默认每个批次的笔数。
    default_count = float(default["transaction_count"])
    # 销售回款存在覆盖项时取覆盖值，否则取默认值。
    sales_count = float((overrides.get("sales_collection") or {}).get("transaction_count", default_count))
    # 供应商付款存在覆盖项时取覆盖值，否则取默认值。
    supplier_count = float((overrides.get("supplier_payment") or {}).get("transaction_count", default_count))
    # 返回两类经营性收付的计划笔数。
    return sales_count, supplier_count


# 依合同的低、中、高阈值标记计划交易频率。
def frequency_band(value: float) -> str:
    # 平均每批不超过3笔属于低频档。
    if value <= 3.0:
        # 返回机器和中文兼具的稳定标签。
        return "low（低频：每经营批次平均不超过3笔）"
    # 平均每批少于8笔属于常规档。
    if value < 8.0:
        # 返回常规标签。
        return "regular（常规：每经营批次平均大于3笔且小于8笔）"
    # 其余属于高频档。
    return "high（高频：每经营批次平均不少于8笔）"


# 依合同的短、中、长阈值标记平均收付款账期。
def term_band(value: float) -> str:
    # 不超过30日为短账期。
    if value <= 30.0:
        # 返回短账期标签。
        return "short（短账期：平均收付款账期不超过30日）"
    # 不超过60日为中账期。
    if value <= 60.0:
        # 返回中账期标签。
        return "medium（中账期：平均收付款账期大于30日且不超过60日）"
    # 其余为长账期。
    return "long（长账期：平均收付款账期大于60日）"


# 依合同的期初缓冲代理阈值标记薄、中、厚缓冲。
def liquidity_band(value: float) -> str:
    # 小于0.30个首月销售额为薄缓冲。
    if value < 0.30:
        # 返回薄缓冲标签。
        return "thin（薄缓冲：首次注资/首月销售额小于0.30）"
    # 小于0.80个首月销售额为中缓冲。
    if value < 0.80:
        # 返回中缓冲标签。
        return "medium（中缓冲：首次注资/首月销售额为0.30至0.80）"
    # 其余为厚缓冲。
    return "thick（厚缓冲：首次注资/首月销售额不少于0.80）"


# 依合同的年化销售水平标记营收档。
def revenue_band(value: float) -> str:
    # 不超过一亿元为既有中低营收覆盖档。
    if value <= 100000000.0:
        # 返回第一档。
        return "up_to_100m（声明年化销售不超过1亿元）"
    # 不超过三亿元为新增第一档。
    if value <= 300000000.0:
        # 返回第二档。
        return "from_100m_to_300m（声明年化销售大于1亿元且不超过3亿元）"
    # 不超过五亿元为新增第二档。
    if value <= 500000000.0:
        # 返回第三档。
        return "from_300m_to_500m（声明年化销售大于3亿元且不超过5亿元）"
    # 超过五亿元时保留原值单列，不能为满足显示档位而静默压回3至5亿元。
    return "over_500m（声明年化销售超过5亿元；按证据门单独报告）"


# 将每户JSON画像转换成仅供报告的静态标签与融资合同信息。
def build_reporting_profiles(root: Path, cohort_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    # 读取既有两批画像分配清单。
    allocations = read_allocation_rows(root)
    # 创建最终企业到报告画像的映射。
    profiles: dict[str, dict[str, Any]] = {}
    # 遍历C3B最终52户血缘清单。
    for cohort in cohort_rows:
        # 取得企业编号。
        enterprise_id = cohort["enterprise_id"]
        # 该企业必须有既有画像分配行。
        if enterprise_id not in allocations:
            # 阻断未经登记的画像。
            raise C4ProfileAnalysisError(f"C4最终企业缺少画像分配：{enterprise_id}")
        # 定位正式运行保存的JSON画像；只读，不写回。
        profile_path = root / cohort["run_directory"] / "scenario_profile.json"
        # 读取最外层画像对象。
        profile = read_json(profile_path)
        # 读取用户输入画像；其字段是生成前已声明参数的副本。
        input_profile = profile.get("input_profile")
        # 输入画像不是对象时不能安全分层。
        if not isinstance(input_profile, dict):
            # 阻断错误结构。
            raise C4ProfileAnalysisError(f"C4画像缺少input_profile：{enterprise_id}")
        # 读取经营对象。
        operating = input_profile.get("operating")
        # 经营对象必须存在。
        if not isinstance(operating, dict):
            # 阻断错误结构。
            raise C4ProfileAnalysisError(f"C4画像缺少operating：{enterprise_id}")
        # 读取经营状态计划。
        regime = operating.get("regime_plan")
        # 状态计划必须是对象。
        if not isinstance(regime, dict):
            # 阻断错误结构。
            raise C4ProfileAnalysisError(f"C4画像缺少regime_plan：{enterprise_id}")
        # 读取交易规则对象。
        transactions = input_profile.get("transactions")
        # 交易规则必须是对象。
        if not isinstance(transactions, dict):
            # 阻断错误结构。
            raise C4ProfileAnalysisError(f"C4画像缺少transactions：{enterprise_id}")
        # 计算两类经营收付的计划笔数。
        sales_count, supplier_count = planned_operating_transaction_counts(transactions)
        # 计算两个计划笔数的平均值。
        transaction_mean = (sales_count + supplier_count) / 2.0
        # 读取回款计划列表。
        collection_schedule = (operating.get("collection_policy") or {}).get("base_schedule")
        # 读取供应商付款计划列表。
        supplier_schedule = operating.get("supplier_payment_schedule")
        # 要求两类计划都是列表，避免字符串被错误迭代。
        if not isinstance(collection_schedule, list) or not isinstance(supplier_schedule, list):
            # 阻断缺失计划。
            raise C4ProfileAnalysisError(f"C4画像收付款计划缺失：{enterprise_id}")
        # 计算加权平均回款天数。
        collection_days = weighted_delay_days(collection_schedule)
        # 计算加权平均付款天数。
        supplier_days = weighted_delay_days(supplier_schedule)
        # 计算两种账期的共同平均，作为已登记“收付款账期档”的透明代理。
        term_mean = (collection_days + supplier_days) / 2.0
        # 读取首次注资对象。
        capital = input_profile.get("initial_capital") or {}
        # 计算期初缓冲代理分子。
        capital_amount = float(capital["amount_cny"])
        # 读取首月潜在月销售额。
        base_sales = float(regime["base_latent_monthly_sales_cny"])
        # 销售额必须为正，否则比率没有业务意义。
        if base_sales <= 0:
            # 阻断错误画像。
            raise C4ProfileAnalysisError(f"C4画像首月销售额不是正数：{enterprise_id}")
        # 计算首次注资相对于首月销售额的缓冲代理。
        liquidity_ratio = capital_amount / base_sales
        # 计算画像声明的年化销售水平。
        annual_sales = base_sales * 12.0
        # 读取已有分配清单，转折月字段为空代表无声明转折。
        allocation = allocations[enterprise_id]
        # 读取所有银行贷款合同；空列表代表无贷款。
        loans = input_profile.get("loan_contracts") or []
        # 要求贷款合同为列表。
        if not isinstance(loans, list):
            # 阻断错误画像。
            raise C4ProfileAnalysisError(f"C4画像贷款合同不是列表：{enterprise_id}")
        # 保存只用于报告分组的静态事实和标签。
        profiles[enterprise_id] = {
            # 保存画像文件指纹，方便事后核对输入未漂移。
            "scenario_profile_sha256": sha256_file(profile_path),
            # 保存经营趋势标签。
            "operating_trend": str(regime["shape"]),
            # 保存计划笔数的可读诊断值。
            "planned_sales_transactions_per_tranche": sales_count,
            # 保存计划供应商笔数诊断值。
            "planned_supplier_transactions_per_tranche": supplier_count,
            # 保存交易频率分层标签。
            "transaction_frequency_band": frequency_band(transaction_mean),
            # 保存加权回款账期诊断值。
            "weighted_collection_delay_days": collection_days,
            # 保存加权付款账期诊断值。
            "weighted_supplier_delay_days": supplier_days,
            # 保存收付款账期分层标签。
            "receivable_and_payable_term_band": term_band(term_mean),
            # 保存期初缓冲代理数值。
            "initial_capital_to_monthly_sales_ratio": liquidity_ratio,
            # 保存缓冲分层标签。
            "liquidity_buffer_band": liquidity_band(liquidity_ratio),
            # 保存声明年化销售数值。
            "declared_annual_sales_cny": annual_sales,
            # 保存营收分层标签。
            "annual_revenue_band": revenue_band(annual_sales),
            # 保存声明转折月份，空字符串代表没有声明转折。
            "declared_turn_month": allocation.get("turn_month", ""),
            # 保存贷款合同；后续仅判断日期相交，不作模型输入。
            "loan_contracts": loans,
            # 结束当前企业的报告专用画像字典。
        }
    # 只接受刚好52户唯一报告画像。
    if len(profiles) != EXPECTED_FINAL_ENTERPRISES:
        # 阻断最终群体不完整。
        raise C4ProfileAnalysisError("C4报告画像没有覆盖最终52户")
    # 返回报告用画像映射。
    return profiles


# 读取一个企业公开日度表，建立银行工作日步骤到日期的映射。
def business_dates_by_step(root: Path, cohort: dict[str, str]) -> dict[int, date]:
    # 定位该企业公开日度表；C4不读取人工备注表。
    daily_path = root / cohort["run_directory"] / "account_daily_total.csv"
    # 日度表哈希必须仍与T2冻结清单一致。
    if sha256_file(daily_path) != cohort["account_daily_total_sha256"]:
        # 阻断企业历史数据漂移。
        raise C4ProfileAnalysisError(f"C4企业日度表哈希不一致：{cohort['enterprise_id']}")
    # 创建业务步骤到日期的映射。
    dates: dict[int, date] = {}
    # 逐行读取公开日度表。
    for row in read_csv(daily_path):
        # 只保留银行工作日；非工作日没有business_step预测坐标。
        if row["is_bank_workday"] != "True":
            # 跳过自然日补齐行。
            continue
        # 解析业务步骤整数。
        step = int(row["business_step"])
        # 同一步骤重复会破坏C3B样本键解释。
        if step in dates:
            # 阻断重复业务步骤。
            raise C4ProfileAnalysisError(f"C4日度表业务步骤重复：{cohort['enterprise_id']}")
        # 保存该工作日日期。
        dates[step] = date.fromisoformat(row["calendar_date"])
    # 返回日期索引。
    return dates


# 读取公开逐笔流水并按自然日保存最大单笔金额；绝不读取人工备注表。
def maximum_transaction_by_day(root: Path, cohort: dict[str, str]) -> dict[date, float]:
    # 定位公开逐笔流水表。
    transaction_path = root / cohort["run_directory"] / "transactions_total.csv"
    # 创建日期到当天最大收付金额的映射。
    amounts: dict[date, float] = {}
    # 逐笔读取公开交易，不读取备注、类型或任何人工字段。
    for row in read_csv(transaction_path):
        # 只提取记账日期部分。
        booking_day = date.fromisoformat(row["booking_datetime"].split("T", 1)[0])
        # 一笔交易只可能为借或贷，取二者绝对金额的较大值。
        amount = max(abs(number(row["debit_cny"])), abs(number(row["credit_cny"])))
        # 保存同一天的最大单笔金额。
        amounts[booking_day] = max(amounts.get(booking_day, 0.0), amount)
    # 返回只用于“大额窗口”标签的金额映射。
    return amounts


# 从C3B样本键解析预测截止日的银行工作日步骤。
def parse_business_step(sample_key: str) -> int:
    # C3B样本键必须以__bd加四位或更多步骤结尾。
    if "__bd" not in sample_key:
        # 阻断未知格式，避免猜测预测日。
        raise C4ProfileAnalysisError(f"C4无法解析C3B样本键：{sample_key}")
    # 取最后一个分隔符后的数字文本。
    value = sample_key.rsplit("__bd", 1)[1]
    # 数字文本必须全部是数字。
    if not value.isdigit():
        # 阻断格式错误。
        raise C4ProfileAnalysisError(f"C4样本键步骤不是整数：{sample_key}")
    # 返回步骤整数。
    return int(value)


# 判断贷款合同是否在未来预测窗口中仍然有效。
def financing_window_label(loans: list[dict[str, Any]], future_start: date, future_end: date) -> str:
    # 逐份既有贷款合同检查其存续期间是否与窗口相交。
    for loan in loans:
        # 解析已声明提款日期。
        draw_date = date.fromisoformat(str(loan["draw_date"]))
        # 解析已声明到期日期。
        maturity_date = date.fromisoformat(str(loan["maturity_date"]))
        # 两个闭区间相交说明窗口处于贷款存续期间。
        if draw_date <= future_end and maturity_date >= future_start:
            # 返回有活动贷款合同标签。
            return "active_contract（窗口内有存续银行贷款合同）"
    # 没有任一合同相交时返回无活动合同标签。
    return "no_active_contract（窗口内没有存续银行贷款合同）"


# 判断声明转折月份是否与未来预测窗口中的任一天相交。
def turning_window_label(turn_month: str, future_dates: list[date]) -> str:
    # 空转折月代表该企业没有声明转折。
    if not turn_month:
        # 返回其他窗口标签。
        return "other_window（无声明转折或窗口不覆盖声明转折月）"
    # 将每个未来日期转换为YYYY-MM并检查是否命中。
    if any(item.strftime("%Y-%m") == turn_month for item in future_dates):
        # 返回命中声明转折月份标签。
        return "declared_turn_month_in_window（窗口覆盖声明转折月份）"
    # 其余情况为未命中。
    return "other_window（无声明转折或窗口不覆盖声明转折月）"


# 判断未来窗口是否出现相对于预测起点账户尺度50%及以上的大额单笔。
def large_transaction_label(day_amounts: dict[date, float], future_dates: list[date], balance_scale: float) -> str:
    # 计算窗口内每个自然日最大单笔金额中的最大值。
    largest = max((day_amounts.get(item, 0.0) for item in future_dates), default=0.0)
    # 账户尺度应由C3B固定为正数。
    if balance_scale <= 0:
        # 阻断错误尺度。
        raise C4ProfileAnalysisError("C4大额窗口遇到非正账户尺度")
    # 单笔达到50%阈值时标记为存在大额交易。
    if largest / balance_scale >= 0.50:
        # 返回存在标签。
        return "present（窗口内存在单笔不少于账户尺度50%的交易）"
    # 否则返回不存在标签。
    return "absent（窗口内不存在单笔不少于账户尺度50%的交易）"


# 核验C3B报告、哈希和最终人口后返回冻结指标行与血缘行。
def read_and_verify_c3b(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    # 定位本地合成数据根目录。
    data = root / "data" / "synthetic_generated" / "syn_b1_v2"
    # 定位C3B唯一最终测试目录。
    c3b = data / "_post_t6_c3b_final_test_v1"
    # 定位T2最终群体血缘清单。
    cohort_path = data / "_pretrain_t2_14bd_feature_split_v1" / "cohort_manifest.csv"
    # 读取C3B最终报告。
    report = read_json(c3b / "c3b_final_test_report.json")
    # C3B状态必须证明唯一最终测试已完成一次。
    if report.get("status") != "completed_final_test" or report.get("final_test_open_count") != 1:
        # 阻断未完成或被重开的最终测试。
        raise C4ProfileAnalysisError("C4发现C3B并非已完成的一次最终测试")
    # C3B必须明确禁止结果后重训或重选。
    if report.get("no_retraining_or_reselection_after_open") is not True:
        # 阻断冻结边界不成立。
        raise C4ProfileAnalysisError("C4发现C3B结果后冻结声明缺失")
    # 逐项复算C3B自带输出哈希，保证C4读取的还是原始结果。
    for filename, expected in report.get("output_sha256", {}).items():
        # 当前文件哈希必须与C3B报告保存的值一致。
        if sha256_file(c3b / filename) != expected:
            # 阻断C3B源结果漂移。
            raise C4ProfileAnalysisError(f"C4发现C3B输出哈希不一致：{filename}")
    # 读取C3B最终指标行。
    metric_rows = read_csv(c3b / "c3b_final_prediction_metrics.csv")
    # 指标行数必须与报告和双期限人口一致。
    if len(metric_rows) != EXPECTED_METRIC_ROWS or len(metric_rows) != int(report["metric_row_count"]):
        # 阻断指标文件被替换或截断。
        raise C4ProfileAnalysisError("C4发现C3B最终指标行数不一致")
    # 读取T2最终群体血缘清单并只保留final_test行。
    cohort_rows = [row for row in read_csv(cohort_path) if row.get("split_group") == "final_test"]
    # 最终群体必须恰为52户且编号唯一。
    if len(cohort_rows) != EXPECTED_FINAL_ENTERPRISES or len({row["enterprise_id"] for row in cohort_rows}) != EXPECTED_FINAL_ENTERPRISES:
        # 阻断最终人口变化。
        raise C4ProfileAnalysisError("C4最终企业血缘清单不是52户唯一企业")
    # 指标企业集合必须正好等于血缘清单集合。
    if {row["enterprise_id"] for row in metric_rows} != {row["enterprise_id"] for row in cohort_rows}:
        # 阻断C3B结果与最终群体错配。
        raise C4ProfileAnalysisError("C4发现C3B指标企业集合与最终血缘不一致")
    # 返回已核验指标、血缘与报告。
    return metric_rows, cohort_rows, report


# 为每条C3B指标追加八类报告标签；本函数不改写原C3B文件。
def enrich_metric_rows(root: Path, metric_rows: list[dict[str, str]], cohort_rows: list[dict[str, str]], profiles: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    # 建立企业编号到血缘行的映射。
    cohorts = {row["enterprise_id"]: row for row in cohort_rows}
    # 建立每户的银行工作日日期索引。
    dates = {enterprise_id: business_dates_by_step(root, cohort) for enterprise_id, cohort in cohorts.items()}
    # 建立每户的公开逐笔最大金额日期索引。
    amounts = {enterprise_id: maximum_transaction_by_day(root, cohort) for enterprise_id, cohort in cohorts.items()}
    # 创建追加标签后的指标行列表。
    enriched: list[dict[str, str]] = []
    # 逐条处理C3B已冻结指标。
    for row in metric_rows:
        # 读取当前企业编号。
        enterprise_id = row["enterprise_id"]
        # 解析当前期限。
        horizon = int(row["average_horizon_business_days"])
        # 只接受预登记10日或30日。
        if horizon not in HORIZONS:
            # 阻断新增期限。
            raise C4ProfileAnalysisError("C4发现合同外预测期限")
        # 解析C3B样本键中的截止日步骤。
        cutoff_step = parse_business_step(row["sample_key"])
        # 取得未来1日至期限末的银行工作日日期。
        future_dates = [dates[enterprise_id][cutoff_step + offset] for offset in range(1, horizon + 1)]
        # 复制原始C3B指标列，绝不改其数值。
        output = dict(row)
        # 取得当前企业报告画像。
        profile = profiles[enterprise_id]
        # 追加静态交易频率档。
        output["transaction_frequency_band"] = str(profile["transaction_frequency_band"])
        # 追加静态经营趋势。
        output["operating_trend"] = str(profile["operating_trend"])
        # 追加静态收付款账期档。
        output["receivable_and_payable_term_band"] = str(profile["receivable_and_payable_term_band"])
        # 追加静态期初缓冲代理档。
        output["liquidity_buffer_band"] = str(profile["liquidity_buffer_band"])
        # 追加静态声明年化销售档。
        output["annual_revenue_band"] = str(profile["annual_revenue_band"])
        # 追加动态大额交易窗口标签。
        output["large_transaction_window"] = large_transaction_label(amounts[enterprise_id], future_dates, number(row["balance_scale_cny"]))
        # 追加动态声明转折窗口标签。
        output["turning_point_window"] = turning_window_label(str(profile["declared_turn_month"]), future_dates)
        # 追加动态融资窗口标签。
        output["financing_window"] = financing_window_label(profile["loan_contracts"], future_dates[0], future_dates[-1])
        # 保存当前已分层行。
        enriched.append(output)
    # 返回不改变C3B原文件的派生行。
    return enriched


# 将一组行先按企业平均，再以企业等权方式汇总并给出95%自助抽样范围。
def summarize_slice(rows: list[dict[str, str]], axis: str, label: str, horizon: int) -> dict[str, str]:
    # 创建企业到其行列表的分组映射。
    by_enterprise: dict[str, list[dict[str, str]]] = {}
    # 逐行放入对应企业。
    for row in rows:
        # 追加到当前企业的行列表。
        by_enterprise.setdefault(row["enterprise_id"], []).append(row)
    # 为每户计算内部均值，防止预测起点更多的企业权重更大。
    enterprise_metrics: list[dict[str, float]] = []
    # 按企业编号排序使自助抽样输入顺序稳定。
    for enterprise_id in sorted(by_enterprise):
        # 取得这一户在当前分层的所有预测起点。
        items = by_enterprise[enterprise_id]
        # 计算该户P50绝对误差均值。
        p50_error = mean([number(item["normalized_p50_absolute_error"]) for item in items])
        # 计算该户WIS80均值。
        wis80 = mean([number(item["normalized_wis80"]) for item in items])
        # 计算该户覆盖率均值；true记为1，false记为0。
        coverage = mean([1.0 if item["covered_by_interval"].lower() == "true" else 0.0 for item in items])
        # 计算该户区间宽度均值。
        width = mean([number(item["normalized_interval_width"]) for item in items])
        # 计算该户原始区间评分均值。
        interval_score = mean([number(item["normalized_interval_score"]) for item in items])
        # 保存当前企业等权前的五项值。
        enterprise_metrics.append({"p50_error": p50_error, "wis80": wis80, "coverage": coverage, "width": width, "interval_score": interval_score})
    # 取每户P50误差值列表。
    p50_values = [item["p50_error"] for item in enterprise_metrics]
    # 取每户WIS80值列表。
    wis_values = [item["wis80"] for item in enterprise_metrics]
    # 取每户覆盖率值列表。
    coverage_values = [item["coverage"] for item in enterprise_metrics]
    # 取每户区间宽度值列表。
    width_values = [item["width"] for item in enterprise_metrics]
    # 取每户区间评分值列表。
    score_values = [item["interval_score"] for item in enterprise_metrics]
    # 用分组文字生成稳定但不同的随机种子偏移。
    seed_offset = sum(ord(character) for character in f"{axis}|{label}|{horizon}")
    # 计算P50误差的企业整块95%范围。
    p50_low, p50_high = bootstrap_mean_interval(p50_values, seed_offset)
    # 计算WIS80的企业整块95%范围。
    wis_low, wis_high = bootstrap_mean_interval(wis_values, seed_offset + 1)
    # 判断预登记最少企业数与最少预测起点数是否同时满足。
    sufficient = len(enterprise_metrics) >= MIN_ENTERPRISES and len(rows) >= MIN_ORIGINS
    # 返回可直接写入CSV的一行汇总。
    return {
        # 保存分析轴机器名。
        "slice_axis": axis,
        # 保存类别名称及中文解释。
        "slice_label": label,
        # 保存预测期限。
        "average_horizon_business_days": str(horizon),
        # 保存企业等权的企业数。
        "enterprise_count": str(len(enterprise_metrics)),
        # 保存该分层中的预测起点数。
        "forecast_origin_count": str(len(rows)),
        # 保存企业等权P50绝对误差。
        "enterprise_equal_normalized_p50_absolute_error": f"{mean(p50_values):.8f}",
        # 保存P50误差企业整块95%范围。
        "p50_error_enterprise_block_95pct_low": f"{p50_low:.8f}",
        # 保存P50误差企业整块95%范围上界。
        "p50_error_enterprise_block_95pct_high": f"{p50_high:.8f}",
        # 保存企业等权WIS80；它是当前轴内排序值，越低越好。
        "enterprise_equal_normalized_wis80": f"{mean(wis_values):.8f}",
        # 保存WIS80企业整块95%范围下界。
        "wis80_enterprise_block_95pct_low": f"{wis_low:.8f}",
        # 保存WIS80企业整块95%范围上界。
        "wis80_enterprise_block_95pct_high": f"{wis_high:.8f}",
        # 保存企业等权覆盖率。
        "enterprise_equal_empirical_coverage": f"{mean(coverage_values):.8f}",
        # 保存企业等权区间宽度。
        "enterprise_equal_normalized_interval_width": f"{mean(width_values):.8f}",
        # 保存企业等权原始区间评分。
        "enterprise_equal_normalized_interval_score": f"{mean(score_values):.8f}",
        # 保存证据等级；不足时只允许描述，不允许下适用性结论。
        "evidence_level": "有足够描述证据（合成最终测试范围）" if sufficient else "证据不足",
        # 明确C3B没有冻结余额保持法最终输出，因此不事后补相对Skill。
        "relative_baseline_skill_status": "未计算：C3B未冻结最终余额保持法结果，C4不事后补跑",
        # 明确本行不能被解释为因果或真实企业外推。
        "interpretation_boundary": "仅描述冻结合成最终测试；不表示因果、不改变模型或产品范围",
        # 结束当前分层汇总CSV行字典。
    }


# 按八个预登记轴生成分层汇总表，并在同一轴同一期限内以WIS80由低到高排名。
def build_slice_summaries(enriched_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 固定八个已预登记分析轴，禁止按结果新增或删除轴。
    axes = ("transaction_frequency_band", "operating_trend", "receivable_and_payable_term_band", "liquidity_buffer_band", "annual_revenue_band", "large_transaction_window", "turning_point_window", "financing_window")
    # 创建最终汇总行列表。
    summaries: list[dict[str, str]] = []
    # 逐个期限处理，避免把10日和30日混在同一排名中。
    for horizon in HORIZONS:
        # 只选择当前期限行。
        horizon_rows = [row for row in enriched_rows if int(row["average_horizon_business_days"]) == horizon]
        # 逐个预登记轴分组。
        for axis in axes:
            # 创建当前轴的类别到行列表映射。
            grouped: dict[str, list[dict[str, str]]] = {}
            # 逐行读取当前轴标签。
            for row in horizon_rows:
                # 按标签追加到对应组。
                grouped.setdefault(row[axis], []).append(row)
            # 对每个标签计算企业等权汇总。
            for label in sorted(grouped):
                # 追加当前类别汇总行。
                summaries.append(summarize_slice(grouped[label], axis, label, horizon))
    # 为每个“期限+轴”分别建立从低WIS80到高WIS80的描述性排名。
    for horizon in HORIZONS:
        # 逐轴找出同一期限的汇总行。
        for axis in axes:
            # 筛选当前排名组。
            candidates = [row for row in summaries if row["average_horizon_business_days"] == str(horizon) and row["slice_axis"] == axis]
            # 依WIS80数值升序、标签升序排序，数值低代表误差/区间综合损失较低。
            ordered = sorted(candidates, key=lambda item: (number(item["enterprise_equal_normalized_wis80"]), item["slice_label"]))
            # 从1开始写排名。
            for rank, row in enumerate(ordered, start=1):
                # 增加排名字段；它只可在同轴同期限内阅读。
                row["wis80_rank_within_same_axis_and_horizon"] = str(rank)
    # 固定输出字段顺序要求排名字段放在末尾前；此循环已对每行赋值。
    return summaries


# 构造每户静态报告画像表；它不输出随机种子，避免把无关真值扩散到报告中。
def enterprise_profile_rows(profiles: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    # 创建输出行列表。
    rows: list[dict[str, str]] = []
    # 按企业编号稳定排序。
    for enterprise_id in sorted(profiles):
        # 取得当前报告画像。
        profile = profiles[enterprise_id]
        # 写出分层标签和诊断值，不写贷款金额、种子或人工备注。
        rows.append({
            # 保存企业编号。
            "enterprise_id": enterprise_id,
            # 保存画像文件指纹。
            "scenario_profile_sha256": str(profile["scenario_profile_sha256"]),
            # 保存经营趋势。
            "operating_trend": str(profile["operating_trend"]),
            # 保存销售回款计划笔数。
            "planned_sales_transactions_per_tranche": f"{float(profile['planned_sales_transactions_per_tranche']):.2f}",
            # 保存供应商付款计划笔数。
            "planned_supplier_transactions_per_tranche": f"{float(profile['planned_supplier_transactions_per_tranche']):.2f}",
            # 保存交易频率档。
            "transaction_frequency_band": str(profile["transaction_frequency_band"]),
            # 保存加权回款天数。
            "weighted_collection_delay_days": f"{float(profile['weighted_collection_delay_days']):.2f}",
            # 保存加权付款天数。
            "weighted_supplier_delay_days": f"{float(profile['weighted_supplier_delay_days']):.2f}",
            # 保存收付款账期档。
            "receivable_and_payable_term_band": str(profile["receivable_and_payable_term_band"]),
            # 保存期初缓冲代理数值。
            "initial_capital_to_monthly_sales_ratio": f"{float(profile['initial_capital_to_monthly_sales_ratio']):.8f}",
            # 保存缓冲档。
            "liquidity_buffer_band": str(profile["liquidity_buffer_band"]),
            # 保存声明年化销售额。
            "declared_annual_sales_cny": f"{float(profile['declared_annual_sales_cny']):.2f}",
            # 保存营收档。
            "annual_revenue_band": str(profile["annual_revenue_band"]),
            # 保存声明转折月份；空值明确代表未声明。
            "declared_turn_month": str(profile["declared_turn_month"]),
            # 保存是否存在任何贷款合同，而不输出合同金额。
            "has_declared_bank_loan_contract": "true" if profile["loan_contracts"] else "false",
            # 再次声明画像只用于报告。
            "reporting_only_notice": "画像仅用于C4分组，未进入C3B模型、特征或指标",
            # 结束当前企业画像CSV行字典并作为append参数。
        })
    # 返回每户一行的报告画像表。
    return rows


# 执行一次完整C4只读分析，并将派生结果写到全新的独立目录。
def run(root: Path, output_directory: Path) -> dict[str, Any]:
    # 历史输出目录存在时拒绝覆盖，保持每次证据物理隔离。
    if output_directory.exists():
        # 抛出不覆盖错误。
        raise FileExistsError(f"C4输出目录已存在，禁止覆盖：{output_directory}")
    # 核验并读取C3B冻结指标、最终血缘和报告。
    metric_rows, cohort_rows, c3b_report = read_and_verify_c3b(root)
    # 读取仅供报告分层的既有画像。
    profiles = build_reporting_profiles(root, cohort_rows)
    # 为每条冻结指标追加八类报告标签。
    enriched_rows = enrich_metric_rows(root, metric_rows, cohort_rows, profiles)
    # 构造八轴、两期限的企业等权汇总与证据等级。
    summaries = build_slice_summaries(enriched_rows)
    # 构造每户静态报告画像表。
    profile_rows = enterprise_profile_rows(profiles)
    # 先创建全新输出目录，随后只写C4派生文件。
    output_directory.mkdir(parents=True, exist_ok=False)
    # 写出可追溯的每户报告画像表。
    write_csv(output_directory / "c4_enterprise_reporting_profiles.csv", profile_rows)
    # 写出逐预测起点的派生标签与原始C3B指标副本。
    write_csv(output_directory / "c4_enriched_final_metrics.csv", enriched_rows)
    # 写出按轴、类别、期限的企业等权排名和证据门结果。
    write_csv(output_directory / "c4_profile_slice_summary.csv", summaries)
    # 创建C4报告的主要机器可读内容。
    report: dict[str, Any] = {
        # 保存任务编号。
        "stage_id": STAGE_ID,
        # 保存完成状态。
        "status": "completed_reporting_only_analysis",
        # 明确C4读取的唯一模型名。
        "selected_method_id": c3b_report["selected_method_id"],
        # 明确C3B最终测试没有被重开。
        "final_test_reopened": False,
        # 明确没有重训、调参或重新选择模型。
        "retraining_tuning_or_reselection_performed": False,
        # 明确没有修改C3B原始结果。
        "c3b_original_result_modified": False,
        # 明确人工备注表没有读取。
        "cash_flow_review_notes_read_or_loaded": False,
        # 明确没有写数据库。
        "database_created_or_written": False,
        # 明确画像和生成真值只在报告分组使用。
        "profile_seed_and_generator_truth_reporting_only": True,
        # 保存最终企业数。
        "final_enterprise_count": len(cohort_rows),
        # 保存C3B原始指标行数。
        "c3b_metric_row_count": len(metric_rows),
        # 保存C4派生指标行数，应等于C3B行数。
        "c4_enriched_metric_row_count": len(enriched_rows),
        # 保存C4分层摘要行数。
        "profile_slice_summary_row_count": len(summaries),
        # 明确相对基线Skill为何不报告。
        "relative_baseline_skill_status_cn": "C3B未冻结最终余额保持法结果；C4为避免事后新增对手，不计算相对基线Skill",
        # 固定证据门解释。
        "evidence_gate": {"minimum_enterprises": MIN_ENTERPRISES, "minimum_forecast_origins": MIN_ORIGINS, "below_minimum_label": "证据不足"},
        # 固定结论边界。
        "conclusion_boundary_cn": "全部结果仅适用于冻结的52户合成最终测试，不外推真实企业，不表示因果，不反馈修改模型、样本或产品范围。",
        # 保存C3B报告哈希，使C4与原结果建立只读血缘。
        "c3b_final_test_report_sha256": sha256_file(root / "data" / "synthetic_generated" / "syn_b1_v2" / "_post_t6_c3b_final_test_v1" / "c3b_final_test_report.json"),
        # 结束C4主机器报告字典。
    }
    # 为三个C4派生CSV生成文件指纹。
    report["c4_output_sha256"] = {
        # 保存每户画像表哈希。
        "c4_enterprise_reporting_profiles.csv": sha256_file(output_directory / "c4_enterprise_reporting_profiles.csv"),
        # 保存逐行派生标签表哈希。
        "c4_enriched_final_metrics.csv": sha256_file(output_directory / "c4_enriched_final_metrics.csv"),
        # 保存分层汇总表哈希。
        "c4_profile_slice_summary.csv": sha256_file(output_directory / "c4_profile_slice_summary.csv"),
        # 结束C4输出哈希清单字典。
    }
    # 最后写机器报告；报告本身不放进自己的哈希清单，避免自引用。
    write_json(output_directory / "c4_profile_analysis_report.json", report)
    # 返回报告对象供命令入口和测试读取。
    return report
