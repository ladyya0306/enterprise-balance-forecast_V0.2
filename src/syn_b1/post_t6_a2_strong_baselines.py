"""POST-T6-A2：在A1开发池上运行五种预登记多期限强基线。"""

# 导入csv标准库；它只读取A1派生表和写出可审计的基线预测明细。
import csv
# 导入json标准库；它读取A0/A1报告并写出锁定基线的机器可读记录。
import json
# 从collections导入defaultdict；它按样本键和折角色收集少量索引信息。
from collections import defaultdict
# 从 dataclasses 导入dataclass；它把公开历史、标签和折角色表示为字段明确的对象。
from dataclasses import dataclass
# 从 decimal 导入Decimal与舍入规则；金额误差和指标保持精确财务精度。
from decimal import Decimal, ROUND_HALF_UP
# 从 pathlib 导入Path；它定位冻结A0合同、A1输入和新A2输出。
from pathlib import Path
# 从 typing 导入Iterable和Iterator；它说明大CSV按流式逐行处理而不是一次全部装入内存。
from typing import Iterable, Iterator

# 导入yaml；它读取A0冻结的五种候选、平局顺序和四折定义。
import yaml

# 固定A2实现版本；未来改变基线公式、评估指标或锁定规则必须升级版本。
A2_VERSION = "syn_b1_post_t6_a2_strong_baseline_1_0"
# 固定最长历史序列长度；五种基线的所有输入均来自A1的60银行工作日序列表。
HISTORY_BUSINESS_DAYS = 60
# 固定完整预测路径长度；每种方法一次产生从T+1到T+30的余额路径。
FORECAST_PATH_BUSINESS_DAYS = 30
# 固定四个预登记评价项目；T+5采用5日路径而不是只看第5天。
REGISTERED_MEASURES = (
    ("t_plus_1_endpoint", 1),
    ("t_plus_5_path", 5),
    ("t_plus_10_endpoint", 10),
    ("t_plus_30_endpoint", 30),
)
# 固定五种强基线的预注册顺序；同分时只能按这个顺序锁定，不能看复杂模型或最终测试。
BASELINE_ORDER = (
    "last_balance_path",
    "repeat_last_5bd_changes",
    "historical_horizon_net_flow_mean",
    "exponential_smoothed_daily_change",
    "intermittent_event_amount",
)
# 固定指数平滑系数；0.20来自A0合同，A2不得根据结果调整。
SES_ALPHA = Decimal("0.20")
# 固定A2只输出校准和验证角色的预测；fit角色只用于未来复杂模型拟合而不需要基线误差表。
PREDICTION_ROLES = {"calibration", "validation"}


# 声明A2专用异常；它将输入指纹、密封边界和预测公式失败与一般模型成绩区分开。
class A2ContractError(ValueError):
    """当A2违反A0合同、A1血缘或强基线完整性时抛出。"""


# 用不可变对象保存一个公开历史工作日；它不含企业画像、种子或未来金额。
@dataclass(frozen=True)
class HistoryDay:
    # 保存已发生的日终余额；它是当前可见余额历史。
    closing_balance_cny: Decimal
    # 保存已发生的日净流量；它等于该工作日公开收支的净额。
    net_flow_cny: Decimal


# 用不可变对象保存一个开发池样本的未来答案；它只存在于开发池，不涉及最终测试。
@dataclass(frozen=True)
class TargetPath:
    # 保存样本键；它只连接输入、标签与折索引，不作为模型特征。
    sample_key: str
    # 保存企业编号；它只用于企业等权指标和企业隔离核验。
    enterprise_id: str
    # 保存截止日余额；它是所有预测路径的可见起点。
    cutoff_balance_cny: Decimal
    # 保存账户归一化尺度；它来自截止日可见余额与冻结下限。
    balance_scale_cny: Decimal
    # 保存未来30个银行工作日的实际期末余额；它只用于开发池评价。
    target_balances_cny: tuple[Decimal, ...]


# 用不可变对象保存一个样本在一折中的角色；它只用于数据切分和输出审计。
@dataclass(frozen=True)
class FoldRole:
    # 保存折编号A/B/C/D。
    fold_id: str
    # 保存角色calibration或validation；fit不进入A2预测输出。
    fold_role: str


# 计算文件的SHA-256；它证明A1输入在A2前没有被改写。
def sha256_file(path: Path) -> str:
    # 读取原始字节并计算哈希；本函数不修改输入文件。
    import hashlib
    # 返回固定长度的十六进制摘要。
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 将Decimal金额按人民币两位小数输出；格式化不改变内部精确计算。
def money(value: Decimal) -> str:
    # 按普通财务规则四舍五入到分。
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# 将比例按八位小数输出；账户归一化MAE和基线比较需要稳定精度。
def ratio(value: Decimal) -> str:
    # 使用定点格式避免小数在CSV和Excel中变成科学计数法。
    return f"{value.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP):f}"


# 读取UTF-8或UTF-8-BOM CSV；大文件调用方可选择逐行迭代器而不是读取成列表。
def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    # 以utf-8-sig打开兼容普通UTF-8和带BOM的CSV。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # 逐行产出字典，避免A2将600多万行序列表整体放入内存。
        yield from csv.DictReader(handle)


# 读取并核验A0合同；A2只接受已经冻结且与预注册公式一致的版本。
def load_a0_contract(contract_path: Path) -> dict[str, object]:
    # 安全解析A0 YAML文本，不执行任何配置对象。
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    # 要求阶段编号准确，防止误用其他实验合同。
    if contract.get("stage_id") != "POST-T6-A0":
        # 阻断错误合同。
        raise A2ContractError("A0合同阶段编号不正确")
    # 要求状态为frozen，草稿不能作为实际基线实验依据。
    if contract.get("status") != "frozen":
        # 阻断未冻结合同。
        raise A2ContractError("A0合同未冻结")
    # 取出A0登记的基线方法标识。
    configured_order = tuple(item["id"] for item in contract["strong_baseline_contract"]["candidates"])
    # 要求方法集合和固定平局顺序都与A2实现一一对应。
    if configured_order != BASELINE_ORDER or tuple(contract["strong_baseline_contract"]["exact_tie_order"]) != BASELINE_ORDER:
        # 阻断基线候选或平局顺序漂移。
        raise A2ContractError("A0五种强基线或平局顺序与A2不一致")
    # 要求指数平滑系数保持冻结值。
    if Decimal(str(contract["strong_baseline_contract"]["candidates"][3]["alpha"])) != SES_ALPHA:
        # 阻断按结果改动的平滑参数。
        raise A2ContractError("A0指数平滑系数与A2不一致")
    # 返回已核验合同供调用方读取A1输入指纹和四折信息。
    return contract


# 读取A1报告与特征清单，并核验最终测试仍密封、输入指纹和文件哈希全部一致。
def validate_a1_inputs(a1_directory: Path, contract: dict[str, object], contract_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    # 定位A1阶段报告。
    report_path = a1_directory / "a1_derivative_report.json"
    # 定位A1特征清单。
    manifest_path = a1_directory / "feature_manifest.json"
    # 缺少任一证据文件时拒绝执行基线。
    if not report_path.is_file() or not manifest_path.is_file():
        # 阻断不完整A1输入。
        raise A2ContractError("缺少A1报告或特征清单")
    # 解析A1报告JSON。
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 解析A1特征清单JSON。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 要求A1已经通过并覆盖208户开发企业。
    if report.get("status") != "passed" or report.get("development_enterprise_count") != 208:
        # 阻断未通过或人口错误的派生数据。
        raise A2ContractError("A1不是208户通过状态")
    # 要求A1共同截止点数仍与A0绑定的D0事实一致。
    if report.get("common_sample_count") != 102964 or report.get("d0_expected_common_sample_count") != 102964:
        # 阻断资格标准漂移。
        raise A2ContractError("A1共同截止点数量与D0不一致")
    # 要求最终测试从未被A1读取或组装。
    if any(report.get(field) is not False for field in ("final_test_run_directories_opened", "final_test_daily_files_read", "final_test_labels_read_or_assembled", "final_test_features_created")):
        # 阻断密封边界已被破坏的输入。
        raise A2ContractError("A1报告显示最终测试已被打开或组装")
    # 要求A1所引用的A0合同指纹仍与当前冻结合同相同。
    if report["input_sha256"]["a0_contract"] != sha256_file(contract_path):
        # 阻断A1与当前A0合同不一致。
        raise A2ContractError("A1绑定的A0合同指纹不一致")
    # 要求特征清单本身仍与报告中登记的指纹一致。
    if report.get("feature_manifest_sha256") != sha256_file(manifest_path):
        # 阻断特征字段说明被静默改写。
        raise A2ContractError("A1特征清单指纹不一致")
    # 要求60日序列和独立标签表均在A1清单中登记。
    if manifest.get("input_files", {}).get("60", {}).get("sequence") != "lstm_sequences_60bd.csv" or manifest.get("label_file") != "development_path_targets_t1_to_t30.csv":
        # 阻断错误的A1文件布局。
        raise A2ContractError("A1未登记60日序列或30日标签表")
    # 固定A2实际读取的三份A1数据文件；它们都必须有A1报告登记的输出指纹。
    required_files = (manifest["label_file"], manifest["input_files"]["60"]["sequence"], manifest["fold_index_file"])
    # 逐个核验文件存在及其SHA-256与A1报告一致。
    for name in required_files:
        # 将文件名解析到A1输出目录。
        path = a1_directory / name
        # 要求文件存在且哈希未发生变化。
        if not path.is_file() or report.get("output_sha256", {}).get(name) != sha256_file(path):
            # 阻断A1输入文件缺失或被改写。
            raise A2ContractError(f"A1输入文件缺失或指纹改变：{name}")
    # 要求A1明确排除身份和target字段作为模型输入。
    tree_inputs = set(manifest.get("tree_model_input_columns", []))
    # 要求树模型输入没有企业身份或目标字段。
    if {"enterprise_id", "sample_key", "cutoff_date"}.intersection(tree_inputs) or any(name.startswith("target_") for name in tree_inputs):
        # 阻断A1模型输入边界不安全。
        raise A2ContractError("A1树模型输入清单包含身份或目标字段")
    # 返回经核验的报告和特征清单。
    return report, manifest


# 读取四折索引中需要A2预测的校准和验证角色；fit角色绝不使用未来标签输出预测。
def read_prediction_roles(fold_index_path: Path) -> dict[str, list[FoldRole]]:
    # 初始化样本键到折角色列表的映射。
    roles_by_key: dict[str, list[FoldRole]] = defaultdict(list)
    # 逐行读取不含金额的四折索引。
    for row in iter_csv(fold_index_path):
        # 读取当前折角色。
        role = row["fold_role"]
        # 只保留区间校准和验证角色；fit不需要生成基线预测明细。
        if role not in PREDICTION_ROLES:
            # 跳过fit行。
            continue
        # 读取样本键。
        sample_key = row["sample_key"]
        # 将当前折和角色追加到该样本键下。
        roles_by_key[sample_key].append(FoldRole(row["fold_id"], role))
    # 要求至少有一个校准或验证样本。
    if not roles_by_key:
        # 阻断空索引。
        raise A2ContractError("A1四折索引没有校准或验证角色")
    # 返回只含A2需要预测的样本角色映射。
    return roles_by_key


# 读取被校准或验证角色引用的开发池30日答案；最终测试样本键若出现立即拒绝。
def read_selected_target_paths(target_path: Path, selected_keys: set[str]) -> dict[str, TargetPath]:
    # 初始化样本键到目标路径的映射。
    targets: dict[str, TargetPath] = {}
    # 逐行读取A1独立标签表。
    for row in iter_csv(target_path):
        # 读取当前样本键。
        sample_key = row["sample_key"]
        # 未被校准或验证引用的开发样本无需加载，节省内存。
        if sample_key not in selected_keys:
            # 跳过fit专用样本。
            continue
        # 要求开发池企业编号符合SYN-B1正式企业命名。
        if not row["enterprise_id"].startswith("syn_b1_p0_"):
            # 阻断未知企业来源。
            raise A2ContractError(f"A1标签表存在未知企业：{row['enterprise_id']}")
        # 逐期限读取30个实际余额答案。
        balances = tuple(Decimal(row[f"target_balance_t_plus_{horizon:02d}_cny"]) for horizon in range(1, FORECAST_PATH_BUSINESS_DAYS + 1))
        # 要求恰有30个答案，避免部分路径与完整路径比较。
        if len(balances) != FORECAST_PATH_BUSINESS_DAYS:
            # 阻断标签路径长度错误。
            raise A2ContractError(f"A1标签路径长度错误：{sample_key}")
        # 保存强类型目标路径。
        targets[sample_key] = TargetPath(
            # 保存样本键。
            sample_key=sample_key,
            # 保存企业编号。
            enterprise_id=row["enterprise_id"],
            # 解析截止日余额。
            cutoff_balance_cny=Decimal(row["cutoff_balance_cny"]),
            # 解析账户误差归一化尺度。
            balance_scale_cny=Decimal(row["balance_scale_cny"]),
            # 保存未来30日实际余额路径。
            target_balances_cny=balances,
        )
    # 要求每个需要预测的样本键都成功加载标签。
    if set(targets) != selected_keys:
        # 列出少量缺失样本键帮助定位而不输出任何金额。
        missing = sorted(selected_keys.difference(targets))[:5]
        # 阻断标签和折索引不一致。
        raise A2ContractError(f"A1标签表缺少校准或验证样本：{missing}")
    # 返回仅开发池校准/验证样本的目标路径。
    return targets


# 将60日序列表按样本键流式分组；排序错误、缺行或重复位置都会被拒绝。
def iter_history_windows(sequence_path: Path, selected_keys: set[str]) -> Iterator[tuple[str, tuple[HistoryDay, ...]]]:
    # 保存当前分组样本键，初始为空。
    current_key: str | None = None
    # 保存当前样本的60日公开历史。
    current_rows: list[HistoryDay] = []
    # 保存当前样本的期望序列位置。
    expected_position = 1
    # 逐行读取600多万行60日序列表。
    for row in iter_csv(sequence_path):
        # 读取当前行样本键。
        sample_key = row["sample_key"]
        # 只要样本键发生变化，就完成上一组的严格校验和可能输出。
        if current_key is not None and sample_key != current_key:
            # 要求上一组恰好60行。
            if len(current_rows) != HISTORY_BUSINESS_DAYS:
                # 阻断不完整的历史窗口。
                raise A2ContractError(f"A1 60日序列不是60行：{current_key}")
            # 仅当上一组被校准或验证角色引用时才产出，fit专用样本不占内存。
            if current_key in selected_keys:
                # 产出样本键及其不可变60日公开历史。
                yield current_key, tuple(current_rows)
            # 切换到新样本键并清空历史行。
            current_key = sample_key
            # 重置新样本的期望位置。
            current_rows = []
            # 重置新样本的序列位置检查。
            expected_position = 1
        # 首行时设置当前样本键。
        if current_key is None:
            # 保存第一个样本键。
            current_key = sample_key
        # 要求序列位置连续从1开始，防止窗口被打散或跳行。
        if int(row["sequence_position"]) != expected_position:
            # 阻断位置错误。
            raise A2ContractError(f"A1 60日序列位置不连续：{sample_key}")
        # 追加当前行的公开余额与净流量。
        current_rows.append(HistoryDay(Decimal(row["closing_balance_cny"]), Decimal(row["daily_net_flow_cny"])))
        # 将期望位置加1供下一行校验。
        expected_position += 1
    # 文件结束后仍需处理最后一个样本组。
    if current_key is not None:
        # 要求最后一组同样恰好60行。
        if len(current_rows) != HISTORY_BUSINESS_DAYS:
            # 阻断最后一个不完整窗口。
            raise A2ContractError(f"A1最后一个60日序列不是60行：{current_key}")
        # 仅在最后组被引用时产出。
        if current_key in selected_keys:
            # 产出最后一组公开历史。
            yield current_key, tuple(current_rows)


# 计算签名非零历史净流量的中位数；无事件时返回0，表示不伪造收支金额。
def signed_nonzero_median(values: Iterable[Decimal]) -> Decimal:
    # 收集所有非零日净流量并排序。
    ordered = sorted(value for value in values if value != 0)
    # 没有非零日时中位数为0元。
    if not ordered:
        # 返回精确0元。
        return Decimal("0")
    # 计算中间位置。
    middle = len(ordered) // 2
    # 奇数个值直接返回中间项。
    if len(ordered) % 2 == 1:
        # 返回唯一中间值。
        return ordered[middle]
    # 偶数个值取两项平均，保持签名方向而非绝对金额中位数。
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")


# 根据60日公开历史生成五种方法完整30日余额路径；本函数完全不接受未来答案。
def baseline_paths(history: tuple[HistoryDay, ...]) -> dict[str, tuple[Decimal, ...]]:
    # 要求历史窗口恰为冻结的60个银行工作日。
    if len(history) != HISTORY_BUSINESS_DAYS:
        # 阻断错误窗口长度。
        raise A2ContractError("强基线历史窗口不是60个银行工作日")
    # 读取截止日余额作为所有预测路径的共同起点。
    cutoff_balance = history[-1].closing_balance_cny
    # 提取60个已发生的每日净流量；它们是所有动态基线唯一的金额输入。
    daily_changes = tuple(day.net_flow_cny for day in history)
    # “余额保持”方法让未来每一天都等于截止日余额。
    last_balance_path = tuple(cutoff_balance for _ in range(FORECAST_PATH_BUSINESS_DAYS))
    # 取得最近5个银行工作日的已发生净流量变化节奏。
    last_five_changes = daily_changes[-5:]
    # 初始化重复5日节奏的预测余额列表。
    repeated_path: list[Decimal] = []
    # 初始化重复节奏的累计预测变化。
    repeated_change = Decimal("0")
    # 逐个未来工作日重复最近5日变化。
    for index in range(FORECAST_PATH_BUSINESS_DAYS):
        # 取按模5循环的公开历史变化。
        repeated_change += last_five_changes[index % len(last_five_changes)]
        # 将累计变化加到截止日余额形成当日预测余额。
        repeated_path.append(cutoff_balance + repeated_change)
    # 初始化历史同长度窗口均值路径列表。
    historical_mean_path: list[Decimal] = []
    # 逐个未来期限直接计算同长度历史累计净流量均值。
    for horizon in range(1, FORECAST_PATH_BUSINESS_DAYS + 1):
        # 计算60日内可以取到的完整、不重叠窗口数量。
        block_count = HISTORY_BUSINESS_DAYS // horizon
        # 对每个不重叠历史块求累计净流量。
        block_sums = [sum(daily_changes[index * horizon:(index + 1) * horizon], Decimal("0")) for index in range(block_count)]
        # 取所有同长度块的平均累计变化并加回截止日余额。
        historical_mean_path.append(cutoff_balance + sum(block_sums, Decimal("0")) / Decimal(len(block_sums)))
    # 用第一个历史日变化初始化指数平滑水平。
    smoothed_change = daily_changes[0]
    # 逐日更新固定alpha的指数平滑水平；不读取验证或未来变化。
    for change in daily_changes[1:]:
        # 当前水平等于近期变化权重0.20加旧水平权重0.80。
        smoothed_change = SES_ALPHA * change + (Decimal("1") - SES_ALPHA) * smoothed_change
    # 用最终平滑日变化逐日累加构造未来路径。
    ses_path = tuple(cutoff_balance + smoothed_change * Decimal(horizon) for horizon in range(1, FORECAST_PATH_BUSINESS_DAYS + 1))
    # 计算60日内发生非零净流量事件的比例。
    event_probability = Decimal(sum(change != 0 for change in daily_changes)) / Decimal(HISTORY_BUSINESS_DAYS)
    # 计算已发生非零净流量的带符号中位数。
    median_nonzero_change = signed_nonzero_median(daily_changes)
    # 计算每个未来工作日的期望变化。
    expected_daily_change = event_probability * median_nonzero_change
    # 用期望每日变化线性累加构造间歇性事件金额路径。
    intermittent_path = tuple(cutoff_balance + expected_daily_change * Decimal(horizon) for horizon in range(1, FORECAST_PATH_BUSINESS_DAYS + 1))
    # 返回五种方法的完整路径，键顺序保持A0预注册顺序。
    return {
        "last_balance_path": last_balance_path,
        "repeat_last_5bd_changes": tuple(repeated_path),
        "historical_horizon_net_flow_mean": tuple(historical_mean_path),
        "exponential_smoothed_daily_change": ses_path,
        "intermittent_event_amount": intermittent_path,
    }


# 产生一条逐样本、逐折、逐方法、逐未来日的基线预测明细；它只用于开发池校准和验证。
def prediction_row(target: TargetPath, fold_role: FoldRole, method: str, forecast_step: int, predicted_balance: Decimal) -> dict[str, str]:
    # 取得当前未来日的实际余额答案；forecast_step从1开始所以数组索引减1。
    actual_balance = target.target_balances_cny[forecast_step - 1]
    # 计算余额绝对误差。
    absolute_error = abs(predicted_balance - actual_balance)
    # 计算账户归一化绝对误差。
    normalized_error = absolute_error / target.balance_scale_cny
    # 返回不含任何模型输入字段的开发池评价明细。
    return {
        "fold_id": fold_role.fold_id,
        "fold_role": fold_role.fold_role,
        "baseline_method": method,
        "sample_key": target.sample_key,
        "enterprise_id": target.enterprise_id,
        "forecast_step": str(forecast_step),
        "actual_target_balance_cny": money(actual_balance),
        "predicted_target_balance_cny": money(predicted_balance),
        "absolute_error_target_balance_cny": money(absolute_error),
        "account_normalized_absolute_error": ratio(normalized_error),
        "baseline_version": A2_VERSION,
    }


# 用嵌套字典累计验证指标；每家企业先独立平均，最终才对企业等权平均。
def update_validation_aggregates(aggregates: dict[str, dict[str, dict[str, dict[str, list[Decimal]]]]], fold_role: FoldRole, method: str, target: TargetPath, predicted_path: tuple[Decimal, ...]) -> None:
    # 校准行只保留预测明细供将来区间校准，不参与“谁是最强基线”的锁定。
    if fold_role.fold_role != "validation":
        # 直接返回，不更新验证选择指标。
        return
    # 逐个预登记指标项目更新本折、该方法、该企业的误差和计数。
    for measure_name, horizon in REGISTERED_MEASURES:
        # T+5路径需要累计第1至5日误差；其他期限只评价各自期限末日。
        steps = range(1, horizon + 1) if measure_name == "t_plus_5_path" else (horizon,)
        # 逐个需要评价的未来日累加误差。
        for step in steps:
            # 计算当前日余额绝对误差。
            error = abs(predicted_path[step - 1] - target.target_balances_cny[step - 1])
            # 使用截止日账户尺度归一化。
            normalized_error = error / target.balance_scale_cny
            # 取得当前企业的可变累计桶；第0项是误差和、第1项是观察数。
            bucket = aggregates[fold_role.fold_id][method][measure_name][target.enterprise_id]
            # 累加归一化误差。
            bucket[0] += normalized_error
            # 累加当前企业用于平均的预测日数。
            bucket[1] += Decimal("1")


# 将验证累计桶转换为每折每方法每指标的企业等权指标行。
def fold_metric_rows(aggregates: dict[str, dict[str, dict[str, dict[str, list[Decimal]]]]]) -> list[dict[str, str]]:
    # 初始化可写入CSV的指标行列表。
    rows: list[dict[str, str]] = []
    # 按折编号字典排序以稳定输出A、B、C、D。
    for fold_id in sorted(aggregates):
        # 按预注册方法顺序输出，避免运行字典顺序改变平局规则。
        for method in BASELINE_ORDER:
            # 按预登记评价项目顺序输出。
            for measure_name, _ in REGISTERED_MEASURES:
                # 读取当前折、方法和指标按企业累计的桶。
                enterprise_buckets = aggregates[fold_id][method][measure_name]
                # 要求至少有一家验证企业。
                if not enterprise_buckets:
                    # 阻断空验证指标。
                    raise A2ContractError(f"折{fold_id}方法{method}指标{measure_name}没有验证企业")
                # 逐企业先求自己的平均归一化误差。
                enterprise_means = [values[0] / values[1] for values in enterprise_buckets.values()]
                # 再对企业等权平均，防止记录多的企业支配结果。
                macro_mae = sum(enterprise_means, Decimal("0")) / Decimal(len(enterprise_means))
                # 计算该折该方法该指标实际评价的预测日总数。
                prediction_count = sum(int(values[1]) for values in enterprise_buckets.values())
                # 写入指标行。
                rows.append({
                    "fold_id": fold_id,
                    "baseline_method": method,
                    "registered_measure": measure_name,
                    "enterprise_count": str(len(enterprise_buckets)),
                    "prediction_count": str(prediction_count),
                    "enterprise_macro_account_normalized_mae": ratio(macro_mae),
                    "baseline_version": A2_VERSION,
                })
    # 返回四折乘五方法乘四指标的完整指标行。
    return rows


# 计算Decimal列表中位数；四折选择采用中位数，避免某一个季度样本量较大而支配锁定。
def decimal_median(values: list[Decimal]) -> Decimal:
    # 空列表没有中位数定义。
    if not values:
        # 阻断没有四折指标的选择。
        raise A2ContractError("空列表不能计算基线选择中位数")
    # 从小到大排序。
    ordered = sorted(values)
    # 找到中间位置。
    middle = len(ordered) // 2
    # 奇数个值直接返回中间项。
    if len(ordered) % 2 == 1:
        # 返回中间值。
        return ordered[middle]
    # 偶数四折取中间两折的平均值。
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")


# 汇总四折指标并按A0固定顺序锁定每个期限的一种最强基线。
def summarize_and_lock(fold_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    # 初始化指标到方法到四折行的嵌套映射。
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    # 逐行按指标和方法聚集四折结果。
    for row in fold_rows:
        # 追加当前折指标行。
        grouped[row["registered_measure"]][row["baseline_method"]].append(row)
    # 初始化汇总指标行列表。
    summary_rows: list[dict[str, str]] = []
    # 初始化锁定结果字典。
    locks: dict[str, object] = {}
    # 按预登记指标顺序处理。
    for measure_name, horizon in REGISTERED_MEASURES:
        # 初始化当前指标的方法到中位数映射。
        method_scores: dict[str, Decimal] = {}
        # 按预注册顺序汇总五种方法。
        for method in BASELINE_ORDER:
            # 读取该方法四折指标行。
            rows = grouped[measure_name][method]
            # 要求恰好四折，缺折不能少算困难季度。
            if len(rows) != 4:
                # 阻断不完整四折比较。
                raise A2ContractError(f"{measure_name}的{method}不是四折完整指标")
            # 取得四折企业等权误差。
            scores = [Decimal(row["enterprise_macro_account_normalized_mae"]) for row in rows]
            # 计算四折中位数作为冻结选择分数。
            median_score = decimal_median(scores)
            # 保存该方法中位数供锁定比较。
            method_scores[method] = median_score
            # 写入汇总指标行，便于人工查看每折波动范围。
            summary_rows.append({
                "registered_measure": measure_name,
                "horizon_business_days": str(horizon),
                "baseline_method": method,
                "fold_count": "4",
                "median_enterprise_macro_account_normalized_mae": ratio(median_score),
                "minimum_fold_macro_mae": ratio(min(scores)),
                "maximum_fold_macro_mae": ratio(max(scores)),
                "selection_metric": "four_fold_median_enterprise_macro_account_normalized_mae",
                "baseline_version": A2_VERSION,
            })
        # 按中位误差升序和A0候选顺序解决精确同分。
        winner = min(BASELINE_ORDER, key=lambda method: (method_scores[method], BASELINE_ORDER.index(method)))
        # 保存当前期限的唯一锁定记录。
        locks[measure_name] = {
            "horizon_business_days": horizon,
            "locked_baseline_method": winner,
            "selection_metric": "four_fold_median_enterprise_macro_account_normalized_mae",
            "selection_metric_value": ratio(method_scores[winner]),
            "tie_breaker_cn": "中位误差完全相同则按A0预注册五种基线顺序锁定；未读取复杂模型或最终测试。",
        }
    # 返回完整汇总指标和四项锁定结果。
    return summary_rows, locks


# 写入CSV表；输出目录由调用方确保全新，因此不会覆盖历史实验记录。
def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, str]]) -> None:
    # 以UTF-8创建新CSV。
    with path.open("w", encoding="utf-8", newline="") as handle:
        # 按固定字段顺序建立写入器。
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        # 写入表头。
        writer.writeheader()
        # 流式写入所有记录。
        writer.writerows(rows)
