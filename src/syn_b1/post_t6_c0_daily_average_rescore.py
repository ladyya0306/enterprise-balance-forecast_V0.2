"""POST-T6-C0：只读复算既有LSTM预测的10日与30日日均余额误差。"""

# 导入CSV标准库；它流式读取约3GB的既有逐日预测，避免一次装入内存。
import csv
# 从collections导入defaultdict；它按种子、折、期限和企业累计误差。
from collections import defaultdict
# 从decimal导入Decimal；它让人民币平均值和比例计算不受二进制浮点误差影响。
from decimal import Decimal
# 从pathlib导入Path；它用跨平台对象表示项目文件路径。
from pathlib import Path

# 复用既有SHA256函数；它核对B2预测明细没有被事后改写。
from syn_b1.post_t6_a3_xgboost_windows import ratio, sha256_file

# 固定本阶段实现版本；以后改变复算公式必须另建版本，不能覆盖本结论。
C0_VERSION = "syn_b1_post_t6_c0_daily_average_rescore_1_0"
# 固定只复算B2已经锁定的LSTM内部配置，禁止看完新指标后改选其它配置。
SELECTED_CONFIGURATION = "lstm_direct_selected_window_h16_scaled_l1"
# 固定三个原始训练种子；复算不得删去表现不好的种子。
TRAINING_SEEDS = (2026083102, 2026083103, 2026083104)
# 固定四个滚动验证折；每折代表不同企业组和不同未来季度。
FOLDS = ("A", "B", "C", "D")
# 固定用户关心的两个日均余额期限。
AVERAGE_HORIZONS = (10, 30)


# 声明本阶段专用异常；它把合同或文件不完整与模型效果不好区分开。
class C0RescoreError(ValueError):
    """当既有预测、样本路径或复算分组不满足冻结条件时抛出。"""


# 读取A1标签文件中的账户尺度；该尺度只用于评价，不会进入模型。
def read_balance_scales(label_path: Path) -> dict[str, Decimal]:
    # 初始化样本键到账户尺度的映射。
    scales: dict[str, Decimal] = {}
    # 以UTF-8和CSV换行规则打开既有开发池标签。
    with label_path.open("r", encoding="utf-8", newline="") as handle:
        # 创建按表头取值的流式读取器。
        reader = csv.DictReader(handle)
        # 逐个预测起点读取样本键和截止日可见尺度。
        for row in reader:
            # 取得不会进入模型的连接键。
            sample_key = row["sample_key"]
            # 拒绝重复样本，避免同一预测起点被重复计权。
            if sample_key in scales:
                # 抛出清晰异常。
                raise C0RescoreError(f"A1标签存在重复样本键：{sample_key}")
            # 保存正数账户尺度；A1已把极小余额提升到冻结下限。
            scales[sample_key] = Decimal(row["balance_scale_cny"])
    # 拒绝空标签文件。
    if not scales:
        # 抛出清晰异常。
        raise C0RescoreError("A1标签文件为空")
    # 返回全部开发池样本的只读评价尺度。
    return scales


# 对单个预测起点的完整30日逐日记录计算10日和30日日均余额误差。
def calculate_sample_average_errors(rows: list[dict[str, str]], balance_scale: Decimal) -> dict[int, tuple[Decimal, Decimal, Decimal, Decimal]]:
    # 要求每个预测起点恰有30个未来银行工作日。
    if len(rows) != 30:
        # 阻断缺日或多日路径。
        raise C0RescoreError("单个预测起点不是完整30日路径")
    # 读取并核验未来步号必须连续为1至30。
    steps = tuple(int(row["forecast_step"]) for row in rows)
    # 与冻结的完整步号比较。
    if steps != tuple(range(1, 31)):
        # 阻断乱序、重复或缺失未来日。
        raise C0RescoreError("单个预测起点的未来步号不是连续1至30")
    # 要求账户尺度严格为正，避免无定义归一化误差。
    if balance_scale <= 0:
        # 阻断错误尺度。
        raise C0RescoreError("账户评价尺度必须大于0")
    # 初始化两个期限的结果字典。
    results: dict[int, tuple[Decimal, Decimal, Decimal, Decimal]] = {}
    # 依次计算10日和30日日均余额。
    for horizon in AVERAGE_HORIZONS:
        # 只截取当前期限覆盖的逐日路径。
        horizon_rows = rows[:horizon]
        # 将期限天数转换为精确十进制分母。
        denominator = Decimal(horizon)
        # 计算真实的未来日均余额。
        actual_average = sum((Decimal(row["actual_target_balance_cny"]) for row in horizon_rows), Decimal("0")) / denominator
        # 计算模型预测的未来日均余额。
        model_average = sum((Decimal(row["predicted_target_balance_cny"]) for row in horizon_rows), Decimal("0")) / denominator
        # 计算余额保持基线的未来日均余额；逐日值虽相同，仍按同一公式核对。
        baseline_average = sum((Decimal(row["locked_baseline_predicted_balance_cny"]) for row in horizon_rows), Decimal("0")) / denominator
        # 计算模型日均余额的人民币绝对误差。
        model_raw_error = abs(model_average - actual_average)
        # 计算基线日均余额的人民币绝对误差。
        baseline_raw_error = abs(baseline_average - actual_average)
        # 保存人民币误差和按账户尺度归一化的误差。
        results[horizon] = (model_raw_error, baseline_raw_error, model_raw_error / balance_scale, baseline_raw_error / balance_scale)
    # 返回两个期限的样本级误差。
    return results


# 读取既有预测并按企业等权口径形成种子×折指标。
def rescore_prediction_file(prediction_path: Path, balance_scales: dict[str, Decimal]) -> list[dict[str, str]]:
    # 建立种子、折、期限、企业到误差累计值的四层分组。
    aggregates: dict[tuple[int, str, int, str], list[Decimal]] = defaultdict(lambda: [Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0")])
    # 记录已经完成的种子、折、样本组合，防止同一预测起点重复出现。
    completed_sample_groups: set[tuple[int, str, str]] = set()
    # 初始化当前连续样本分组键。
    current_key: tuple[int, str, str] | None = None
    # 初始化当前样本的30行缓冲。
    current_rows: list[dict[str, str]] = []
    # 定义关闭当前样本并累计误差的内层函数。
    def finalize_current_sample() -> None:
        # 声明要读取外层当前键和缓冲。
        nonlocal current_key, current_rows
        # 没有当前样本时无需处理。
        if current_key is None:
            # 直接返回。
            return
        # 拒绝同一种子同一折下重复出现同一样本。
        if current_key in completed_sample_groups:
            # 抛出重复异常。
            raise C0RescoreError(f"既有预测存在重复样本组：{current_key}")
        # 解包种子、折和样本键。
        seed, fold_id, sample_key = current_key
        # 要求样本尺度存在于A1开发池标签。
        if sample_key not in balance_scales:
            # 阻断未知或越界样本。
            raise C0RescoreError(f"预测样本不在A1开发池标签中：{sample_key}")
        # 计算该预测起点的两种日均余额误差。
        errors = calculate_sample_average_errors(current_rows, balance_scales[sample_key])
        # 从逐日记录读取企业编号。
        enterprise_id = current_rows[0]["enterprise_id"]
        # 逐期限累计到企业桶。
        for horizon, values in errors.items():
            # 取得当前企业累计桶。
            bucket = aggregates[(seed, fold_id, horizon, enterprise_id)]
            # 累加模型人民币绝对误差。
            bucket[0] += values[0]
            # 累加基线人民币绝对误差。
            bucket[1] += values[1]
            # 累加模型账户归一化误差。
            bucket[2] += values[2]
            # 累加基线账户归一化误差。
            bucket[3] += values[3]
            # 累加该企业预测起点数。
            bucket[4] += Decimal("1")
        # 标记该样本组已经完成。
        completed_sample_groups.add(current_key)
        # 清空缓冲，为下一组做准备。
        current_rows = []
    # 以UTF-8和CSV换行规则流式打开B2预测明细。
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        # 创建按表头读取的CSV迭代器。
        reader = csv.DictReader(handle)
        # 逐行扫描全部既有配置，但只解析锁定配置的验证行金额。
        for row in reader:
            # 跳过未被B2锁定的其它内部配置。
            if row["configuration_id"] != SELECTED_CONFIGURATION:
                # 继续读取下一行。
                continue
            # 跳过训练企业的校准行；它们不能作为效果评价。
            if row["fold_role"] != "validation":
                # 继续读取下一行。
                continue
            # 将当前训练种子转为整数。
            seed = int(row["training_seed"])
            # 读取当前滚动折号。
            fold_id = row["fold_id"]
            # 拒绝未登记种子或折号。
            if seed not in TRAINING_SEEDS or fold_id not in FOLDS:
                # 阻断实验矩阵漂移。
                raise C0RescoreError("预测明细包含未登记种子或滚动折")
            # 组成连续样本分组键。
            row_key = (seed, fold_id, row["sample_key"])
            # 分组切换时先结算上一完整样本。
            if current_key is not None and row_key != current_key:
                # 结算上一样本。
                finalize_current_sample()
            # 当前没有活动分组时登记新键。
            if current_key != row_key:
                # 保存新分组键。
                current_key = row_key
            # 保存当前未来日记录。
            current_rows.append(row)
    # 文件结束后结算最后一个样本。
    finalize_current_sample()
    # 初始化种子×折×期限指标行。
    metric_rows: list[dict[str, str]] = []
    # 逐种子、折和期限形成完整24行。
    for seed in TRAINING_SEEDS:
        # 遍历四个滚动折。
        for fold_id in FOLDS:
            # 遍历两个日均期限。
            for horizon in AVERAGE_HORIZONS:
                # 筛出当前组合下所有企业桶。
                members = {key[3]: values for key, values in aggregates.items() if key[:3] == (seed, fold_id, horizon)}
                # 验证折必须包含企业。
                if not members:
                    # 阻断缺失种子折。
                    raise C0RescoreError("日均余额复算缺少完整种子折")
                # 初始化企业等权人民币模型误差列表。
                model_raw_means: list[Decimal] = []
                # 初始化企业等权人民币基线误差列表。
                baseline_raw_means: list[Decimal] = []
                # 初始化企业等权归一化模型误差列表。
                model_normalized_means: list[Decimal] = []
                # 初始化企业等权归一化基线误差列表。
                baseline_normalized_means: list[Decimal] = []
                # 初始化模型优于基线的企业数。
                improved_enterprises = 0
                # 逐企业把样本累计值转换为企业内均值。
                for bucket in members.values():
                    # 读取该企业预测起点数。
                    sample_count = bucket[4]
                    # 计算企业人民币模型平均绝对误差。
                    model_raw = bucket[0] / sample_count
                    # 计算企业人民币基线平均绝对误差。
                    baseline_raw = bucket[1] / sample_count
                    # 计算企业账户归一化模型平均绝对误差。
                    model_normalized = bucket[2] / sample_count
                    # 计算企业账户归一化基线平均绝对误差。
                    baseline_normalized = bucket[3] / sample_count
                    # 保存四项企业级均值。
                    model_raw_means.append(model_raw)
                    # 保存人民币基线误差。
                    baseline_raw_means.append(baseline_raw)
                    # 保存归一化模型误差。
                    model_normalized_means.append(model_normalized)
                    # 保存归一化基线误差。
                    baseline_normalized_means.append(baseline_normalized)
                    # 企业级模型误差严格更小时记为改善。
                    if model_normalized < baseline_normalized:
                        # 增加改善企业数。
                        improved_enterprises += 1
                # 将企业数转换为十进制分母。
                enterprise_denominator = Decimal(len(members))
                # 计算企业等权归一化模型误差。
                model_macro = sum(model_normalized_means, Decimal("0")) / enterprise_denominator
                # 计算企业等权归一化基线误差。
                baseline_macro = sum(baseline_normalized_means, Decimal("0")) / enterprise_denominator
                # 拒绝零基线，避免技能分数无定义。
                if baseline_macro == 0:
                    # 阻断异常组合。
                    raise C0RescoreError("日均余额基线误差为0，无法计算Skill Score")
                # 写出当前种子折期限的完整指标。
                metric_rows.append({
                    # 保存折号。
                    "fold_id": fold_id,
                    # 保存训练种子。
                    "training_seed": str(seed),
                    # 保存日均期限。
                    "average_horizon_business_days": str(horizon),
                    # 保存锁定配置编号。
                    "configuration_id": SELECTED_CONFIGURATION,
                    # 保存验证企业数。
                    "enterprise_count": str(len(members)),
                    # 保存预测起点数。
                    "forecast_origin_count": str(sum(int(bucket[4]) for bucket in members.values())),
                    # 保存企业等权人民币模型误差。
                    "model_enterprise_macro_mae_cny": f"{sum(model_raw_means, Decimal('0')) / enterprise_denominator:.2f}",
                    # 保存企业等权人民币基线误差。
                    "baseline_enterprise_macro_mae_cny": f"{sum(baseline_raw_means, Decimal('0')) / enterprise_denominator:.2f}",
                    # 保存企业等权账户归一化模型误差。
                    "model_enterprise_macro_account_normalized_mae": ratio(model_macro),
                    # 保存企业等权账户归一化基线误差。
                    "baseline_enterprise_macro_account_normalized_mae": ratio(baseline_macro),
                    # 保存相对同期限余额保持基线的技能分数。
                    "skill_score_vs_balance_hold_baseline": ratio(Decimal("1") - model_macro / baseline_macro),
                    # 保存模型改善企业数。
                    "improved_enterprise_count": str(improved_enterprises),
                    # 保存模型改善企业比例。
                    "improved_enterprise_rate": ratio(Decimal(improved_enterprises) / enterprise_denominator),
                    # 保存复算版本。
                    "c0_version": C0_VERSION,
                })
    # 要求3种子×4折×2期限恰为24行。
    if len(metric_rows) != 24:
        # 阻断不完整结果。
        raise C0RescoreError("日均余额种子折指标不是完整24行")
    # 返回完整种子折指标。
    return metric_rows


# 计算奇数或偶数个十进制数的标准中位数。
def decimal_median(values: list[Decimal]) -> Decimal:
    # 拒绝空列表。
    if not values:
        # 抛出无定义中位数异常。
        raise C0RescoreError("不能计算空列表的中位数")
    # 从小到大排序副本，避免改写调用者数据。
    ordered = sorted(values)
    # 计算列表长度。
    size = len(ordered)
    # 奇数项直接返回中间值。
    if size % 2 == 1:
        # 返回唯一中间项。
        return ordered[size // 2]
    # 偶数项返回中间两项平均值。
    return (ordered[size // 2 - 1] + ordered[size // 2]) / Decimal("2")


# 把每折三个固定种子合成为种子中位折指标。
def build_fold_metrics(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 初始化折指标结果。
    fold_rows: list[dict[str, str]] = []
    # 遍历四个滚动折。
    for fold_id in FOLDS:
        # 遍历两个日均期限。
        for horizon in AVERAGE_HORIZONS:
            # 取得当前折期限的三个种子行。
            members = [row for row in seed_rows if row["fold_id"] == fold_id and int(row["average_horizon_business_days"]) == horizon]
            # 要求种子数量和编号完整。
            if len(members) != 3 or {int(row["training_seed"]) for row in members} != set(TRAINING_SEEDS):
                # 阻断缺种子或换种子。
                raise C0RescoreError("折指标缺少三个固定训练种子")
            # 计算三种子模型归一化误差中位数。
            model_error = decimal_median([Decimal(row["model_enterprise_macro_account_normalized_mae"]) for row in members])
            # 计算三种子基线归一化误差中位数。
            baseline_error = decimal_median([Decimal(row["baseline_enterprise_macro_account_normalized_mae"]) for row in members])
            # 写出折级种子中位结果。
            fold_rows.append({
                # 保存折号。
                "fold_id": fold_id,
                # 保存日均期限。
                "average_horizon_business_days": str(horizon),
                # 保存锁定配置。
                "configuration_id": SELECTED_CONFIGURATION,
                # 记录种子数。
                "seed_count": "3",
                # 保存人民币模型误差中位数。
                "median_model_enterprise_macro_mae_cny": f"{decimal_median([Decimal(row['model_enterprise_macro_mae_cny']) for row in members]):.2f}",
                # 保存人民币基线误差中位数。
                "median_baseline_enterprise_macro_mae_cny": f"{decimal_median([Decimal(row['baseline_enterprise_macro_mae_cny']) for row in members]):.2f}",
                # 保存归一化模型误差中位数。
                "median_model_enterprise_macro_account_normalized_mae": ratio(model_error),
                # 保存归一化基线误差中位数。
                "median_baseline_enterprise_macro_account_normalized_mae": ratio(baseline_error),
                # 保存折级技能分数。
                "skill_score_vs_balance_hold_baseline": ratio(Decimal("1") - model_error / baseline_error),
                # 保存改善企业比例的种子中位数。
                "median_improved_enterprise_rate": ratio(decimal_median([Decimal(row["improved_enterprise_rate"]) for row in members])),
                # 保存复算版本。
                "c0_version": C0_VERSION,
            })
    # 返回四折×两期限的8行。
    return fold_rows


# 把四折结果合成为最终两行日均余额诊断摘要。
def build_summary_metrics(seed_rows: list[dict[str, str]], fold_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # 初始化两个期限摘要。
    summary_rows: list[dict[str, str]] = []
    # 逐期限处理。
    for horizon in AVERAGE_HORIZONS:
        # 取得当前期限四个折行。
        horizon_folds = [row for row in fold_rows if int(row["average_horizon_business_days"]) == horizon]
        # 取得当前期限12个种子折行。
        horizon_seed_rows = [row for row in seed_rows if int(row["average_horizon_business_days"]) == horizon]
        # 要求四折和12种子折完整。
        if len(horizon_folds) != 4 or len(horizon_seed_rows) != 12:
            # 阻断缺失汇总来源。
            raise C0RescoreError("日均余额摘要缺少完整四折或12个种子折")
        # 计算四折模型归一化误差中位数。
        model_error = decimal_median([Decimal(row["median_model_enterprise_macro_account_normalized_mae"]) for row in horizon_folds])
        # 计算四折基线归一化误差中位数。
        baseline_error = decimal_median([Decimal(row["median_baseline_enterprise_macro_account_normalized_mae"]) for row in horizon_folds])
        # 写出当前期限摘要。
        summary_rows.append({
            # 保存日均期限。
            "average_horizon_business_days": str(horizon),
            # 保存锁定配置。
            "configuration_id": SELECTED_CONFIGURATION,
            # 保存折数。
            "fold_count": "4",
            # 保存每折种子数。
            "seed_count_per_fold": "3",
            # 保存四折人民币模型误差中位数。
            "median_model_enterprise_macro_mae_cny": f"{decimal_median([Decimal(row['median_model_enterprise_macro_mae_cny']) for row in horizon_folds]):.2f}",
            # 保存四折人民币基线误差中位数。
            "median_baseline_enterprise_macro_mae_cny": f"{decimal_median([Decimal(row['median_baseline_enterprise_macro_mae_cny']) for row in horizon_folds]):.2f}",
            # 保存四折归一化模型误差中位数。
            "median_model_enterprise_macro_account_normalized_mae": ratio(model_error),
            # 保存四折归一化基线误差中位数。
            "median_baseline_enterprise_macro_account_normalized_mae": ratio(baseline_error),
            # 保存最终日均余额技能分数。
            "skill_score_vs_balance_hold_baseline": ratio(Decimal("1") - model_error / baseline_error),
            # 保存正向折数。
            "positive_fold_count": str(sum(Decimal(row["skill_score_vs_balance_hold_baseline"]) > 0 for row in horizon_folds)),
            # 保存正向种子折数。
            "positive_seed_fold_count": str(sum(Decimal(row["skill_score_vs_balance_hold_baseline"]) > 0 for row in horizon_seed_rows)),
            # 保存最弱折技能分数，防止只看中位数掩盖失败季度。
            "minimum_fold_skill_score": ratio(min(Decimal(row["skill_score_vs_balance_hold_baseline"]) for row in horizon_folds)),
            # 保存最弱种子折技能分数。
            "minimum_seed_fold_skill_score": ratio(min(Decimal(row["skill_score_vs_balance_hold_baseline"]) for row in horizon_seed_rows)),
            # 保存四折改善企业比例中位数。
            "median_improved_enterprise_rate": ratio(decimal_median([Decimal(row["median_improved_enterprise_rate"]) for row in horizon_folds])),
            # 明确本阶段只是诊断，不锁模型。
            "candidate_selection_use": "diagnostic_only",
            # 保存复算版本。
            "c0_version": C0_VERSION,
        })
    # 返回10日和30日两行。
    return summary_rows
