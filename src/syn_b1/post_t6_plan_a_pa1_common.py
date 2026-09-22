# 模块说明：本文件建立Plan A共同事实表所需的只读连接和完整性检查。
"""Plan A PA1共同事实表的流式读取、日期补全与连续段规则。"""

# 导入CSV标准库；PA1用它流式读取大预测表并写出共同事实记录。
import csv
# 从collections导入defaultdict；PA1按预测起点和连续段组织少量索引信息。
from collections import defaultdict
# 从datetime导入date；PA1只按银行工作日序号连接日期。
from datetime import date
# 从decimal导入Decimal；余额必须精确比较到分。
from decimal import Decimal
# 从pathlib导入Path；PA1用它定位已获准的每日余额表。
from pathlib import Path

# 导入共同输入保护、方法编号和固定折种子集合。
from syn_b1.post_t6_plan_a_common_records import DEVELOPMENT_ENTERPRISE_COUNT, PlanAInputError, SELECTED_B2_CONFIGURATION, SELECTED_B2_FOLDS, SELECTED_B2_TRAINING_SEEDS, project_relative_path

# 固定每个已保存B2预测起点的未来银行工作日数量。
FORECAST_STEPS = tuple(range(1, 31))
# 固定每次训练在四个时间检查中的验证起点数量。
EXPECTED_ORIGIN_COUNTS = {"A": 2251, "B": 2188, "C": 2248, "D": 2413}
# 固定每个训练起始状态应保留的逐日预测行数。
EXPECTED_ROWS_PER_SEED = sum(EXPECTED_ORIGIN_COUNTS.values()) * len(FORECAST_STEPS)


# 把金额文字转换为精确到分的十进制数；非金额会以输入错误形式停止。
def money(value: str) -> Decimal:
    # 尝试将文字转换为十进制金额。
    try:
        # 量化到分，确保一分钱差异能被识别。
        return Decimal(value).quantize(Decimal("0.01"))
    # 金额格式错误不能被悄悄当作零。
    except Exception as error:
        # 抛出带原值的输入边界错误。
        raise PlanAInputError(f"余额金额不能按分读取：{value}") from error


# 流式读取四折索引，只保留validation起点并验证A1企业与日期连接一致。
def read_validation_index(index_path: Path, reader) -> dict[str, dict[str, str]]:
    # 通过输入守卫打开A1四折索引。
    reader.assert_allowed(index_path)
    # 初始化按样本编号保存的开发检查索引。
    validation: dict[str, dict[str, str]] = {}
    # 只在守卫通过后流式打开CSV。
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        # 逐行读取而不把完整A1索引装入内存。
        for row in csv.DictReader(handle):
            # 只保存允许用于Plan A的开发检查角色。
            if row["fold_role"] == "validation":
                # 同一样本不能在不同时间检查中重复成为validation。
                if row["sample_key"] in validation:
                    # 阻断无法唯一连接的样本。
                    raise PlanAInputError(f"A1开发检查样本重复：{row['sample_key']}")
                # 固定四个时间检查之外的记录不能进入共同表。
                if row["fold_id"] not in SELECTED_B2_FOLDS:
                    # 阻断未知时间检查。
                    raise PlanAInputError(f"A1开发检查时间编号异常：{row['fold_id']}")
                # 保存样本编号、企业和截止日等唯一连接信息。
                validation[row["sample_key"]] = row
    # 四折起点合计必须是固定9100个。
    if len(validation) != sum(EXPECTED_ORIGIN_COUNTS.values()):
        # 阻断开发检查样本数量漂移。
        raise PlanAInputError(f"A1开发检查起点数不是9100：实际为{len(validation)}")
    # 分别核对每个时间检查的固定数量。
    for fold_id, expected_count in EXPECTED_ORIGIN_COUNTS.items():
        # 统计当前时间检查样本数。
        actual_count = sum(1 for row in validation.values() if row["fold_id"] == fold_id)
        # 任何少行或多行都不能继续。
        if actual_count != expected_count:
            # 抛出可定位的数量错误。
            raise PlanAInputError(f"A1开发检查{fold_id}起点数不一致：{actual_count}")
    # 返回已验证的9100个开发检查起点。
    return validation


# 流式读取A1未来答案，只保存开发检查起点的30个目标余额和截止信息。
def read_validation_targets(target_path: Path, reader, validation_index: dict[str, dict[str, str]]) -> dict[str, dict[str, object]]:
    # 通过输入守卫打开A1未来答案表。
    reader.assert_allowed(target_path)
    # 初始化仅含9100个开发检查起点的答案映射。
    targets: dict[str, dict[str, object]] = {}
    # 只在守卫通过后流式打开CSV。
    with target_path.open("r", encoding="utf-8", newline="") as handle:
        # 逐行读取102,964个A1样本。
        for row in csv.DictReader(handle):
            # 不是开发检查起点的行不保留，节约内存且不进入Plan A评价。
            if row["sample_key"] not in validation_index:
                # 跳过fit和calibration对应起点。
                continue
            # 样本不能被保存两次。
            if row["sample_key"] in targets:
                # 阻断答案重复。
                raise PlanAInputError(f"A1未来答案样本重复：{row['sample_key']}")
            # A1答案企业和截止日必须同四折索引一致。
            index_row = validation_index[row["sample_key"]]
            # 任一连接字段不一致即无法使用。
            if row["enterprise_id"] != index_row["enterprise_id"] or row["cutoff_date"] != index_row["cutoff_date"]:
                # 阻断A1内部血缘不一致。
                raise PlanAInputError(f"A1答案与四折索引不一致：{row['sample_key']}")
            # 依固定1至30顺序读取所有未来余额答案。
            future_balances = tuple(money(row[f"target_balance_t_plus_{step:02d}_cny"]) for step in FORECAST_STEPS)
            # 保存后续连接所需的少量事实。
            targets[row["sample_key"]] = {"enterprise_id": row["enterprise_id"], "cutoff_date": row["cutoff_date"], "future_start_date": row["future_start_date"], "cutoff_balance_cny": money(row["cutoff_balance_cny"]), "balance_scale_cny": money(row["balance_scale_cny"]), "future_balances": future_balances}
    # 每个开发检查起点都必须有唯一答案。
    if set(targets) != set(validation_index):
        # 阻断缺失或多余A1未来答案。
        raise PlanAInputError("A1开发检查起点与未来答案集合不一致")
    # 返回已核对的9100个未来答案。
    return targets


# 读取一户每日余额表，构建自然日余额和银行工作日顺序两种只读索引。
def read_daily_balance_index(daily_path: Path, reader) -> tuple[dict[str, Decimal], dict[str, int], tuple[str, ...]]:
    # 通过输入守卫核对许可后才打开每日余额表。
    reader.assert_allowed(daily_path)
    # 初始化按自然日期查询的余额映射。
    balances: dict[str, Decimal] = {}
    # 初始化按银行工作日查询的序号映射。
    business_steps: dict[str, int] = {}
    # 初始化按银行工作日顺序排列的日期列表。
    business_dates: list[str] = []
    # 只在守卫通过后以能识别文件开头UTF-8标记的方式打开每天余额表。
    with daily_path.open("r", encoding="utf-8-sig", newline="") as handle:
        # 逐行读取日度表，绝不读取同目录逐笔流水。
        for row in csv.DictReader(handle):
            # 每日余额表必须含有既有日度合同的关键字段，不能按猜测字段名兼容。
            if not {"calendar_date", "ending_balance_cny", "is_bank_workday", "business_step"}.issubset(row):
                # 抛出具体路径和可见字段，供人工判断资料合同是否已变化。
                raise PlanAInputError(f"每日余额表缺少固定字段：{daily_path}；实际字段为{sorted(row)}")
            # 日期不能重复，否则预测日期无法唯一连接。
            if row["calendar_date"] in balances:
                # 阻断重复自然日。
                raise PlanAInputError(f"每日余额表日期重复：{daily_path}")
            # 保存此日结束余额。
            balances[row["calendar_date"]] = money(row["ending_balance_cny"])
            # 只有明确银行工作日才进入连续序号索引。
            if row["is_bank_workday"] == "True":
                # 读取并保存已验收的银行工作日序号。
                business_steps[row["calendar_date"]] = int(row["business_step"])
                # 按文件原始日期顺序追加工作日日期。
                business_dates.append(row["calendar_date"])
    # 银行工作日日期必须严格递增，防止用自然日错误拼接。
    if tuple(business_dates) != tuple(sorted(business_dates)):
        # 阻断乱序每日余额表。
        raise PlanAInputError(f"每日余额表银行工作日日期未递增：{daily_path}")
    # 相邻工作日序号必须精确加一，保证节假日不被看作中断。
    for previous, current in zip(business_dates, business_dates[1:]):
        # 读取相邻日期的已验收序号。
        previous_step = business_steps[previous]
        # 读取后一天的已验收序号。
        current_step = business_steps[current]
        # 序号不连续说明日历资料不完整。
        if current_step != previous_step + 1:
            # 阻断缺失银行工作日。
            raise PlanAInputError(f"每日余额表银行工作日序号不连续：{daily_path}")
    # 返回两种索引和稳定日期顺序。
    return balances, business_steps, tuple(business_dates)


# 根据A1来源清单逐户核对指纹并读取208户每日余额索引。
def read_all_daily_indices(manifest_path: Path, reader, project_root: Path) -> dict[str, dict[str, object]]:
    # 通过输入守卫读取A1来源清单。
    manifest_rows = reader.read_csv_rows(manifest_path)
    # 清单必须恰为开发池208户。
    if len(manifest_rows) != DEVELOPMENT_ENTERPRISE_COUNT:
        # 阻断非208户来源清单。
        raise PlanAInputError("A1来源清单不是208户")
    # 初始化按企业编号保存的日期索引。
    indices: dict[str, dict[str, object]] = {}
    # 逐户核对登记指纹并打开唯一允许的每日余额表。
    for row in manifest_rows:
        # 取出企业编号。
        enterprise_id = row["enterprise_id"]
        # 企业编号不得重复。
        if enterprise_id in indices:
            # 阻断重复企业。
            raise PlanAInputError(f"A1来源清单企业重复：{enterprise_id}")
        # 定位登记运行目录内的每日余额表。
        daily_path = project_root / row["run_directory"] / "account_daily_total.csv"
        # 动态加入这份已由A1清单逐户指明的唯一每日余额表权限。
        reader.allowed_relative_paths = frozenset(set(reader.allowed_relative_paths) | {project_relative_path(project_root, daily_path)})
        # 计算在许可后读取的文件指纹。
        actual_hash = reader.hash_file(daily_path)
        # 指纹必须与A1清单逐字一致。
        if actual_hash != row["account_daily_total_sha256"]:
            # 阻断来源每日余额表变动。
            raise PlanAInputError(f"PA1每日余额表指纹不一致：{enterprise_id}")
        # 读取此企业的自然日和银行工作日索引。
        balances, business_steps, business_dates = read_daily_balance_index(daily_path, reader)
        # 保存仅含日期和余额的公开事实索引。
        indices[enterprise_id] = {"balances": balances, "business_steps": business_steps, "business_dates": business_dates}
    # 返回完整208户每日余额索引。
    return indices


# 根据固定四折、三种子和企业银行工作日序号为每个起点建立连续段编号。
def build_segments(validation_index: dict[str, dict[str, str]], daily_indices: dict[str, dict[str, object]]) -> tuple[dict[tuple[str, int, str], str], list[dict[str, object]]]:
    # 初始化每个预测起点对应连续段编号的映射。
    origin_segments: dict[tuple[str, int, str], str] = {}
    # 初始化连续段摘要行。
    segment_rows: list[dict[str, object]] = []
    # 对每个时间检查和训练起始状态独立建立段，禁止跨折或跨种子连接。
    for fold_id in SELECTED_B2_FOLDS:
        # 取得当前时间检查全部企业编号并稳定排序。
        enterprises = sorted({row["enterprise_id"] for row in validation_index.values() if row["fold_id"] == fold_id})
        # 每个种子都使用相同起点日期但保留独立段身份。
        for training_seed in SELECTED_B2_TRAINING_SEEDS:
            # 分企业建立连续段，绝不跨企业连接。
            for enterprise_id in enterprises:
                # 取出当前企业当前时间检查的全部样本并按截止日排序。
                samples = sorted((row for row in validation_index.values() if row["fold_id"] == fold_id and row["enterprise_id"] == enterprise_id), key=lambda row: row["cutoff_date"])
                # 取得每日余额表中的银行工作日序号映射。
                business_steps = daily_indices[enterprise_id]["business_steps"]
                # 初始化当前段样本列表。
                current_segment: list[dict[str, str]] = []
                # 初始化当前段编号计数。
                segment_number = 0
                # 逐日期判断相邻预测起点是否刚好相邻银行工作日。
                for sample in samples:
                    # 当前段为空时直接开始新段。
                    if not current_segment:
                        # 追加第一条样本。
                        current_segment.append(sample)
                        # 继续读取下一样本。
                        continue
                    # 取出上一样本截止日。
                    previous = current_segment[-1]
                    # 只有银行工作日序号恰加一才允许连接。
                    if business_steps[sample["cutoff_date"]] == business_steps[previous["cutoff_date"]] + 1:
                        # 追加到同一连续段。
                        current_segment.append(sample)
                    # 缺少应有银行工作日时结束旧段并开始新段。
                    else:
                        # 段编号从一开始递增。
                        segment_number += 1
                        # 生成跨运行唯一且可读的连续段编号。
                        segment_id = f"{enterprise_id}__{fold_id}__seed_{training_seed}__segment_{segment_number:04d}"
                        # 为旧段内所有样本写入该段编号。
                        for member in current_segment:
                            # 记录样本、种子和折到段的唯一连接。
                            origin_segments[(member["sample_key"], training_seed, fold_id)] = segment_id
                        # 保存旧段摘要。
                        segment_rows.append({"continuous_segment_id": segment_id, "enterprise_id": enterprise_id, "fold_id": fold_id, "training_seed": training_seed, "origin_count": len(current_segment), "start_cutoff_date": current_segment[0]["cutoff_date"], "end_cutoff_date": current_segment[-1]["cutoff_date"], "connection_rule_cn": "相邻预测起点在每日余额表中的银行工作日序号恰好相差1"})
                        # 用当前样本开始新段。
                        current_segment = [sample]
                # 每个企业至少应有一段开发检查起点。
                if current_segment:
                    # 处理循环结束后尚未写出的最后一段。
                    segment_number += 1
                    # 生成最后一段编号。
                    segment_id = f"{enterprise_id}__{fold_id}__seed_{training_seed}__segment_{segment_number:04d}"
                    # 为最后一段全部样本写段编号。
                    for member in current_segment:
                        # 保存该样本的段连接。
                        origin_segments[(member["sample_key"], training_seed, fold_id)] = segment_id
                    # 保存最后一段摘要。
                    segment_rows.append({"continuous_segment_id": segment_id, "enterprise_id": enterprise_id, "fold_id": fold_id, "training_seed": training_seed, "origin_count": len(current_segment), "start_cutoff_date": current_segment[0]["cutoff_date"], "end_cutoff_date": current_segment[-1]["cutoff_date"], "connection_rule_cn": "相邻预测起点在每日余额表中的银行工作日序号恰好相差1"})
    # 每个9100起点在三种子下都必须得到唯一段编号。
    if len(origin_segments) != len(validation_index) * len(SELECTED_B2_TRAINING_SEEDS):
        # 阻断连续段映射遗漏。
        raise PlanAInputError("连续预测起点分段数量不完整")
    # 返回逐起点段映射与段摘要。
    return origin_segments, segment_rows
