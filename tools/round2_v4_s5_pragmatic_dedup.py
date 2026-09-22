# 中文教学注释：模块入口负责本轮S5机械任务，保持业务语义不变。
"""S5 大考前的只读 Q4 双相关去重筛查；本工具不生成或修改任何流水。"""

# 中文逐行注释：导入命令行参数模块，仅允许显式指定只读清单和新诊断目录。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
import argparse
# 中文逐行注释：导入 CSV 模块以读取已经生成的日余额与现金流复核表。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
import csv
# 中文逐行注释：导入 JSON 模块以读取清单并写出新的可审计证据。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
import json
# 中文逐行注释：导入路径工具以避免手工拼接 Windows 路径。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
from pathlib import Path
# 中文逐行注释：导入类型标注以说明每项证据的结构。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
from typing import Any

# 中文逐行注释：导入数组库供成熟 compare 公式使用的数组容器。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
import numpy as np

# 中文逐行注释：从既有 2026 签名工具复用全年经营相似硬门的签名和判定。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
from round2_v4_s5_orchestrator import obvious_duplicate_2026, signature_2026
# 中文逐行注释：只导入成熟 compare 的相关公式，不调用其旧日期索引或历史入口。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
from shape_contract_search import compare

# 中文逐行注释：固定本轮批准的 Q4 闭区间，禁止由输入数据自行推断季度。
Q4_START = "2026-10-01"
# 中文逐行注释：固定本轮批准的 Q4 结束日期，确保恰好是 92 个自然日。
Q4_END = "2026-12-31"
# 中文逐行注释：固定本轮复用的余额相关处置线，含等号。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
BALANCE_LINE = 0.90
# 中文逐行注释：固定本轮复用的经营收支相关处置线，含等号。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
OPERATING_LINE = 0.65
# 中文逐行注释：明确排除融资项目，保留工资、采购、税费和设备等真实经营项目。
FINANCING_WORDS = ("股东", "银行借款", "贷款", "借款本金", "利息")


# 中文逐行注释：读取 JSON 文件并返回对象，任何格式错误都让调用者停止而非猜测。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
def read_json(path: Path) -> Any:
    # 中文逐行注释：以 UTF-8 解码冻结清单或已有回执。
    return json.loads(path.read_text(encoding="utf-8"))


# 中文逐行注释：建立 Q4 签名，数组字段与成熟 compare 保持一致但不借用其 365 日布尔结果。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
def q4_signature(directory: Path) -> dict[str, np.ndarray]:
    # 中文逐行注释：定位中央生成器实际写出的日表。
    daily_path = directory / "account_daily_total.csv"
    # 中文逐行注释：定位中央生成器实际写出的融资分类复核表。
    notes_path = directory / "cash_flow_review_notes.csv"
    # 中文逐行注释：缺任一原件即列为异常，绝不以零数组替代。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    if not daily_path.exists() or not notes_path.exists():
        # 中文逐行注释：错误信息包含缺失路径，便于 Luna 仅核原件而不生成。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        raise FileNotFoundError(f"Q4签名缺少原件：daily={daily_path.exists()} notes={notes_path.exists()}")
    # 中文逐行注释：读取并仅保留批准日期范围内的日表行。
    with daily_path.open(encoding="utf-8-sig", newline="") as handle:
        q4_rows = [row for row in csv.DictReader(handle) if Q4_START <= row["calendar_date"] <= Q4_END] # 中文教学注释：本行属于S5机械流程，保持原逻辑。
    # 中文逐行注释：按日历排序，禁止依赖 CSV 输出的偶然行序。
    q4_rows.sort(key=lambda row: row["calendar_date"])
    # 中文逐行注释：完整 Q4 是新处置线的前提，缺日或重复日均进入异常复核。
    if len(q4_rows) != 92 or len({row["calendar_date"] for row in q4_rows}) != 92:
        # 中文逐行注释：直接失败，避免不完整数据被当作低相关。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        raise ValueError(f"Q4必须有92个唯一自然日，实际为{len(q4_rows)}")
    # 中文逐行注释：建立日期下标，使现金流复核表可准确累加到对应自然日。
    index_by_date = {row["calendar_date"]: index for index, row in enumerate(q4_rows)}
    # 中文逐行注释：余额来自日表期末余额，不由流入流出反推。
    balance = np.asarray([float(row["ending_balance_cny"]) for row in q4_rows], dtype=float)
    # 中文逐行注释：初始化经营收支为零，随后逐笔累加非融资复核项。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    operating = np.zeros(92, dtype=float)
    # 中文逐行注释：读取已有现金流复核明细，保持既有经营口径。
    with notes_path.open(encoding="utf-8-sig", newline="") as handle:
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        for note in csv.DictReader(handle):
            # 中文逐行注释：截取记账日期以和日表自然日精确对应。
            booking_date = note["booking_datetime"][:10]
            # 中文逐行注释：不在 Q4 的记录不参与本次季度诊断。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
            index = index_by_date.get(booking_date)
            # 中文逐行注释：沿用项目的明确融资词排除规则。
            financing = any(word in note["cash_flow_type_cn"] for word in FINANCING_WORDS)
            # 中文逐行注释：工资、采购、税费、设备等非融资流仍按流入流出方向保留。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
            if index is not None and not financing:
                # 中文逐行注释：把原件金额累加为日经营净收支。
                operating[index] += float(note["amount_cny"]) * (1.0 if note["direction_cn"] == "流入" else -1.0)
    # 中文逐行注释：零方差相关性没有意义，必须进入异常复核而不是判作低相似。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    if float(np.std(balance)) == 0.0 or float(np.std(operating)) == 0.0:
        # 中文逐行注释：明确指出是哪个数组没有方差。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        raise ValueError(f"Q4相关性不可计算：balance_std={np.std(balance)} operating_std={np.std(operating)}")
    # 中文逐行注释：compare 需要三类数组；现金字段不用于本轮 Q4 处置但保持结构完整。
    return {"balance": balance, "cash": operating.copy(), "operating": operating}


# 中文逐行注释：判定源目录是否已经使用过 r1，以执行细则禁止本轮复活的约束。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
def old_r1_used(directory: Path) -> bool:
    # 中文逐行注释：目录名是现有原件的版本证据，不试图从身份名称猜测。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    name = directory.name.lower()
    # 中文逐行注释：同时识别首六 T05 的 revision01 和批量目录的 r01 标识。
    return "revision01" in name or name.endswith("_r01")


# 中文逐行注释：构造一对 Q4 证据，并明确手工应用 92 日双相关线而非 compare 的365日布尔值。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
def q4_pair_evidence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    # 中文逐行注释：复用成熟公式计算余额、现金和经营收支相关性。
    result = compare(left["q4_signature"], right["q4_signature"])
    # 中文逐行注释：提取可 JSON 序列化的余额相关值。
    balance_corr = result["balance_correlation"]
    # 中文逐行注释：提取可 JSON 序列化的排融资经营收支相关值。
    operating_corr = result["operating_cash_correlation"]
    # 中文逐行注释：季度缺日或空相关不得静默作为低相似。
    invalid = result["common_days"] != 92 or balance_corr is None or operating_corr is None
    # 中文逐行注释：本轮新线必须两项同时达到，余额单高只属于共同趋势提示。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    trigger = bool(not invalid and balance_corr >= BALANCE_LINE and operating_corr >= OPERATING_LINE)
    # 中文逐行注释：给余额单高对单独标识，避免把共同方向误写成模板重复。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    common_trend = bool(not invalid and balance_corr >= BALANCE_LINE and operating_corr < OPERATING_LINE)
    # 中文逐行注释：返回每对的原始相关值与明确处置类别。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    return {
        "left": left["sample_id"], # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "right": right["sample_id"], # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "common_days": result["common_days"], # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "q4_balance_correlation": balance_corr, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "q4_operating_cash_correlation": operating_corr, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "q4_double_high_trigger": trigger, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "q4_common_trend_only": common_trend, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "invalid_for_review": invalid, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    }


# 中文逐行注释：按稀少类别优先、同类固定企业 ID 顺序生成不可随成绩变化的贪心顺序。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
def greedy_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 中文逐行注释：先统计源清单中的类别数量，数量少的类别拥有更高保留优先级。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    counts: dict[str, int] = {}
    # 中文逐行注释：逐行累计类别数，不读模型或考试结果。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    for row in rows:
        # 中文逐行注释：使用冻结清单的走势类别字段。
        counts[row["planned_shape"]] = counts.get(row["planned_shape"], 0) + 1
    # 中文逐行注释：稳定排序确保相同类别始终按企业 ID 决定先后。
    return sorted(rows, key=lambda row: (counts[row["planned_shape"]], row["planned_shape"], row["sample_id"]))


# 中文逐行注释：以已保留集为基准贪心处置，避免 A-B 和 B-C 导致 A-C 被错误连坐。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
def disposition(rows: list[dict[str, Any]], triggers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 中文逐行注释：把触发边按样本 ID 建立邻接表，保留逐对证据而不是连通分量猜测。
    neighbours: dict[str, list[dict[str, Any]]] = {row["sample_id"]: [] for row in rows}
    # 中文逐行注释：逐个登记触发边到两端。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    for pair in triggers:
        # 中文逐行注释：左端保存该成对证据。
        neighbours[pair["left"]].append(pair)
        # 中文逐行注释：右端保存同一成对证据。
        neighbours[pair["right"]].append(pair)
    # 中文逐行注释：保存已经保留的 ID 集合，作为下一户的唯一比较对象。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    retained: set[str] = set()
    # 中文逐行注释：初始化最终可审计的逐户处置列表。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    output: list[dict[str, Any]] = []
    # 中文逐行注释：按固定优先级处理每一户。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    for row in greedy_order(rows):
        # 中文逐行注释：找出该户与已保留户之间真正触发双高线的逐对证据。
        conflicts = [pair for pair in neighbours[row["sample_id"]] if (pair["left"] if pair["right"] == row["sample_id"] else pair["right"]) in retained]
        # 中文逐行注释：没有冲突时立即保留该户。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        if not conflicts:
            # 中文逐行注释：登记为保留并把 ID 放入固定保留集。
            action = "retain"
            # 中文逐行注释：说明该户没有与已保留户发生 Q4 双高冲突。
            reason = "按稀少类别和ID顺序优先保留；不存在与已保留户的Q4双高证据"
            # 中文逐行注释：更新保留集合供后续户比较。
            retained.add(row["sample_id"])
        # 中文逐行注释：有逐对冲突时先作为冗余候选排除，不按连通组扩大排除范围。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        else:
            # 中文逐行注释：默认不入卷且不自动重做，符合执行细则。
            action = "exclude_candidate"
            # 中文逐行注释：原因只引用与实际已保留户的触发证据。
            reason = "与已保留户存在Q4余额及经营收支双高成对证据，默认作为冗余候选"
        # 中文逐行注释：写出执行表固定字段和回查所需的原件路径。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        output.append({
            "sample_id": row["sample_id"], # 中文教学注释：本行属于S5机械流程，保持原逻辑。
            "planned_shape": row["planned_shape"], # 中文教学注释：本行属于S5机械流程，保持原逻辑。
            "source_status": row.get("status"), # 中文教学注释：本行属于S5机械流程，保持原逻辑。
            "source_quality_receipt_reference": row["source_quality_receipt_reference"], # 中文教学注释：本行属于S5机械流程，保持原逻辑。
            "generation_directory": row["generation_directory"], # 中文教学注释：本行属于S5机械流程，保持原逻辑。
            "source_revision_r1_or_later": old_r1_used(Path(row["generation_directory"])), # 中文教学注释：本行属于S5机械流程，保持原逻辑。
            "action": action, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
            "reason": reason, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
            "trigger_pairs_with_retained": conflicts, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        })
    # 中文逐行注释：统计贪心后的每类保留数量，决定是否有稀少类别需要有限调整。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    retained_counts: dict[str, int] = {}
    # 中文逐行注释：逐项累计已保留的类别数量。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    for item in output:
        # 中文逐行注释：仅保留项参与类别覆盖数。
        if item["action"] == "retain":
            # 中文逐行注释：更新当前类别保留数。
            retained_counts[item["planned_shape"]] = retained_counts.get(item["planned_shape"], 0) + 1
    # 中文逐行注释：初始化全轮最多六户的受控调整名额。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    adjustment_slots = 6
    # 中文逐行注释：按同一固定顺序只从冗余候选中挑选类别仍少于三户的样本。
    for item in sorted(output, key=lambda item: (retained_counts.get(item["planned_shape"], 0), item["planned_shape"], item["sample_id"])):
        # 中文逐行注释：只处理默认冗余候选，保留项不重复进入队列。
        if item["action"] != "exclude_candidate":
            # 中文逐行注释：跳过已保留项。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
            continue
        # 中文逐行注释：类别已有三户时不为了数量使用调整名额。
        if retained_counts.get(item["planned_shape"], 0) >= 3:
            # 中文逐行注释：该冗余候选继续维持排除建议。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
            continue
        # 中文逐行注释：名额耗尽后不扩容，保持本轮最多六次新增生成约束。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        if adjustment_slots == 0:
            # 中文逐行注释：停止挑选后续候选。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
            break
        # 中文逐行注释：旧 r1 已用尽身份不得借本轮改名或换种子复活。
        if item["source_revision_r1_or_later"]:
            # 中文逐行注释：保留排除建议并记录不能调整的具体原因。
            item["reason"] += "；源原件已使用r1额度，本轮不复活"
            # 中文逐行注释：继续检查下一户。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
            continue
        # 中文逐行注释：升级为仅供主审批准的受控调整候选，不自动调用预算或生成。
        item["action"] = "adjustment_recommended"
        # 中文逐行注释：明确推荐原因来自类别去重后少于三户。
        item["reason"] += "；本类贪心保留不足3户，推荐进入最多6户的单变量组调整队列"
        # 中文逐行注释：附上批准细则限定的单一可选字段组。
        item["allowed_adjustment_fields"] = ["stage_settlement_slices", "stage_collection_credit_days", "stage_supplier_credit_days", "staggered_service_and_statement_calendar_v1"]
        # 中文逐行注释：附上不允许改变的经济与身份字段，供后续执行者断言。
        item["prohibited_adjustment_fields"] = ["identity", "random_seed", "planned_shape", "initial_capital", "financing", "tax_rules", "customer_supplier_counts", "business_nodes_monthly_totals", "contract_total"]
        # 中文逐行注释：占用一户有限调整名额。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        adjustment_slots -= 1
        # 中文逐行注释：把已推荐调整计入预期覆盖数，达到三户后不再无必要扩充队列。
        retained_counts[item["planned_shape"]] = retained_counts.get(item["planned_shape"], 0) + 1
    # 中文逐行注释：返回逐户处置表，后续不修改原件或主报告。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    return output


# 中文逐行注释：运行全部只读检查并把三项新产物写入独立目录。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
def run(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    # 中文逐行注释：读取批准范围内的 88 户通过清单。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    manifest = read_json(manifest_path)
    # 中文逐行注释：复制行对象，避免本工具意外修改读取到的原清单结构。
    rows = [dict(row) for row in manifest["rows"]]
    # 中文逐行注释：本轮细则固定以 88 户为输入，数量不符即停止而不猜范围。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    if len(rows) != 88:
        # 中文逐行注释：报告实际数量，方便主审核对清单版本。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        raise ValueError(f"只读筛查要求88户，实际为{len(rows)}")
    # 中文逐行注释：为每户保留原通过清单中质量状态的可回查引用，不重写任何旧回执。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    for row in rows:
        # 中文逐行注释：引用由清单绝对路径和样本ID组成，Luna可直接定位原状态行。
        row["source_quality_receipt_reference"] = f"{manifest_path.resolve()}#sample_id={row['sample_id']}"
    # 中文逐行注释：创建新的诊断目录，不触碰最终检查汇总或源原件。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 中文逐行注释：初始化完整全年签名缓存。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    full_signatures: dict[str, dict[str, Any]] = {}
    # 中文逐行注释：初始化 Q4 签名缓存。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    q4_signatures: dict[str, dict[str, np.ndarray]] = {}
    # 中文逐行注释：初始化逐户异常回执。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    exceptions: list[dict[str, str]] = []
    # 中文逐行注释：逐户验证原清单状态和必要原件。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    for row in rows:
        # 中文逐行注释：把生成目录转换为路径对象。
        directory = Path(row["generation_directory"])
        # 中文逐行注释：只认清单中的程序通过状态，状态缺失也记录异常。
        if "程序检查通过" not in str(row.get("status", "")):
            # 中文逐行注释：状态异常不继续签名，避免把未通过原件纳入保留集。
            exceptions.append({"sample_id": row["sample_id"], "stage": "manifest_status", "reason": str(row.get("status"))})
            # 中文逐行注释：转至下一行保留全部异常证据。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
            continue
        # 中文逐行注释：全年签名复用既有排融资口径并要求完整365日。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        try:
            # 中文逐行注释：读取实际全年原件，绝不触发 preview 或生成器。
            full_signatures[row["sample_id"]] = signature_2026(directory)
            # 中文逐行注释：构造本轮92日季度签名。
            q4_signatures[row["sample_id"]] = q4_signature(directory)
        # 中文逐行注释：任何日期、文件或方差错误都进入异常，而不是被替代为低相关。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        except Exception as error:
            # 中文逐行注释：保存明确异常原因供主审和Luna原件核对。
            exceptions.append({"sample_id": row["sample_id"], "stage": "signature", "reason": str(error)})
    # 中文逐行注释：有异常时停止处置，避免不完整88户被悄悄缩小范围。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    if exceptions:
        # 中文逐行注释：写出异常证据，仍不写任何源文件。
        (output_dir / "pair_evidence.json").write_text(json.dumps({"status": "blocked_input_exception", "exceptions": exceptions}, ensure_ascii=False, indent=2), encoding="utf-8")
        # 中文逐行注释：让调用方显式看到阻断原因。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        raise RuntimeError(f"88户只读筛查遇到{len(exceptions)}项输入异常")
    # 中文逐行注释：把 Q4 签名附到内存行对象，供成对公式和处置使用。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    for row in rows:
        # 中文逐行注释：不把数组写回清单，仅保留在本次进程内。
        row["q4_signature"] = q4_signatures[row["sample_id"]]
    # 中文逐行注释：初始化全年的原硬门违规证据列表。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    full_hard_failures: list[dict[str, Any]] = []
    # 中文逐行注释：初始化 Q4 双高触发对列表。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    q4_triggers: list[dict[str, Any]] = []
    # 中文逐行注释：初始化 Q4 共同趋势提示对列表。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    q4_common_trends: list[dict[str, Any]] = []
    # 中文逐行注释：按固定 ID 顺序枚举所有两两组合。
    ordered = sorted(rows, key=lambda row: row["sample_id"])
    # 中文逐行注释：遍历左端样本。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    for index, left in enumerate(ordered):
        # 中文逐行注释：遍历右端样本，避免重复或自比。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        for right in ordered[index + 1:]:
            # 中文逐行注释：复用原全年硬门回执公式，绝不以 Q4 结果取代全年规则。
            full_result = obvious_duplicate_2026(full_signatures[left["sample_id"]], full_signatures[right["sample_id"]])
            # 中文逐行注释：原全年强相似或日期平移克隆出现时记为硬门异常。
            if not full_result["passed"]:
                # 中文逐行注释：保存原公式返回的完整证据。
                full_hard_failures.append({"left": left["sample_id"], "right": right["sample_id"], "full_year": full_result})
            # 中文逐行注释：计算独立的 Q4 成对证据。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
            q4_result = q4_pair_evidence(left, right)
            # 中文逐行注释：双高对触发本轮实务处置线。
            if q4_result["q4_double_high_trigger"]:
                # 中文逐行注释：记录这条逐对证据。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
                q4_triggers.append(q4_result)
            # 中文逐行注释：余额单高对只保留为共同趋势提示。
            elif q4_result["q4_common_trend_only"]:
                # 中文逐行注释：记录该提示但不作为贪心排除边。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
                q4_common_trends.append(q4_result)
    # 中文逐行注释：若全年原硬门与通过清单相矛盾则停止，不允许在新规则下悄悄覆盖。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    if full_hard_failures:
        # 中文逐行注释：写入矛盾证据供主审先解决源门问题。
        (output_dir / "pair_evidence.json").write_text(json.dumps({"status": "blocked_full_year_hard_gate", "full_year_hard_failures": full_hard_failures}, ensure_ascii=False, indent=2), encoding="utf-8")
        # 中文逐行注释：明确阻止后续贪心处置。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        raise RuntimeError(f"全年原强相似门复核发现{len(full_hard_failures)}对冲突")
    # 中文逐行注释：按批准的固定保留策略构造处置表和有限调整建议。
    rows_for_disposition = [{key: value for key, value in row.items() if key != "q4_signature"} for row in rows]
    # 中文逐行注释：运行只依赖触发对的贪心，不因余额单高提示排除样本。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    table = disposition(rows_for_disposition, q4_triggers)
    # 中文逐行注释：按动作统计本轮建议数量。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    action_counts: dict[str, int] = {}
    # 中文逐行注释：逐项累计动作数。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    for item in table:
        # 中文逐行注释：更新相应动作计数。
        action_counts[item["action"]] = action_counts.get(item["action"], 0) + 1
    # 中文逐行注释：整理新诊断的全局元数据。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    summary = {
        "version": "s5_pragmatic_dedup_v1", # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "status": "read_only_disposition_recommendation_pending_main_review", # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "source_manifest": str(manifest_path.resolve()), # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "source_rows": len(rows), # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "q4_date_range": [Q4_START, Q4_END], # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "q4_required_days": 92, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "q4_disposition_line": {"balance_correlation_gte": BALANCE_LINE, "operating_cash_correlation_gte": OPERATING_LINE}, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "full_year_original_hard_gate": {"common_days_gte": 365, "balance_correlation_gte": BALANCE_LINE, "operating_cash_correlation_gte": OPERATING_LINE, "recomputed_conflicting_pairs": len(full_hard_failures)}, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "q4_double_high_pair_count": len(q4_triggers), # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "q4_common_trend_only_pair_count": len(q4_common_trends), # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "actions": action_counts, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "no_generation_called": True, # 中文教学注释：本行属于S5机械流程，保持原逻辑。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    }
    # 中文逐行注释：保存成对证据，包含双高、共同趋势及全年硬门复核结果。
    pair_payload = {**summary, "full_year_hard_failures": full_hard_failures, "q4_double_high_pairs": q4_triggers, "q4_common_trend_only_pairs": q4_common_trends, "exceptions": exceptions}
    # 中文逐行注释：以 UTF-8 和缩进写入便于人工审计的新文件。
    (output_dir / "pair_evidence.json").write_text(json.dumps(pair_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 中文逐行注释：保存逐户保留、排除、调整建议，不改变其源状态。
    (output_dir / "disposition.json").write_text(json.dumps({**summary, "rows": table}, ensure_ascii=False, indent=2), encoding="utf-8")
    # 中文逐行注释：生成通俗语言说明，清楚标注它是主审待确认建议。
    markdown = "\n".join([
        "# S5 大考样本去重只读处置建议", # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "", # 中文教学注释：本行属于S5机械流程，保持原逻辑。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        f"本次只读核查 {len(rows)} 户已通过原件。Q4 固定为 {Q4_START} 至 {Q4_END}，每户必须有 92 个自然日。没有调用生成、预演、模型或旧52数据。",
        "", # 中文教学注释：本行属于S5机械流程，保持原逻辑。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        f"全年原硬门按既有公式复算：余额相关≥{BALANCE_LINE:.2f} 且排融资经营收支相关≥{OPERATING_LINE:.2f}，并要求365个共同日；发现冲突 {len(full_hard_failures)} 对。",
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
        f"Q4 新处置线按同一相关公式单独计算：双高触发 {len(q4_triggers)} 对；余额单高但经营收支未达到线的共同趋势提示 {len(q4_common_trends)} 对，后者不参与排除。",
        "", # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        f"固定稀少类别优先、同类按 ID 贪心后的建议：保留 {action_counts.get('retain', 0)} 户，冗余候选 {action_counts.get('exclude_candidate', 0)} 户，建议进入有限单变量调整队列 {action_counts.get('adjustment_recommended', 0)} 户。调整队列最多6户，旧r1原件不会被复活。", # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "", # 中文教学注释：本行属于S5机械流程，保持原逻辑。
        "这是待主审确认的只读处置建议，不是对原88户名单的追认性修改。后续如批准调整，只允许既有交付/结算切片、客户或供应商账期及错峰日历；不得改身份、种子、类别、资本、融资、税、客户供应商数量、24个月经营分类总额或合同总额。", # 中文教学注释：本行属于S5机械流程，保持原逻辑。
    ]) + "\n" # 中文教学注释：本行属于S5机械流程，保持原逻辑。
    # 中文逐行注释：写出独立说明文件，避免修改主报告。
    (output_dir / "处置说明.md").write_text(markdown, encoding="utf-8")
    # 中文逐行注释：返回摘要供命令行打印和后续机械执行器读取。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    return summary


# 中文逐行注释：定义只读命令行入口，默认路径固定在S5批准范围内。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
def main() -> None:
    # 中文逐行注释：创建参数解析器并说明不生成的边界。
    parser = argparse.ArgumentParser(description="只读生成S5 Q4双相关去重证据与处置建议；不调用生成器")
    # 中文逐行注释：允许 Luna 显式传入88户清单，默认使用最终检查汇总版本。
    parser.add_argument("--manifest", type=Path, default=Path("data/research_round2_v4/s5/最终检查汇总/88户通过图册manifest.json"))
    # 中文逐行注释：允许 Luna 显式传入新的独立输出目录。
    parser.add_argument("--output", type=Path, default=Path("data/research_round2_v4/s5/pragmatic_dedup_v1"))
    # 中文逐行注释：解析用户提供的参数。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    args = parser.parse_args()
    # 中文逐行注释：运行只读筛查并取得摘要。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    summary = run(args.manifest, args.output)
    # 中文逐行注释：仅打印摘要，详细证据均在独立输出目录。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 中文逐行注释：仅在脚本直接执行时启动只读入口，导入时没有副作用。
if __name__ == "__main__":
    # 中文逐行注释：调用只读主函数。
# 中文教学注释：说明本行在S5机械流程中的作用，保持原有业务逻辑不变。
    main()
