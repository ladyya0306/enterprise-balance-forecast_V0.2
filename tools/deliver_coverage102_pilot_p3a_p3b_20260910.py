# 首批六户正式交付与自动检查入口；仅读取已冻结参数并写入统一生产目录。
"""将六份已首演的预算事实导出为正式流水、余额图和P3a/P3b检查材料。"""

# 导入命令行工具；入口只接受明确的正式交付动作。
import argparse
# 导入CSV工具；独立检查公开流水而不读取生成器内存对象。
import csv
# 导入哈希工具；核对冻结输入和共享运行环境没有漂移。
import hashlib
# 导入JSON工具；保存可复核的机器记录。
import json
# 导入系统模块；将冻结运行环境放到导入路径首位。
import sys
# 导入轻量命名对象；恢复中断时只描述既有正式文件路径。
from types import SimpleNamespace
# 导入日期时间工具；生成连续日期检查和预测窗口清单。
from datetime import date, datetime, timedelta
# 导入小数工具；金额按分精确比较，避免浮点误差。
from decimal import Decimal
# 导入路径工具；统一处理Windows中文路径。
from pathlib import Path

# 导入绘图库；正式交付需要为每户输出余额折线图。
import matplotlib
# 在无界面环境使用文件绘图后端。
matplotlib.use("Agg")
# 导入绘图接口；只绘制已导出的真实日表。
import matplotlib.pyplot as plt
# 导入字体管理器；为中文交付图明确指定可用字体文件。
from matplotlib import font_manager
# 导入日期刻度；横轴按月份显示。
import matplotlib.dates as mdates

# 计算项目根目录；当前脚本固定放在tools目录。
ROOT = Path(__file__).resolve().parents[1]
# 固定本轮唯一生产根；不向旧验证目录散写。
PACK = ROOT / "data/synthetic_production/coverage102_v1"
# 定位六户冻结控制材料。
CONTROL = PACK / "公共规则与版本/首批六份预算_20260910_v1"
# 定位唯一共享中央运行环境。
RUNTIME = PACK / "公共规则与版本/生成器_c3d_20260909"
# 定位冻结清单；它绑定六户身份、输入和运行环境。
BINDING = CONTROL / "生成前冻结清单.json"
# 将正式流水集中放在本批生产根之下。
FORMAL_ROOT = PACK / "正式流水/训练"
# 将图集中放在本批生产根之下。
FIGURE_ROOT = PACK / "余额图/训练"
# 将检查和复现记录保留在用途隔离目录。
CHECK_ROOT = PACK / "检查与复现/训练"
# 指定本轮正式交付版本名；同一身份只允许生成一次。
DELIVERY_VERSION = "p3a_p3b_delivery_20260910_v1"
# 指定Windows已安装的中文字体；缺失时保留默认字体并由检查阻断图面通过。
CHINESE_FONT_PATH = Path("C:/Windows/Fonts/NotoSansSC-VF.ttf")
# 读取字体文件生成绘图属性；避免默认字体遗漏中文字符。
CHINESE_FONT = font_manager.FontProperties(fname=CHINESE_FONT_PATH) if CHINESE_FONT_PATH.is_file() else None


# 计算文件SHA-256；用于验证封存参数和共享源码。
def sha256(path: Path) -> str:
    # 读取全部字节并计算小写十六进制指纹。
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 读取UTF-8 JSON；所有本入口JSON由本项目写入。
def read_json(path: Path):
    # 解析UTF-8文本为Python对象。
    return json.loads(path.read_text(encoding="utf-8-sig"))


# 写入一次性JSON；防止重跑覆盖首份正式证据。
def write_once(path: Path, payload) -> None:
    # 任何输出都必须留在本轮生产根内。
    if not path.resolve().is_relative_to(PACK.resolve()):
        # 阻止误写到旧目录或工作目录。
        raise ValueError(f"输出越界：{path}")
    # 已存在说明本次证据已经保存，拒绝覆盖。
    if path.exists():
        # 用异常阻止自动重跑篡改历史。
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    # 先建立父目录，便于各户材料分开保存。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写入格式化JSON并保留中文。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 读取带BOM的公开CSV；公开文件是独立核账对象。
def read_csv(path: Path) -> list[dict[str, str]]:
    # 使用utf-8-sig自动去掉BOM。
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # 返回每行字典以便按字段核对。
        return list(csv.DictReader(handle))


# 将公开元金额转换成分；两位小数的正式CSV必须可精确还原。
def fen(value: str) -> int:
    # 使用Decimal避免二进制浮点舍入。
    return int(Decimal(value) * 100)


# 将日期文字转换成日期对象；日表统一使用ISO日期。
def as_date(value: str) -> date:
    # 解析YYYY-MM-DD格式。
    return date.fromisoformat(value)


# 确认冻结输入与共享运行环境保持不变。
def preflight(binding: dict) -> None:
    # 设计文件必须仍是预算首演前冻结的版本。
    if sha256(CONTROL / "六户经营设计.json") != binding["design_sha256"]:
        # 阻止看过预算结果后修改设计。
        raise ValueError("六户经营设计指纹变化")
    # 逐文件核对中央运行环境。
    for relative, expected in binding["runtime_files"].items():
        # 定位一个共享源码文件。
        actual = RUNTIME / relative
        # 缺失或变化都不能继续正式导出。
        if not actual.is_file() or sha256(actual) != expected:
            # 写出受影响相对路径，便于定位。
            raise ValueError(f"共享运行环境变化：{relative}")
    # 逐户核对冻结YAML和事前材料。
    for row in binding["rows"]:
        # 读取该户冻结文件字典。
        for relative, expected in row["files"].items():
            # 定位本批生产根中的冻结文件。
            actual = PACK / relative
            # 任一变化阻断正式导出。
            if not actual.is_file() or sha256(actual) != expected:
                # 明确失败文件而不猜测原因。
                raise ValueError(f"冻结输入变化：{relative}")


# 检查正式导出目录不存在；避免覆盖历史结果。
def ensure_new_target(row: dict) -> tuple[Path, Path, bool]:
    # 本户正式流水目录由中央生成器按身份和运行编号创建。
    formal = FORMAL_ROOT / row["sample_id"] / row["run_id"]
    # 本户余额图目录按相同身份保存。
    figure = FIGURE_ROOT / row["sample_id"] / row["run_id"]
    # 检查本户交付记录是否已经写入开始标记。
    delivery_dir = CHECK_ROOT / row["sample_id"] / row["run_id"] / "正式交付"
    # 只有已生成正式文件但还没有图的技术中断，才允许不重导出地恢复检查。
    resumed = formal.is_dir() and not figure.exists() and (delivery_dir / "正式导出开始.json").is_file()
    # 正常首次导出要求两个目标均不存在。
    new = not formal.exists() and not figure.exists()
    # 其他组合均意味着覆盖风险或半成品事实不明。
    if not (new or resumed):
        # 拒绝覆盖历史或凭猜测接续。
        raise FileExistsError(f"正式交付目标状态不允许写入：{formal}；{figure}")
    # 返回路径及是否只读恢复既有正式文件。
    return formal, figure, resumed


# 对照首演逐笔事实，确认正式CSV没有重新计算出另一套流水。
def verify_transactions(preview: dict, rows: list[dict[str, str]]) -> dict:
    # 取出首演的已支付逐笔账本。
    expected = preview["run"]["ledger"]["transactions"]
    # 正式行数必须等于首演已入账行数。
    if len(rows) != len(expected):
        # 阻断缺笔或多笔的正式交付。
        raise AssertionError("正式逐笔流水行数未匹配首演")
    # 从零余额开始独立滚算正式CSV。
    balance = 0
    # 逐笔比较身份、时间、方向、金额和余额。
    for index, (actual, original) in enumerate(zip(rows, expected, strict=True), start=1):
        # 公开CSV必须保留首演交易标识。
        if actual["transaction_id"] != original["transaction_id"]:
            # 明确指出出现差异的序号。
            raise AssertionError(f"第{index}笔交易编号未匹配首演")
        # 正式导出的秒级日期必须等于首演。
        if actual["booking_datetime"] != original["booking_datetime"]:
            # 阻断随机日期漂移。
            raise AssertionError(f"第{index}笔交易日期未匹配首演")
        # 将正式金额精确还原为分。
        debit, credit = fen(actual["debit_cny"]), fen(actual["credit_cny"])
        # 正式方向和金额必须等于首演。
        if debit != original["debit_fen"] or credit != original["credit_fen"]:
            # 阻断金额或方向漂移。
            raise AssertionError(f"第{index}笔交易金额未匹配首演")
        # 用签名现金流独立更新余额。
        balance += credit - debit
        # 交易后余额必须匹配CSV和首演。
        if balance != fen(actual["post_transaction_balance_cny"]) or balance != original["post_transaction_balance_fen"]:
            # 阻断余额不闭合。
            raise AssertionError(f"第{index}笔交易余额未闭合")
    # 返回适合汇总的独立核账结果。
    return {"transactions_match_preview": True, "transaction_count": len(rows), "ending_balance_fen": balance}


# 对照正式日表检查连续自然日、收支合计和日末余额。
def verify_daily(tx_rows: list[dict[str, str]], daily_rows: list[dict[str, str]], preview: dict) -> dict:
    # 日表不能为空，正式交付至少覆盖一个实际日期。
    if not daily_rows:
        # 阻断空日表被误当成合格文件。
        raise AssertionError("正式日表为空")
    # 将逐笔交易按日累计。
    by_day: dict[str, dict[str, int]] = {}
    # 遍历公开逐笔流水。
    for row in tx_rows:
        # 取交易日期部分，不使用未来任何资料。
        key = row["booking_datetime"][:10]
        # 为首次出现的日期建立累计槽。
        by_day.setdefault(key, {"inflow": 0, "outflow": 0, "count": 0})
        # 累计入账金额。
        by_day[key]["inflow"] += fen(row["credit_cny"])
        # 累计出账金额。
        by_day[key]["outflow"] += fen(row["debit_cny"])
        # 累计当日交易笔数。
        by_day[key]["count"] += 1
    # 从第一天开始独立滚算日末余额。
    balance = 0
    # 记录前一个自然日以验证无缺日。
    previous: date | None = None
    # 遍历正式日表。
    for row in daily_rows:
        # 解析当前自然日。
        current = as_date(row["calendar_date"])
        # 非首行必须正好相隔一天。
        if previous is not None and current != previous + timedelta(days=1):
            # 阻断缺失或乱序日表。
            raise AssertionError("正式日表自然日不连续")
        # 读取当天逐笔聚合。
        total = by_day.get(row["calendar_date"], {"inflow": 0, "outflow": 0, "count": 0})
        # CSV汇总必须等于逐笔聚合。
        if fen(row["inflow_cny"]) != total["inflow"] or fen(row["outflow_cny"]) != total["outflow"] or int(row["transaction_count"]) != total["count"]:
            # 阻断逐笔和日表不一致。
            raise AssertionError(f"正式日表收支未匹配逐笔：{row['calendar_date']}")
        # 用实际日收支独立更新余额。
        balance += total["inflow"] - total["outflow"]
        # 日末余额必须与正式日表一致。
        if balance != fen(row["ending_balance_cny"]):
            # 阻断日余额不闭合。
            raise AssertionError(f"正式日表余额未闭合：{row['calendar_date']}")
        # 记录当前日期给下一次连续性检查。
        previous = current
    # 首演事实给出完整或停止状态。
    complete = bool(preview["run"]["ledger"]["is_complete"])
    # 获取首演最后实际日。
    preview_last = preview["run"]["ledger"]["daily_rows"][-1]["calendar_date"]
    # 完整情景应写满到计划期末，停止情景应写至阻断款当天。
    expected_last = preview["profile"]["request"]["end_date"] if complete else preview["run"]["ledger"]["fact_sheet"]["blocked_booking_datetime"][:10]
    # 统一确认正式日表最后一天符合冻结事实。
    if daily_rows[-1]["calendar_date"] != expected_last:
        # 阻断把停止后未知期间补进来的做法。
        raise AssertionError("正式日表终止日期未匹配首演")
    # 停止日表含阻断当天，可能晚于最后一笔可支付交易日。
    if not complete and preview_last > expected_last:
        # 这表示保存事实本身矛盾，不能继续交付。
        raise AssertionError("首演停止事实日期矛盾")
    # 返回日表独立核验摘要。
    return {"daily_rows_contiguous": True, "natural_day_count": len(daily_rows), "daily_ending_balance_fen": balance, "execution_status": daily_rows[-1]["execution_status"]}


# 核验停止事实或完整现金闭合的正确输出。
def verify_execution_fact(formal: Path, preview: dict) -> dict:
    # 读取首演是否完整。
    complete = bool(preview["run"]["ledger"]["is_complete"])
    # 完整企业应有最终现金闭合文件。
    if complete:
        # 读取正式受限现金闭合材料。
        reconciliation = read_json(formal / "_restricted" / "final_cash_reconciliation.json")
        # 完整首演必须保持闭合。
        if not reconciliation["is_reconciled"]:
            # 阻断未闭合的完整交付。
            raise AssertionError("完整企业最终现金未闭合")
        # 返回完整事实。
        return {"ledger_complete": True, "final_cash_reconciled": True}
    # 停止企业应有正式资金不足事实单。
    fact = read_json(formal / "generation_fact_sheet.json")
    # 读取首演事实单。
    expected = preview["run"]["ledger"]["fact_sheet"]
    # 逐项对照停止日期和资金缺口。
    checks = {"blocked_booking_datetime": fact["blocked_booking_datetime"] == expected["blocked_booking_datetime"], "available_balance_before_block_cny": fen(fact["available_balance_before_block_cny"]) == expected["available_balance_before_block_fen"], "blocked_outflow_cny": fen(fact["blocked_outflow_cny"]) == expected["blocked_outflow_fen"], "shortage_cny": fen(fact["shortage_cny"]) == expected["shortage_fen"], "completed_movement_count": fact["completed_movement_count"] == expected["completed_movement_count"], "planned_movement_count": fact["planned_movement_count"] == expected["planned_movement_count"]}
    # 任一事实不符都不能掩盖。
    if not all(checks.values()):
        # 报告字段级差异供后续排查。
        raise AssertionError(f"停止事实未匹配首演：{checks}")
    # 返回停止事实核验。
    return {"ledger_complete": False, "stopped_fact_matches_preview": True, "stop_checks": checks}


# 在图中只画已正式导出的余额，不把停止后的计划线画成实际。
def render_balance_chart(row: dict, daily_rows: list[dict[str, str]], preview: dict, figure: Path) -> Path:
    # 创建本户图目录。
    figure.mkdir(parents=True, exist_ok=False)
    # 解析横轴日期。
    dates = [datetime.fromisoformat(item["calendar_date"]) for item in daily_rows]
    # 将日末余额从元转换为万元以便阅读。
    balances = [float(Decimal(item["ending_balance_cny"]) / Decimal("10000")) for item in daily_rows]
    # 建立可读的单图画布。
    chart, axis = plt.subplots(figsize=(13, 5.8))
    # 用蓝线表示真实已发生余额。
    axis.plot(dates, balances, color="#1f77b4", linewidth=1.7, label="预演日终余额" if row.get("is_preview") else "实际日终余额")
    # 读取首演完成状态。
    complete = bool(preview["run"]["ledger"]["is_complete"])
    # 停止时只在最后真实日打红点和文字。
    if not complete:
        # 读取资金不足事实。
        fact = preview["run"]["ledger"]["fact_sheet"]
        # 取停止日作为红线位置。
        stop = datetime.fromisoformat(fact["blocked_booking_datetime"])
        # 画出停止日竖线。
        axis.axvline(stop, color="#d62728", linewidth=1.3, linestyle="--", label="资金不足停止")
        # 标出停止前余额。
        axis.scatter([dates[-1]], [balances[-1]], color="#d62728", zorder=4)
        # 写出不可支付项目的缺口。
        shortage = Decimal(fact["shortage_fen"]) / Decimal("1000000")
        # 说明停止后没有余额数据。
        axis.annotate(f"停止：缺口 {shortage:.2f} 万元\n后续无实际流水", xy=(dates[-1], balances[-1]), xytext=(10, 14), textcoords="offset points", color="#a00000", fontsize=9, fontproperties=CHINESE_FONT)
    # 完整企业标出计划期末，帮助阅读全周期。
    else:
        # 在末点画绿色标记。
        axis.scatter([dates[-1]], [balances[-1]], color="#2ca02c", zorder=4)
        # 说明已经完整执行到计划期末。
        axis.annotate(row.get("completion_label", "记录至计划期末"), xy=(dates[-1], balances[-1]), xytext=(-105, 14), textcoords="offset points", color="#176b2c", fontsize=9, fontproperties=CHINESE_FONT)
    # 使用企业编号与组合名作为标题，避免把预期形态误当实际。
    axis.set_title(f"{row['combination_id']}｜{row['sample_id']}｜" + ("预算预演（非正式交付）" if row.get("is_preview") else "正式实际余额"), fontproperties=CHINESE_FONT)
    # 标注横轴为自然日。
    axis.set_xlabel("日期", fontproperties=CHINESE_FONT)
    # 标注纵轴单位为万元。
    axis.set_ylabel("日终余额（万元）", fontproperties=CHINESE_FONT)
    # 每三个月显示一个主刻度。
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    # 使用年月格式显示横轴。
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # 增加浅网格帮助查阅。
    axis.grid(True, alpha=0.25)
    # 显示图例。
    axis.legend(loc="best", prop=CHINESE_FONT)
    # 自动优化留白。
    chart.tight_layout()
    # 固定本户唯一图名。
    output = figure / "余额折线图.png"
    # 输出高分辨率PNG。
    chart.savefig(output, dpi=180, bbox_inches="tight")
    # 关闭图对象防止六图累积占用内存。
    plt.close(chart)
    # 返回图片路径。
    return output


# 从正式日表提取真实可用预测截止日；不从计划未来补答案。
def qualifying_windows(daily_rows: list[dict[str, str]], scenario: dict) -> dict:
    # 读取经营和筹建起点。
    operating_start = as_date(scenario["operating_start"])
    # 读取筹建结束日。
    setup_end = as_date(scenario["setup_end"])
    # 生成按日期查找的正式日表。
    by_date = {as_date(item["calendar_date"]): item for item in daily_rows}
    # 获取已写入的所有银行工作日。
    bankdays = [as_date(item["calendar_date"]) for item in daily_rows if item["is_bank_workday"] in {True, "True", "true"}]
    # 用日期集合快速确认真实连续日。
    available = set(by_date)
    # 经营满六个月的最早候选日。
    try:
        # 将月份向后移动六个月。
        target_month = operating_start.month + 6
        # 推算目标年份。
        year = operating_start.year + (target_month - 1) // 12
        # 推算目标月份。
        month = (target_month - 1) % 12 + 1
        # 逐日退到该月实际存在的日期。
        candidate = date(year, month, operating_start.day)
    # 目标月份没有同日号时改用该月最后一天。
    except ValueError:
        # 月末日期在目标月不存在时使用该月最后一天。
        candidate = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year + 1, 1, 1) - timedelta(days=1)
    # 只在银行工作日上选择预测截止日。
    cutoff_dates = []
    # 逐个银行工作日判断既有历史和未来。
    for index, cutoff in enumerate(bankdays):
        # 截止日必须不早于经营满六个月。
        if cutoff < candidate:
            # 经营年龄不足的日期直接跳过。
            continue
        # 需要前60个银行工作日历史并且历史第一天严格晚于筹建结束。
        history = bankdays[max(0, index - 59): index + 1]
        # 需要后30个银行工作日真实未来。
        future = bankdays[index + 1: index + 31]
        # 检查两段长度。
        if len(history) != 60 or len(future) != 30:
            # 停止末段或过早日期自然不合格。
            continue
        # 历史窗口必须位于筹建结束之后。
        if history[0] <= setup_end:
            # 设备筹建影响还在历史窗口内。
            continue
        # 日表必须在从历史到未来的自然日完整存在。
        if any(day not in available for day in (history + future)):
            # 禁止把缺失停止期当完整未来。
            continue
        # 保存可复核的截止日和固定窗口边界。
        cutoff_dates.append({"cutoff_date": cutoff.isoformat(), "history_bank_workday_start": history[0].isoformat(), "history_bank_workday_end": history[-1].isoformat(), "future_bank_workday_start": future[0].isoformat(), "future_bank_workday_end": future[-1].isoformat()})
    # 返回真实可用窗口，不授予训练权限。
    return {"rule_version": "P3b_60_history_30_future_after_six_months_v1", "operating_start": operating_start.isoformat(), "setup_end": setup_end.isoformat(), "eligible_cutoff_count": len(cutoff_dates), "eligible_cutoffs": cutoff_dates, "training_eligibility_status": "qualified_dates_only_pending_family_isolation_and_training_contract" if cutoff_dates else "no_qualified_dates_under_P3b"}


# 生成六户相似性线索；只输出提示，绝不按图自动删除样本。
def similarity_report(records: list[dict]) -> dict:
    # 创建可读的两两比较结果。
    pairs = []
    # 枚举每一对不同企业。
    for left_index, left in enumerate(records):
        # 右侧只取尚未配对的企业。
        for right in records[left_index + 1:]:
            # 读取两户真实日余额。
            left_values = left["balances"]
            # 读取右侧真实余额数组。
            right_values = right["balances"]
            # 比较共同已发生自然日长度。
            common = min(len(left_values), len(right_values))
            # 共同日不足60天时不强做结论。
            if common < 60:
                # 明确证据不足。
                pairs.append({"left_sample_id": left["sample_id"], "right_sample_id": right["sample_id"], "common_days": common, "status": "insufficient_common_history", "reason_cn": "共同真实日不足60天，不判定走势相似。"})
                # 跳过不足长度的本对比较。
                continue
            # 将共同前缀各自减去首日余额，比较变化形状。
            l = left_values[:common]
            # 取右侧相同长度的共同前缀。
            r = right_values[:common]
            # 计算各自平均变化。
            lm = sum(l) / common
            # 计算右侧共同前缀的平均余额。
            rm = sum(r) / common
            # 计算中心化后的相关分子和分母。
            numerator = sum((a - lm) * (b - rm) for a, b in zip(l, r, strict=True))
            # 计算左侧平方和。
            lss = sum((a - lm) ** 2 for a in l)
            # 计算右侧平方和。
            rss = sum((b - rm) ** 2 for b in r)
            # 恒定曲线没有定义相关系数。
            correlation = None if lss == 0 or rss == 0 else numerator / ((lss * rss) ** 0.5)
            # 比较交易标识完全相同的情况。
            same_tx = left["transaction_fingerprint"] == right["transaction_fingerprint"]
            # 只把完全相同交易判为明确重复。
            status = "exact_transaction_duplicate" if same_tx else "signal_only"
            # 用相似度作为人工复核线索，不设删除门槛。
            pairs.append({"left_sample_id": left["sample_id"], "right_sample_id": right["sample_id"], "common_days": common, "balance_level_pearson": correlation, "same_transaction_fingerprint": same_tx, "status": status, "reason_cn": "只有逐笔指纹相同才确认重复；走势相关只供人工复核，不自动删除。"})
    # 返回本批全对照列表。
    return {"version": "p3a_similarity_signal_v1", "pair_count": len(pairs), "pairs": pairs, "automatic_deletion": False}


# 执行一户正式导出、独立检查、绘图和P3b日期提取。
def deliver_one(row: dict, generate_from_profile) -> dict:
    # 检查正式输出目标尚未存在。
    formal, figure, resumed = ensure_new_target(row)
    # 定位本户YAML。
    yaml_path = PACK / row["folder"] / "完整参数.yaml"
    # 读取已保存首演事实作为唯一对照。
    preview_path = CHECK_ROOT / row["sample_id"] / row["run_id"] / "预算首演/中央完整事实.json"
    # 首演事实必须存在。
    if not preview_path.is_file():
        # 缺少对照事实不能正式导出。
        raise FileNotFoundError(f"缺少首演事实：{preview_path}")
    # 读取首演事实。
    preview = read_json(preview_path)
    # 写入本户正式开始记录，明确这是首份正式导出。
    delivery_dir = CHECK_ROOT / row["sample_id"] / row["run_id"] / "正式交付"
    # 首次交付记录调用前参数，恢复时保持原记录不覆盖。
    if not resumed:
        # 记录调用前的参数指纹。
        write_once(delivery_dir / "正式导出开始.json", {"delivery_version": DELIVERY_VERSION, "formal_calls": 1, "preview_calls_reused": 1, "input_sha256": sha256(yaml_path), "no_budget_adjustment_or_reseed": True})
        # 调用冻结中央生成器写出一次正式文件。
        generated = generate_from_profile(yaml_path, output_root=FORMAL_ROOT, output_run_id=row["run_id"])
        # 正确取得中央正式包对象。
        bundle = generated.bundle
    # 技术中断恢复只读既有正式文件，绝不再次调用中央生成器。
    else:
        # 保存错误原因与不重导出的恢复边界。
        write_once(delivery_dir / "技术中断与恢复.json", {"reason_cn": "首次正式文件已写入后，交付入口错误读取FormalGenerationResult字段而中断；恢复只读取既有正式文件，不再次调用中央生成器。", "formal_calls_total": 1, "re_exported": False})
        # 用同一字段名称包装已有文件路径，后续独立核查无需分支。
        bundle = SimpleNamespace(output_dir=formal, transactions_csv=formal / "transactions_total.csv", daily_csv=formal / "account_daily_total.csv", cash_flow_review_notes_csv=formal / "cash_flow_review_notes.csv", manifest_json=formal / "generation_manifest.json")
    # 读取三份公开正式材料。
    tx_rows = read_csv(bundle.transactions_csv)
    # 读取正式自然日日表。
    daily_rows = read_csv(bundle.daily_csv)
    # 读取人工备注表。
    note_rows = read_csv(bundle.cash_flow_review_notes_csv)
    # 交易备注必须逐笔一一对应。
    if len(note_rows) != len(tx_rows):
        # 阻断人工备注与公开流水不同步。
        raise AssertionError("人工备注表行数未匹配逐笔流水")
    # 独立核对正式逐笔流水。
    transaction_check = verify_transactions(preview, tx_rows)
    # 独立核对正式日表。
    daily_check = verify_daily(tx_rows, daily_rows, preview)
    # 核对完整或停止事实。
    execution_check = verify_execution_fact(bundle.output_dir, preview)
    # 输出正式实际余额图。
    image_path = render_balance_chart(row, daily_rows, preview, figure)
    # 读取首演前冻结的经营和筹建边界。
    scenario = read_json(PACK / row["folder"] / "企业设定.json")
    # 从正式日表提取真实预测窗口清单。
    windows = qualifying_windows(daily_rows, scenario)
    # 保存P3b日期清单但不把它当最终训练准入。
    write_once(delivery_dir / "P3b真实预测窗口.json", windows)
    # 保存本户自动核查证据。
    audit = {"delivery_version": DELIVERY_VERSION, "sample_id": row["sample_id"], "run_id": row["run_id"], "combination_id": row["combination_id"], "family_id": row["family_id"], "purpose": row["purpose"], "formal_path": str(bundle.output_dir.relative_to(PACK)).replace("\\", "/"), "balance_figure": str(image_path.relative_to(PACK)).replace("\\", "/"), "input_sha256": sha256(yaml_path), "preview_fact_sha256": sha256(preview_path), "transaction_check": transaction_check, "daily_check": daily_check, "execution_check": execution_check, "p3b_window_summary": {key: windows[key] for key in ("eligible_cutoff_count", "training_eligibility_status")}, "formal_csv_public_files": ["transactions_total.csv", "account_daily_total.csv", "cash_flow_review_notes.csv"], "model_input_boundary_cn": "cash_flow_review_notes.csv、_restricted目录、画像与预算均不得作为基础余额模型输入。", "target_shape_status": "passed" if row["sample_id"].startswith("business_102_a1") else "unmet_retained", "stop_classification": "complete" if execution_check["ledger_complete"] else "unexpected_early_stop_retained"}
    # 保存一次性自动核查结果。
    write_once(delivery_dir / "自动检查结果.json", audit)
    # 写入正式导出完成事实。
    write_once(delivery_dir / "正式导出完成.json", {"delivery_version": DELIVERY_VERSION, "formal_calls": 1, "formal_output_sha256": sha256(bundle.manifest_json), "ledger_complete": execution_check["ledger_complete"], "eligible_cutoff_count": windows["eligible_cutoff_count"]})
    # 返回汇总和相似性计算所需最少数据。
    return {**audit, "balances": [float(Decimal(item["ending_balance_cny"])) for item in daily_rows], "transaction_fingerprint": sha256(bundle.transactions_csv)}


# 汇总六户并写入本批公共检查目录。
def write_batch_summary(results: list[dict]) -> None:
    # 生成不含大数组的用户可读汇总。
    rows = []
    # 遍历六户结果。
    for result in results:
        # 去掉只用于相似性计算的内存数组。
        item = {key: value for key, value in result.items() if key not in {"balances", "transaction_fingerprint"}}
        # 保留本户汇总。
        rows.append(item)
    # 建立相似性报告。
    similarity = similarity_report(results)
    # 保存整批相似性证据。
    write_once(PACK / "检查与复现/批次/P3a实际相似性线索.json", similarity)
    # 汇总完整和停止数量。
    complete = sum(item["execution_check"]["ledger_complete"] for item in results)
    # 汇总有P3b日期的企业数。
    with_windows = sum(item["p3b_window_summary"]["eligible_cutoff_count"] > 0 for item in results)
    # 记录不掩盖失败的批次状态。
    summary = {"delivery_version": DELIVERY_VERSION, "budget_slots": 6, "formal_generated": 6, "complete_enterprises": complete, "stopped_enterprises": len(results) - complete, "enterprises_with_p3b_dates": with_windows, "similarity_pair_count": similarity["pair_count"], "exact_transaction_duplicates": sum(item["status"] == "exact_transaction_duplicate" for item in similarity["pairs"]), "actual_similarity_status": "signals_generated_no_automatic_deletion", "training_runs": 0, "overall_training_readiness": "pending_family_isolation_and_training_contract", "rows": rows}
    # 保存整批自动检查汇总。
    write_once(PACK / "检查与复现/批次/首批六户P3a_P3b自动检查汇总.json", summary)


# 只重绘因字体缺失而不可读的图片，不触碰已正式导出的流水。
def rerender_figures(binding: dict, revision_name: str) -> None:
    # 没有中文字体时不能宣称修订图可读。
    if CHINESE_FONT is None:
        # 明确阻断而不是再输出缺字图片。
        raise FileNotFoundError(f"缺少中文绘图字体：{CHINESE_FONT_PATH}")
    # 逐户读取既有正式日表和首演事实。
    for row in binding["rows"]:
        # 定位既有正式输出目录。
        formal = FORMAL_ROOT / row["sample_id"] / row["run_id"]
        # 定位首版图片目录。
        original = FIGURE_ROOT / row["sample_id"] / row["run_id"]
        # 定位只读首演事实。
        preview_path = CHECK_ROOT / row["sample_id"] / row["run_id"] / "预算首演/中央完整事实.json"
        # 首版、正式日表或首演事实缺失时不能进行图面修订。
        if not (formal / "account_daily_total.csv").is_file() or not (original / "余额折线图.png").is_file() or not preview_path.is_file():
            # 明确缺少何种前置材料。
            raise FileNotFoundError(f"缺少重绘前置材料：{row['sample_id']}")
        # 修订图独立放入子目录，保留缺字首版不覆盖。
        revised = original / revision_name
        # 修订目录必须此前未产生，防止重绘覆盖。
        if revised.exists():
            # 终止重复修订。
            raise FileExistsError(f"已存在图面修订：{revised}")
        # 读取既有正式日表。
        daily_rows = read_csv(formal / "account_daily_total.csv")
        # 读取首演事实用于停止标注。
        preview = read_json(preview_path)
        # 只据既有CSV重绘修订图。
        image = render_balance_chart(row, daily_rows, preview, revised)
        # 保存修订来源与边界，说明没有重导出。
        write_once(revised / "图面字体修订说明.json", {"reason_cn": "本修订版只读取既有正式日表重绘：首版缺少中文字体，第一次修订版的停止缺口万元换算错误。", "source_daily_csv_sha256": sha256(formal / "account_daily_total.csv"), "source_preview_sha256": sha256(preview_path), "re_exported": False, "retrained": False, "revised_image": str(image.relative_to(PACK)).replace("\\", "/")})


# 执行本次首批正式交付。
def main() -> int:
    # 构建命令行解析器。
    parser = argparse.ArgumentParser(description="按冻结参数正式交付首批六户并执行P3a/P3b自动检查。")
    # 只有明确确认后才允许写入正式交付。
    parser.add_argument("--execute", action="store_true", help="执行一次正式导出；不传入时不写任何材料。")
    # 图面修订只读既有正式日表，不再次导出流水。
    parser.add_argument("--rerender-figures", action="store_true", help="仅重绘已有首版的中文余额图，不再次调用中央生成器。")
    # 第二次图面修订只改正停止缺口的显示单位，仍然只读已有正式日表。
    parser.add_argument("--rerender-figures-v2", action="store_true", help="仅重绘已有图面到第二修订版，不再次调用中央生成器。")
    # 读取命令行参数。
    args = parser.parse_args()
    # 未传任何动作时仅显示帮助并退出。
    if not args.execute and not args.rerender_figures and not args.rerender_figures_v2:
        # 避免误双击就写入正式流水。
        parser.print_help()
        # 正常结束而不改文件。
        return 0
    # 两个写入动作不能在同一进程混用，避免恢复与正式导出相互干扰。
    if args.execute and (args.rerender_figures or args.rerender_figures_v2):
        # 明确要求操作者分开执行两种动作。
        parser.error("--execute不能与图面重绘动作同时使用")
    # 两种修订目标不同，禁止一次同时写入，避免操作者误判当前推荐版本。
    if args.rerender_figures and args.rerender_figures_v2:
        # 要求明确选择一个修订层级。
        parser.error("两种图面重绘动作不能同时使用")
    # 读取冻结绑定。
    binding = read_json(BINDING)
    # 先核对输入和运行环境。
    preflight(binding)
    # 仅重绘时直接处理既有日表后结束。
    if args.rerender_figures or args.rerender_figures_v2:
        # 不导入或调用中央生成器。
        rerender_figures(binding, "修订版" if args.rerender_figures else "修订版_v2")
        # 输出简短重绘摘要。
        print(json.dumps({"rerendered_figures": len(binding["rows"]), "revision_name": "修订版" if args.rerender_figures else "修订版_v2", "formal_calls": 0}, ensure_ascii=False))
        # 正常结束。
        return 0
    # 将冻结中央源码放到导入路径首位。
    sys.path.insert(0, str(RUNTIME / "src"))
    # 只导入中央正式生成接口；本入口不复制业务公式。
    from syn_b1.formal_generator import generate_from_profile
    # 创建本次结果列表。
    results = []
    # 按冻结清单固定顺序逐户正式导出。
    for row in binding["rows"]:
        # 导出、核对并记录一户。
        results.append(deliver_one(row, generate_from_profile))
    # 汇总六户检查和相似性线索。
    write_batch_summary(results)
    # 在终端仅输出短摘要。
    print(json.dumps({"formal_generated": len(results), "complete": sum(item["execution_check"]["ledger_complete"] for item in results), "stopped": sum(not item["execution_check"]["ledger_complete"] for item in results)}, ensure_ascii=False))
    # 返回成功退出码。
    return 0


# 仅直接运行时执行命令入口。
if __name__ == "__main__":
    # 将main返回值交给系统。
    raise SystemExit(main())
