# 固定批量入口只连接已存在的生成、核账、绘图和取样函数。
"""清单驱动的合成流水交付；停止保留，恢复不重复生成。"""
# 命令行分开检查与执行。
import argparse
# 原画像追加实际结果需要保留中文。
import json
# 稳定指纹不含企业或交易编号。
import hashlib
# 文件复制只用于保存实际生效参数。
import shutil
# 共享中央源码必须在导入前明确选择。
import sys
# 合同比较使用多重集合，重复金额不能相互覆盖。
from collections import Counter
# 日历日期用于比较经营阶段。
from datetime import date, datetime, timedelta
# 金额从CSV精确转分。
from decimal import Decimal
# 输出路径由清单和项目根共同确定。
from pathlib import Path
# 复用首批已经使用过的独立核账、绘图和窗口筛选。
import deliver_coverage102_pilot_p3a_p3b_20260910 as delivery
# 复用中央对象序列化，不重新计算经营金额。
from freeze_coverage102_pilot_budget_20260910 import serial

# 所有路径以项目为基准，清单无需写电脑盘符。
ROOT = Path(__file__).resolve().parents[1]
# 每批输出都必须位于唯一生产根。
PRODUCTION = ROOT / "data/synthetic_production"
# 入口版本发生变化时新批次显式保存，不混用旧回执。
VERSION = "synthetic_batch_workflow_v1"

# 计算文件原字节指纹。
def sha(path):
    # 不把解析后的相似内容当成同一版本。
    return hashlib.sha256(path.read_bytes()).hexdigest()

# 清单引用使用项目相对路径。
def locate(relative):
    # 解析真实位置，包含父目录跳转检查。
    path = (ROOT / relative).resolve()
    # 不允许清单访问项目之外的文件。
    if not path.is_relative_to(ROOT.resolve()):
        # 明确拒绝路径越界。
        raise ValueError("清单路径超出项目")
    # 返回已经核对的绝对路径。
    return path

# 输出目录的范围比输入更窄。
def output_path(relative):
    # 先经过项目边界检查。
    path = locate(relative)
    # 生产材料统一保存，不能落入验证目录。
    if not path.is_relative_to(PRODUCTION.resolve()):
        # 错误目标在写入前拒绝。
        raise ValueError("输出必须位于data/synthetic_production")
    # 返回安全位置。
    return path

# 只读JSON兼容既有BOM文件。
def read(path):
    # 不修改任何历史文件编码。
    return json.loads(path.read_text(encoding="utf-8-sig"))

# 稳定文本方便对重复调用做相等比较。
def encoded(value):
    # 对象输出带换行，文本保持原样。
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"

# 已有相同文件复用，有差异则拒绝覆盖。
def stable_write(path, value):
    # 输出文本先固定，避免不同序列化造成误判。
    text = encoded(value)
    # 所有写入仍限制在生产根。
    if not path.resolve().is_relative_to(PRODUCTION.resolve()):
        # 在创建父目录之前拒绝。
        raise ValueError("写入超出统一生产根")
    # 重复执行只允许验证同样内容。
    if path.exists():
        # 差异不能静默覆盖旧回执。
        if path.read_text(encoding="utf-8") != text:
            # 需要新版本或明确的技术修复记录。
            raise ValueError(f"已有材料不同，拒绝覆盖：{path.name}")
        # 相同文件不用再写。
        return
    # 新文件按需要建立目录。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 独占创建，防止两个执行者同时覆盖。
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        # 一次写入确定内容。
        handle.write(text)

# 每户身份不能形成隐含目录层级。
def safe_id(value):
    # 仅接受已有生成器常用的字母数字下划线和连字符。
    if not value or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in value):
        # 不接收路径分隔符或空标识。
        raise ValueError("样本或运行编号无效")

# 校验一个带指纹的输入引用。
def reference(ref):
    # 先解析项目内的实际文件。
    path = locate(ref["path"])
    # 缺文件或字节变化都不能继续使用。
    if not path.is_file() or sha(path) != ref["sha256"]:
        # 只显示相对位置，便于用户定位。
        raise ValueError(f"输入缺失或已变更：{ref['path']}")
    # 返回可复用的路径。
    return path

# 核对中央正式文件清单，包含受限会计材料。
def verify_formal(folder, row):
    # 正式清单必须存在且身份一致。
    manifest = read(folder / "generation_manifest.json")
    # 防止把别户文件放到当前样本名下。
    if (manifest["sample_id"], manifest["run_id"]) != (row["sample_id"], row["run_id"]):
        # 错误对应属于材料错误，不属于经营停止。
        raise ValueError("正式流水身份与清单不一致")
    # 所有已登记文件逐项验证。
    for name, expected in manifest["files"].items():
        # 阻止正式文件清单引用目录之外的内容。
        path = (folder / name).resolve()
        # 路径和指纹必须同时正确。
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or sha(path) != expected:
            # 保留原件，报告材料问题。
            raise ValueError(f"正式文件核验不通过：{name}")
    # 返回实际经营状态与日期。
    return manifest

# 执行前核验环境、全部输入和家族用途。
def preflight(manifest):
    # 只支持明确的清单结构版本。
    if manifest.get("version") != VERSION:
        # 防止预算YAML误当成批次清单。
        raise ValueError("清单版本不支持")
    # 每批上限来自已确认的执行规则。
    if len(manifest["rows"]) > 40:
        # 超额应另分批，不能一次绕过抽查规则。
        raise ValueError("本批最多40个位置")
    # 先检查输出位置。
    output_path(manifest["output"])
    # 运行环境指纹来自同一冻结清单。
    runtime = locate(manifest["runtime"])
    # 逐文件检查，不自动切换到工作目录里的修改版。
    for name, expected in manifest["runtime_files"].items():
        # 源码缺失或变化必须阻断调用。
        if not (runtime / name).resolve().is_relative_to(runtime.resolve()) or sha(runtime / name) != expected:
            # 指出受影响文件。
            raise ValueError(f"冻结生成器已变化：{name}")
    # 明确中央模块导入路径。
    sys.path.insert(0, str(runtime / "src"))
    # 只加载参数校验，不启动经营计算。
    from syn_b1.formal_profile import load_profile
    # 用集合检查重复企业位置。
    ids = set()
    # 家族先固定用途，不能跨训练与考试。
    families = dict(manifest.get("family_registry", {}))
    # 独立收集每个位置的核查结果。
    checks = []
    # 一户输入问题不能把其他户偷偷判成已生成。
    for row in manifest["rows"]:
        # 用户已否决的批次不能被旧完成回执重新放行。
        rejection = PRODUCTION / "coverage102_v1/批次否决登记.json"
        # 新版本使用新运行编号，原失败文件只供只读问题分析。
        if rejection.exists() and row.get("run_id") in read(rejection)["rejected_run_ids"]:
            # 不计入训练、测试、大考或有效覆盖。
            raise ValueError("该运行版本已被用户整批否决，禁止作为有效样本执行或复用")
        # 防止同一批身份重复。
        if row["sample_id"] in ids:
            # 这属于清单错误。
            raise ValueError("清单存在重复样本位置")
        # 记录本户。
        ids.add(row["sample_id"])
        # 校验样本编号不能逃逸目录。
        safe_id(row["sample_id"])
        # 对所有模式限制用途。
        if row["purpose"] not in {"train", "development", "final_exam"}:
            # 未知用途不能混入输出。
            raise ValueError("未知样本用途")
        # 同一家族不跨用途。
        if row["family_id"] in families and families[row["family_id"]] != row["purpose"]:
            # 清单阶段就发现用途冲突。
            raise ValueError("同源家族跨用途")
        # 登记新家族的事前用途。
        families[row["family_id"]] = row["purpose"]
        # 本入口用于日常复核，最终大考须另走密封交付。
        if row["purpose"] == "final_exam":
            # 不读取考试答案或绘制日常可见考试图。
            checks.append({"sample_id": row["sample_id"], "status": "sealed_pending", "reason": "最终大考留待独立密封交付，不进入日常复核入口"})
            # 继续检查其他日常位置。
            continue
        # 单户校验异常保留为材料问题。
        try:
            # 核对当前模式所提供的全部文件。
            for ref in row["inputs"].values():
                # 预算、画像、参数和事实全部绑定。
                reference(ref)
            # 未对接预算只显示具体原因，不伪造可执行参数。
            if row["mode"] == "budget_pending":
                # 这是未执行位置，不是停止或技术执行异常。
                checks.append({"sample_id": row["sample_id"], "status": "budget_pending", "reason": row["pending_reason"]})
                # 保留其预算供相似性预查。
                continue
            # 其他两种模式都要求正式运行编号。
            safe_id(row["run_id"])
            # 未知模式不能落入默认生成分支。
            if row["mode"] not in {"existing", "generate"}:
                # 明确区分借用旧样本与新生成。
                raise ValueError("未知执行模式")
            # 中央解析器核查完整参数。
            profile = load_profile(reference(row["inputs"]["profile"]))
            # 输入身份和执行单必须一致。
            if (profile.sample_id, profile.run_id) != (row["sample_id"], row["run_id"]):
                # 不允许靠输出编号替换另一户。
                raise ValueError("完整参数身份与清单不一致")
            # 日期口径在生成前就必须写清。
            scenario = read(reference(row["inputs"]["scenario"]))
            # 经营年龄与筹建结束必须是实际日期。
            date.fromisoformat(scenario["operating_start"])
            # 不让画像遗漏关键取样边界。
            date.fromisoformat(scenario["setup_end"])
            # 复用旧户时先核对已有文件。
            if row["mode"] == "existing":
                # 旧户只读，绝不调用生成器。
                verify_formal(locate(row["formal"]), row)
            # 这里只表示可以进入相应执行模式。
            checks.append({"sample_id": row["sample_id"], "status": "ready", "mode": row["mode"]})
        # 输入校验问题不冒充经营停止。
        except (ValueError, KeyError, OSError) as error:
            # 原因与位置一起保留，其他户继续检查。
            checks.append({"sample_id": row["sample_id"], "status": "input_error", "reason": str(error)})
    # 返回轻量的清单核查结果。
    return checks

# 只在明确的新户模式调用中央一次。
def ensure_generated(row, root, generator):
    # 输出按用途隔离。
    formal_root = root / "正式流水" / row["purpose"]
    # 中央生成器沿用既有样本和运行编号目录。
    formal = formal_root / row["sample_id"] / row["run_id"]
    # 每户执行日志放在专用检查目录。
    control = root / "检查" / row["sample_id"]
    # 调用开始标记在真正生成之前独占创建。
    started = control / "中央调用开始.json"
    # 完成事实用于中断后复用，不能再跑同户。
    facts = control / "中央核查事实.json"
    # 若正式文件和完整事实都在，验证后直接复用。
    if (formal / "generation_manifest.json").exists() and facts.exists():
        # 确认已有数据不是半成品。
        verify_formal(formal, row)
        # 返回原路径，不调用任何随机生成逻辑。
        return formal, facts
    # 开始过但缺完整证据时不自动第二次生成。
    if started.exists() or formal.exists():
        # 等待明确的技术恢复，保留第一次尝试。
        raise ValueError("已有调用但交付证据不完整；保留现场，不自动再次生成")
    # 创建执行日志父目录。
    control.mkdir(parents=True, exist_ok=True)
    # 独占文件充当一次调用的防重标记。
    with started.open("x", encoding="utf-8") as handle:
        # 参数指纹绑定此次调用。
        json.dump({"profile_sha256": row["inputs"]["profile"]["sha256"], "formal_calls": 1}, handle)
    # 唯一流水生成入口；这里不复制任何金额或余额算法。
    generated = generator(reference(row["inputs"]["profile"]), output_root=formal_root, output_run_id=row["run_id"])
    # 保存中央原始事实供独立核账，不使用另一套模拟器。
    stable_write(facts, {"profile": serial(generated.profile), "run": serial(generated.run)})
    # 正式文件清单应当完整。
    verify_formal(formal, row)
    # 返回同一份正式输出与事实。
    return formal, facts

# 指纹不包括会随企业编号变化的交易ID。
def transaction_fingerprint(rows):
    # 只比较真实时间、借贷方向及交易后余额。
    payload = [[r[k] for k in ("booking_datetime", "debit_cny", "credit_cny", "post_transaction_balance_cny")] for r in rows]
    # 精确相同才标为重复流水。
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

# 相关系数只生成复核线索，不作为删户门槛。
def correlation(left, right):
    # 共同真实长度之外不补数据。
    count = min(len(left), len(right))
    # 太短不能作稳定比较。
    if count < 2:
        # 无法计算时如实留空。
        return None
    # 按共同长度各自中心化。
    a = [float(x) for x in left[:count]]
    # 右侧同样转换。
    b = [float(x) for x in right[:count]]
    # 只在统计中使用浮点，账务仍是整数分。
    am, bm = sum(a) / count, sum(b) / count
    # 计算标准化分母。
    denominator = (sum((x-am)**2 for x in a) * sum((y-bm)**2 for y in b)) ** .5
    # 常数序列没有定义相关系数。
    return None if not denominator else sum((x-am)*(y-bm) for x, y in zip(a, b)) / denominator

# 对已实际交付的样本同时查完整流水和经营阶段。
def actual_similarity(records, threshold):
    # 两两比較先保留全部对照，不只保留漂亮结果。
    pairs = []
    # 每对只计算一次。
    for index, left in enumerate(records):
        # 只遍历之后的企业。
        for right in records[index+1:]:
            # 去掉筹建期后比较余额与日净收支。
            count = min(len(left["operating_balances"]), len(right["operating_balances"]))
            # 不对不足60自然日的片段过度解释。
            balance_corr = correlation(left["operating_balances"], right["operating_balances"]) if count >= 60 else None
            # 日收支相关能补充余额长期趋势的误导。
            cash_corr = correlation(left["operating_cash"], right["operating_cash"]) if count >= 60 else None
            # 显式记录exact与近似线索的不同含义。
            pairs.append({"left": left["sample_id"], "right": right["sample_id"], "exact_transactions": left["fingerprint"] == right["fingerprint"], "common_operating_days": count, "balance_correlation": balance_corr, "cash_change_correlation": cash_corr, "similarity_signal": balance_corr is not None and balance_corr >= threshold})
    # 不删除或重抽任何高相似样本。
    return {"pairs": pairs, "threshold": threshold, "automatic_deletion": False, "alignment": "筹建结束后按真实经营天序比较，不补齐停止期"}

# 预算相似性在生成前即能检查。
def budget_similarity(rows, threshold):
    # 每户只保留数值序列，不用名字证明不同家族。
    records = []
    # 预算未就绪也必须纳入预查。
    for row in rows:
        # 没有月度预算引用时不伪造一个。
        if "budget" not in row["inputs"]:
            # 已有首批只有其正式数据参与实际对照。
            continue
        # 预算YAML使用既有YAML库解析。
        import yaml
        # 仅读取清单绑定的文件。
        budget = yaml.safe_load(reference(row["inputs"]["budget"]).read_text(encoding="utf-8-sig"))
        # 所有月度订单和费用形成明确的经济数值序列。
        values = [[m[k] for k in ("sales_delivery_fen", "purchase_commitment_fen", "payroll_fen", "rent_fen", "other_expense_fen")] for m in budget["monthly_budget"]]
        # 资产付款时点也参与精确预算重复检查。
        assets = [[a["total_fen"], a["ready_date"], a["payments"]] for a in budget["assets"]]
        # 编号与画像文字不进入指纹。
        fingerprint = hashlib.sha256(json.dumps([values, assets, budget["initial_capital_fen"]], sort_keys=True).encode()).hexdigest()
        # 保存预算交付额序列供初步节奏比较。
        records.append({"sample_id": row["sample_id"], "fingerprint": fingerprint, "sales": [m[0] for m in values]})
    # 两两对照只作线索，家族仍须经济来源复审。
    pairs = []
    # 不跨缺失月份补数据。
    for index, left in enumerate(records):
        # 避免重复计算。
        for right in records[index+1:]:
            # 不同规模也可能有相同订单节奏。
            value = correlation(left["sales"], right["sales"])
            # 高相关不是完整预算重复的证明。
            pairs.append({"left": left["sample_id"], "right": right["sample_id"], "exact_selected_budget_fields": left["fingerprint"] == right["fingerprint"], "sales_rhythm_correlation": value, "similarity_signal": value is not None and value >= threshold})
    # 清晰标明这不是实际流水比较。
    return {"kind": "budget_only", "budget_count": len(records), "pairs": pairs, "automatic_deletion": False, "limitation": "精确检查仅覆盖所列经济字段；相关只比较计划销售节奏，不证明企业同源"}

# 成套交付一户，所有账务和绘图复用既有函数。
def deliver(row, root, generator):
    # 每户专用交付文件夹。
    folder = root / "交付" / row["purpose"] / row["sample_id"]
    # 复用既有材料时不触发中央调用。
    if row["mode"] == "existing":
        # 正式文件仍保留在原目录。
        formal = locate(row["formal"])
        # 首演事实也是只读原件。
        facts_path = reference(row["inputs"]["facts"])
    # 新户只经过明确的一次中央入口。
    else:
        # 支持完整证据下恢复，不重复造样本。
        formal, facts_path = ensure_generated(row, root, generator)
    # 独立核验正式清单与身份。
    verify_formal(formal, row)
    # 使用原读取函数，兼容既有CSV编码。
    transactions = delivery.read_csv(formal / "transactions_total.csv")
    # 真实日表是唯一绘图来源。
    daily = delivery.read_csv(formal / "account_daily_total.csv")
    # 核查使用中央原始事实，不根据曲线编故事。
    facts = read(facts_path)
    # 新合同预算在正式候选层独立核对，不只核对导出文件自身。
    if "budget" in row["inputs"]:
        # 原预算仅读，实际税费另列差异。
        import yaml
        # 核查结果与每户交付同处保存。
        stable_write(folder / "原预算与中央检查.json", verify_budget(facts, read(reference(row["inputs"]["scenario"]))["source_budget"]))
    # 原检查逐笔对照日期、金额与余额。
    tx_check = delivery.verify_transactions(facts, transactions)
    # 独立检查日汇总和停止后没有补线。
    day_check = delivery.verify_daily(transactions, daily, facts)
    # 完整或停止各走对应事实核查。
    execution = delivery.verify_execution_fact(formal, facts)
    # 使用事前经营边界筛选真实预测片段。
    scenario = read(reference(row["inputs"]["scenario"]))
    # 停止本身不会使之前的完整预测片段作废。
    windows = delivery.qualifying_windows(daily, scenario)
    # 新增单户内部波形门槛，不再仅比较企业之间。
    from curve_repetition_check import check as check_internal_curve
    # 原始日表直接检查，绝不为过门槛修改余额。
    curve_check = check_internal_curve(daily, scenario["setup_end"])
    # 不合格也保留证据文件，不隐瞒实际生成结果。
    stable_write(folder / "曲线内重复检查.json", curve_check)
    # 保存当前真实可用日期，不启动训练。
    stable_write(folder / "预测片段检查.json", windows)
    # 实际生效的YAML原字节保存到交付目录。
    profile = reference(row["inputs"]["profile"])
    # 目标参数副本保持原始编码，不使用重新排版后的YAML。
    target = folder / "完整参数.yaml"
    # 已有副本必须完全相同。
    if target.exists() and sha(target) != sha(profile):
        # 拒绝将新参数放进旧交付中。
        raise ValueError("交付参数副本已变化")
    # 首次才复制。
    if not target.exists():
        # 保留实际参数原字节。
        shutil.copyfile(profile, target)
    # 已有图片通过回执核对，新图使用固定原函数。
    chart_dir = folder / "余额图"
    # 固定文件名不随人工绘图改变。
    chart = chart_dir / "余额折线图.png"
    # 为修复中断时的临时目录提供安全恢复。
    if not chart.exists():
        # 空目录可以清除后交由原绘图函数创建。
        if chart_dir.exists():
            # 只允许删除空目录，不动已有文件。
            chart_dir.rmdir()
        # 不重写绘图逻辑，不绘制停止后的未知区间。
        delivery.render_balance_chart({**row, "completion_label":"记录至计划期末"}, daily, facts, chart_dir)
    # 读取实际停止事实，避免使用旧建议中的自动重做措辞。
    stop = None if execution["ledger_complete"] else read(formal / "generation_fact_sheet.json")
    # 人工可读结果只列金额、日期与实际支付阻断。
    outcome = "流水记录至计划期末，具体接单、收尾及停产阶段见经营画像。" if stop is None else f"停止经营：{stop['blocked_booking_datetime']}，到期付款{stop['blocked_outflow_cny']}元，可用余额{stop['available_balance_before_block_cny']}元，缺口{stop['shortage_cny']}元。停止记录保留，不自动补资或重做。"
    # 原企业画像明确属于事前计划。
    portrait = reference(row["inputs"]["portrait"]).read_text(encoding="utf-8-sig")
    # 同一个文件将背景与实际结果连起来供用户阅读。
    portrait += "\n\n## 本次实际经营结果\n\n" + outcome + f"\n\n真实可用预测起点：{windows['eligible_cutoff_count']}个；不等于独立企业数。原画像后续计划若未兑现，不当成已发生。\n\n![实际余额图]({chart.as_posix()})\n"
    # 画像不作为基础预测模型输入。
    stable_write(folder / "企业画像.md", portrait)
    # 成套材料用索引连接，旧流水不复制也不改写。
    artifacts = {"流水明细": formal / "transactions_total.csv", "日余额": formal / "account_daily_total.csv", "余额折线图": chart, "完整参数YAML": target, "企业画像": folder / "企业画像.md", "预测片段检查": folder / "预测片段检查.json"}
    # 交付回执绑定这户的新增质量检查。
    artifacts["曲线内重复检查"] = folder / "曲线内重复检查.json"
    # 新预算的差异检查也纳入交付指纹。
    if "budget" in row["inputs"]:
        # 方便用户直接查看原预算与正式计算的对应情况。
        artifacts["预算与中央检查"] = folder / "原预算与中央检查.json"
    # 保存全部核心交付的内容指纹，重复运行必须一致。
    receipt = {"sample_id": row["sample_id"], "run_id": row["run_id"], "mode": row["mode"], "purpose": row["purpose"], "status": "complete" if stop is None else "stopped", "stop": {k: stop[k] for k in ("blocked_booking_datetime", "blocked_outflow_cny", "available_balance_before_block_cny", "shortage_cny")} if stop else None, "transaction_check": tx_check, "daily_check": day_check, "execution_check": execution, "eligible_cutoffs": windows["eligible_cutoff_count"], "artifacts": {k: {"path": v.relative_to(ROOT).as_posix(), "sha256": sha(v)} for k, v in artifacts.items()}}
    # 回执本身不可被二次调用覆盖。
    # 技术生成完成不再等同样本质量通过。
    receipt["curve_quality"] = {"passed": curve_check["passed"], "sufficient_observation": curve_check["sufficient_observation"], "status": curve_check["status"]}
    # 保留可解释停止，与机械重复分别判定。
    stable_write(folder / "交付回执.json", receipt)
    # 只取筹建结束后的真实经营日用于近似检查。
    operating = [d for d in daily if d["calendar_date"] > scenario["setup_end"]]
    # 大数组只在内存比较，报告保持简短。
    return {"receipt": receipt, "sample_id": row["sample_id"], "fingerprint": transaction_fingerprint(transactions), "operating_balances": [delivery.fen(d["ending_balance_cny"]) for d in operating], "operating_cash": [delivery.fen(d["inflow_cny"])-delivery.fen(d["outflow_cny"]) for d in operating]}

# 核验合同金额和日期，允许税费由正式模型替代原暂估。
def verify_budget(facts, budget):
    # 月度确认值独立与原预算对照，停止后月份只属于计划。
    months = facts["run"]["operating_cycle"]["monthly_rows"]
    # 月份不能缺失后靠zip静默忽略。
    if len(months) != len(budget["monthly_budget"]):
        # 计划不完整属于材料错误。
        raise ValueError("中央计划月数与预算不同")
    # 逐一核对销售、成本、库存、采购和各项费用。
    fields = {"sales_confirmed_fen": "sales_delivery_fen", "cost_of_goods_sold_fen": "material_consumed_fen", "material_purchases_fen": "purchase_commitment_fen", "ending_inventory_fen": "closing_inventory_fen", "payroll_expense_fen": "payroll_fen", "rent_and_utilities_expense_fen": "rent_fen", "other_operating_expense_fen": "other_expense_fen"}
    # 不通过事后修改预算凑一致。
    for actual, source in zip(months, budget["monthly_budget"], strict=True):
        # 日期和所有金额必须完全相同。
        if actual["month"] != source["month"] or any(actual[a] != source[b] for a, b in fields.items()):
            # 报告具体月份，原件保留。
            raise ValueError(f"中央确认与原预算不同：{source['month']}")
    # 非税收付种类使用中央类别名称。
    categories = {"initial_capital": "shareholder_funding", "additional_capital": "shareholder_funding", "sales_collection": "sales_collection", "supplier_payment": "supplier_payment", "payroll": "payroll", "rent": "rent_and_utilities", "operating_expense": "other_declared_event", "equipment_payment": "fixed_asset_payment"}
    # 期内原合同构成候选，不把期外应收提前计入。
    expected = Counter((e["date"], categories[e["type"]], abs(e["amount_fen"])) for e in budget["planned_cash_events"] if e["type"] in categories and e["amount_fen"] and budget["period"]["start_date"] <= e["date"] <= budget["period"]["end_date"])
    # 所有计划候选包括停止后未执行项，不能写成实际流水。
    candidates = facts["run"]["transaction_plan"]["candidates"]
    # 回款依赖供应商款用原到期日核对预算，实际付款时刻另行逐笔核验。
    actual = Counter()
    # 保存实际延期证据数量，供报告直接披露。
    deferred_supplier_checks = 0
    # 逐笔检查候选，不允许工资租赁或普通付款借用延期字段。
    for e in candidates:
        # 非税类别才属于本段预算合同核对。
        if e["category"] not in categories.values():
            continue
        # 读取中央明确标记，缺省保持普通原到期规则。
        applied = bool(e.get("receipt_dependency_applied", False))
        # 仅供应商付款可以真实延期。
        if applied:
            # 类别、原到期、指定回款、宽限和原因缺一不可。
            if e["category"] != "supplier_payment" or not e.get("original_due_booking_datetime") or not e.get("dependent_receipt_event_id") or not e.get("payment_deferral_reason_cn") or not e.get("negotiated_grace_days"):
                raise ValueError("回款依赖延期合同字段无效")
            # 在全候选中精确找到指定客户回款，避免只按金额或日期模糊匹配。
            receipt = next((item for item in candidates if item.get("business_order_id") == e["dependent_receipt_event_id"] and item["category"] == "sales_collection"), None)
            # 实际付款只能是指定回款后一秒。
            if receipt is None or e["booking_datetime"] != (datetime.fromisoformat(receipt["booking_datetime"]) + timedelta(seconds=1)).isoformat():
                raise ValueError("回款依赖实际付款未在指定回款后一秒")
            # 实际付款需晚于原到期且不超过已协商自然日宽限。
            original = datetime.fromisoformat(e["original_due_booking_datetime"])
            actual_time = datetime.fromisoformat(e["booking_datetime"])
            if actual_time <= original or (actual_time.date() - original.date()).days > e["negotiated_grace_days"]:
                raise ValueError("回款依赖实际付款超出原到期或宽限")
            # 预算日期仍是原合同到期日，金额和供应商类别不能变化。
            actual[(original.date().isoformat(), e["category"], e["amount_fen"])] += 1
            # 计数用于可读审计。
            deferred_supplier_checks += 1
        else:
            # 未延期的全部类别继续按实际候选日期严格核对。
            actual[(e["booking_datetime"][:10], e["category"], e["amount_fen"])] += 1
    # 不靠月总额相同掩盖日期被挪动。
    if actual != expected:
        # 明确差异数量，避免输出巨大清单。
        raise ValueError(f"非税合同日期金额不一致：缺{sum((expected-actual).values())}项、多{sum((actual-expected).values())}项")
    # 新税费与原暂估分开披露，停止后候选不是实缴。
    tax_planned = sum(e["amount_fen"] for e in candidates if e["category"] in {"corporate_income_tax_payment", "simulated_vat_payment", "surcharge_payment"})
    # 原2%暂估仅作比较，绝不重复入账。
    estimate = -sum(e["amount_fen"] for e in budget["planned_cash_events"] if e["type"] == "tax_estimate" and e["date"] <= budget["period"]["end_date"])
    # 留存中央期外尾款，后续核对可以明确应收应付。
    return {"monthly_plan_checks": len(months) * len(fields), "non_tax_contract_candidates": sum(actual.values()), "non_tax_dates_and_amounts_match": True, "receipt_dependent_supplier_checks": deferred_supplier_checks, "original_tax_estimate_fen": estimate, "central_planned_tax_fen": tax_planned, "planned_tax_difference_fen": tax_planned-estimate, "deferred_cash_due": facts["run"]["operating_cycle"]["deferred_cash_due"], "scope": "此处核对全期事前计划；回款依赖供应商款按原到期核预算、按实际回款后一秒核执行；停止之后的候选与税款不是已执行流水，实际以CSV可支付前缀为准"}

# 历史样本只参与比较，不重复生成、绘图或复制原材料。
def historical_similarity_rows(manifest):
    # 未指定历史范围时保持原工作流行为。
    if "comparison_manifest" not in manifest:
        # 不隐式扫描最终大考或其他目录。
        return []
    # 历史范围由冻结清单明确引用。
    source = read(reference(manifest["comparison_manifest"]))
    # 仅接受已有且非最终大考的企业。
    records = []
    # 跳过原清单中尚未生成的预算。
    for row in source["rows"]:
        # 日常复核不能访问最终大考答案。
        if row["mode"] != "existing" or row["purpose"] == "final_exam":
            # 不把待生成位置当作历史样本。
            continue
        # 原正式文件逐项核验。
        formal = locate(row["formal"])
        # 身份与字节必须匹配。
        verify_formal(formal, row)
        # 使用原筹建边界。
        scenario = read(reference(row["inputs"]["scenario"]))
        # 只取实际经营期间，停止后不补线。
        daily = [d for d in delivery.read_csv(formal / "account_daily_total.csv") if d["calendar_date"] > scenario["setup_end"]]
        # 沿用相同相似性定义。
        records.append({"sample_id": row["sample_id"], "fingerprint": transaction_fingerprint(delivery.read_csv(formal / "transactions_total.csv")), "operating_balances": [delivery.fen(d["ending_balance_cny"]) for d in daily], "operating_cash": [delivery.fen(d["inflow_cny"])-delivery.fen(d["outflow_cny"]) for d in daily]})
    # 返回只读比较资料。
    return records

# 批次执行与报告共用同一个固定入口。
def run(manifest, manifest_path, generator=None):
    # 核对环境和输入后才能开始写交付。
    checks = preflight(manifest)
    # 历史比较也在新生成前完成材料核查。
    historical = historical_similarity_rows(manifest)
    # 明确生产输出目录。
    root = output_path(manifest["output"])
    # 固定工具及依赖版本，重复运行不能暗换逻辑。
    stable_write(root / "工作流绑定.json", {"manifest_sha256": sha(manifest_path), "workflow_sha256": sha(Path(__file__)), "delivery_helper_sha256": sha(Path(delivery.__file__)), "curve_gate_sha256": sha(Path(__file__).with_name("curve_repetition_check.py")), "version": VERSION})
    # 已有批次回执仍会复核所有输入和交付指纹。
    finished = root / "批次汇总.json"
    # 重复调用走验证，不重画图、不重生成。
    if finished.exists():
        # 输入复核有异常时不能返回已完成。
        if any(c["status"] == "input_error" for c in checks):
            # 原回执保留，要求核对变化来源。
            raise ValueError("已有批次输入已变化，拒绝确认复用")
        # 读取原汇总检查交付核心材料。
        summary = read(finished)
        # 已交付的每户文件都必须仍可核对。
        for item in summary["delivered"]:
            # 不因停止跳过其材料。
            for ref in item["artifacts"].values():
                # 相同回执只能对应相同文件。
                reference(ref)
        # 明确本次调用没有新增生成。
        return {"reused_completed_batch": True, "new_generation_calls_this_invocation": 0, "summary": summary}
    # 预算相似性在任何新中央调用之前保存。
    planned_similarity = budget_similarity([r for r, c in zip(manifest["rows"], checks) if c["status"] in {"ready", "budget_pending"}], manifest["similarity_threshold"])
    # 所有近似仅提示，不能阻断合理经营结果。
    stable_write(root / "预算相似性检查.json", planned_similarity)
    # 没有测试替身时使用唯一中央生成器。
    if generator is None:
        # 直到实际执行才导入正式生成边界。
        from syn_b1.formal_generator import generate_from_profile
        # 不引入第二套流水计算方法。
        generator = generate_from_profile
    # 按清单定位核查结果。
    by_id = {x["sample_id"]: x for x in checks}
    # 保留已交付、未就绪和技术异常各自的结果。
    records, pending, errors = [], [], []
    # 材料损坏属于技术问题，不混在正常待接入预算里。
    errors.extend({"sample_id": c["sample_id"], "type": "InputError", "reason": c["reason"]} for c in checks if c["status"] == "input_error")
    # 每个位置最多进入一次对应动作。
    for row in manifest["rows"]:
        # 未就绪预算不产生假流水。
        if by_id[row["sample_id"]]["status"] != "ready":
            # 保存具体输入或能力差异。
            pending.append(by_id[row["sample_id"]])
            # 不将未执行算成停止经营。
            continue
        # 单户技术错误不与经营停止混淆。
        try:
            # 复用与生成都经过同样的成套交付检查。
            records.append(deliver(row, root, generator))
        # 保留本次技术异常，不自动换输入重试。
        except Exception as error:
            # 报告异常类型与原因，但不中断无关位置检查。
            errors.append({"sample_id": row["sample_id"], "type": type(error).__name__, "reason": str(error)})
    # 只比较真正已交付的流水。
    similarity = actual_similarity(records + historical, manifest["similarity_threshold"])
    # 近似对全部留档。
    # 异常尝试单独保存，避免修复后覆盖或冲突。
    report_root = root if not errors else root / "执行尝试" / str(len(list((root / "执行尝试").glob("*"))) + 1)
    # 每次技术异常都有独立证据，不改成功户的交付。
    stable_write(report_root / "实际流水相似性检查.json", similarity)
    # 成套交付数量不能与新生成次数混为一谈。
    receipts = [r["receipt"] for r in records]
    # 汇总完整和停止，技术异常另列。
    summary = {"version": VERSION, "manifest_rows": len(manifest["rows"]), "new_generated_deliveries": sum(r["mode"] == "generate" for r in receipts), "existing_deliveries": sum(r["mode"] == "existing" for r in receipts), "complete_enterprises": sum(r["status"] == "complete" for r in receipts), "stopped_enterprises": sum(r["status"] == "stopped" for r in receipts), "new_stopped_enterprises": sum(r["status"] == "stopped" and r["mode"] == "generate" for r in receipts), "pending": pending, "technical_errors": errors, "delivered": receipts, "exact_duplicate_pairs": sum(p["exact_transactions"] for p in similarity["pairs"]), "similarity_signal_pairs": sum(p["similarity_signal"] for p in similarity["pairs"]), "training_runs": 0}
    # 历史比较不计入本批新增或重新交付数量。
    summary["historical_comparison_enterprises"] = len(historical)
    # 预测起点与独立家族分别计数。
    summary["eligible_cutoffs"] = sum(r["eligible_cutoffs"] for r in receipts)
    # 这是事前登记家族数，不声称已证明所有经济背景相互独立。
    summary["registered_families"] = len({r["family_id"] for r in manifest["rows"] if r["sample_id"] in {s["sample_id"] for s in receipts}})
    # 机械重复未通过者不进入有效样本名单。
    summary["curve_rejected"] = [r["sample_id"] for r in receipts if not r["curve_quality"]["passed"]]
    # 短记录留待人工判断，不以观察不足冒充通过。
    summary["curve_short_review"] = [r["sample_id"] for r in receipts if r["curve_quality"]["passed"] and not r["curve_quality"]["sufficient_observation"]]
    # 只有新增门槛通过且观察足够的材料才列为可用候选。
    valid = [r for r in receipts if r["curve_quality"]["passed"] and r["curve_quality"]["sufficient_observation"]]
    # 实际训练还要满足家族隔离和预测片段条件，不在这里启动。
    stable_write(report_root / "有效样本候选清单.json", {"rows": valid, "curve_gate_version": "within_curve_repetition_v1", "training_started": False})
    # 报告内容由同一摘要固定生成。
    write_report(report_root, summary)
    # 存在技术错误时留作中断结果，允许之后明确恢复。
    stable_write(report_root / ("本次技术异常汇总.json" if errors else "批次汇总.json"), summary)
    # 控制台返回本次执行事实。
    return {"reused_completed_batch": False, "summary": summary}

# 固定报告模板，日后每批不重新写图文代码。
def write_report(root, summary):
    # 先说明新旧样本数量，避免重复计入2040。
    lines = ["# 流水批量交付", "", f"新生成并成套交付：{summary['new_generated_deliveries']}户；复核已有样本：{summary['existing_deliveries']}户；待接入：{len(summary['pending'])}户；技术异常：{len(summary['technical_errors'])}户。", "", f"本页已交付样本中，完整经营{summary['complete_enterprises']}户、停止经营{summary['stopped_enterprises']}户；其中新生成的停止经营{summary['new_stopped_enterprises']}户。停止属于经营结果，不自动删除或补资。", "", "|企业|实际结果|查看材料|", "|---|---|---|"]
    # 每户各类材料保持可直接查阅。
    for item in summary["delivered"]:
        # 最终大考材料不进入日常审图报告。
        if item["purpose"] == "final_exam":
            # 保存状态但不在普通报告披露余额或停止答案。
            lines.append(f"|{item['sample_id']}|最终大考材料密封|不展示图和经营结果|")
            # 不追加可打开的材料链接。
            continue
        # 逐项链接绝对位置，兼容桌面文件查看。
        links = " · ".join(f"[{name}]({locate(ref['path']).as_posix()})" for name, ref in item["artifacts"].items() if name != "预测片段检查")
        # 停止原因在图和画像之外再列一次。
        status = "完整经营" if item["stop"] is None else f"停止：{item['stop']['blocked_booking_datetime']}；付款{item['stop']['blocked_outflow_cny']}元、可用{item['stop']['available_balance_before_block_cny']}元、缺口{item['stop']['shortage_cny']}元"
        # 一行可以访问本户全套材料。
        lines.append(f"|{item['sample_id']}|{status}|{links}|")
    # 相似性不是删户指令。
    lines.extend(["", f"实际流水精确重复对：{summary['exact_duplicate_pairs']}；筹建后余额相似线索对：{summary['similarity_signal_pairs']}。相关度只提示复核，不证明企业同源，不自动删除。", "", "预算和实际流水的完整两两检查保存在同目录JSON；预算相似不冒充实际流水相似。", "", "## 待接入位置", ""])
    # 单户内部重复是强制门槛，不能与企业间提示混淆。
    lines.extend(["", f"单户曲线内部重复未过门槛：{summary.get('curve_rejected', [])}；观察期不足待复核：{summary.get('curve_short_review', [])}。未过者不进入有效样本候选清单。", ""])
    # 明确哪户尚未产生流水及原因。
    lines.extend(f"- {item['sample_id']}：{item['reason']}" for item in summary["pending"])
    # 技术异常单独显示，不写成企业停止。
    lines.extend(["", "## 技术异常", ""] + [f"- {item['sample_id']}：{item['type']}，{item['reason']}" for item in summary["technical_errors"]])
    # 已完成报告不可被不同内容覆盖。
    stable_write(root / "查看本批交付.md", "\n".join(lines) + "\n")

# 日常只需指定保存好的清单，检查或运行。
def main():
    # 不再每批临时编写脚本。
    parser = argparse.ArgumentParser(description="固定合成流水批量工作流")
    # 清单来自已明确的样本范围和输入。
    parser.add_argument("--manifest", type=Path, required=True)
    # 默认只检查，不隐式触发新中央生成。
    parser.add_argument("--action", choices=("check", "run"), default="check")
    # 读取命令行。
    args = parser.parse_args()
    # 清单内容在执行前读取一次。
    manifest = read(args.manifest)
    # 检查不会生成、绘图或改写预算。
    if args.action == "check":
        # 输出紧凑的核查状态。
        print(json.dumps({"checks": preflight(manifest), "new_generation": 0}, ensure_ascii=False))
    # 明确run才进入固定交付流程。
    else:
        # 主过程本身负责重复调用和中断保护。
        # 命令行生成统一经过R1授权及正式后检查，不能绕开门槛。
        from execute_pressure_budget_batch import run_frozen_manifest
        # 同一正式入口负责零、一、二退出码。
        from preflight_evidence import BusinessPause
        # 业务暂停必须保持二而非异常默认的一。
        try:
            # 直接结束旧命令，避免第二次执行流水。
            raise SystemExit(run_frozen_manifest(manifest, args.manifest.resolve().parent))
        # 保留业务暂停说明供用户复核。
        except BusinessPause as error:
            # 打印实际原因。
            print(str(error))
            # 不进入后续批次。
            raise SystemExit(2)

# 导入测试时不运行批次。
if __name__ == "__main__":
    # 普通脚本入口。
    main()
