"""从现有文件树生成只读核对导览；不会读取或改写任何样本内容。"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ABS = ROOT.as_posix()
EXCLUDED_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv"}
NEW_DOCS = {"项目文件夹与逐文件阅读指南.md", "项目文件逐项索引_样本.md", "项目文件逐项索引_其他材料.md"}

SAMPLE_FILE_HELP = {
    "来源映射.json": "来源追溯。看 source_path、source_sha256、样本身份；与清单的来源和校验值一致为正常。",
    "account_daily_total.csv": "每日余额主表。`calendar_date` 是日期，`inflow_cny`/`outflow_cny` 是当天流入/流出，`transaction_count` 是笔数，`ending_balance_cny` 是日末余额；日期应连续，金额应能由流水汇总解释。",
    "transactions_total.csv": "逐笔流水。看 transaction_id、booking_datetime、debit_cny、credit_cny、post_transaction_balance_cny；借/贷金额和余额应能汇总到日余额。",
    "cash_flow_review_notes.csv": "逐笔用途备注。按 transaction_id 对照流水，重点看用途、日期、金额和方向；未知用途是允许值，不应擅自补造。",
    "generation_manifest.json": "生成/导出留痕。看样本标识、版本、输入和输出路径或指纹；它说明来源过程，不是训练结果。",
    "quality_report.json": "已有质量检查结果。看通过/失败项、账务或日期检查和异常说明；发现失败或缺字段需回到对应 CSV 核对。",
    "scenario_profile.json": "企业情景设定。看企业业务、资金、周期和参数；用于理解样本机制，不能替代实际流水。",
}

FIELD_CN = {
    "sample_id": "样本编号", "family_id": "家族编号", "v3_role": "V3 分组", "original_family_id": "原家族编号",
    "图页": "图册页码", "图中位置": "页内位置", "日余额CSV": "对应日余额文件",
    "average_horizon_business_days": "预测期限工作日数", "calibration_record_count": "校准记录数", "calibration_enterprise_count": "校准企业数",
    "finite_sample_conformal_rank": "有限样本校准排名", "normalized_conformal_adjustment": "区间上下界调整量",
    "calibration_target_date_start": "校准目标起日", "calibration_target_date_end": "校准目标止日", "c2f_version": "定型版本",
    "epoch": "训练轮次", "train_account_scaled_pinball_loss": "训练误差", "calibration_account_scaled_pinball_loss": "校准误差",
    "training_seed": "随机种子", "transaction_id": "交易编号", "booking_datetime": "入账日期时间",
    "debit_cny": "借方金额", "credit_cny": "贷方金额", "post_transaction_balance_cny": "交易后余额",
}

def link(path: Path, label: str | None = None) -> str:
    absolute = path.resolve().as_posix()
    return f"[{label or path.name}]({absolute})"

def project_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.name.endswith(".pyc") or path.suffix == ".log":
            continue
        if path.parent == REPORTS and path.name in NEW_DOCS:
            continue
        yield path

def text_opening(path: Path) -> str:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return "直接点击链接预览；看页码、图中样本标签和横轴日期，再回到对应 CSV 查具体数值。"
    if path.name == ".gitignore":
        return "用记事本打开；每一行是 Git 不上传的路径或通配规则，`!` 开头表示例外保留。"
    if path.suffix not in {".py", ".md", ".json", ".yaml", ".yml", ".csv"}:
        return "用与格式匹配的软件打开；二进制模型文件不能用记事本。"
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        shown = [f"`{item}`（{FIELD_CN.get(item, '原始字段名')}）" for item in header[:8]]
        return "用 Excel/WPS 打开；先看：" + "、".join(shown) + ("……" if len(header) > 8 else "")
    if path.suffix == ".py":
        lines = path.read_text(encoding="utf-8").splitlines()
        symbols = [line.strip().split("(")[0].replace("def ", "").replace("class ", "") for line in lines if line.startswith(("def ", "class "))][:5]
        return "代码编辑器打开；先看文件首部、导入项和入口函数" + ("：" + "、".join(symbols) if symbols else "。")
    return "用记事本或代码编辑器以 UTF-8 打开；用搜索查找 `status`、`version`、`path`、`sha256` 等关键词。"

def classify(path: Path) -> tuple[str, str]:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("data/manifests/"):
        return "清单/核对结果", manifest_purpose(path.name)
    if rel.startswith("reports/"):
        if path.suffix == ".png":
            return "余额图册图页", "本图册第 %s 页，展示 8 户日末余额曲线；样本、页码和 CSV 对应关系见 `图册CSV对应.csv`。" % path.stem.split("_")[-1]
        state = "；历史或被修正情况须以 03_样本整理收尾_20260922.md 为准" if path.name in {"02_补齐完成记录_20260922.md", "第三轮V3.0样本整理交付报告.md", "样本整理复核与下一步_20260922.md"} else ""
        return "报告/图册入口", report_purpose(path.name) + state
    if rel.startswith("src/") or rel.startswith("tools/"):
        return "程序代码", code_purpose(path)
    if rel.startswith("configs/"):
        return "配置", "此 YAML 固化 `%s` 的参数和规则；看 `version`、日期范围、预算/阈值及引用路径。" % path.stem
    if rel.startswith("assets/"):
        asset = asset_purpose(path)
        if asset:
            return "第一轮发布模型资产", asset
        if path.suffix in {".pt", ".pth", ".bin"}:
            return "模型权重（二进制）", "用 PyTorch/对应程序加载；不要用记事本。它是第一轮已发布资产，非 V3 新训练结果。"
        return "资产", "看清单中的指纹和用途；模板用文本/表格打开，训练记录 CSV 看轮次和 loss。"
    if rel.startswith("研究材料/"):
        return "历史研究材料", "作为背景读取；不等同于本轮完成状态。"
    if rel.startswith("data/synthetic_generated/"):
        return "历史生成资料", "用于追溯旧生成/审计，不是本轮新增或训练输入结论。"
    if path.name == ".gitignore":
        return "Git 忽略规则", "规定不提交缓存、虚拟环境、Python 编译文件和普通日志；特别将本地完整 `历史用途输入.csv` 留在本机，远程改交两份可重组分片。"
    return "项目文件", "按文件格式打开并结合 README 或同目录清单阅读。"

def manifest_purpose(name: str) -> str:
    return {
        "样本处理清单.json": "780 户的总追溯表：样本号、家族、来源目录、指纹、日期、账务核对、用途和 V3 分组。",
        "样本分组名单.csv": "按样本列出学习/区间校准/开发测试组及家族；先按 sample_id 找户。",
        "图册CSV对应.csv": "把每个样本号定位到图册页、页内位置和权威日余额 CSV。",
        "按走势查看图册.csv": "按实测走势标签筛选样本，并定位图册页和 CSV。",
        "实测走势与曲线重复.json": "全量实测走势标签与两户曲线重复线索的证据结果。",
        "近复制证据.json": "17 对近复制线索的相似证据，供复核而非自动删样。",
        "近复制与曲线复核处置.json": "17 对近复制和两户曲线重复的最终保留/家族处置依据。",
        "资产复制清单.json": "迁入资产的旧/新路径、指纹、依赖、用途和基础检查记录。",
        "历史用途输入.csv": "本地完整的逐笔历史用途表；人工可查看，但不能把预测日后的整段数据塞入模型。",
        "历史用途输入_第1部分.csv": "完整用途表的交错行第 1 分片；必须和第 2 分片用重组工具交错恢复。",
        "历史用途输入_第2部分.csv": "完整用途表的交错行第 2 分片；不能直接接在第 1 分片后。",
        "不用或待查样本清单.json": "保留给异常或不用样本的入口；当前为空数组。",
    }.get(name, "来源/历史清单：用于追溯本轮引用的样本、模型或分组来源；先看顶层版本、路径和样本列表。")

def report_purpose(name: str) -> str:
    return {
        "03_样本整理收尾_20260922.md": "当前权威收尾结论：780 条样本已整理，未训练；并记录近复制最终修正和用途表重组方法。",
        "02_补齐完成记录_20260922.md": "补齐阶段记录；其中把 17 对近复制线索先写作‘2 户待人工复核’，后由收尾记录给出最终处置。",
        "第三轮V3.0样本整理交付报告.md": "阶段交付概览；它把最终状态指向 02 记录，因此当前结论应再以 03 收尾记录为准。",
        "全样本余额图册入口.md": "98 页余额图册的样本导航入口。",
        "01_整理.md": "早期整理阶段的自动汇总记录。",
        "样本整理复核与下一步_20260922.md": "复核过程和下一步建议，不能替代收尾结论。",
    }.get(name, "规则或背景报告；用于了解当时建议和过程，不覆盖当前收尾结论。")

def asset_purpose(path: Path) -> str | None:
    rel = path.relative_to(ROOT).as_posix()
    exact = {
        "assets/first_round_published_model/c2f_final_fit_report.json": "第一轮最终拟合通过报告；看状态、三份模型数量、时间角色、输入/输出指纹及是否打开最终测试。",
        "assets/first_round_published_model/final_fit_conformal_adjustments.csv": "10/30 日预测区间的保序校准调整量；看期限、有限样本排名和调整值，数值用于修正预测上下范围。",
        "assets/first_round_published_model/final_quantile_lstm_model_manifest.json": "三份固定种子 LSTM 权重、输入特征、缩放器和训练参数的对应清单；`feature_scales`/`target_scales` 是把不同量纲数值缩放到可训练尺度的系数。",
        "assets/first_round_published_model/output_sha256.json": "上述模型包输出文件的 SHA-256 指纹表，用于核对文件是否被改动。",
        "assets/templates/company_product_template.csv": "企业产品信息录入模板；看公司编号、资产负债、收入利润、现金和主产品字段。",
        "assets/templates/entity_expansion_template.json": "扩展企业情景的 JSON 模板；看样本身份、业务设定和参数位置。",
        "assets/templates/sample_registration_template.yaml": "新样本登记 YAML 模板；看来源、家族、角色与路径字段。",
    }
    if rel in exact:
        return exact[rel]
    if path.name == "training_loss_history.csv":
        return "该随机种子每一轮的训练误差和校准误差记录；看 epoch、两类 loss 与 improved，误差下降/标记改进表示该轮被保留为更优候选。"
    return None

def code_purpose(path: Path) -> str:
    """按实际模块名和已抽取的函数入口给出保守、可核查的说明。"""
    name = path.stem
    exact = {
        "reassemble_usage_csv": "按交错行重组两份用途 CSV，并可做完整表校验；本轮不需要运行。",
        "split_usage_csv": "把用途输入表拆为可交付分片的工具；与重组工具配套。",
        "render_batch_atlas": "从日余额 CSV 批量绘制图册页的工具。",
        "curve_repetition_check": "检查余额曲线的重复或相似线索的工具。",
        "v3_prepare_samples": "V3 样本整理与清单生成的批处理入口。",
        "v3_finalize_checks": "V3 收尾核对的批处理入口。",
        "v3_closeout": "V3 整理交付收尾的工具。",
        "v3_asset_inventory": "记录迁入资产、路径和指纹的工具。",
        "runtime_bank_calendar": "银行工作日/日历运行时适配模块。",
        "runtime_daily_aggregation": "把交易归并为日粒度数据的运行时模块。",
        "runtime_field_mapping": "统一字段名映射的运行时模块。",
        "runtime_transaction_normalizer": "交易记录规范化的运行时模块。",
        "models": "第二轮模型定义模块；只表示既有代码，不表示 V3 已训练。",
        "scoring": "第二轮预测评分计算模块。",
        "cli": "syn_b1 的命令行入口参数与调度。",
        "scenario_runner": "按情景请求构建一次企业运行的主调度。",
        "formal_generator": "正式样本导出的生成主模块。",
        "formal_export": "把生成结果导出为正式文件的模块。",
        "formal_history": "正式样本历史记录组织模块。",
        "formal_profile": "正式企业情景画像模块。",
        "profile_document": "企业画像文档读写与中文注释渲染模块。",
        "profile_workspace": "企业画像工作目录组织模块。",
        "teaching_comment_audit": "扫描代码/文本中的教学注释并生成审计结果。",
        "statement_rollforward": "滚动生成损益、现金流和资产负债表的模块。",
        "ledger_rollforward": "逐笔交易后的账本和余额滚动模块。",
        "final_cash_reconciliation": "最终现金/账务核对模块。",
        "transaction_planner": "把业务与资金事件安排为带日期的交易候选项。",
        "tax_routing": "税款计算、付款时间和现金流路由模块。",
        "tax_projection": "替换税款假设后的报表/现金预测模块。",
    }
    if name in exact:
        return exact[name]
    if name.startswith("pretrain_"):
        return "训练前数据、基线、特征、LSTM/XGBoost 或审计的既有模块；由文件中的入口函数和配置决定具体步骤，V3 尚未运行训练。"
    if name.startswith("post_t6_"):
        return "第一轮 T6 后的评测、区间、拟合或方案比较模块；文件名保留其阶段编号，当前只作历史资产阅读。"
    if name.startswith("round2_v4_"):
        return "第二轮 V4 的编排或去重工具；用于历史样本处理留痕。"
    if name.startswith("render_"):
        return "根据已有 CSV/清单渲染图或摘要的工具。"
    if name.startswith("audit_"):
        return "读取已有资料并输出审计/走势检查结果的工具。"
    if name.endswith("calendar_adapter"):
        return "把日历规则接入企业运行过程的模块。"
    if name in {"__init__", "__main__"}:
        return "Python 包初始化或模块启动入口；需结合相邻 README/CLI 阅读。"
    return "既有 syn_b1 生成器组成模块；请先看文件首部、导入对象和列出的函数，职责未在文件名中直接说明时不要仅凭名称推断。"

def write_samples(sample_files: list[Path]):
    samples = sorted((ROOT / "data/samples").iterdir())
    manifest = json.loads((ROOT / "data/manifests/样本处理清单.json").read_text(encoding="utf-8"))
    by_folder = {row.get("v3_folder", ""): row for row in manifest}
    lines = ["# 项目文件逐项索引：样本", "", "本页逐目录列出当前实际存在的样本文件。每户 7 个同构文件；共 %d 户、%d 个文件。文件链接均指向本机绝对路径。" % (len(samples), len(sample_files)), "", "## 共用读法（适用于每一户）", ""]
    for name, explanation in SAMPLE_FILE_HELP.items():
        lines.append(f"- `{name}`：{explanation}")
    lines += ["", "完整历史样本可以人工逐户查看；后续模型读取时只能使用预测日当时已经知道的历史段，不能把完整未来段整体输入模型。", "", "## 逐户实际文件", ""]
    for folder in samples:
        files = sorted(p for p in folder.iterdir() if p.is_file())
        row = by_folder.get(folder.relative_to(ROOT).as_posix())
        if row:
            sample_id = row.get("sample_id", "清单缺少样本编号")
            group = row.get("v3_role", "清单缺少分组")
            family = row.get("family_id", "清单缺少家族")
            title = f"### `{sample_id}`｜{group}组｜家族 `{family}`"
        else:
            title = f"### `待查样本`｜目录 `{folder.name}`（样本处理清单未找到对应项）"
        lines += [title, "", f"- 资料目录：`{folder.name}`；共 {len(files)} 个实际文件。"]
        for p in files:
            lines.append(f"- {link(p, p.name)}")
        lines.append("")
    (REPORTS / "项目文件逐项索引_样本.md").write_text("\n".join(lines), encoding="utf-8")

def write_other(files: list[Path]):
    lines = ["# 项目文件逐项索引：程序与其他材料", "", "本页逐项列出非 `data/samples/` 的正常项目文件（不含本次新导览文件，以避免自引用）。链接指向本机绝对路径。", ""]
    for path in files:
        category, purpose = classify(path)
        rel = path.relative_to(ROOT).as_posix()
        lines += [f"## `{rel}`", "", f"- 文件：{link(path, rel)}", f"- 类别与意义：{category}；{purpose}", f"- 怎么看：{text_opening(path)}", ""]
    (REPORTS / "项目文件逐项索引_其他材料.md").write_text("\n".join(lines), encoding="utf-8")

def write_main(files: list[Path], sample_files: list[Path]):
    dirs = Counter(p.relative_to(ROOT).parts[0] for p in files)
    lines = ["# 项目文件夹与逐文件阅读指南", "", "这是一份**导航说明**，不是新的当前状态来源。当前状态、样本数量和最终复核结论只以 [03_样本整理收尾_20260922.md](" + (REPORTS / "03_样本整理收尾_20260922.md").resolve().as_posix() + ") 为准。", "", "## 最短核对路线", "", "1. 点击 [样本整理收尾记录](" + (REPORTS / "03_样本整理收尾_20260922.md").resolve().as_posix() + ") 预览 Markdown，先确认“整理完成、未训练”。", "2. 点击 [全样本余额图册入口](" + (REPORTS / "全样本余额图册入口.md").resolve().as_posix() + ")，按页看余额曲线。", "3. 用 Excel/WPS 打开 [样本分组名单](" + (ROOT / "data/manifests/样本分组名单.csv").resolve().as_posix() + ")，按样本编号挑一户，再到样本索引打开该户日余额。", "4. 用同户流水和用途备注核对日期、借/贷金额和用途；用途资料仅限预测日前已知内容。", "", "## 怎么打开，先不要运行", "", "- `.md`：直接点击链接预览；`.csv`：用 Excel 或 WPS；`.json`/`.yaml`：用记事本或代码编辑器，搜索 `status`、`version`、`path`。", "- `.py` 是程序源码，`.pt` 是模型二进制权重；本轮只阅读，不要双击运行，更不要用记事本打开权重。", "", "## 实际目录树（重复样本折叠）", "", "```text", "V3.0/", "├─ README.md                         项目入口与当前进度链接", "├─ reports/                          当前结论、图册入口、阶段与背景报告", "├─ data/manifests/                   样本分组、来源、图册、相似性、用途清单", "├─ data/samples/                     780 户 × 7 个实际文件（见样本逐项索引）", "├─ data/synthetic_generated/         实际存在的历史生成/审计资料", "├─ assets/                           旧模型、模板和可复用资产", "├─ configs/                          生成与规则配置", "├─ src/、tools/                      既有程序和批处理工具", "├─ 研究材料/                         实际存在的历史背景材料", "└─ runs/logs/                        追加式执行记录", "```", "", "## 建议阅读顺序", "", "1. **当前结论**：[README.md](" + (ROOT / "README.md").resolve().as_posix() + ") 与上面的收尾记录。", "2. **图册**：全样本余额图册入口，以及 `data/manifests/图册CSV对应.csv`、`按走势查看图册.csv`。图是余额 CSV 的可视化，曲线标签只是浏览线索。", "3. **样本分组**：`样本分组名单.csv` 的 sample_id 是样本编号，v3_role 是学习/区间校准/开发测试分组；`样本处理清单.json` 有来源路径与处理留痕。", "4. **样本原件**：每户先日余额，再流水、备注、配置、质量报告和来源映射。完整链接见 [样本逐项索引](" + (REPORTS / "项目文件逐项索引_样本.md").resolve().as_posix() + ")。", "5. **用途资料**：`历史用途输入.csv` 是本地完整表（被 Git 忽略但重要）；远程交付为第 1、2 分片。两片按交错行重组，用 `tools/reassemble_usage_csv.py`，不能直接拼接。", "6. **资产、代码、历史报告、Git**：逐项入口见 [程序与其他材料索引](" + (REPORTS / "项目文件逐项索引_其他材料.md").resolve().as_posix() + ")。权重 `.pt` 必须用对应程序加载。", "", "## 哪些报告是当前或历史记录", "", "- **当前权威**：`03_样本整理收尾_20260922.md`；README 只作入口。", "- **已被后续收尾覆盖**：`02_补齐完成记录_20260922.md` 仍将两户曲线重复列为待人工复核；收尾记录已给出两户均保留的结论。它也明确承认早期的‘17 对都在开发测试组’为组别误报。", "- **阶段交付记录**：`第三轮V3.0样本整理交付报告.md` 将最终状态指向 02，因此应继续以 03 收尾记录复核。", "- **历史/背景**：`背景_*`、`01_整理.md`、`样本整理复核与下一步_20260922.md` 和极简复审报告用于了解过程或规则，不覆盖收尾记录。", "", "## 计数与排除范围", "", f"本次机械枚举了 {len(files)} 个正常项目文件，其中样本文件 {len(sample_files)} 个（780 户）。按一级目录计数：" + "、".join(f"{key} {value}" for key, value in sorted(dirs.items())) + "。", "", "逐文件目录排除 `.git/` 内部对象、`__pycache__/`、`.pytest_cache/`、`.venv/`、`.pyc` 与普通运行日志缓存；这些不是用户阅读的项目材料。`runs/logs/*.md` 保留。三份本次新导览文件不在各自逐项名单中，避免索引自引用。", "", "## 各类文件的正常与需关注情况", "", "- CSV：用 Excel、WPS 或文本编辑器打开；确认表头、日期格式、金额单位和空值含义。日余额应能由流水汇总解释。", "- JSON/YAML：用代码编辑器折叠查看；重点是版本、样本标识、路径、指纹、规则和检查状态。路径不存在、校验值不一致、失败项未说明时需关注。", "- PNG 图页：看标题中的样本和日期范围，并回到对应 CSV；图不能代替原始数值。", "- Markdown：优先分清当前报告与历史报告的日期和纠正说明。", "- `.pt` 权重：二进制文件，不能记事本阅读；它们属于第一轮发布资产，V3 尚未训练。", ""]
    (REPORTS / "项目文件夹与逐文件阅读指南.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    files = list(project_files())
    sample_files = [p for p in files if p.relative_to(ROOT).as_posix().startswith("data/samples/")]
    write_samples(sample_files)
    write_other([p for p in files if p not in sample_files])
    write_main(files, sample_files)
    print(f"normal={len(files)} samples={len(sample_files)}")

if __name__ == "__main__":
    main()
