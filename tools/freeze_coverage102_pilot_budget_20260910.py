# 本入口只冻结首批六户预算并调用原中央内存预演，不提供正式导出或训练动作。
"""首批六户：版本绑定、完整参数、一次预算首演与事实预算留存。"""
# 导入命令行工具以明确区分冻结和一次预算预演。
import argparse
# 导入精确复制以隔离共用模板与各户独有参数。
import copy
# 导入数据类转换以保存中央计算对象。
from dataclasses import asdict, is_dataclass
# 导入日期以保存固定经营观察期。
from datetime import date, timedelta
# 导入十进制金额以避免浮点误差。
from decimal import Decimal
# 导入枚举以稳定保存中央分类。
from enum import Enum
# 导入哈希以绑定输入和执行代码。
import hashlib
# 导入JSON以保存机器事实。
import json
# 导入路径以限制全部新材料的保存范围。
from pathlib import Path
# 导入复制工具以只保存一份共享运行环境。
import shutil
# 导入系统以禁止旧环境写入字节码。
import sys
# 在导入任何项目模块前禁用缓存写入。
sys.dont_write_bytecode = True
# 导入YAML以展开完整运行参数。
import yaml

# 固定项目根目录，不受调用终端影响。
ROOT = Path(__file__).resolve().parents[1]
# 将全部新材料统一放入已批准的生产根。
PACK = ROOT / "data/synthetic_production/coverage102_v1"
# 保存本次预算及共用规则的独立版本。
CONTROL = PACK / "公共规则与版本/首批六份预算_20260910_v1"
# 采用已有代表实际调用的中央环境，不猜测工作目录代码版本。
SOURCE = ROOT / "docs/C3_D参数与预算定稿_20260909_v1"
# 共用源码只在整个新数据集中保存一次。
RUNTIME = PACK / "公共规则与版本/生成器_c3d_20260909"
# 只继承已经明示的通用参数结构，经营节点由新预算替换。
BASE = ROOT / "docs/有限批量执行规则_20260909_v1/待执行完整参数/G02_01.yaml"
# 固定六个获准位置，任何其他格不在本入口范围内。
SLOTS = {(branch, slot) for branch in ("A1-中-充裕", "B3-中-紧张", "C4-中-紧张") for slot in ("T01", "T02")}

# 把中央对象稳定转为JSON，不改动任何数值。
def serial(value):
    # 展开数据类的原始字段。
    if is_dataclass(value):
        # 递归保留所有字段。
        return serial(asdict(value))
    # 将枚举保存成已定义值。
    if isinstance(value, Enum):
        # 返回枚举数据。
        return value.value
    # 日期使用标准格式。
    if isinstance(value, date):
        # 保留真实日期。
        return value.isoformat()
    # 精确金额保存为十进制字符串。
    if isinstance(value, Decimal):
        # 不经浮点转换。
        return str(value)
    # 方法仅保存身份，行为由源码指纹绑定。
    if callable(value):
        # 不保存含随机地址的字符串。
        return {"callable_reference": f"{value.__module__}.{value.__qualname__}"}
    # 递归处理键值结构。
    if isinstance(value, dict):
        # 字典键统一为文本。
        return {str(k): serial(v) for k, v in value.items()}
    # 递归处理中央列表和元组。
    if isinstance(value, (list, tuple)):
        # 保留原顺序。
        return [serial(v) for v in value]
    # 其他基础类型原样保存。
    return value

# 计算原字节指纹。
def sha(path):
    # 同时覆盖金额、日期和换行变化。
    return hashlib.sha256(path.read_bytes()).hexdigest()

# 保存新文件，禁止覆盖任何既有版本。
def write_once(path, value):
    # 检查解析后的写入路径仍在统一新根内。
    if not path.resolve().is_relative_to(PACK.resolve()):
        # 拒绝路径越界。
        raise ValueError("写入超出统一生产根")
    # 创建必要的父目录。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 使用独占创建保证重跑不能静默覆盖。
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        # 文本直接写入，结构化事实稳定序列化。
        handle.write(value if isinstance(value, str) else json.dumps(serial(value), ensure_ascii=False, indent=2) + "\n")

# 读取已保存的JSON。
def read(path):
    # 兼容已有UTF-8标记而不修改源文件。
    return json.loads(path.read_text(encoding="utf-8-sig"))

# 根据冻结来源复制并逐项核对唯一共享运行环境。
def snapshot():
    # 读取旧来源的执行文件清单。
    binding = read(SOURCE / "冻结输入SHA256.json")["runtime_files"]
    # 逐个检查源文件没有变化。
    for name, expected in binding.items():
        # 定位只读源文件。
        source = SOURCE / "运行环境" / name
        # 源文件错误必须在复制前停止。
        if sha(source) != expected:
            # 不接受漂移的生成器。
            raise ValueError(f"原运行环境变化：{name}")
        # 定位共享副本。
        target = RUNTIME / name
        # 已有相同内容可复用，不写第二份。
        if target.exists():
            # 已有不同内容不得覆盖。
            if sha(target) != expected:
                # 明确环境冲突。
                raise ValueError(f"共享环境冲突：{name}")
        # 缺失副本才复制。
        else:
            # 建立对应目录。
            target.parent.mkdir(parents=True, exist_ok=True)
            # 保持文件原始内容。
            shutil.copy2(source, target)
    # 返回逐文件绑定。
    return binding

# 从显式经营设计展开中央可解析的完整参数。
def profile_from(item):
    # 读取旧通用结构，不采用其订单或资本。
    p = yaml.safe_load(BASE.read_text(encoding="utf-8-sig"))
    # 身份、期间和种子均由本户设计给定。
    p.update({k: item[k] for k in ("sample_id", "run_id", "random_seed")})
    # 采用本户完整经营期间。
    p.update(item["period"])
    # 首次实缴由本户预算确定。
    p["initial_capital"]["amount_cny"] = item["capital"]["amount_cny"]
    # 后续注资必须显式声明。
    p["additional_capital_plans"] = item["capital"]["additional_capital_plans"]
    # 借款必须显式声明。
    p["loan_contracts"] = item["capital"]["loan_contracts"]
    # 拆分安排也按本户冻结值保存。
    p["transactions"] = item["transactions"]
    # 逐项替换全部与本户经营有关的收付安排。
    for key in ("procurement_inertia", "collection_schedule", "supplier_payment_schedule", "payroll_payment_schedule", "rent_and_utilities_payment_schedule", "other_operating_expense_payment_schedule"):
        # 防止共享结构被其他户修改。
        p["operating"][key] = copy.deepcopy(item[key])
    # 产能属于节点能力，不能放入设备合同的未知字段。
    p["operating"]["fixed_assets"] = [{k: v for k, v in asset.items() if k != "monthly_units"} for asset in item["assets"]]
    # 全部月份、理由和产能显式展开。
    p["operating"]["business_nodes"] = {"version": "monthly_business_nodes_v1", "nodes": item["nodes"], "asset_capacities": [{"event_id": a["event_id"], "monthly_units": a["monthly_units"]} for a in item["assets"]]}
    # 返回完整新企业配置。
    return p

# 冻结六份参数及日期计划，此动作不调用中央计算。
def freeze():
    # 读取Sol准备的明确经营设计。
    design = read(CONTROL / "六户经营设计.json")
    # 取得六户方案。
    items = design["scenarios"]
    # 验证位置既不缺失也不越界。
    if len(items) != 6 or {(x["combination_id"], x["slot"]) for x in items} != SLOTS:
        # 禁止偷偷新增或替换组合。
        raise ValueError("必须恰好是已批准的六个位置")
    # 身份及家族在生成前必须唯一明确。
    if len({x["sample_id"] for x in items}) != 6 or len({x["family_id"] for x in items}) != 6:
        # 本六户设计要求六个独立预算，不能以重复编号冒充。
        raise ValueError("六户身份或事前家族重复")
    # 先绑定共用执行环境。
    runtime = snapshot()
    # 插入共享源码供静态解析，不运行企业。
    sys.path.insert(0, str(RUNTIME / "src"))
    # 只加载中央参数校验器。
    from syn_b1.formal_profile import load_profile, resolve_profile
    # 全部六份先通过中央静态解析，防止写了一半才发现漏月或产能越界。
    prepared = {item["sample_id"]: profile_from(item) for item in items}
    # 此接口只解析合同，不生成任何计划或逐笔数据。
    for profile in prepared.values():
        # 六份全部合法后才开始写逐户冻结文件。
        resolve_profile(profile)
    # 准备逐户索引。
    rows = []
    # 逐户展开与验证参数。
    for item in items:
        # 经营日必须紧接或晚于筹建结束，不能事后挪到恢复日期。
        assert date.fromisoformat(item["setup_end"]) < date.fromisoformat(item["operating_start"])
        # 固定资金观察点至少两个，不允许事后从全期挑点。
        assert len(set(item["funds_check"]["observation_dates"])) >= 2
        # 每个观察点都必须在经营期且计划未来30自然日完整。
        assert all(date.fromisoformat(item["operating_start"]) <= date.fromisoformat(d) <= date.fromisoformat(item["period"]["end_date"]) - timedelta(days=30) for d in item["funds_check"]["observation_dates"])
        # 参数、画像、预算最终都留在本户唯一主目录。
        folder = PACK / "样本/训练" / item["sample_id"] / item["run_id"]
        # 构造完整参数。
        p = prepared[item["sample_id"]]
        # 输出完整YAML，不加入中央不认识的经营描述字段。
        text = "# 本户完整冻结参数；改动金额或日期须另版登记，资金不足不自动救助。\n" + yaml.safe_dump(p, allow_unicode=True, sort_keys=False)
        # 将精确配置只写一次。
        write_once(folder / "完整参数.yaml", text)
        # 将真实经营解释单独保存为侧车。
        write_once(folder / "企业设定.json", item)
        # 使用同版本中央解析器核合同与能力。
        resolved = load_profile(folder / "完整参数.yaml")
        # 固定日历可行性只在计划期间检查，不声称已有真实窗口。
        cal = resolved.request.runtime_settings.bank_calendar
        # 逐自然日验证日历版本支持完整期间。
        dates = [date.fromisoformat(p["start_date"]) + timedelta(days=i) for i in range((date.fromisoformat(p["end_date"]) - date.fromisoformat(p["start_date"])).days + 1)]
        # 保存计划银行工作日，后续P3b仍须核真实未来。
        bankdays = [d.isoformat() for d in dates if cal.is_bank_workday(d)]
        # 每户保存计划日历而非伪造余额。
        write_once(folder / "计划日历.json", {"bank_workdays": bankdays, "calendar": p["calendar"], "actual_window_eligibility": "not_checked"})
        # 写入可读画像，明确全部为事前合成设定。
        write_once(folder / "企业画像.md", f"# {item['sample_id']} 企业画像\n\n组合：{item['combination_id']}；位置：{item['slot']}；家族：{item['family_id']}；版本：{item['run_id']}。\n\n{item['short_profile_cn']}\n\n本画像为事前合成设定，不是现实企业实际流水；预算首演结果另列，不据图改写故事。\n\n{item['business_contract_cn']}\n\n独立预算依据：{item['independence_reason']}\n")
        # 绑定所有事前材料，禁止预演后修改预算。
        files = {f.relative_to(PACK).as_posix(): sha(f) for f in folder.iterdir() if f.is_file()}
        # 登记用途、位置与冻结指纹。
        rows.append({"combination_id": item["combination_id"], "slot": item["slot"], "sample_id": item["sample_id"], "run_id": item["run_id"], "family_id": item["family_id"], "purpose": "train", "folder": folder.relative_to(PACK).as_posix(), "files": files})
    # 保存共同参数的明确来源，不把复制当作隐含默认。
    base = yaml.safe_load(BASE.read_text(encoding="utf-8-sig"))
    # 公共税费为已用的合成简化值，不宣称真实法定税费。
    common = {"source": BASE.relative_to(ROOT).as_posix(), "source_sha256": sha(BASE), "calendar": base["calendar"], "account": base["account"], "tax": base["tax"], "production_cycle_days": base["operating"]["production_cycle_days"], "inventory_target_days": base["operating"]["inventory_target_days"], "note": "税费为既有研究简化设定；每户YAML全部展开，节点模式采购按耗用与库存差计算，procurement_inertia不作为有效变化旋钮。"}
    # 保留公共参数供Sol逐项核对。
    write_once(CONTROL / "共用参数与能力说明.json", common)
    # 形成一次预算首演前的不可覆盖指纹清单。
    tool_files = [Path(__file__), ROOT / "tools/report_coverage102_pilot_budget_20260910.py"]
    # 自有检查入口也保存一次源码，不仅保存一个无法找回代码的指纹。
    for tool in tool_files:
        # 快照仅供版本恢复，后续正式入口仍按完整YAML调用共享中央。
        write_once(CONTROL / "预算工具源码" / tool.name, tool.read_text(encoding="utf-8"))
    # 输入、中央源码、两入口和依赖一起绑定，避免静默漂移。
    write_once(CONTROL / "生成前冻结清单.json", {"design_sha256": sha(CONTROL / "六户经营设计.json"), "builder_sha256": sha(Path(__file__)), "tool_files": {p.name: sha(p) for p in tool_files}, "python_version": sys.version, "pyyaml_version": yaml.__version__, "runtime_files": runtime, "rows": rows, "formal_generated": 0, "training_runs": 0})
    # 返回简短状态，不把六份YAML灌入对话。
    print(json.dumps({"frozen": len(rows), "preview_calls": 0}, ensure_ascii=False))

# 一次预算首演，完整留存中央结果且不正式导出流水。
def preview(sample_id):
    # 读取首演前已冻结的全部输入绑定。
    binding = read(CONTROL / "生成前冻结清单.json")
    # 执行源码变化时必须先留技术修正记录，不能静默换代码。
    if sha(Path(__file__)) != binding["builder_sha256"]:
        # 保护首演执行器。
        raise ValueError("预算执行器指纹变化")
    # 核对设计没有在预演间变化。
    if sha(CONTROL / "六户经营设计.json") != binding["design_sha256"]:
        # 阻止临时调参。
        raise ValueError("六户设计指纹变化")
    # 核对共享执行版本。
    for name, expected in binding["runtime_files"].items():
        # 任一变化都阻断本次调用。
        if sha(RUNTIME / name) != expected:
            # 不自动采用工作目录的新代码。
            raise ValueError(f"运行环境变化：{name}")
    # 身份必须来自六个已冻结的位置。
    row = next(r for r in binding["rows"] if r["sample_id"] == sample_id)
    # 检查本户事前文件保持原始字节。
    for name, expected in row["files"].items():
        # 参数及侧车不能在看过图后修改。
        if sha(PACK / name) != expected:
            # 说明受影响的文件。
            raise ValueError(f"本户冻结输入变化：{name}")
    # 每户首演事实位于用途隔离的检查目录。
    dest = PACK / "检查与复现/训练" / sample_id / row["run_id"] / "预算首演"
    # 调用前先落开始记录，异常也不能消失。
    write_once(dest / "调用开始.json", {"sample_id": sample_id, "attempt": 1, "kind": "budget_preview_only", "input_sha256": sha(PACK / row["folder"] / "完整参数.yaml")})
    # 将唯一共享执行代码置于搜索首位。
    sys.path.insert(0, str(RUNTIME / "src"))
    # 只导入内存预演，不导入正式导出函数。
    from syn_b1.formal_generator import preview_from_profile
    # 捕获实际异常并留存，禁止自动重抽。
    try:
        # 本户只调用这一次中央预算预演。
        resolved, run = preview_from_profile(PACK / row["folder"] / "完整参数.yaml")
        # 保存完整结果，含停止前缀与期外应收应付，金额未重新生成。
        write_once(dest / "中央完整事实.json", {"profile": serial(resolved), "run": serial(run)})
        # 保存真实完成状态，不能将未来计划充当已执行流水。
        write_once(dest / "调用结束.json", {"complete": run.ledger.is_complete, "ledger_days": len(run.ledger.daily_rows), "transactions": len(run.ledger.transactions), "stop": serial(run.ledger.fact_sheet), "formal_exports": 0})
        # 输出便于模型复核的摘要。
        print(json.dumps({"sample_id": sample_id, "complete": run.ledger.is_complete, "days": len(run.ledger.daily_rows), "transactions": len(run.ledger.transactions)}, ensure_ascii=False))
    # 程序或输入异常单独保留，不解释成真实经营失败。
    except Exception as error:
        # 记录实际失败类型和说明。
        write_once(dest / "技术异常.json", {"type": type(error).__name__, "message": str(error)})
        # 向调用者报告失败，不能继续假装预算通过。
        raise

# 显式入口只有冻结和预算预演两个动作。
if __name__ == "__main__":
    # 定义用户可以执行的有限动作。
    parser = argparse.ArgumentParser(description="首批六户预算冻结与一次首演，不正式生成")
    # 冻结动作只解析参数和保存计划。
    parser.add_argument("--freeze", action="store_true")
    # 预演动作必须明确给出已冻结企业身份。
    parser.add_argument("--preview")
    # 读取命令行。
    args = parser.parse_args()
    # 不允许一次同时选择两个阶段或未选择阶段。
    if bool(args.freeze) == bool(args.preview):
        # 说明正确阶段选择。
        parser.error("只能选择--freeze或--preview 企业编号")
    # 冻结阶段不生成流水。
    if args.freeze:
        # 展开并绑定已审六户设计。
        freeze()
    # 另一阶段只做单户预算首演。
    else:
        # 严格读取冻结身份。
        preview(args.preview)
