"""第三轮开发样本的只读迁移、核验、汇总与余额图册。

只使用来源 split manifest 中明确列出的企业；不生成、训练或评分。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/来源_round2_v4_split_manifest.json"
SAMPLES = ROOT / "data/samples"
OUT = ROOT / "data/manifests"
REPORTS = ROOT / "reports"
LOG = ROOT / "runs/logs/执行记录.md"
OLD_ROOT = Path("D:/MiniProj/深度学习制作企业产品预测系统")
COPY_NAMES = (
    "account_daily_total.csv", "transactions_total.csv", "cash_flow_review_notes.csv",
    "generation_manifest.json", "quality_report.json", "scenario_profile.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def money(value: str) -> float:
    return float(value or 0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def legacy_rows() -> list[dict]:
    """从磁盘精简入口和已封存 84 户反查真实 CSV，保留其旧使用历史。"""
    rows = []
    stage = OLD_ROOT / "交付整理/20260914_磁盘精简整理/第一阶段_260户_156训练_52验证_52旧最终"
    for label, role in (("156训练_目录链接", "fit"), ("52验证_目录链接", "calibration")):
        for folder in sorted((stage / label).iterdir()):
            daily = folder / "account_daily_total.csv"
            if daily.is_file():
                sid = folder.name
                rows.append({"sample_id": sid, "directory": str(folder.resolve()), "family_id": f"F-ROUND1-{sid}",
                             "shape": "旧来源标签不参与分类", "role": role, "daily_sha256": digest(daily), "usage_history": "第一轮旧开发"})
    index = stage / "52旧最终_仅元数据索引_不读取原件/52旧最终_元数据身份索引.csv"
    for item in read_csv(index):
        folder = Path(item["real_source_directory"])
        daily = folder / "account_daily_total.csv"
        if daily.is_file():
            sid = item["sample_id"]
            rows.append({"sample_id": sid, "directory": str(folder), "family_id": f"F-ROUND1-{sid}",
                         "shape": "旧来源标签不参与分类", "role": "development", "daily_sha256": digest(daily), "usage_history": "第一轮旧最终；本轮仅作开发准备"})
    sealed = json.loads((ROOT / "data/manifests/来源_final_population_manifest.json").read_text(encoding="utf-8-sig"))
    for item in sealed["rows"]:
        folder = Path(item["mechanical_receipt"]["directory"])
        daily = folder / "account_daily_total.csv"
        if daily.is_file():
            sid = item["sample_id"]
            rows.append({"sample_id": sid, "directory": str(folder), "family_id": f"F-ROUND2-FINAL-{sid}",
                         "shape": item.get("planned_shape", "旧来源标签不参与分类"), "role": "development",
                         "daily_sha256": digest(daily), "usage_history": "第二轮最终 84 户；本轮仅作开发准备"})
    return rows


def copy_authoritative(row: dict, seen: dict[str, str]) -> tuple[Path | None, str]:
    source = Path(row["directory"])
    daily = source / "account_daily_total.csv"
    if not daily.is_file():
        return None, "来源目录或日余额缺失"
    actual = digest(daily)
    if actual != row["daily_sha256"]:
        return None, "日余额指纹与来源清单不一致"
    if actual in seen:
        return SAMPLES / seen[actual], "复用同一权威资料"
    folder = f"source_{actual[:16]}"
    target = SAMPLES / folder
    target.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in COPY_NAMES:
        item = source / name
        if item.is_file():
            dest = target / name
            if not dest.exists():
                shutil.copy2(item, dest)
            if digest(item) != digest(dest):
                raise RuntimeError(f"复制校验失败：{row['sample_id']} {name}")
        else:
            missing.append(name)
    (target / "来源映射.json").write_text(json.dumps({
        "source_directory": str(source), "sample_ids": [row["sample_id"]],
        "daily_sha256": actual, "missing_expected_files": missing,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    seen[actual] = folder
    return target, "新复制"


def metrics(folder: Path) -> dict:
    daily = read_csv(folder / "account_daily_total.csv")
    txns = read_csv(folder / "transactions_total.csv")
    notes_path = folder / "cash_flow_review_notes.csv"
    notes = read_csv(notes_path) if notes_path.exists() else []
    balances = [money(x["ending_balance_cny"]) for x in daily]
    inflow = sum(money(x["inflow_cny"]) for x in daily)
    outflow = sum(money(x["outflow_cny"]) for x in daily)
    first, last = balances[0], balances[-1]
    net = inflow - outflow
    tolerance = 0.02
    # 首日通常含期初注资；首末日余额变化只应对应首日之后的净收支。
    net_after_first_day = sum(money(x["inflow_cny"]) - money(x["outflow_cny"]) for x in daily[1:])
    daily_ok = abs((last - first) - net_after_first_day) <= tolerance
    txn_net = sum(money(x["credit_cny"]) - money(x["debit_cny"]) for x in txns)
    txn_ok = abs(net - txn_net) <= tolerance
    ids = {x["transaction_id"]: x for x in txns}
    matched = 0
    categories = Counter()
    unknown = 0
    for note in notes:
        txn = ids.get(note.get("transaction_id", ""))
        direction_ok = txn and ((note.get("direction_cn") == "流入" and money(txn["credit_cny"]) == money(note["amount_cny"])) or (note.get("direction_cn") == "流出" and money(txn["debit_cny"]) == money(note["amount_cny"])))
        date_ok = txn and txn["booking_datetime"][:10] == note.get("booking_datetime", "")[:10]
        if direction_ok and date_ok:
            matched += 1
            cat = note.get("cash_flow_type_cn", "").strip() or "未知"
            categories[cat] += 1
        else:
            unknown += 1
    n = len(balances)
    # 分段首尾中位数避免将全年趋势和最后阶段趋势混为一谈。
    k = max(1, n // 5)
    start_avg = sum(balances[:k]) / k
    end_avg = sum(balances[-k:]) / k
    change = (end_avg - start_avg) / max(abs(start_avg), 1)
    diffs = [balances[i] - balances[i - 1] for i in range(1, n)]
    volatility = (sum((x - (sum(diffs) / max(len(diffs), 1))) ** 2 for x in diffs) / max(len(diffs), 1)) ** 0.5
    if change > 0.08:
        shape = "全程末段上升"
    elif change < -0.08:
        shape = "全程末段下降"
    elif max(balances) - min(balances) < max(abs(sum(balances) / n), 1) * 0.05:
        shape = "全程较平稳"
    else:
        shape = "全程有波动、末段变化有限"
    return {
        "date_start": daily[0]["calendar_date"], "date_end": daily[-1]["calendar_date"], "days": n,
        "balance_first_cny": round(first, 2), "balance_last_cny": round(last, 2),
        "balance_min_cny": round(min(balances), 2), "balance_max_cny": round(max(balances), 2),
        "balance_median_cny": round(sorted(balances)[n // 2], 2), "net_change_cny": round(last - first, 2),
        "inflow_cny": round(inflow, 2), "outflow_cny": round(outflow, 2), "transaction_count": len(txns),
        "daily_reconciliation_pass": daily_ok, "transaction_reconciliation_pass": txn_ok,
        "notes_total": len(notes), "notes_matched": matched, "notes_unmatched": unknown,
        "usage_categories": dict(categories), "measured_shape_full_period": shape,
        "ending_window_change_ratio": round(change, 5), "daily_change_volatility_cny": round(volatility, 2),
    }


def plot_pages(rows: list[dict]) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # 使用已安装的中文字体，图册标题与企业走势描述不显示成方框。
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pages = REPORTS / "余额图册"
    pages.mkdir(parents=True, exist_ok=True)
    output = []
    for page_no, start in enumerate(range(0, len(rows), 8), 1):
        chunk = rows[start:start + 8]
        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        name = f"余额图册_{page_no:02d}.png"
        target = pages / name
        # 输入清单未变时，已核验页直接复用；中断重跑不会浪费时间也不改图源。
        if target.exists():
            plt.close(fig)
            output.append(f"余额图册/{name}")
            continue
        for axis, row in zip(axes.flat, chunk):
            daily = read_csv(ROOT / row["v3_folder"] / "account_daily_total.csv")
            dates = [datetime.strptime(x["calendar_date"], "%Y-%m-%d") for x in daily]
            balances = [money(x["ending_balance_cny"]) / 10000 for x in daily]
            axis.plot(dates, balances, color="#2463a5", linewidth=0.75)
            axis.set_title(f"{row['sample_id']}｜{row['measured_shape_full_period']}", fontsize=8)
            axis.set_ylabel("余额（万元）", fontsize=7)
            axis.tick_params(axis="both", labelsize=6)
            axis.grid(alpha=.25)
        for axis in list(axes.flat)[len(chunk):]:
            axis.set_visible(False)
        fig.suptitle(f"第三轮 V3.0：整理后全部样本真实日余额（第 {page_no} 页）", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(target, dpi=160)
        plt.close(fig)
        output.append(f"余额图册/{name}")
    index = ["# 第三轮 V3.0 全样本余额图册", "", "每条线均直接读取 `data/samples` 中对应权威 CSV 的自然日日余额；未平滑、未补造未来。", ""]
    for path in output:
        index += [f"## {Path(path).stem}", "", f"![{Path(path).stem}]({path})", ""]
    (REPORTS / "全样本余额图册入口.md").write_text("\n".join(index), encoding="utf-8")
    return output


def main() -> None:
    source = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    source["rows"] = source["rows"] + legacy_rows()
    seen: dict[str, str] = {}
    prepared, excluded = [], []
    for row in source["rows"]:
        folder, action = copy_authoritative(row, seen)
        if folder is None:
            excluded.append({**row, "reason": action})
            continue
        # 添加后续同指纹企业映射，绝不覆盖权威文件。
        mapping = json.loads((folder / "来源映射.json").read_text(encoding="utf-8"))
        if row["sample_id"] not in mapping["sample_ids"]:
            mapping["sample_ids"].append(row["sample_id"])
            (folder / "来源映射.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result = metrics(folder)
        role = {"fit": "学习", "calibration": "区间校准", "development": "开发测试"}.get(row["role"], "待查")
        prepared.append({"sample_id": row["sample_id"], "family_id": row["family_id"], "old_role": row["role"], "usage_history": row.get("usage_history", "第二轮开发池"), "v3_role": role,
                         "source_shape_label": row["shape"], "v3_folder": str(folder.relative_to(ROOT)).replace("\\", "/"),
                         "daily_sha256": row["daily_sha256"], "copy_action": action, **result})
    # 同一日余额指纹属于同一资料来源；同家族不跨三种开发角色。
    fingerprint_groups = defaultdict(list)
    family_roles = defaultdict(set)
    for row in prepared:
        fingerprint_groups[row["daily_sha256"]].append(row["sample_id"])
        family_roles[row["family_id"]].add(row["v3_role"])
    # 先按同源资料绑定家族，再以确定性顺序给 V3 开发准备分组，约 70/15/15。
    ordered_families = sorted(family_roles)
    n_families = len(ordered_families)
    cut_fit, cut_cal = round(n_families * .70), round(n_families * .85)
    proposed = {family: ("学习" if i < cut_fit else "区间校准" if i < cut_cal else "开发测试") for i, family in enumerate(ordered_families)}
    for row in prepared:
        row["v3_role"] = proposed[row["family_id"]]
        same = fingerprint_groups[row["daily_sha256"]]
        row["same_source_sample_ids"] = same
        row["same_source_count"] = len(same)
        row["family_role_isolated"] = len(family_roles[row["family_id"]]) == 1
        row["status"] = "可用" if row["daily_reconciliation_pass"] and row["transaction_reconciliation_pass"] and row["notes_unmatched"] == 0 and row["family_role_isolated"] else "待查"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "样本处理清单.json").write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "不用或待查样本清单.json").write_text(json.dumps(excluded + [x for x in prepared if x["status"] != "可用"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pages = plot_pages(prepared)
    status = Counter(x["status"] for x in prepared)
    roles = Counter(x["v3_role"] for x in prepared)
    shapes = Counter(x["measured_shape_full_period"] for x in prepared)
    cats = Counter(cat for x in prepared for cat in x["usage_categories"])
    report = ["# 第三轮 V3.0 样本整理报告", "", "本报告只描述既有开发样本整理；未训练、未考试、未生成新企业。", "",
              "## 盘点结果", "", f"- 来源清单企业记录：{len(source['rows'])} 条。", f"- 独立权威日余额来源：{len(seen)} 套；同指纹记录已合并映射，未重复复制。",
              f"- 可用：{status['可用']}；待查：{status['待查']}；来源缺失/指纹不符：{len(excluded)}。", f"- 开发角色：学习 {roles['学习']}，区间校准 {roles['区间校准']}，开发测试 {roles['开发测试']}。", "",
              "## 实测走势（全程）", ""] + [f"- {k}：{v} 户。" for k,v in sorted(shapes.items())] + ["", "全程走势和末段走势分别按 CSV 的首末五分之一日余额计算；不采用旧标题或旧形状标签决定入组。", "", "## 已发生收支用途信息", ""] + [f"- {k}：{v} 户有已核对记录。" for k,v in cats.most_common()] + ["", "用途只来自与逐笔流水交易编号、金额、方向和日期都核对成功的备注。未知或核对失败不会推测，更不会加入未来合同计划。", "", "## 交付入口", "", "- [全样本余额图册入口](全样本余额图册入口.md)", "- [样本处理清单](../data/manifests/样本处理清单.json)", "- [待查清单](../data/manifests/不用或待查样本清单.json)", ""]
    (REPORTS / "01_整理.md").write_text("\n".join(report), encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"| 2026-09-22 | 执行 V3 只读样本迁移、账务/备注核验与图册 | 来源 {len(source['rows'])} 条，独立资料 {len(seen)} 套，可用 {status['可用']}，待查 {status['待查']}；图册 {len(pages)} 页 |\n")


if __name__ == "__main__":
    main()
