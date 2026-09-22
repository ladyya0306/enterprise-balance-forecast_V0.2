"""形状引导预算搜索的P0--P3可恢复入口。

本工具只向新的实验目录写入：它读取421户已通过清单来缓存特征，
再用唯一中央生成器进行四个目标的小型候选预演。它不改写历史流水，
也不把目标余额线作为流水输入。
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import subprocess
import sys
import time
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from syn_b1.formal_generator import generate_from_profile, preview_from_profile
from syn_b1.formal_profile import default_profile_v2

OUT = ROOT / "data/synthetic_production/coverage102_v1/形状引导预算搜索_20260913_v1"
HISTORY = ROOT / "data/synthetic_production/coverage102_v1/现行421户历史清单_20260912_v1/现行历史manifest_421.json"
ALGORITHM_VERSION = "shape_business_index_v1"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def monthly_balances(daily_csv: Path) -> list[float]:
    """读取真实日表月末余额；不从图像反读，也不重新生成历史。"""
    by_month: dict[str, float] = {}
    with daily_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            by_month[row["calendar_date"][:7]] = float(row["ending_balance_cny"])
    return [by_month[key] for key in sorted(by_month)]


def normalized(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    return [0.0 for _ in values] if hi == lo else [(value - lo) / (hi - lo) for value in values]


def trend_signature(values: list[float]) -> dict[str, object]:
    deltas = [right - left for left, right in zip(values, values[1:])]
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in deltas]
    turns = sum(left and right and left != right for left, right in zip(signs, signs[1:]))
    # 主要转折不应由税费日、月末结算等小幅反向月变动触发；阈值随该户实际月变动规模自适应。
    nonzero = sorted(abs(value) for value in deltas if value)
    major_threshold = max(20_000.0, (nonzero[len(nonzero) // 2] * 0.25) if nonzero else 20_000.0)
    major_signs = [1 if value >= major_threshold else -1 if value <= -major_threshold else 0 for value in deltas]
    compact = [sign for sign in major_signs if sign]
    major_turns = sum(left != right for left, right in zip(compact, compact[1:]))
    return {
        "month_count": len(values), "start_cny": round(values[0], 2), "end_cny": round(values[-1], 2),
        "normalized_month_end": [round(x, 5) for x in normalized(values)],
        "mean_monthly_change_cny": round(sum(deltas) / len(deltas), 2) if deltas else 0,
        "turn_count": turns,
        "major_turn_count": major_turns,
        "major_turn_threshold_cny": round(major_threshold, 2),
    }


def business_signature(notes_csv: Path) -> dict[str, object] | None:
    """只读现有现金流复核表，形成与余额形状分开的经营指纹。"""
    if not notes_csv.exists():
        return None
    monthly_in, monthly_out, types = defaultdict(float), defaultdict(float), defaultdict(float)
    with notes_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            amount = float(row["amount_cny"]); month = row["booking_datetime"][:7]
            if row["direction_cn"] == "流入": monthly_in[month] += amount
            else: monthly_out[month] += amount
            types[row["cash_flow_type_cn"]] += amount
    months = sorted(set(monthly_in) | set(monthly_out))
    return {"months": months, "inflow_share": [round(monthly_in[m] / max(1.0, sum(monthly_in.values())), 6) for m in months], "outflow_share": [round(monthly_out[m] / max(1.0, sum(monthly_out.values())), 6) for m in months], "type_share": {key: round(value / max(1.0, sum(types.values())), 6) for key, value in sorted(types.items())}}


def mean_l1(left: list[float], right: list[float]) -> float:
    size = max(len(left), len(right), 1)
    return sum(abs((left[i] if i < len(left) else 0.0) - (right[i] if i < len(right) else 0.0)) for i in range(size)) / size


def novelty_distance(candidate_shape: dict[str, object], candidate_business: dict[str, object], history_row: dict[str, object]) -> dict[str, float] | None:
    historical_business = history_row.get("business_signature")
    if not historical_business:
        return None
    shape = mean_l1(candidate_shape["normalized_month_end"], history_row["balance_signature"]["normalized_month_end"])
    business = (mean_l1(candidate_business["inflow_share"], historical_business["inflow_share"]) + mean_l1(candidate_business["outflow_share"], historical_business["outflow_share"])) / 2
    keys = set(candidate_business["type_share"]) | set(historical_business["type_share"])
    type_distance = sum(abs(candidate_business["type_share"].get(key, 0) - historical_business["type_share"].get(key, 0)) for key in keys) / max(1, len(keys))
    return {"shape_distance": round(shape, 6), "business_cash_timing_distance": round(business, 6), "business_category_distance": round(type_distance, 6)}


def build_history_index() -> dict[str, object]:
    """仅补缺失特征；清单的既有通过状态原样引用，不作重验。"""
    history = json.loads(HISTORY.read_text(encoding="utf-8-sig"))
    cache_path = OUT / "P1_历史通过特征索引.json"
    old = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"algorithm_version": ALGORITHM_VERSION, "rows": []}
    cached = {row["content_sha256"]: row for row in old.get("rows", []) if row.get("algorithm_version") == ALGORITHM_VERSION}
    rows = []
    for item in history["rows"]:
        daily = ROOT / item["formal"] / "account_daily_total.csv"
        if not daily.exists():
            rows.append({"sample_id": item["sample_id"], "approval_reference": "现行421户历史manifest_421.json", "feature_status": "daily_missing_not_reaudited"})
            continue
        digest = sha(daily)
        cached_row = cached.get(digest)
        if cached_row:
            if "business_signature" not in cached_row:
                cached_row = {**cached_row, "business_signature": business_signature(ROOT / item["formal"] / "cash_flow_review_notes.csv")}
            rows.append(cached_row)
            continue
        values = monthly_balances(daily)
        rows.append({"sample_id": item["sample_id"], "family_id": item.get("family_id"), "approval_reference": "现行421户历史manifest_421.json", "approval_scope": "引用既有正式历史通过结论；本轮未重验", "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "content_sha256": digest, "algorithm_version": ALGORITHM_VERSION, "balance_signature": trend_signature(values), "business_signature": business_signature(ROOT / item["formal"] / "cash_flow_review_notes.csv")})
    result = {"algorithm_version": ALGORITHM_VERSION, "source_manifest": str(HISTORY.relative_to(ROOT)).replace("\\", "/"), "source_manifest_sha256": sha(HISTORY), "history_rows": len(history["rows"]), "cached_or_computed_rows": len(rows), "rows": rows}
    dump(cache_path, result)
    return result


def targets() -> list[dict[str, object]]:
    """冻结24个数值合同。数值是验收骨架而非修改余额的指令。"""
    items: list[dict[str, object]] = []
    def add(key: str, shape: str, breakpoints: list[int], slopes: list[int], volatility: str, explanations: list[str]) -> None:
        items.append({"target_id": key, "months": 24, "shape": shape, "initial_balance_cny": 1800000, "breakpoint_months": breakpoints, "monthly_net_cash_slope_cny": slopes, "volatility": volatility, "major_turn_limit": len(breakpoints), "explanations_max": 2, "business_explanations": explanations})
    add("T01", "持续上行", [], [85000], "低", ["年度框架续约", "渠道备货周转"]); add("T02", "持续上行", [], [85000], "中", ["年度框架续约", "渠道备货周转"])
    add("T03", "近平台", [], [5000], "低", ["维保服务组合", "按需备件配送"]); add("T04", "近平台", [], [5000], "中", ["维保服务组合", "按需备件配送"])
    add("T05", "持续下行", [], [-70000], "低", ["应收回款拉长", "新品备货消耗"]); add("T06", "持续下行", [], [-70000], "中", ["应收回款拉长", "新品备货消耗"])
    for key, shape, bp, sl, ex in [("T07", "V早转", [7], [-110000, 125000], ["项目验收恢复", "维保合同恢复"]), ("T08", "V晚转", [15], [-90000, 130000], ["项目验收恢复", "维保合同恢复"]), ("T09", "U早转", [7], [-35000, 45000], ["缓慢去库存后续约", "淡季后渠道补货"]), ("T10", "U晚转", [15], [-30000, 50000], ["缓慢去库存后续约", "淡季后渠道补货"]), ("T11", "倒V早转", [7], [110000, -115000], ["阶段性交付完工", "一次性渠道铺货完成"]), ("T12", "倒V晚转", [15], [90000, -120000], ["阶段性交付完工", "一次性渠道铺货完成"]), ("T13", "倒U早转", [7], [35000, -45000], ["临时高毛利服务结束", "短期价格优惠结束"]), ("T14", "倒U晚转", [15], [30000, -50000], ["临时高毛利服务结束", "短期价格优惠结束"])]: add(key, shape, bp, sl, "中", ex)
    for key, shape, bp, sl in [("T15", "降升降", [6, 16], [-90000, 120000, -70000]), ("T16", "降升降", [8, 17], [-70000, 105000, -90000]), ("T17", "降升降", [10, 18], [-100000, 95000, -65000]), ("T18", "升降升", [6, 16], [90000, -120000, 70000]), ("T19", "升降升", [8, 17], [70000, -105000, 90000]), ("T20", "升降升", [10, 18], [100000, -95000, 65000])]: add(key, shape, bp, sl, "中", ["多阶段项目交付", "季节性渠道补货"])
    add("T21", "升降升降", [5, 11, 18], [80000, -100000, 110000, -70000], "低", ["季度项目与维保组合", "分批招标订单"]); add("T22", "升降升降", [6, 13, 19], [65000, -85000, 95000, -85000], "中", ["季度项目与维保组合", "分批招标订单"])
    add("T23", "降升降升", [5, 12, 18], [-80000, 100000, -110000, 70000], "低", ["检修与恢复交替", "渠道去库存与回补"]); add("T24", "降升降升", [6, 13, 19], [-65000, 85000, -95000, 85000], "中", ["检修与恢复交替", "渠道去库存与回补"])
    return items


def multipliers(target: dict[str, object]) -> list[float]:
    # 销售/成本/库存均由同一经营节点同步驱动，非余额折线。
    slopes = target["monthly_net_cash_slope_cny"]
    breaks = [0] + target["breakpoint_months"] + [24]
    result: list[float] = []
    for month in range(24):
        segment = next(i for i in range(len(breaks) - 1) if breaks[i] <= month < breaks[i + 1])
        sign = 1 if slopes[segment] >= 0 else -1
        result.append(max(0.45, 1.0 + sign * (0.10 + 0.025 * (month - breaks[segment]))))
    return result


def profile_for(target: dict[str, object], explanation: str, ordinal: int, variant: int = 0) -> dict[str, object]:
    project = explanation in {"项目验收恢复", "阶段性交付完工", "多阶段项目交付", "季度项目与维保组合"}
    channel = explanation in {"一次性渠道铺货完成", "季节性渠道补货", "分批招标订单"}
    base = 1150000 if project else 820000
    margin = 0.42 if project else 0.34
    collection, supplier = (45, 20) if project else ((30, 10) if channel else (15, 35))
    suffix = f"_r2v{variant}" if variant else ""
    raw = default_profile_v2(f"shape_guided_{target['target_id'].lower()}_{ordinal}{suffix}", "p3r2_budget_search_v1" if variant else "p3_exploration_v1")
    raw.update({"start_date": "2024-01-01", "end_date": "2025-12-31", "random_seed": 2026091300 + int(target["target_id"][1:]) * 100 + ordinal * 10 + variant})
    raw["calendar"]["prediction_cutoff"] = "2025-12-31"; raw["initial_capital"]["amount_cny"] = "2200000.00" if project else "1800000.00"
    operating = raw["operating"]
    for key in ("regime_plan", "sales_seasonality", "collection_policy", "procurement_inertia", "monthly_payroll_cny", "monthly_rent_and_utilities_cny", "monthly_other_operating_expense_cny"):
        operating.pop(key, None)
    operating["collection_schedule"] = [{"delay_days": collection, "share_percent": "100"}]
    operating["supplier_payment_schedule"] = [{"delay_days": supplier, "share_percent": "100"}]
    reasons = {"项目验收恢复": "既有项目在验收节点恢复，材料已按合同批次采购", "维保合同恢复": "维保续约恢复，备件按工单消耗", "阶段性交付完工": "阶段性交付完成后订单回落，固定班组仍连续在岗", "渠道备货周转": "渠道补货随经销库存周期调整", "检修与恢复交替": "设备检修窗口降低交付，检修完成恢复接单"}
    nodes = []
    slopes = target["monthly_net_cash_slope_cny"]; breaks = [0] + target["breakpoint_months"] + [24]
    for month, factor in enumerate(multipliers(target), 1):
        segment = next(i for i in range(len(breaks) - 1) if breaks[i] <= month - 1 < breaks[i + 1])
        declining_cash = slopes[segment] < 0
        # R2 将“下行”映射为已说明的返工/备货耗用期；变量只在冻结范围内离散搜索。
        sales = round(base * factor * (1 - 0.025 * variant if declining_cash else 1 + 0.015 * variant))
        cost_ratio = (1.02 + 0.04 * variant) if declining_cash else ((1 - margin) - 0.015 * variant)
        cost = round(sales * cost_ratio); inventory = round(cost * ((0.42 + 0.04 * variant) if declining_cash else (0.32 if project else 0.22)))
        payroll = (115000 if project else 80000) + (18000 * variant if declining_cash else 4000 * variant)
        phase_reason = "返工、备货和连续班组支出在已签项目回款前发生" if declining_cash else "交付与结算恢复，仍维持连续工资、租赁和采购"
        nodes.append({"month": f"{2024 + (month - 1) // 12}-{(month - 1) % 12 + 1:02d}", "stage_id": f"{target['target_id']}-M{month:02d}", "business_reason": reasons.get(explanation, f"{explanation}：订单、交付与采购按已声明合同节奏连续执行") + "；" + phase_reason, "sales_fen": sales * 100, "variable_cost_fen": cost * 100, "target_inventory_fen": inventory * 100, "payroll_fen": payroll * 100, "rent_fen": 28000 * 100, "other_expense_fen": (22000 + 5000 * variant) * 100, "minimum_purchase_fen": 0})
    # 已实现的 monthly_business_nodes_v1 是本轮最小兼容适配：
    # 使用明确单位、设备合同和产能，而不是写一套平行合同流水器。
    operating["fixed_assets"] = [{"event_id": "EQ-01", "asset_type": "equipment", "purchase_month": "2024-01", "purchase_amount_cny": "180000.00", "payment_schedule": [{"delay_days": 0, "share_percent": "100"}], "ready_for_use_date": "2024-01-01", "useful_life_months": 60, "residual_value_percent": "5"}]
    operating["business_nodes"] = {"version": "monthly_business_nodes_v1", "asset_capacities": [{"event_id": "EQ-01", "monthly_units": 4000}], "nodes": [{"month": row["month"], "stage_id": row["stage_id"], "business_reason": row["business_reason"], "sales_units": 1000, "unit_price_cny": f"{row['sales_fen'] // 100000:.2f}", "unit_variable_cost_cny": f"{row['variable_cost_fen'] // 100000:.2f}", "target_inventory_cny": f"{row['target_inventory_fen'] / 100:.2f}", "payroll_cny": f"{row['payroll_fen'] / 100:.2f}", "rent_cny": f"{row['rent_fen'] / 100:.2f}", "other_expense_cny": f"{row['other_expense_fen'] / 100:.2f}"} for row in nodes]}
    return raw


def score(target: dict[str, object], actual: list[float]) -> dict[str, object]:
    actual_slopes = [right - left for left, right in zip(actual, actual[1:])]
    cuts = [0] + target["breakpoint_months"] + [24]
    desired = target["monthly_net_cash_slope_cny"]
    segments = []
    for index, expected in enumerate(desired):
        observed = actual_slopes[cuts[index]:cuts[index + 1] - 1]
        mean = sum(observed) / len(observed) if observed else 0.0
        segments.append({"expected_slope_cny": expected, "actual_mean_slope_cny": round(mean, 2), "direction_match": (mean >= 0) == (expected >= 0)})
    return {"segments": segments, "direction_hits": sum(row["direction_match"] for row in segments), "direction_total": len(segments), "shape_signature": trend_signature(actual)}


def certification_decline_profile(target: dict[str, object], variant: int) -> dict[str, object]:
    """T12R3 的事前合同：认证期收入下降，固定开支连续；不以单件亏损造下行。"""
    raw = profile_for(target, "一次性渠道铺货完成", 2, variant)
    raw["sample_id"] = f"shape_guided_t12_cert_r3v{variant}"
    raw["run_id"] = "p3r3_certification_fixed_cost_v1"
    raw["random_seed"] = 2026092400 + variant
    operating = raw["operating"]
    operating["collection_schedule"] = [{"delay_days": 30, "share_percent": "100"}]
    operating["supplier_payment_schedule"] = [{"delay_days": 10, "share_percent": "100"}]
    for month_index, node in enumerate(operating["business_nodes"]["nodes"], 1):
        if month_index <= 15:
            node["business_reason"] = "原型号渠道铺货及日常补货按已签订单执行；工资、租赁和常规质保费用连续发生。"
            continue
        # 认证/换代期不再以高于售价的材料成本制造亏损。销售降低，保留班组和场地并发生认证费。
        certification_sales = round(820000 * (0.34 - 0.015 * (variant - 1)))
        unit_price = certification_sales / 1000
        unit_cost = unit_price * 0.66
        node.update({"business_reason": "原型号停止新增铺货，新型号在认证和渠道换代期仅交付存量订单；保留核心班组、租赁场地并承担认证测试费用。", "sales_units": 1000, "unit_price_cny": f"{unit_price:.2f}", "unit_variable_cost_cny": f"{unit_cost:.2f}", "target_inventory_cny": f"{unit_cost * 1000 * 0.06:.2f}", "payroll_cny": f"{120000 + 4000 * variant:.2f}", "rent_cny": "32000.00", "other_expense_cny": f"{50000 + 5000 * variant:.2f}"})
    return raw


def main() -> None:
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "P2_24目标骨架.json", {"version": "shape_contract_24_v1", "frozen_at": "2026-09-13", "targets": targets()})
    index = build_history_index()
    pilot_ids = {"T07", "T12", "T15", "T21"}
    pilot = [item for item in targets() if item["target_id"] in pilot_ids]
    results = []
    for target in pilot:
        for ordinal, explanation in enumerate(target["business_explanations"], 1):
            profile = profile_for(target, explanation, ordinal)
            profile_path = OUT / "P3_候选参数" / target["target_id"] / f"方案{ordinal}.json"
            dump(profile_path, profile)
            destination = OUT / "P3_候选流水" / target["target_id"]
            try:
                daily = destination / profile["sample_id"] / f"{target['target_id'].lower()}_p3_{ordinal}" / "account_daily_total.csv"
                # 可恢复：已存在的中央产物只读取，不因报告字段修复重生成。
                if not daily.exists():
                    bundle = generate_from_profile(profile_path, output_root=destination, output_run_id=f"{target['target_id'].lower()}_p3_{ordinal}")
                    reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else:
                    reconciled = True
                values = monthly_balances(daily)
                scored = score(target, values)
                runtime_status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]
                scored["sufficient_observation"] = len(values) == 24 and runtime_status == "complete"
                # 形状与账务是独立门：账务正确不能掩盖形状未命中。
                scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                results.append({"target_id": target["target_id"], "explanation": explanation, "status": "generated_and_accounting_reconciled", "actual_execution_status": runtime_status, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
            except Exception as error:  # 记录失败，不把搜索未命中误报为生成器能力不足。
                results.append({"target_id": target["target_id"], "explanation": explanation, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results: by_target[row["target_id"]].append(row)
    selected = []
    for key, rows in by_target.items():
        successful = [row for row in rows if row["status"] == "generated_and_accounting_reconciled"]
        passed = [row for row in successful if row["score"]["shape_contract_passed"]]
        selected.append({"target_id": key, "attempts": len(rows), "shape_accepted": passed, "not_accepted": successful})
    report = {"stage": "P3_completed_with_shape_misses", "acceptance_scope": "仅四个目标的中央生成、资金守恒回执及目标方向比较；不等于经营有效、去重通过、训练可用或24目标完成", "history_index_rows": index["history_rows"], "pilot_targets": sorted(pilot_ids), "central_exploration_calls": len(results), "budget_proxy_evaluations": 0, "results": results, "selected_per_target": selected, "shape_accepted_count": sum(len(row["shape_accepted"]) for row in selected), "elapsed_seconds": round(time.monotonic() - started, 2)}
    dump(OUT / "P3_四目标结果.json", report)
    miss_rows = [row for row in results if row["status"] == "generated_and_accounting_reconciled" and not row["score"]["shape_contract_passed"]]
    review = "# P3 四目标复审\n\n范围：仅检查新生成八个候选的中央生成回执和月末目标方向；421户历史只作只读特征索引，未重验。\n\n- 中央生成且对账通过：%d / %d\n- 形状合同完整命中：%d / %d\n- 结论：P4 不启动；没有候选进入通过名单。\n\n失败分类：形状未命中（非生成器能力结论）%d；技术/约束失败 %d。\n" % (sum(row["status"] == "generated_and_accounting_reconciled" for row in results), len(results), len(results) - len(miss_rows), len(results), len(miss_rows), sum(row["status"] != "generated_and_accounting_reconciled" for row in results))
    (OUT / "P3_四目标复审.md").write_text(review, encoding="utf-8")
    dump(OUT / "progress.json", {"stage": "P3_reviewed", "completed": ["P0目录与进程盘点", "P1历史只读特征索引", "P2冻结24目标骨架", "P3四目标中央预演与复审"], "next": "在不改余额的前提下，为实际余额首段下行的原因建立预算搜索维度与输入映射单测；通过后才可重启受影响四目标的有限搜索", "failure_categories": {"shape_miss": len(miss_rows), "technical_or_constraint_failure": sum(row["status"] != "generated_and_accounting_reconciled" for row in results)}, "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"output": str(OUT), "pilot_attempts": len(results), "successes": sum(row["status"] == "generated_and_accounting_reconciled" for row in results)}, ensure_ascii=False))


def main_r2() -> None:
    """P3R2：先证明字段传导，再跑冻结的 4×2×4 有限搜索。"""
    OUT.mkdir(parents=True, exist_ok=True)
    history_index = build_history_index()
    pilot = [item for item in targets() if item["target_id"] in {"T07", "T12", "T15", "T21"}]
    results = []
    mapping = []
    for target in pilot:
        for ordinal, explanation in enumerate(target["business_explanations"], 1):
            for variant in range(1, 5):
                profile = profile_for(target, explanation, ordinal, variant)
                profile_path = OUT / "P3R2_候选参数" / target["target_id"] / f"方案{ordinal}_预算{variant}.json"
                dump(profile_path, profile)
                # 两个极端预算先走内存中央链；差异不足会阻断对应搜索。
                if target["target_id"] == "T07" and ordinal == 1 and variant in {1, 4}:
                    _, preview = preview_from_profile(profile_path)
                    first_six = [row.ending_balance_fen for row in preview.ledger.daily_rows if row.calendar_date.month <= 6][-1]
                    mapping.append({"variant": variant, "first_half_year_ending_balance_fen": first_six, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/")})
                destination = OUT / "P3R2_候选流水" / target["target_id"]
                daily = destination / profile["sample_id"] / f"{target['target_id'].lower()}_r2_{ordinal}_{variant}" / "account_daily_total.csv"
                try:
                    if not daily.exists():
                        bundle = generate_from_profile(profile_path, output_root=destination, output_run_id=f"{target['target_id'].lower()}_r2_{ordinal}_{variant}")
                        reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                    else:
                        reconciled = True
                    values = monthly_balances(daily)
                    scored = score(target, values)
                    runtime_status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]
                    scored["sufficient_observation"] = len(values) == 24 and runtime_status == "complete"
                    scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                    results.append({"target_id": target["target_id"], "explanation": explanation, "budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": runtime_status, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
                except Exception as error:
                    results.append({"target_id": target["target_id"], "explanation": explanation, "budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    mapping_ok = len(mapping) == 2 and mapping[0]["first_half_year_ending_balance_fen"] != mapping[1]["first_half_year_ending_balance_fen"]
    dump(OUT / "P3R2_字段映射单测.json", {"scope": "同一T07经营解释，改变已声明的成本、库存和连续费用预算后，经中央内存链读取六月末实际余额", "passed": mapping_ok, "observations": mapping})
    passed = [row for row in results if row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]
    report = {"stage": "P3R2_limited_budget_search", "acceptance_scope": "仅字段传导、中央生成/对账和形状合同；尚未完成对421历史的业务距离排重，未进入正式通过名单", "mapping_test_passed": mapping_ok, "central_exploration_calls": len(results), "budget_evaluations_per_target": 8, "results": results, "shape_contract_passed": passed, "shape_contract_passed_count": len(passed)}
    dump(OUT / "P3R2_有限预算搜索结果.json", report)
    # 同目标的预算搜索变体不是独立经营解释：最多留下目标内得分最高的一份进入排重。
    representatives = []
    for target_id in sorted({row["target_id"] for row in passed}):
        options = [row for row in passed if row["target_id"] == target_id]
        representatives.append(min(options, key=lambda row: (sum(abs(s["actual_mean_slope_cny"] - s["expected_slope_cny"]) for s in row["score"]["segments"]), row["budget_variant"])))
    novelty = []
    for candidate in representatives:
        candidate_business = business_signature(ROOT / candidate["daily_csv"].replace("account_daily_total.csv", "cash_flow_review_notes.csv"))
        pairs = [] if candidate_business is None else [
            {"history_sample_id": row["sample_id"], **distance}
            for row in history_index["rows"]
            if row.get("balance_signature") and (distance := novelty_distance(candidate["score"]["shape_signature"], candidate_business, row)) is not None
        ]
        pairs.sort(key=lambda item: (item["shape_distance"] + item["business_cash_timing_distance"], item["history_sample_id"]))
        double_high = [item for item in pairs if item["shape_distance"] <= 0.05 and item["business_cash_timing_distance"] <= 0.05 and item["business_category_distance"] <= 0.08]
        novelty.append({"candidate": candidate, "review_scope": "与421户只读索引比较余额形状、月度收付时点和现金流类别；非历史重验", "nearest_pairs": pairs[:5], "double_high_similarity_pairs": double_high, "decision": "reject_double_high_similarity" if double_high else "provisional_novelty_clear_pending_business_manual_review"})
    dump(OUT / "P3R2_增量排重复审.json", {"algorithm_version": ALGORITHM_VERSION, "history_rows": history_index["history_rows"], "representative_candidates": len(representatives), "rows": novelty, "acceptance_scope": "仅增量形状/经营现金流指纹排重；不等于正式样本批准"})
    cleared = [row for row in novelty if row["decision"] != "reject_double_high_similarity"]
    # 人工复审只检验已声明解释能否覆盖实际参数；不为已生成流水补写故事。
    manual_reviews = []
    for row in cleared:
        candidate = row["candidate"]
        profile = json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8"))
        nodes = profile["operating"]["business_nodes"]["nodes"]
        loss_nodes = [node["month"] for node in nodes if float(node["unit_variable_cost_cny"]) >= float(node["unit_price_cny"])]
        explanation = candidate["explanation"]
        needs_return_or_rework_basis = bool(loss_nodes) and explanation == "一次性渠道铺货完成"
        manual_reviews.append({"candidate": candidate, "loss_or_rework_months": loss_nodes, "continuous_payroll_and_rent_checked": all(float(node["payroll_cny"]) > 0 and float(node["rent_cny"]) > 0 for node in nodes), "capacity_checked": all(node["sales_units"] <= 4000 for node in nodes), "decision": "reject_business_explanation_incomplete" if needs_return_or_rework_basis else "business_constraints_review_passed", "reason": "实际预算在后段以单位变动成本不低于售价形成下行，但已声明的“渠道铺货完成”没有退货、返工或保修耗用合同依据；不得在结果后补造解释。" if needs_return_or_rework_basis else "声明的经营解释覆盖已使用的参数范围。"})
    dump(OUT / "P3R2_经营约束人工复审.json", {"scope": "仅复审已清除双高相似的代表候选；不重验421历史", "rows": manual_reviews})
    manual_passed = [row for row in manual_reviews if row["decision"] == "business_constraints_review_passed"]
    stage = "P3R2_ready_for_next_stage" if mapping_ok and manual_passed else "P3R2_manual_business_review_failed"
    dump(OUT / "progress.json", {"stage": stage, "completed": ["P0目录与进程盘点", "P1历史只读特征索引", "P2冻结24目标骨架", "P3初轮中央预演与复审", "P3R2字段映射单测、有限预算搜索、增量排重和经营约束复审"], "next": "为该目标重写事前经营解释与合同约束后再搜索；不改已生成流水" if not manual_passed else "对经营约束复审通过的候选生成图册并进入下一阶段", "failure_categories": {"shape_miss": len([row for row in results if row["status"] == "generated_and_accounting_reconciled" and not row["score"]["shape_contract_passed"]]), "technical_or_constraint_failure": len([row for row in results if row["status"] != "generated_and_accounting_reconciled"]), "double_high_similarity": len([row for row in novelty if row["decision"] == "reject_double_high_similarity"]), "business_explanation_incomplete": len([row for row in manual_reviews if row["decision"] != "business_constraints_review_passed"])}, "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"output": str(OUT), "mapping_test_passed": mapping_ok, "central_runs": len(results), "shape_passed": len(passed), "novelty_cleared_representatives": len(cleared), "manual_business_passed": len(manual_passed)}, ensure_ascii=False))


def main_r3() -> None:
    """对已拒绝的 T12 仅替换事前经营解释，运行四个预算点，不改任何 R2 产物。"""
    OUT.mkdir(parents=True, exist_ok=True)
    history_index = build_history_index()
    target = next(row for row in targets() if row["target_id"] == "T12")
    contract = {"contract_version": "T12_certification_channel_replacement_v1", "target_id": "T12", "business_explanation": "产品认证与渠道换代：原型号停止新增铺货，新型号认证期间仅交付存量订单；核心班组、场地和认证测试费用连续发生。", "certification_period": "2025-04至2025-12", "permitted_search_fields": {"certification_monthly_sales_cny": [242000, 278800], "monthly_payroll_cny": [124000, 136000], "monthly_rent_cny": [32000, 32000], "monthly_certification_expense_cny": [55000, 70000], "unit_variable_cost_percent_of_price": 66}, "prohibitions": ["不以单位变动成本不低于售价制造下行", "不按余额改流水", "不新增融资", "不重写P3R2候选或历史"]}
    dump(OUT / "P3R3_T12_经营合同_v1.json", contract)
    results, mapping = [], []
    for variant in range(1, 5):
        profile = certification_decline_profile(target, variant)
        profile_path = OUT / "P3R3_候选参数" / f"T12_认证换代_预算{variant}.json"
        dump(profile_path, profile)
        _, preview = preview_from_profile(profile_path)
        june_2025 = next(row.ending_balance_fen for row in preview.ledger.daily_rows if row.calendar_date.isoformat() == "2025-06-30")
        mapping.append({"variant": variant, "june_2025_ending_balance_fen": june_2025})
        destination = OUT / "P3R3_候选流水"
        run_id = f"t12_r3_cert_{variant}"
        daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists():
                bundle = generate_from_profile(profile_path, output_root=destination, output_run_id=run_id)
                reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else:
                reconciled = True
            values = monthly_balances(daily)
            runtime_status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]
            scored = score(target, values)
            scored["sufficient_observation"] = len(values) == 24 and runtime_status == "complete"
            scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
            results.append({"budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": runtime_status, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
        except Exception as error:
            results.append({"budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    mapping_ok = len({row["june_2025_ending_balance_fen"] for row in mapping}) > 1
    passed = [row for row in results if row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]
    chosen = min(passed, key=lambda row: (sum(abs(s["actual_mean_slope_cny"] - s["expected_slope_cny"]) for s in row["score"]["segments"]), row["budget_variant"])) if passed else None
    novelty = []
    if chosen:
        candidate_business = business_signature(ROOT / chosen["daily_csv"].replace("account_daily_total.csv", "cash_flow_review_notes.csv"))
        novelty = [{"history_sample_id": row["sample_id"], **distance} for row in history_index["rows"] if row.get("balance_signature") and candidate_business and (distance := novelty_distance(chosen["score"]["shape_signature"], candidate_business, row)) is not None]
        novelty.sort(key=lambda item: (item["shape_distance"] + item["business_cash_timing_distance"], item["history_sample_id"]))
    double_high = [row for row in novelty if row["shape_distance"] <= 0.05 and row["business_cash_timing_distance"] <= 0.05 and row["business_category_distance"] <= 0.08]
    business_ok = False
    if chosen:
        profile = json.loads((ROOT / chosen["profile"]).read_text(encoding="utf-8")); nodes = profile["operating"]["business_nodes"]["nodes"]
        business_ok = all(float(node["unit_variable_cost_cny"]) < float(node["unit_price_cny"]) and float(node["payroll_cny"]) > 0 and float(node["rent_cny"]) > 0 for node in nodes) and all("认证" in node["business_reason"] or "铺货" in node["business_reason"] for node in nodes)
    decision = "limited_candidate_passed_not_formal" if chosen and not double_high and business_ok and mapping_ok else "not_accepted"
    report = {"stage": "P3R3_T12_certification_fixed_cost_search", "acceptance_scope": "仅T12一份候选的字段传导、中央生成/对账、24月形状、增量排重与经营约束；不等于正式样本或训练可用", "mapping_test": {"passed": mapping_ok, "observations": mapping}, "central_exploration_calls": len(results), "results": results, "chosen_candidate": chosen, "nearest_history_pairs": novelty[:5], "double_high_similarity_pairs": double_high, "business_constraints_passed": business_ok, "decision": decision}
    dump(OUT / "P3R3_T12_认证换代搜索结果.json", report)
    dump(OUT / "progress.json", {"stage": "P3R3_limited_candidate_passed" if decision.startswith("limited") else "P3R3_no_candidate", "completed": ["P0至P3R2", "P3R3认证换代事前合同", "P3R3字段传导、中央生成、形状/排重/经营复审"], "next": "为该单一有限候选制作目标/实际对照图和画像；不启动P4或扩量" if decision.startswith("limited") else "记录失败并返回其他目标，不强行调整余额", "failure_categories": {"technical_or_constraint_failure": len([row for row in results if row["status"] != "generated_and_accounting_reconciled"]), "shape_miss": len([row for row in results if row["status"] == "generated_and_accounting_reconciled" and not row["score"]["shape_contract_passed"]]), "double_high_similarity": len(double_high), "business_constraint": 0 if business_ok else 1}, "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"output": str(OUT), "runs": len(results), "shape_passed": len(passed), "decision": decision}, ensure_ascii=False))


def main_r3_visual() -> None:
    """P3R3 的只读图文交付：不生成流水、不改预算。"""
    report = json.loads((OUT / "P3R3_T12_认证换代搜索结果.json").read_text(encoding="utf-8"))
    chosen = report.get("chosen_candidate")
    if not chosen or report.get("decision") != "limited_candidate_passed_not_formal":
        raise ValueError("仅对已通过P3R3有限候选生成图文材料")
    output = OUT / "P3R3_查看材料_字体修复"
    if output.exists():
        raise FileExistsError("P3R3查看材料已存在，拒绝覆盖")
    output.mkdir()
    daily = ROOT / chosen["daily_csv"]
    values, months = monthly_balances(daily), []
    with daily.open(encoding="utf-8-sig", newline="") as handle:
        last = {}
        for row in csv.DictReader(handle): last[row["calendar_date"][:7]] = float(row["ending_balance_cny"])
        months = sorted(last)
    target = next(row for row in targets() if row["target_id"] == "T12")
    expected, break_month = target["monthly_net_cash_slope_cny"], target["breakpoint_months"][0]
    rail = [values[0]]
    for index in range(1, len(values)):
        rail.append(rail[-1] + (expected[0] if index < break_month else expected[1]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.plot(months, values, color="#1769aa", marker="o", linewidth=2, label="中央流水实际月末余额")
    axis.plot(months, rail, color="#ef6c00", linestyle="--", linewidth=2, label="目标斜率参照线（锚定首个实际月末）")
    axis.axvline(months[break_month], color="#777777", linestyle=":", label="计划转折：2025-04")
    axis.set_title("T12：产品认证与渠道换代｜目标/实际余额对照（有限候选）")
    axis.set_ylabel("余额（元）")
    axis.set_xlabel("月份")
    axis.tick_params(axis="x", rotation=45); axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    chart = output / "T12_目标实际余额对照.png"; figure.savefig(chart, dpi=160); plt.close(figure)
    profile = json.loads((ROOT / chosen["profile"]).read_text(encoding="utf-8")); nodes = profile["operating"]["business_nodes"]["nodes"]
    portrait = "# T12 产品认证与渠道换代｜企业画像\n\n这是一份有限实验候选，不是正式训练样本。原型号在 2025 年 4 月后停止新增铺货；新型号进入认证和渠道换代，企业只交付存量订单，同时保留核心班组、租赁场地并承担认证测试费用。\n\n- 认证期：2025-04 至 2025-12。\n- 认证期单件变动成本始终低于售价；余额下行来自较低销售额无法覆盖连续工资、租金和认证费用，而非逐单亏损。\n- 实际月均变化：转折前约 +{:.2f} 万元，转折后约 {:.2f} 万元。\n- 中央流水完整运行 24 个月且已对账；本材料不代表正式批准、训练可用或可扩量。\n\n![目标与实际余额]({})\n".format(chosen["score"]["segments"][0]["actual_mean_slope_cny"] / 10000, chosen["score"]["segments"][1]["actual_mean_slope_cny"] / 10000, chart.as_posix())
    (output / "T12_企业画像.md").write_text(portrait, encoding="utf-8")
    (output / "查看说明.md").write_text("# P3R3 有限候选材料\n\n" + portrait + "\n原始参数：`" + chosen["profile"] + "`\n实际日余额：`" + chosen["daily_csv"] + "`\n", encoding="utf-8")
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
    progress.update({"stage": "P3R3_limited_candidate_materials_ready", "completed": progress["completed"] + ["P3R3目标/实际图与企业画像（中文字体修复版）"], "next": "等待用户查看有限候选材料后决定是否为其他目标创建事前经营合同；不启动P4或扩量", "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    dump(OUT / "progress.json", progress)
    print(json.dumps({"output": str(output), "central_generation_calls": 0}, ensure_ascii=False))


def fixed_cost_phase_profile(target: dict[str, object], contract: dict[str, object], variant: int) -> dict[str, object]:
    """把已冻结的经营阶段映射为中央节点；下行只靠低销售与连续固定费用。"""
    raw = profile_for(target, "维保合同恢复", 1, variant)
    raw["sample_id"] = f"shape_guided_{target['target_id'].lower()}_{contract['short_id']}_r4v{variant}"
    raw["run_id"] = "p3r4_fixed_cost_phase_v1"; raw["random_seed"] = 2026092600 + int(target["target_id"][1:]) * 10 + variant
    operating = raw["operating"]
    operating["collection_schedule"] = contract.get("collection_schedule", [{"delay_days": contract["collection_delay_days"], "share_percent": "100"}])
    operating["supplier_payment_schedule"] = contract.get("supplier_schedule", [{"delay_days": contract["supplier_delay_days"], "share_percent": "100"}])
    if "startup_asset_cny" in contract:
        operating["fixed_assets"][0]["purchase_amount_cny"] = f"{contract['startup_asset_cny']:.2f}"
    breaks = [0] + target["breakpoint_months"] + [24]
    for month_index, node in enumerate(operating["business_nodes"]["nodes"], 1):
        segment = next(i for i in range(len(breaks) - 1) if breaks[i] < month_index <= breaks[i + 1])
        declining = target["monthly_net_cash_slope_cny"][segment] < 0
        sales = round(contract["base_sales_cny"] * ((0.19 - 0.01 * (variant - 1)) if declining else (1.08 + 0.02 * variant)))
        unit_price = sales / 1000; unit_cost = unit_price * contract["variable_cost_ratio"]
        node.update({"business_reason": contract["phase_reasons"][segment], "sales_units": 1000, "unit_price_cny": f"{unit_price:.2f}", "unit_variable_cost_cny": f"{unit_cost:.2f}", "target_inventory_cny": f"{unit_cost * 1000 * (0.60 if declining else 0.22):.2f}", "payroll_cny": f"{(contract['low_payroll_cny'] + 3000 * variant) if declining else (contract['normal_payroll_cny'] + 2000 * variant):.2f}", "rent_cny": f"{contract['rent_cny']:.2f}", "other_expense_cny": f"{(contract['low_other_cny'] + 4000 * variant) if declining else (contract['normal_other_cny'] + 2000 * variant):.2f}"})
    return raw


def main_r4() -> None:
    """补齐P3其余三目标；每目标仅一份事前解释、四个预算点。"""
    OUT.mkdir(parents=True, exist_ok=True); history_index = build_history_index()
    contracts = {
        "T07": {"short_id": "service_launch", "business_explanation": "开业前服务能力投入后框架订单兑现", "collection_delay_days": 45, "supplier_delay_days": 15, "base_sales_cny": 820000, "variable_cost_ratio": 0.66, "low_payroll_cny": 122000, "normal_payroll_cny": 84000, "rent_cny": 32000, "low_other_cny": 50000, "normal_other_cny": 24000, "phase_reasons": ["新服务团队招聘、备件库启用和客户验收准备期；交付量有限，但核心人员与场地持续投入。", "框架订单通过验收后按工单交付，备件按实际耗用补货，连续班组和场地照常维持。"]},
        "T15": {"short_id": "tender_project_warranty", "business_explanation": "招标准备—项目交付—保修期的现金节奏", "collection_delay_days": 30, "supplier_delay_days": 20, "base_sales_cny": 860000, "variable_cost_ratio": 0.64, "low_payroll_cny": 118000, "normal_payroll_cny": 82000, "rent_cny": 30000, "low_other_cny": 46000, "normal_other_cny": 22000, "phase_reasons": ["项目投标、样机测试与客户入围期，交付有限；报价工程师、租赁工位和测试费用连续发生。", "中标后按里程碑交付，采购与回款依照已签订单和账期安排。", "项目主体交付完成，进入保修与下一轮招标准备；新增交付下降，但核心售后班组、场地和质保支出持续。"]},
        "T21": {"short_id": "maintenance_tender", "business_explanation": "项目交付—计划维护—恢复—续标等待的现金节奏", "collection_delay_days": 40, "supplier_delay_days": 15, "base_sales_cny": 900000, "variable_cost_ratio": 0.65, "low_payroll_cny": 125000, "normal_payroll_cny": 88000, "rent_cny": 34000, "low_other_cny": 52000, "normal_other_cny": 26000, "phase_reasons": ["首批项目按计划交付，材料按批次采购，回款按合同账期兑现。", "设备计划维护和人员安全复训期，现场交付减少；保留班组、场地和维护支出。", "维护结束后恢复项目交付，采用已确认订单和原有供应商账期。", "渠道年度续标等待期，新增交付下降；保留核心服务能力、场地和投标准备费用。"]},
    }
    target_by_id = {row["target_id"]: row for row in targets()}; dump(OUT / "P3R4_三目标经营合同_v1.json", {"version": "predeclared_fixed_cost_phases_v1", "contracts": contracts, "prohibitions": ["单位变动成本不得不低于售价", "不得改余额或事后补写解释", "不得新增融资"]})
    results, mapping = [], []
    for target_id, contract in contracts.items():
        target = target_by_id[target_id]
        for variant in range(1, 5):
            profile = fixed_cost_phase_profile(target, contract, variant)
            profile_path = OUT / "P3R4_候选参数" / f"{target_id}_{contract['short_id']}_预算{variant}.json"; dump(profile_path, profile)
            _, preview = preview_from_profile(profile_path)
            mapping.append({"target_id": target_id, "variant": variant, "ending_balance_fen": preview.ledger.daily_rows[-1].ending_balance_fen})
            destination = OUT / "P3R4_候选流水" / target_id; run_id = f"{target_id.lower()}_r4_{variant}"
            daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists():
                    bundle = generate_from_profile(profile_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                values = monthly_balances(daily); runtime_status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]
                scored = score(target, values); scored["sufficient_observation"] = len(values) == 24 and runtime_status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                results.append({"target_id": target_id, "business_explanation": contract["business_explanation"], "budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": runtime_status, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
            except Exception as error: results.append({"target_id": target_id, "budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    mapping_ok = all(len({row["ending_balance_fen"] for row in mapping if row["target_id"] == target_id}) > 1 for target_id in contracts)
    passed = [row for row in results if row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]
    chosen = []
    for target_id in contracts:
        options = [row for row in passed if row["target_id"] == target_id]
        if options: chosen.append(min(options, key=lambda row: (sum(abs(s["actual_mean_slope_cny"] - s["expected_slope_cny"]) for s in row["score"]["segments"]), row["budget_variant"])))
    review = []
    for candidate in chosen:
        business = business_signature(ROOT / candidate["daily_csv"].replace("account_daily_total.csv", "cash_flow_review_notes.csv")); pairs = [{"history_sample_id": row["sample_id"], **distance} for row in history_index["rows"] if row.get("balance_signature") and business and (distance := novelty_distance(candidate["score"]["shape_signature"], business, row)) is not None]; pairs.sort(key=lambda item: (item["shape_distance"] + item["business_cash_timing_distance"], item["history_sample_id"]))
        profile = json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8")); nodes = profile["operating"]["business_nodes"]["nodes"]
        business_ok = all(float(node["unit_variable_cost_cny"]) < float(node["unit_price_cny"]) and float(node["payroll_cny"]) > 0 and float(node["rent_cny"]) > 0 for node in nodes)
        double_high = [pair for pair in pairs if pair["shape_distance"] <= 0.05 and pair["business_cash_timing_distance"] <= 0.05 and pair["business_category_distance"] <= 0.08]
        review.append({"candidate": candidate, "nearest_history_pairs": pairs[:5], "double_high_similarity_pairs": double_high, "business_constraints_passed": business_ok, "decision": "limited_candidate_passed_not_formal" if business_ok and not double_high else "not_accepted"})
    accepted = [row for row in review if row["decision"].startswith("limited")]
    dump(OUT / "P3R4_三目标搜索与复审.json", {"acceptance_scope": "仅补齐P3另外三目标的字段映射、中央生成/对账、形状、增量排重和经营约束；不等于正式样本", "mapping_test_passed": mapping_ok, "central_exploration_calls": len(results), "results": results, "review": review, "accepted_count": len(accepted)})
    dump(OUT / "progress.json", {"stage": "P3_four_target_validation_completed" if len(accepted) == 3 and mapping_ok else "P3_remaining_targets_need_revision", "completed": ["P0至P3R3", "P3R4三目标事前合同、有限搜索和复审"], "next": "按24目标计划启动P4的首个受控小组" if len(accepted) == 3 and mapping_ok else "仅修订未命中目标的事前经营合同，不改已有流水", "failure_categories": {"shape_miss": len([row for row in results if row["status"] == "generated_and_accounting_reconciled" and not row["score"]["shape_contract_passed"]]), "technical_or_constraint_failure": len([row for row in results if row["status"] != "generated_and_accounting_reconciled"]), "business_or_similarity_reject": len(review) - len(accepted)}, "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"output": str(OUT), "runs": len(results), "shape_passed": len(passed), "accepted": len(accepted), "mapping_test_passed": mapping_ok}, ensure_ascii=False))


def main_r5() -> None:
    """仅修订P3R4明确失败的T07/T21合同；不触碰已通过的T12/T15。"""
    OUT.mkdir(parents=True, exist_ok=True); history_index = build_history_index(); target_by_id = {row["target_id"]: row for row in targets()}
    contracts = {
        "T07": {"short_id": "service_launch_v2", "business_explanation": "精简启动团队与备件服务框架订单兑现", "collection_delay_days": 45, "supplier_delay_days": 15, "base_sales_cny": 820000, "variable_cost_ratio": 0.66, "low_payroll_cny": 96000, "normal_payroll_cny": 80000, "rent_cny": 25000, "low_other_cny": 28000, "normal_other_cny": 22000, "phase_reasons": ["精简启动团队、租赁小型备件工位并完成首批客户验收；交付量有限，但基础服务能力连续保留。", "框架订单验收后按工单交付，备件按实际耗用补货，服务团队和场地持续运营。"]},
        "T21": {"short_id": "maintenance_tender_v2", "business_explanation": "分期回款的项目交付—维护—恢复—续标等待", "collection_delay_days": 0, "collection_schedule": [{"delay_days": 0, "share_percent": "50"}, {"delay_days": 30, "share_percent": "50"}], "supplier_delay_days": 15, "base_sales_cny": 900000, "variable_cost_ratio": 0.65, "low_payroll_cny": 118000, "normal_payroll_cny": 84000, "rent_cny": 30000, "low_other_cny": 44000, "normal_other_cny": 24000, "phase_reasons": ["首批项目分两期收款：签约首款与三十天验收款；材料按批次采购。", "设备计划维护和人员安全复训期，现场交付减少；保留班组、场地和维护支出。", "维护结束后恢复项目交付，继续采用签约首款与验收款的分期回款。", "渠道年度续标等待期，新增交付下降；保留核心服务能力、场地和投标准备费用。"]},
    }
    dump(OUT / "P3R5_T07_T21修订经营合同_v1.json", {"version": "targeted_contract_revision_v1", "contracts": contracts, "revision_reason": "T07防止订单兑现前停止；T21将整笔回款改为已声明分期回款以消除非主要月末折返", "prohibitions": ["不改变P3R4既有流水", "不以单位亏损制造下行", "不新增融资"]})
    results = []
    for target_id, contract in contracts.items():
        target = target_by_id[target_id]
        for variant in range(1, 5):
            profile = fixed_cost_phase_profile(target, contract, variant); profile["run_id"] = "p3r5_targeted_contract_revision_v1"
            profile_path = OUT / "P3R5_候选参数" / f"{target_id}_{contract['short_id']}_预算{variant}.json"; dump(profile_path, profile)
            destination = OUT / "P3R5_候选流水" / target_id; run_id = f"{target_id.lower()}_r5_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists():
                    bundle = generate_from_profile(profile_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                values = monthly_balances(daily); runtime_status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; scored = score(target, values)
                scored["sufficient_observation"] = len(values) == 24 and runtime_status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                results.append({"target_id": target_id, "business_explanation": contract["business_explanation"], "budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": runtime_status, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
            except Exception as error: results.append({"target_id": target_id, "budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    passed = [row for row in results if row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]; review = []
    for target_id in contracts:
        options = [row for row in passed if row["target_id"] == target_id]
        if not options: continue
        candidate = min(options, key=lambda row: (sum(abs(s["actual_mean_slope_cny"] - s["expected_slope_cny"]) for s in row["score"]["segments"]), row["budget_variant"])); business = business_signature(ROOT / candidate["daily_csv"].replace("account_daily_total.csv", "cash_flow_review_notes.csv")); pairs = [{"history_sample_id": row["sample_id"], **distance} for row in history_index["rows"] if row.get("balance_signature") and business and (distance := novelty_distance(candidate["score"]["shape_signature"], business, row)) is not None]; pairs.sort(key=lambda item: (item["shape_distance"] + item["business_cash_timing_distance"], item["history_sample_id"])); profile = json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8")); nodes = profile["operating"]["business_nodes"]["nodes"]; business_ok = all(float(node["unit_variable_cost_cny"]) < float(node["unit_price_cny"]) and float(node["payroll_cny"]) > 0 and float(node["rent_cny"]) > 0 for node in nodes); double_high = [pair for pair in pairs if pair["shape_distance"] <= .05 and pair["business_cash_timing_distance"] <= .05 and pair["business_category_distance"] <= .08]; review.append({"candidate": candidate, "nearest_history_pairs": pairs[:5], "double_high_similarity_pairs": double_high, "business_constraints_passed": business_ok, "decision": "limited_candidate_passed_not_formal" if business_ok and not double_high else "not_accepted"})
    accepted = [row for row in review if row["decision"].startswith("limited")]
    dump(OUT / "P3R5_T07_T21搜索与复审.json", {"acceptance_scope": "仅T07/T21修订合同的中央生成/对账、形状、增量排重与经营约束；不等于正式样本", "central_exploration_calls": len(results), "results": results, "review": review, "accepted_count": len(accepted)})
    dump(OUT / "progress.json", {"stage": "P3_four_target_validation_completed" if len(accepted) == 2 else "P3_targeted_revision_incomplete", "completed": ["P0至P3R4", "P3R5 T07/T21定向合同修订、有限搜索和复审"], "next": "开始P4首个受控小组（其余20目标）" if len(accepted) == 2 else "仅继续修订未命中的事前合同", "failure_categories": {"shape_miss": len([row for row in results if row["status"] == "generated_and_accounting_reconciled" and not row["score"]["shape_contract_passed"]]), "technical_or_constraint_failure": len([row for row in results if row["status"] != "generated_and_accounting_reconciled"]), "business_or_similarity_reject": len(review) - len(accepted)}, "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"output": str(OUT), "runs": len(results), "shape_passed": len(passed), "accepted": len(accepted)}, ensure_ascii=False))


def main_r6() -> None:
    """根据P3R5最小复现修正启动资产与供应商付款节奏。"""
    OUT.mkdir(parents=True, exist_ok=True); history_index = build_history_index(); target_by_id = {row["target_id"]: row for row in targets()}
    contracts = {
        "T07": {"short_id": "service_launch_v3_leased", "business_explanation": "租赁诊断工位的精简服务启动与框架订单兑现", "collection_delay_days": 45, "supplier_delay_days": 15, "startup_asset_cny": 60000, "base_sales_cny": 820000, "variable_cost_ratio": .66, "low_payroll_cny": 96000, "normal_payroll_cny": 80000, "rent_cny": 25000, "low_other_cny": 28000, "normal_other_cny": 22000, "phase_reasons": ["以租赁诊断工位替代大额自购设备；精简团队完成客户验收，交付量有限但基础服务能力持续保留。", "框架订单验收后按工单交付，备件按实际耗用补货，服务团队和租赁工位持续运营。"]},
        "T21": {"short_id": "maintenance_tender_v3_supplier_current", "business_explanation": "分期回款、到货当月结算的项目维护与续标等待", "collection_delay_days": 0, "collection_schedule": [{"delay_days": 0, "share_percent": "50"}, {"delay_days": 30, "share_percent": "50"}], "supplier_delay_days": 0, "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "base_sales_cny": 900000, "variable_cost_ratio": .65, "low_payroll_cny": 118000, "normal_payroll_cny": 84000, "rent_cny": 30000, "low_other_cny": 44000, "normal_other_cny": 24000, "phase_reasons": ["首批项目采取签约首款与三十天验收款；供应商材料到货当月结算，避免跨月集中付款。", "设备计划维护和人员安全复训期，现场交付减少；保留班组、场地和维护支出。", "维护结束后恢复项目交付，继续采用分期客户回款及到货当月供应商结算。", "渠道年度续标等待期，新增交付下降；保留核心服务能力、场地和投标准备费用。"]},
    }
    dump(OUT / "P3R6_最小修订经营合同_v1.json", {"version": "minimal_reproduction_fixes_v1", "contracts": contracts, "basis": {"T07": "P3R5仅在订单兑现前停止，故降低并冻结启动资产投入，不改变销售/余额数据", "T21": "P3R5月末尖峰由15天供应商账期跨月聚集导致，改为到货当月结算"}, "prohibitions": ["不改P3R5或历史流水", "不以单位亏损制造下行", "不新增融资"]})
    results = []
    for target_id, contract in contracts.items():
        target = target_by_id[target_id]
        for variant in range(1, 5):
            profile = fixed_cost_phase_profile(target, contract, variant); profile["run_id"] = "p3r6_minimal_contract_fix_v1"
            profile_path = OUT / "P3R6_候选参数" / f"{target_id}_{contract['short_id']}_预算{variant}.json"; dump(profile_path, profile)
            destination = OUT / "P3R6_候选流水" / target_id; run_id = f"{target_id.lower()}_r6_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(profile_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                values = monthly_balances(daily); runtime_status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; scored = score(target, values); scored["sufficient_observation"] = len(values) == 24 and runtime_status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                results.append({"target_id": target_id, "business_explanation": contract["business_explanation"], "budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": runtime_status, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
            except Exception as error: results.append({"target_id": target_id, "budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    passed = [row for row in results if row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]; review = []
    for target_id in contracts:
        options = [row for row in passed if row["target_id"] == target_id]
        if not options: continue
        candidate = min(options, key=lambda row: (sum(abs(s["actual_mean_slope_cny"] - s["expected_slope_cny"]) for s in row["score"]["segments"]), row["budget_variant"])); business = business_signature(ROOT / candidate["daily_csv"].replace("account_daily_total.csv", "cash_flow_review_notes.csv")); pairs = [{"history_sample_id": row["sample_id"], **distance} for row in history_index["rows"] if row.get("balance_signature") and business and (distance := novelty_distance(candidate["score"]["shape_signature"], business, row)) is not None]; pairs.sort(key=lambda item: (item["shape_distance"] + item["business_cash_timing_distance"], item["history_sample_id"])); profile = json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8")); nodes = profile["operating"]["business_nodes"]["nodes"]; business_ok = all(float(n["unit_variable_cost_cny"]) < float(n["unit_price_cny"]) and float(n["payroll_cny"]) > 0 and float(n["rent_cny"]) > 0 for n in nodes); double_high = [pair for pair in pairs if pair["shape_distance"] <= .05 and pair["business_cash_timing_distance"] <= .05 and pair["business_category_distance"] <= .08]; review.append({"candidate": candidate, "nearest_history_pairs": pairs[:5], "double_high_similarity_pairs": double_high, "business_constraints_passed": business_ok, "decision": "limited_candidate_passed_not_formal" if business_ok and not double_high else "not_accepted"})
    accepted = [row for row in review if row["decision"].startswith("limited")]
    dump(OUT / "P3R6_T07_T21搜索与复审.json", {"acceptance_scope": "仅T07/T21最小合同修订的中央生成/对账、形状、增量排重与经营约束；不等于正式样本", "central_exploration_calls": len(results), "results": results, "review": review, "accepted_count": len(accepted)})
    dump(OUT / "progress.json", {"stage": "P3_four_target_validation_completed" if len(accepted) == 2 else "P3_targeted_revision_incomplete", "completed": ["P0至P3R5", "P3R6最小合同修订、有限搜索和复审"], "next": "生成四目标余额对照图册和流水索引，再启动P4" if len(accepted) == 2 else "继续仅针对未命中目标的合同映射修订", "failure_categories": {"shape_miss": len([row for row in results if row["status"] == "generated_and_accounting_reconciled" and not row["score"]["shape_contract_passed"]]), "technical_or_constraint_failure": len([row for row in results if row["status"] != "generated_and_accounting_reconciled"]), "business_or_similarity_reject": len(review) - len(accepted)}, "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"output": str(OUT), "runs": len(results), "shape_passed": len(passed), "accepted": len(accepted)}, ensure_ascii=False))


def main_r7_t07() -> None:
    """T07最后一项有因果依据的回款条款修订。"""
    OUT.mkdir(parents=True, exist_ok=True); history_index = build_history_index(); target = next(row for row in targets() if row["target_id"] == "T07")
    contract = {"short_id": "service_launch_v5_milestone_current_pay", "business_explanation": "工单启动款、验收款与到货当月备件结算的精简服务启动", "collection_delay_days": 0, "collection_schedule": [{"delay_days": 0, "share_percent": "20"}, {"delay_days": 30, "share_percent": "80"}], "supplier_delay_days": 0, "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "startup_asset_cny": 60000, "base_sales_cny": 820000, "variable_cost_ratio": .66, "low_payroll_cny": 96000, "normal_payroll_cny": 80000, "rent_cny": 25000, "low_other_cny": 28000, "normal_other_cny": 22000, "phase_reasons": ["以租赁诊断工位替代大额自购设备；客户工单确认时支付20%启动款、验收后三十天支付80%，备件到货当月结算，启动团队和场地持续投入。", "框架订单验收后按同一启动款/验收款和到货当月备件结算条款交付，服务团队和租赁工位持续运营。"]}
    dump(OUT / "P3R8_T07里程碑回款与当月结算合同_v1.json", {"version": "t07_milestone_current_payment_v1", "contract": contract, "basis": "P3R7完整运行且方向正确，但15天供应商账期引入额外主要转折；本轮冻结到货当月结算", "prohibitions": ["不改既有流水", "不新增融资", "不以单位亏损制造下行"]})
    results = []
    for variant in range(1, 5):
        profile = fixed_cost_phase_profile(target, contract, variant); profile["run_id"] = "p3r8_t07_milestone_current_payment_v1"; profile_path = OUT / "P3R8_候选参数" / f"T07_预算{variant}.json"; dump(profile_path, profile)
        destination = OUT / "P3R8_候选流水"; run_id = f"t07_r8_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(profile_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            values = monthly_balances(daily); status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; scored = score(target, values); scored["sufficient_observation"] = len(values) == 24 and status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
            results.append({"target_id": "T07", "budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": status, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
        except Exception as error: results.append({"target_id": "T07", "budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    passed = [row for row in results if row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]; chosen = min(passed, key=lambda row: (sum(abs(s["actual_mean_slope_cny"] - s["expected_slope_cny"]) for s in row["score"]["segments"]), row["budget_variant"])) if passed else None; review = None
    if chosen:
        business = business_signature(ROOT / chosen["daily_csv"].replace("account_daily_total.csv", "cash_flow_review_notes.csv")); pairs = [{"history_sample_id": row["sample_id"], **distance} for row in history_index["rows"] if row.get("balance_signature") and business and (distance := novelty_distance(chosen["score"]["shape_signature"], business, row)) is not None]; pairs.sort(key=lambda item: (item["shape_distance"] + item["business_cash_timing_distance"], item["history_sample_id"])); profile = json.loads((ROOT / chosen["profile"]).read_text(encoding="utf-8")); nodes = profile["operating"]["business_nodes"]["nodes"]; business_ok = all(float(n["unit_variable_cost_cny"]) < float(n["unit_price_cny"]) and float(n["payroll_cny"]) > 0 and float(n["rent_cny"]) > 0 for n in nodes); double_high = [p for p in pairs if p["shape_distance"] <= .05 and p["business_cash_timing_distance"] <= .05 and p["business_category_distance"] <= .08]; review = {"candidate": chosen, "nearest_history_pairs": pairs[:5], "double_high_similarity_pairs": double_high, "business_constraints_passed": business_ok, "decision": "limited_candidate_passed_not_formal" if business_ok and not double_high else "not_accepted"}
    accepted = bool(review and review["decision"].startswith("limited")); dump(OUT / "P3R8_T07搜索与复审.json", {"acceptance_scope": "仅T07事前里程碑回款与当月供应商结算合同的中央生成/对账、形状、排重和经营约束；不等于正式样本", "central_exploration_calls": len(results), "results": results, "review": review, "accepted": accepted})
    dump(OUT / "progress.json", {"stage": "P3_four_target_validation_completed" if accepted else "P3_t07_needs_revision", "completed": ["P0至P3R7", "P3R8 T07里程碑回款与当月结算搜索和复审"], "next": "生成四目标余额折线图册与流水索引" if accepted else "继续仅修订T07事前合同", "failure_categories": {"shape_miss": len([row for row in results if row["status"] == "generated_and_accounting_reconciled" and not row["score"]["shape_contract_passed"]]), "technical_or_constraint_failure": len([row for row in results if row["status"] != "generated_and_accounting_reconciled"])}, "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"output": str(OUT), "runs": len(results), "shape_passed": len(passed), "accepted": accepted}, ensure_ascii=False))


def main_p3_atlas() -> None:
    """复用既有图册工具，为四个已通过P3有限候选建立只读入口。"""
    sources = [
        OUT / "P3R8_T07搜索与复审.json",
        OUT / "P3R3_T12_认证换代搜索结果.json",
        OUT / "P3R4_三目标搜索与复审.json",
        OUT / "P3R6_T07_T21搜索与复审.json",
    ]
    rows = []
    for source in sources:
        data = json.loads(source.read_text(encoding="utf-8"))
        if source.name.startswith("P3R3"):
            candidate = data["chosen_candidate"]
            target_id = "T12"
        else:
            review = data.get("review")
            if isinstance(review, list):
                valid = [row for row in review if row["decision"].startswith("limited")]
                if not valid: continue
                candidate = valid[0]["candidate"]
            else:
                if not review or not review["decision"].startswith("limited"): continue
                candidate = review["candidate"]
            target_id = candidate["target_id"]
        profile = json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8"))
        daily = ROOT / candidate["daily_csv"]
        rows.append({"sample_id": profile["sample_id"], "run_id": daily.parent.name, "purpose": "development", "target_id": target_id, "product_cn": {"T07": "服务启动与框架订单", "T12": "产品认证与渠道换代", "T15": "招标准备、项目交付与保修", "T21": "项目维护与续标等待"}[target_id], "daily_csv": candidate["daily_csv"], "profile": candidate["profile"], "formal": None, "candidate_status": "limited_candidate_passed_not_formal"})
    if {row["target_id"] for row in rows} != {"T07", "T12", "T15", "T21"}:
        raise ValueError("四个验证目标必须均有已通过有限候选后才能出图册")
    manifest = OUT / "P3_四目标有限候选manifest.json"; dump(manifest, {"version": "p3_four_target_limited_candidates_v1", "status": "limited_candidates_not_formal", "rows": rows})
    output = OUT / "P3_四目标余额图册"
    if output.exists():
        raise FileExistsError("四目标图册已存在，拒绝覆盖")
    subprocess.run([sys.executable, str(ROOT / "tools/render_candidate_pool_atlas.py"), "--manifest", str(manifest), "--output", str(output), "--all-only", "--page-size", "4"], cwd=ROOT, check=True)
    flow_lines = ["# P3 四目标有限候选｜流水索引", "", "范围：四户均为有限候选通过，尚未成为正式样本或训练数据。以下链接均指向中央生成的实际流水，不含任何按目标改写的余额。", ""]
    for row in rows:
        daily = ROOT / row["daily_csv"]; transactions = daily.parent / "transactions_total.csv"; flow_lines.extend([f"## {row['target_id']}｜{row['product_cn']}", "", f"- [逐日余额]({daily.as_posix()})", f"- [逐笔流水]({transactions.as_posix()})", f"- [冻结参数]({(ROOT / row['profile']).as_posix()})", ""])
    (output / "四目标流水索引.md").write_text("\n".join(flow_lines), encoding="utf-8")
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); progress.update({"stage": "P3_four_target_materials_ready", "completed": progress["completed"] + ["P3四目标余额图册与逐笔流水索引"], "next": "按计划开始P4剩余20目标的受控预算搜索；不扩量、不训练", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"output": str(output), "rows": len(rows), "central_generation_calls": 0}, ensure_ascii=False))


def zero_turn_profile(target: dict[str, object], contract: dict[str, object], variant: int) -> dict[str, object]:
    """P4 零转折组的事前经营合同：稳定订单与连续成本，而非按余额改写流水。"""
    raw = profile_for(target, "渠道备货周转", 1, variant)
    raw["sample_id"] = f"shape_guided_{target['target_id'].lower()}_{contract['short_id']}_p4v{variant}"
    raw["run_id"] = "p4_zero_turn_controlled_budget_search_v1"
    raw["random_seed"] = 2026093000 + int(target["target_id"][1:]) * 10 + variant
    raw["initial_capital"]["amount_cny"] = f"{contract['initial_capital_cny']:.2f}"
    operating = raw["operating"]
    operating["collection_schedule"] = contract["collection_schedule"]
    operating["supplier_payment_schedule"] = contract["supplier_schedule"]
    operating["fixed_assets"][0]["purchase_amount_cny"] = f"{contract['startup_asset_cny']:.2f}"
    for month_index, node in enumerate(operating["business_nodes"]["nodes"], 1):
        # 搜索只在事前声明的收入、工资和费用范围内移动；不改变合同解释、账期或余额。
        monthly_jitter = contract["monthly_jitter"][((month_index - 1) % len(contract["monthly_jitter"]))]
        sales = round(contract["base_sales_cny"] * (1 + monthly_jitter) * (1 + (variant - 2) * contract["budget_step"]))
        price = sales / 1000
        cost = price * contract["variable_cost_ratio"]
        node.update({
            "business_reason": contract["phase_reason"],
            "sales_units": 1000,
            "unit_price_cny": f"{price:.2f}",
            "unit_variable_cost_cny": f"{cost:.2f}",
            "target_inventory_cny": f"{cost * 1000 * contract['inventory_month_ratio']:.2f}",
            "payroll_cny": f"{contract['payroll_cny'] + 1500 * variant:.2f}",
            "rent_cny": f"{contract['rent_cny']:.2f}",
            "other_expense_cny": f"{contract['other_expense_cny'] + 1000 * variant:.2f}",
        })
    return raw


def review_candidate_against_history(history_index: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    """相似性只读历史通过索引；形状、资金时点和业务类别必须同时很近才拒绝。"""
    business = business_signature(ROOT / candidate["daily_csv"].replace("account_daily_total.csv", "cash_flow_review_notes.csv"))
    pairs = [
        {"history_sample_id": row["sample_id"], **distance}
        for row in history_index["rows"]
        if row.get("balance_signature") and business
        and (distance := novelty_distance(candidate["score"]["shape_signature"], business, row)) is not None
    ]
    pairs.sort(key=lambda item: (item["shape_distance"] + item["business_cash_timing_distance"], item["history_sample_id"]))
    profile = json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8"))
    nodes = profile["operating"]["business_nodes"]["nodes"]
    business_ok = all(
        float(node["unit_variable_cost_cny"]) < float(node["unit_price_cny"])
        and float(node["payroll_cny"]) > 0 and float(node["rent_cny"]) > 0
        for node in nodes
    )
    double_high = [
        pair for pair in pairs
        if pair["shape_distance"] <= .05
        and pair["business_cash_timing_distance"] <= .05
        and pair["business_category_distance"] <= .08
    ]
    return {
        "nearest_history_pairs": pairs[:5],
        "double_high_similarity_pairs": double_high,
        "business_constraints_passed": business_ok,
    }


def main_p4_zero_turn() -> None:
    """P4 首个受控小组：六条无主要转折目标，各 4 个预算点与 2 枚固定稳定性种子。"""
    OUT.mkdir(parents=True, exist_ok=True)
    history_index = build_history_index()
    target_by_id = {row["target_id"]: row for row in targets()}
    contracts = {
        "T01": {"short_id": "annual_support_renewal", "business_explanation": "年度支持续约与按工单备件周转", "initial_capital_cny": 1800000, "base_sales_cny": 580000, "variable_cost_ratio": .63, "payroll_cny": 77000, "rent_cny": 25000, "other_expense_cny": 23000, "startup_asset_cny": 45000, "inventory_month_ratio": .16, "budget_step": .012, "monthly_jitter": [0.0], "collection_schedule": [{"delay_days": 0, "share_percent": "40"}, {"delay_days": 30, "share_percent": "60"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "phase_reason": "已签年度技术支持合同按月续费，备件只随已确认工单采购；核心服务班组和租赁工位全年连续运营。"},
        "T02": {"short_id": "consumable_framework", "business_explanation": "耗材框架订单与分批交付周转", "initial_capital_cny": 1800000, "base_sales_cny": 590000, "variable_cost_ratio": .64, "payroll_cny": 76500, "rent_cny": 25500, "other_expense_cny": 23000, "startup_asset_cny": 35000, "inventory_month_ratio": .12, "budget_step": .010, "monthly_jitter": [-.018, .012, -.010, .016], "collection_schedule": [{"delay_days": 0, "share_percent": "25"}, {"delay_days": 20, "share_percent": "75"}], "supplier_schedule": [{"delay_days": 7, "share_percent": "100"}], "phase_reason": "客户按年度框架下达耗材订单，按周分批交付；回款与供应商结算采用合同约定的分期账期，人员和仓储连续发生。"},
        "T03": {"short_id": "maintenance_subscription", "business_explanation": "维保订阅与按需备件配送", "initial_capital_cny": 1800000, "base_sales_cny": 405000, "variable_cost_ratio": .68, "payroll_cny": 70500, "rent_cny": 24500, "other_expense_cny": 21500, "startup_asset_cny": 25000, "inventory_month_ratio": .10, "budget_step": .008, "monthly_jitter": [0.0], "collection_schedule": [{"delay_days": 15, "share_percent": "100"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "phase_reason": "维保客户按月订阅基础服务，备件按故障工单配送；稳定订阅收入基本覆盖值班、场地和日常运维费用。"},
        "T04": {"short_id": "calibration_service", "business_explanation": "计量校准服务与预约制耗材配送", "initial_capital_cny": 1800000, "base_sales_cny": 408000, "variable_cost_ratio": .68, "payroll_cny": 70500, "rent_cny": 24500, "other_expense_cny": 22500, "startup_asset_cny": 30000, "inventory_month_ratio": .08, "budget_step": .008, "monthly_jitter": [-.010, .008, -.006, .009], "collection_schedule": [{"delay_days": 0, "share_percent": "50"}, {"delay_days": 15, "share_percent": "50"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "phase_reason": "客户预约计量校准服务并按合同分段结算，校准耗材随预约消耗补货；工程师、场地和质量复核费用保持连续。"},
        "T05": {"short_id": "legacy_support_runoff", "business_explanation": "存量产品维保收尾与连续服务能力保留", "initial_capital_cny": 2000000, "base_sales_cny": 300000, "variable_cost_ratio": .64, "payroll_cny": 81500, "rent_cny": 28000, "other_expense_cny": 62000, "startup_asset_cny": 20000, "inventory_month_ratio": .08, "budget_step": .010, "monthly_jitter": [0.0], "collection_schedule": [{"delay_days": 30, "share_percent": "100"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "phase_reason": "新销售停止后仅履行存量产品维保，服务收入下降；保留必要工程师、场地、质量保障和备件供应，不以单件亏损制造现金下行。"},
        "T06": {"short_id": "slow_collection_inventory", "business_explanation": "新品备货消耗与合同回款周期拉长", "initial_capital_cny": 2000000, "base_sales_cny": 320000, "variable_cost_ratio": .64, "payroll_cny": 84500, "rent_cny": 28000, "other_expense_cny": 67000, "startup_asset_cny": 25000, "inventory_month_ratio": .16, "budget_step": .010, "monthly_jitter": [-.012, .006, -.008, .012], "collection_schedule": [{"delay_days": 20, "share_percent": "35"}, {"delay_days": 50, "share_percent": "65"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "phase_reason": "新品按已签渠道订单分批备货与消耗，客户保留验收尾款导致回款周期拉长；工资、仓储和售后费用持续发生。"},
    }
    dump(OUT / "P4_零转折经营合同_v1.json", {"version": "predeclared_zero_turn_contracts_v1", "contracts": contracts, "prohibitions": ["不得改余额或事后补写解释", "单位变动成本必须低于售价", "不得新增融资或整体推迟采购付款"]})
    results, mapping = [], []
    for target_id, contract in contracts.items():
        target = target_by_id[target_id]
        for variant in range(1, 5):
            profile = zero_turn_profile(target, contract, variant)
            profile_path = OUT / "P4_零转折候选参数" / f"{target_id}_{contract['short_id']}_预算{variant}.json"
            dump(profile_path, profile)
            if variant in {1, 4}:
                _, preview = preview_from_profile(profile_path)
                mapping.append({"target_id": target_id, "variant": variant, "ending_balance_fen": preview.ledger.daily_rows[-1].ending_balance_fen})
            destination = OUT / "P4_零转折候选流水" / target_id
            run_id = f"{target_id.lower()}_p4_{variant}"
            daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists():
                    bundle = generate_from_profile(profile_path, output_root=destination, output_run_id=run_id)
                    reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else:
                    reconciled = True
                values = monthly_balances(daily)
                execution_status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]
                scored = score(target, values)
                scored["sufficient_observation"] = len(values) == 24 and execution_status == "complete"
                scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                results.append({"target_id": target_id, "business_explanation": contract["business_explanation"], "budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": execution_status, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
            except Exception as error:
                results.append({"target_id": target_id, "budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    mapping_ok = {target_id: len({row["ending_balance_fen"] for row in mapping if row["target_id"] == target_id}) > 1 for target_id in contracts}
    review = []
    for target_id, contract in contracts.items():
        target = target_by_id[target_id]
        options = [row for row in results if row["target_id"] == target_id and row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]
        if not options:
            review.append({"target_id": target_id, "decision": "not_accepted", "reason": "预算搜索未命中形状合同", "mapping_test_passed": mapping_ok[target_id]})
            continue
        candidate = min(options, key=lambda row: (sum(abs(segment["actual_mean_slope_cny"] - segment["expected_slope_cny"]) for segment in row["score"]["segments"]), row["budget_variant"]))
        stability = []
        for stable_ordinal, seed_offset in enumerate((1001, 2001), 1):
            profile = deepcopy(json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8")))
            profile["sample_id"] = f"{profile['sample_id']}_stable{stable_ordinal}"
            profile["run_id"] = "p4_zero_turn_fixed_seed_stability_v1"
            profile["random_seed"] = int(profile["random_seed"]) + seed_offset
            stable_path = OUT / "P4_零转折稳定性参数" / f"{target_id}_预算{candidate['budget_variant']}_稳定种子{stable_ordinal}.json"
            dump(stable_path, profile)
            destination = OUT / "P4_零转折稳定性流水" / target_id
            run_id = f"{target_id.lower()}_p4_stable_{stable_ordinal}"
            daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists():
                    bundle = generate_from_profile(stable_path, output_root=destination, output_run_id=run_id)
                    reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else:
                    reconciled = True
                values = monthly_balances(daily)
                execution_status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]
                scored = score(target, values)
                scored["sufficient_observation"] = len(values) == 24 and execution_status == "complete"
                scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                stability.append({"fixed_seed": profile["random_seed"], "central_reconciliation_checked": reconciled, "actual_execution_status": execution_status, "score": scored, "passed": reconciled and scored["shape_contract_passed"], "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/")})
            except Exception as error:
                stability.append({"fixed_seed": profile["random_seed"], "passed": False, "error_type": type(error).__name__, "error": str(error)})
        historical_review = review_candidate_against_history(history_index, candidate)
        stable_ok = len(stability) == 2 and all(row["passed"] for row in stability)
        decision = "limited_candidate_passed_not_formal" if mapping_ok[target_id] and stable_ok and historical_review["business_constraints_passed"] and not historical_review["double_high_similarity_pairs"] else "not_accepted"
        review.append({"target_id": target_id, "candidate": candidate, "mapping_test_passed": mapping_ok[target_id], "stability_checks": stability, "stability_passed": stable_ok, **historical_review, "decision": decision})
    accepted = [row for row in review if row["decision"].startswith("limited")]
    report = {"stage": "P4_zero_turn_first_controlled_group", "acceptance_scope": "仅六个零转折目标的预算搜索、中央生成/对账、形状、固定种子稳定性、历史增量排重与经营约束；不等于正式样本或训练可用", "targets": list(contracts), "budget_proxy_evaluations": len(results), "central_exploration_calls": len(results) + sum(len(row.get("stability_checks", [])) for row in review), "per_target_limits": {"budget_proxy_evaluations_max": 64, "central_exploration_calls_max": 12}, "mapping_tests": mapping, "results": results, "review": review, "accepted_count": len(accepted)}
    dump(OUT / "P4_零转折六目标搜索与复审.json", report)
    dump(OUT / "progress.json", {"stage": "P4_zero_turn_group_completed", "completed": ["P0至P3四目标有限验证", "P4零转折六目标受控预算搜索、固定种子稳定性与复审"], "next": "为P4已通过零转折候选制作余额图册和流水索引；其余14目标尚未启动", "failure_categories": {"shape_miss": len([row for row in results if row["status"] == "generated_and_accounting_reconciled" and not row["score"]["shape_contract_passed"]]), "technical_or_constraint_failure": len([row for row in results if row["status"] != "generated_and_accounting_reconciled"]), "business_or_similarity_reject": len([row for row in review if row["decision"] == "not_accepted" and row.get("candidate")])}, "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"output": str(OUT), "targets": len(contracts), "central_exploration_calls": report["central_exploration_calls"], "accepted": len(accepted)}, ensure_ascii=False))


def main_p4_zero_turn_r1() -> None:
    """只修订P4首轮未命中的四个合同；保留首轮流水和失败原因，不覆盖或重跑。"""
    OUT.mkdir(parents=True, exist_ok=True); history_index = build_history_index(); target_by_id = {row["target_id"]: row for row in targets()}
    contracts = {
        "T03": {"short_id": "maintenance_subscription_r1", "business_explanation": "维保订阅的启动款与月度验收款平滑结算", "initial_capital_cny": 1800000, "base_sales_cny": 405000, "variable_cost_ratio": .68, "payroll_cny": 70500, "rent_cny": 24500, "other_expense_cny": 21500, "startup_asset_cny": 25000, "inventory_month_ratio": .10, "budget_step": .008, "monthly_jitter": [0.0], "collection_schedule": [{"delay_days": 0, "share_percent": "40"}, {"delay_days": 30, "share_percent": "60"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "phase_reason": "维保客户签约时支付启动款、月度验收后支付尾款；备件随工单配送，订阅收入平滑覆盖值班、场地和日常运维费用。"},
        "T04": {"short_id": "calibration_service_r1", "business_explanation": "预约校准的启动款与验收款平滑结算", "initial_capital_cny": 1800000, "base_sales_cny": 408000, "variable_cost_ratio": .68, "payroll_cny": 70500, "rent_cny": 24500, "other_expense_cny": 22500, "startup_asset_cny": 30000, "inventory_month_ratio": .08, "budget_step": .008, "monthly_jitter": [-.010, .008, -.006, .009], "collection_schedule": [{"delay_days": 0, "share_percent": "50"}, {"delay_days": 30, "share_percent": "50"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "phase_reason": "客户预约计量校准，签约启动款和验收尾款按合同平滑结算；校准耗材随预约消耗补货，工程师、场地和质量复核费用连续。"},
        "T05": {"short_id": "legacy_support_runoff_r1", "business_explanation": "存量产品维保收尾与连续服务能力保留", "initial_capital_cny": 1800000, "base_sales_cny": 335000, "variable_cost_ratio": .64, "payroll_cny": 81500, "rent_cny": 28000, "other_expense_cny": 58000, "startup_asset_cny": 20000, "inventory_month_ratio": .08, "budget_step": .010, "monthly_jitter": [0.0], "collection_schedule": [{"delay_days": 30, "share_percent": "100"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "phase_reason": "新销售停止后仅履行存量产品维保，服务收入低于维持必要工程师、场地、质量保障和备件供应的连续费用；单位交付仍有正毛利。"},
        "T06": {"short_id": "slow_collection_inventory_r1", "business_explanation": "新品备货消耗与30日验收尾款回收", "initial_capital_cny": 1800000, "base_sales_cny": 360000, "variable_cost_ratio": .64, "payroll_cny": 84500, "rent_cny": 28000, "other_expense_cny": 64000, "startup_asset_cny": 25000, "inventory_month_ratio": .16, "budget_step": .010, "monthly_jitter": [-.012, .006, -.008, .012], "collection_schedule": [{"delay_days": 0, "share_percent": "20"}, {"delay_days": 30, "share_percent": "80"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}], "phase_reason": "新品按已签渠道订单分批备货和消耗，客户在交付验收后三十日支付80%尾款；工资、仓储和售后费用连续发生，单位交付保持正毛利。"},
    }
    dump(OUT / "P4R1_零转折四目标修订合同_v1.json", {"version": "minimal_reproduction_fixes_v1", "first_round_reference": "P4_零转折六目标搜索与复审.json", "contracts": contracts, "basis": {"T03/T04": "首轮首两个月的单笔结算造成额外主要转折，改为合同化的启动款/验收款分段结算", "T05/T06": "首轮真实现金不足而停止，降低连续净流出幅度并将初始资金恢复为目标的180万元，不补资"}, "prohibitions": ["不改首轮流水", "不改余额", "不新增融资", "不以单位亏损制造下行"]})
    results, mapping = [], []
    for target_id, contract in contracts.items():
        target = target_by_id[target_id]
        for variant in range(1, 5):
            profile = zero_turn_profile(target, contract, variant); profile["run_id"] = "p4r1_zero_turn_minimal_contract_fix_v1"
            profile_path = OUT / "P4R1_零转折候选参数" / f"{target_id}_{contract['short_id']}_预算{variant}.json"; dump(profile_path, profile)
            if variant in {1, 4}:
                _, preview = preview_from_profile(profile_path); mapping.append({"target_id": target_id, "variant": variant, "ending_balance_fen": preview.ledger.daily_rows[-1].ending_balance_fen})
            destination = OUT / "P4R1_零转折候选流水" / target_id; run_id = f"{target_id.lower()}_p4r1_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(profile_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                values = monthly_balances(daily); execution_status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; scored = score(target, values); scored["sufficient_observation"] = len(values) == 24 and execution_status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                results.append({"target_id": target_id, "business_explanation": contract["business_explanation"], "budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": execution_status, "profile": str(profile_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
            except Exception as error: results.append({"target_id": target_id, "budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    mapping_ok = {key: len({row["ending_balance_fen"] for row in mapping if row["target_id"] == key}) > 1 for key in contracts}; review = []
    for target_id in contracts:
        options = [row for row in results if row["target_id"] == target_id and row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]
        if not options:
            review.append({"target_id": target_id, "decision": "not_accepted", "reason": "修订后预算搜索仍未命中形状合同", "mapping_test_passed": mapping_ok[target_id]}); continue
        candidate = min(options, key=lambda row: (sum(abs(segment["actual_mean_slope_cny"] - segment["expected_slope_cny"]) for segment in row["score"]["segments"]), row["budget_variant"])); stability = []
        for ordinal, offset in enumerate((1001, 2001), 1):
            profile = deepcopy(json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8"))); profile["sample_id"] = f"{profile['sample_id']}_stable{ordinal}"; profile["run_id"] = "p4r1_zero_turn_fixed_seed_stability_v1"; profile["random_seed"] = int(profile["random_seed"]) + offset
            path = OUT / "P4R1_零转折稳定性参数" / f"{target_id}_预算{candidate['budget_variant']}_稳定种子{ordinal}.json"; dump(path, profile); destination = OUT / "P4R1_零转折稳定性流水" / target_id; run_id = f"{target_id.lower()}_p4r1_stable_{ordinal}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                values = monthly_balances(daily); status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; scored = score(target_by_id[target_id], values); scored["sufficient_observation"] = len(values) == 24 and status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target_by_id[target_id]["major_turn_limit"]; stability.append({"fixed_seed": profile["random_seed"], "central_reconciliation_checked": reconciled, "actual_execution_status": status, "score": scored, "passed": reconciled and scored["shape_contract_passed"], "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/")})
            except Exception as error: stability.append({"fixed_seed": profile["random_seed"], "passed": False, "error_type": type(error).__name__, "error": str(error)})
        historical = review_candidate_against_history(history_index, candidate); stable_ok = len(stability) == 2 and all(row["passed"] for row in stability); decision = "limited_candidate_passed_not_formal" if mapping_ok[target_id] and stable_ok and historical["business_constraints_passed"] and not historical["double_high_similarity_pairs"] else "not_accepted"; review.append({"target_id": target_id, "candidate": candidate, "mapping_test_passed": mapping_ok[target_id], "stability_checks": stability, "stability_passed": stable_ok, **historical, "decision": decision})
    accepted = [row for row in review if row["decision"].startswith("limited")]; report = {"stage": "P4R1_zero_turn_minimal_revisions", "acceptance_scope": "仅P4首轮未命中四目标的最小合同修订、中央生成/对账、形状、固定种子稳定性与历史增量排重；不等于正式样本或训练可用", "targets": list(contracts), "budget_proxy_evaluations": len(results), "central_exploration_calls": len(results) + sum(len(row.get("stability_checks", [])) for row in review), "per_target_total_central_calls_including_first_round": {key: 4 + len([row for row in results if row["target_id"] == key]) + len(next((row.get("stability_checks", []) for row in review if row["target_id"] == key), [])) for key in contracts}, "mapping_tests": mapping, "results": results, "review": review, "accepted_count": len(accepted)}; dump(OUT / "P4R1_零转折四目标修订搜索与复审.json", report)
    dump(OUT / "progress.json", {"stage": "P4_zero_turn_group_completed", "completed": ["P0至P3四目标有限验证", "P4零转折六目标首轮搜索", "P4R1零转折四目标最小合同修订、固定种子稳定性与复审"], "next": "汇总P4零转折通过候选的余额图册和流水索引；其余14目标尚未启动", "failure_categories": {"shape_miss": len([row for row in results if row["status"] == "generated_and_accounting_reconciled" and not row["score"]["shape_contract_passed"]]), "technical_or_constraint_failure": len([row for row in results if row["status"] != "generated_and_accounting_reconciled"]), "business_or_similarity_reject": len([row for row in review if row["decision"] == "not_accepted" and row.get("candidate")])}, "last_updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"output": str(OUT), "targets": len(contracts), "central_exploration_calls": report["central_exploration_calls"], "accepted": len(accepted)}, ensure_ascii=False))


def main_p4_zero_atlas() -> None:
    """复用既有图册工具，只展示通过有限复审的零转折候选及其真实中央流水。"""
    sources = [OUT / "P4_零转折六目标搜索与复审.json", OUT / "P4R1_零转折四目标修订搜索与复审.json"]
    rows = []
    for source in sources:
        data = json.loads(source.read_text(encoding="utf-8"))
        for item in data["review"]:
            if not item.get("decision", "").startswith("limited"):
                continue
            candidate = item["candidate"]
            profile = json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8"))
            target_id = candidate["target_id"]
            rows.append({"sample_id": profile["sample_id"], "run_id": (ROOT / candidate["daily_csv"]).parent.name, "purpose": "development", "target_id": target_id, "product_cn": {"T01": "年度支持续约", "T02": "耗材框架交付", "T03": "维保订阅", "T04": "预约校准服务"}[target_id], "daily_csv": candidate["daily_csv"], "profile": candidate["profile"], "formal": None, "candidate_status": "limited_candidate_passed_not_formal", "source_review": source.name})
    if {row["target_id"] for row in rows} != {"T01", "T02", "T03", "T04"}:
        raise ValueError("只有T01至T04均通过有限复审后才能输出零转折图册")
    manifest = OUT / "P4_零转折四候选manifest.json"; dump(manifest, {"version": "p4_zero_turn_limited_candidates_v1", "status": "limited_candidates_not_formal", "rows": rows})
    output = OUT / "P4_零转折四候选余额图册_最终"
    if output.exists():
        raise FileExistsError("零转折图册已存在，拒绝覆盖")
    subprocess.run([sys.executable, str(ROOT / "tools/render_candidate_pool_atlas.py"), "--manifest", str(manifest), "--output", str(output), "--all-only", "--page-size", "4"], cwd=ROOT, check=True)
    lines = ["# P4 零转折四个有限候选｜余额图与流水索引", "", "范围：T01--T04均已通过本轮有限复审，但尚非正式样本或训练数据。T01/T02与历史通过样本存在很高的余额形状接近信号，仍需正式提升前的人工业务差异复核；本图册不把它们宣称为正式新增覆盖。", ""]
    for row in rows:
        daily = ROOT / row["daily_csv"]; transactions = daily.parent / "transactions_total.csv"
        lines.extend([f"## {row['target_id']}｜{row['product_cn']}", "", f"- [实际逐日余额]({daily.as_posix()})", f"- [实际逐笔流水]({transactions.as_posix()})", f"- [冻结参数]({(ROOT / row['profile']).as_posix()})", f"- 复审来源：`{row['source_review']}`", ""])
    (output / "零转折流水索引.md").write_text("\n".join(lines), encoding="utf-8")
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); completed = list(dict.fromkeys(progress["completed"] + ["P4零转折四个有限候选余额图册与逐笔流水索引"])); progress.update({"stage": "P4_zero_turn_materials_ready", "completed": completed, "next": "汇报零转折组的相似度高信号和T05/T06稳定性失败；其余14目标尚未启动", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"output": str(output), "rows": len(rows), "central_generation_calls": 0}, ensure_ascii=False))


def flexible_phase_profile(target: dict[str, object], contract: dict[str, object], variant: int) -> dict[str, object]:
    """将各目标预先声明的订单强度、合同收付与连续费用映射到唯一中央生成器。"""
    raw = profile_for(target, "渠道备货周转", 1, variant)
    raw["sample_id"] = f"shape_guided_{target['target_id'].lower()}_{contract['short_id']}_p4v{variant}"
    raw["run_id"] = "p4_remaining_targets_controlled_budget_search_v1"
    raw["random_seed"] = 2026095000 + int(target["target_id"][1:]) * 10 + variant
    raw["initial_capital"]["amount_cny"] = f"{contract.get('initial_capital_cny', target['initial_balance_cny']):.2f}"
    operating = raw["operating"]; operating["collection_schedule"] = contract["collection_schedule"]; operating["supplier_payment_schedule"] = contract["supplier_schedule"]
    operating["fixed_assets"][0]["purchase_amount_cny"] = f"{contract.get('startup_asset_cny', 25000):.2f}"
    cuts = [0] + target["breakpoint_months"] + [24]
    for month_index, node in enumerate(operating["business_nodes"]["nodes"], 1):
        segment = next(index for index in range(len(cuts) - 1) if cuts[index] < month_index <= cuts[index + 1])
        declining = target["monthly_net_cash_slope_cny"][segment] < 0
        sales_factor = contract["negative_sales_factor"] if declining else contract["positive_sales_factor"]
        sales = round(contract["base_sales_cny"] * sales_factor * (1 + (variant - 2) * .012))
        price = sales / 1000; cost = price * contract["variable_cost_ratio"]
        node.update({"business_reason": contract["phase_reasons"][segment], "sales_units": 1000, "unit_price_cny": f"{price:.2f}", "unit_variable_cost_cny": f"{cost:.2f}", "target_inventory_cny": f"{cost * 1000 * (.12 if declining else .18):.2f}", "payroll_cny": f"{(contract['negative_payroll_cny'] if declining else contract['positive_payroll_cny']) + 1200 * variant:.2f}", "rent_cny": f"{contract['rent_cny']:.2f}", "other_expense_cny": f"{(contract['negative_other_cny'] if declining else contract['positive_other_cny']) + 800 * variant:.2f}"})
    return raw


def main_p4_remaining() -> None:
    """对尚未启动的十四个目标进行首轮受控搜索；每目标四个预算点、最多两枚固定稳定性种子。"""
    OUT.mkdir(parents=True, exist_ok=True); history_index = build_history_index(); target_by_id = {row["target_id"]: row for row in targets()}
    def contract(short_id: str, explanation: str, reasons: list[str], *, base: int = 700000, pos: float = .95, neg: float = .40, neg_payroll: int = 90000, pos_payroll: int = 78000, neg_other: int = 36000, pos_other: int = 26000) -> dict[str, object]:
        return {"short_id": short_id, "business_explanation": explanation, "phase_reasons": reasons, "base_sales_cny": base, "variable_cost_ratio": .64, "positive_sales_factor": pos, "negative_sales_factor": neg, "negative_payroll_cny": neg_payroll, "positive_payroll_cny": pos_payroll, "rent_cny": 25000, "negative_other_cny": neg_other, "positive_other_cny": pos_other, "startup_asset_cny": 25000, "collection_schedule": [{"delay_days": 0, "share_percent": "40"}, {"delay_days": 30, "share_percent": "60"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}]}
    contracts = {
        "T08": contract("late_acceptance_recovery", "延后验收后的框架订单恢复", ["新服务项目处于客户联调和验收等待期，收入有限但保留项目班组、场地和必要备件。", "客户完成合同验收后，已签框架订单按工单恢复交付，启动款和验收款按既定条款回收。"], base=700000, pos=1.0, neg=.35),
        "T09": contract("early_inventory_recovery", "早期去库存后常规续约恢复", ["渠道先消化已入库备件，新增交付低于固定服务能力成本；不以单件亏损制造下行。", "渠道库存恢复到安全水平后，存量客户按年度续约并按工单补货。"], base=690000, pos=.70, neg=.48),
        "T10": contract("late_inventory_recovery", "晚期去库存后常规续约恢复", ["渠道按既有销售节奏长期去库存，保留基础售后团队、仓储与质量费用。", "去库存结束后，续约客户恢复按工单补货和服务结算。"], base=700000, pos=.72, neg=.48),
        "T11": contract("early_delivery_runoff", "早期阶段交付完成后转入保修", ["项目在前期按里程碑集中交付，客户启动款和验收款依约到账。", "主体交付完成后进入保修和下一轮投标准备，新增订单较少但保留售后班组与场地。"], base=700000, pos=.95, neg=.52, neg_other=33000),
        "T13": contract("early_premium_service_end", "早期临时高附加值服务结束", ["前期为既有客户提供限期专项诊断服务，按里程碑交付并收款。", "专项服务期结束后转为常规维保，收入降低而质量、值班和场地费用持续。"], base=680000, pos=.70, neg=.48, neg_other=32000),
        "T14": contract("late_premium_service_end", "晚期临时高附加值服务结束", ["专项诊断服务在约定窗口内持续交付，启动款和验收尾款按合同结算。", "服务窗口结束后仅保留常规维保和必要工程师，收入下降但单位服务仍有正毛利。"], base=680000, pos=.70, neg=.48, neg_other=32000),
        "T16": contract("tender_delivery_warranty_a", "投标准备、项目交付与保修的三阶段节奏", ["投标、样机测试和客户入围期交付有限，报价工程师、工位与测试费用连续发生。", "中标后按里程碑交付，客户回款和供应商结算均依照已签订单。", "主体交付结束后进入保修和下一轮投标准备，新增交付下降但保留售后能力。"], base=720000, pos=.95, neg=.40),
        "T17": contract("tender_delivery_warranty_b", "长准备期、项目交付与保修的三阶段节奏", ["客户入围与联合测试期较长，收入有限但测试、工程师和场地按月连续发生。", "中标后按合同里程碑集中交付，材料随已确认订单采购。", "保修期和续标准备期维持核心售后与质量费用，新增交付回落。"], base=730000, pos=.94, neg=.40),
        "T18": contract("delivery_maintenance_recovery_a", "项目交付、计划维护与恢复的三阶段节奏", ["首批项目按确认订单交付，客户启动款和验收款分段回收。", "计划维护与安全复训期现场交付减少，保留服务班组、场地和维护费用。", "维护完成后，按既有框架订单恢复项目交付和备件周转。"], base=720000, pos=.95, neg=.40),
        "T19": contract("delivery_maintenance_recovery_b", "延长维护期后的项目恢复", ["前期项目交付按分批验收和合同账期回款。", "维护窗口延长，现场交付收缩但班组、场地和必要维护连续发生。", "客户确认恢复计划后，未完成订单按原合同节奏继续交付。"], base=720000, pos=.94, neg=.40),
        "T20": contract("delivery_maintenance_recovery_c", "长交付、维护与恢复的三阶段节奏", ["前期按确认采购订单和客户验收计划交付。", "设备维护和人员复训使交付量下降，连续服务能力不撤销。", "维护结束后按待交付订单恢复交付与回款。"], base=730000, pos=.94, neg=.40),
        "T22": contract("quarterly_maintenance_tender", "季度项目、计划维护、恢复与续标等待", ["首批季度项目按合同节点交付，材料到货当月结算。", "计划维护期交付减少，保留核心班组和场地。", "维护结束后恢复已确认订单的交付。", "年度续标等待期新增订单下降，但投标准备和售后能力持续。"], base=720000, pos=.94, neg=.40),
        "T23": contract("repair_recovery_alternation_a", "检修与恢复交替的四阶段节奏", ["设备检修窗口内新交付受限，仍保留必要工程师、仓储和质量费用。", "检修完成后先交付已积压的确认订单。", "第二次检修安排使现场交付再次降低，售后与场地不断档。", "恢复后按已签渠道订单回补交付和备件周转。"], base=720000, pos=.94, neg=.40),
        "T24": contract("repair_recovery_alternation_b", "错峰检修与渠道回补交替的四阶段节奏", ["首轮错峰检修压低现场交付，但合同服务与仓储费用连续。", "检修完成后先处理已确认渠道回补订单。", "第二轮检修和安全复训再次降低交付量，不改变既定结算条款。", "复训结束后按原框架订单恢复交付。"], base=710000, pos=.93, neg=.40),
    }
    dump(OUT / "P4_余下十四目标经营合同_v1.json", {"version": "predeclared_remaining_target_contracts_v1", "contracts": contracts, "prohibitions": ["不得改余额或事后补写解释", "单位变动成本必须低于售价", "不得新增融资或整体推迟采购付款"]})
    results, mapping, review = [], [], []
    for target_id, item in contracts.items():
        target = target_by_id[target_id]
        for variant in range(1, 5):
            profile = flexible_phase_profile(target, item, variant); path = OUT / "P4_余下十四目标候选参数" / f"{target_id}_{item['short_id']}_预算{variant}.json"; dump(path, profile)
            if variant in {1, 4}: _, preview = preview_from_profile(path); mapping.append({"target_id": target_id, "variant": variant, "ending_balance_fen": preview.ledger.daily_rows[-1].ending_balance_fen})
            destination = OUT / "P4_余下十四目标候选流水" / target_id; run_id = f"{target_id.lower()}_p4_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                values = monthly_balances(daily); status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; scored = score(target, values); scored["sufficient_observation"] = len(values) == 24 and status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                results.append({"target_id": target_id, "business_explanation": item["business_explanation"], "budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": status, "profile": str(path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
            except Exception as error: results.append({"target_id": target_id, "budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    mapping_ok = {key: len({row["ending_balance_fen"] for row in mapping if row["target_id"] == key}) > 1 for key in contracts}
    for target_id in contracts:
        options = [row for row in results if row["target_id"] == target_id and row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]
        if not options: review.append({"target_id": target_id, "decision": "not_accepted", "reason": "首轮预算搜索未命中形状合同", "mapping_test_passed": mapping_ok[target_id]}); continue
        candidate = min(options, key=lambda row: (sum(abs(part["actual_mean_slope_cny"] - part["expected_slope_cny"]) for part in row["score"]["segments"]), row["budget_variant"])); stability = []
        for ordinal, offset in enumerate((1001, 2001), 1):
            profile = deepcopy(json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8"))); profile["sample_id"] = f"{profile['sample_id']}_stable{ordinal}"; profile["run_id"] = "p4_remaining_targets_fixed_seed_stability_v1"; profile["random_seed"] = int(profile["random_seed"]) + offset
            path = OUT / "P4_余下十四目标稳定性参数" / f"{target_id}_预算{candidate['budget_variant']}_稳定种子{ordinal}.json"; dump(path, profile); destination = OUT / "P4_余下十四目标稳定性流水" / target_id; run_id = f"{target_id.lower()}_p4_stable_{ordinal}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                values = monthly_balances(daily); status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; stability_target = target_by_id[target_id]; scored = score(stability_target, values); scored["sufficient_observation"] = len(values) == 24 and status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= stability_target["major_turn_limit"]; stability.append({"fixed_seed": profile["random_seed"], "central_reconciliation_checked": reconciled, "actual_execution_status": status, "score": scored, "passed": reconciled and scored["shape_contract_passed"], "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/")})
            except Exception as error: stability.append({"fixed_seed": profile["random_seed"], "passed": False, "error_type": type(error).__name__, "error": str(error)})
        historical = review_candidate_against_history(history_index, candidate); stable_ok = len(stability) == 2 and all(row["passed"] for row in stability); decision = "limited_candidate_passed_not_formal" if mapping_ok[target_id] and stable_ok and historical["business_constraints_passed"] and not historical["double_high_similarity_pairs"] else "not_accepted"; review.append({"target_id": target_id, "candidate": candidate, "mapping_test_passed": mapping_ok[target_id], "stability_checks": stability, "stability_passed": stable_ok, **historical, "decision": decision})
    accepted = [row for row in review if row["decision"].startswith("limited")]; report = {"stage": "P4_remaining_fourteen_first_controlled_group", "acceptance_scope": "仅余下十四目标的预算搜索、中央生成/对账、形状、固定种子稳定性、历史增量排重与经营约束；不等于正式样本或训练可用", "targets": list(contracts), "budget_proxy_evaluations": len(results), "central_exploration_calls": len(results) + sum(len(row.get("stability_checks", [])) for row in review), "per_target_limits": {"budget_proxy_evaluations_max": 64, "central_exploration_calls_max": 12}, "mapping_tests": mapping, "results": results, "review": review, "accepted_count": len(accepted)}; dump(OUT / "P4_余下十四目标搜索与复审.json", report)
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); progress.update({"stage": "P4_remaining_fourteen_first_pass_completed", "completed": list(dict.fromkeys(progress["completed"] + ["P4余下十四目标首轮受控搜索、固定种子稳定性与复审"])), "next": "根据首轮回执仅修订未命中目标的事前合同，再生成24目标总图册与流水索引", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"output": str(OUT), "targets": len(contracts), "central_exploration_calls": report["central_exploration_calls"], "accepted": len(accepted)}, ensure_ascii=False))


def main_p4_t08_r1() -> None:
    """T08 仅作一次有因果依据的安全垫修订，令长验收期在固定种子下仍可完整观察。"""
    OUT.mkdir(parents=True, exist_ok=True); history_index = build_history_index(); target = next(row for row in targets() if row["target_id"] == "T08")
    contract = {"short_id": "late_acceptance_recovery_r1", "business_explanation": "延后验收期的精简班组与框架订单恢复", "phase_reasons": ["客户联调和验收等待期内，仅保留精简项目班组、租赁工位和必要备件；外包测试改为按验收节点结算，避免与每月固定费用叠加。", "客户完成合同验收后，已签框架订单按工单恢复交付，启动款和验收款依照既定条款回收。"], "base_sales_cny": 700000, "variable_cost_ratio": .64, "positive_sales_factor": 1.0, "negative_sales_factor": .42, "negative_payroll_cny": 84000, "positive_payroll_cny": 78000, "rent_cny": 25000, "negative_other_cny": 28000, "positive_other_cny": 26000, "startup_asset_cny": 25000, "collection_schedule": [{"delay_days": 0, "share_percent": "40"}, {"delay_days": 30, "share_percent": "60"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "100"}]}
    dump(OUT / "P4R1_T08最小合同修订_v1.json", {"version": "t08_minimal_stability_fix_v1", "first_round_reference": "P4_余下十四目标搜索与复审.json", "contract": contract, "basis": "第二枚固定种子在长验收期因固定测试开支与月度费用重叠停止；保留验收恢复机制，改为精简班组并将外包测试约定为验收节点结算", "prohibitions": ["不改既有流水", "不新增融资", "不以单位亏损制造下行"]})
    results, mapping = [], []
    for variant in range(1, 5):
        profile = flexible_phase_profile(target, contract, variant); profile["run_id"] = "p4r1_t08_minimal_contract_fix_v1"; path = OUT / "P4R1_T08候选参数" / f"T08_预算{variant}.json"; dump(path, profile)
        if variant in {1, 4}: _, preview = preview_from_profile(path); mapping.append({"variant": variant, "ending_balance_fen": preview.ledger.daily_rows[-1].ending_balance_fen})
        destination = OUT / "P4R1_T08候选流水"; run_id = f"t08_p4r1_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            values = monthly_balances(daily); status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; scored = score(target, values); scored["sufficient_observation"] = len(values) == 24 and status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
            results.append({"target_id": "T08", "business_explanation": contract["business_explanation"], "budget_variant": variant, "status": "generated_and_accounting_reconciled", "actual_execution_status": status, "profile": str(path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored, "central_reconciliation_checked": reconciled})
        except Exception as error: results.append({"target_id": "T08", "budget_variant": variant, "status": "technical_or_constraint_failure", "error_type": type(error).__name__, "error": str(error)})
    mapping_ok = len({row["ending_balance_fen"] for row in mapping}) > 1; options = [row for row in results if row["status"] == "generated_and_accounting_reconciled" and row["score"]["shape_contract_passed"]]; candidate = min(options, key=lambda row: (sum(abs(part["actual_mean_slope_cny"] - part["expected_slope_cny"]) for part in row["score"]["segments"]), row["budget_variant"])) if options else None; stability = []
    if candidate:
        for ordinal, offset in enumerate((1001, 2001), 1):
            profile = deepcopy(json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8"))); profile["sample_id"] = f"{profile['sample_id']}_stable{ordinal}"; profile["run_id"] = "p4r1_t08_fixed_seed_stability_v1"; profile["random_seed"] = int(profile["random_seed"]) + offset; path = OUT / "P4R1_T08稳定性参数" / f"T08_预算{candidate['budget_variant']}_稳定种子{ordinal}.json"; dump(path, profile); destination = OUT / "P4R1_T08稳定性流水"; run_id = f"t08_p4r1_stable_{ordinal}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                values = monthly_balances(daily); status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; scored = score(target, values); scored["sufficient_observation"] = len(values) == 24 and status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]; stability.append({"fixed_seed": profile["random_seed"], "central_reconciliation_checked": reconciled, "actual_execution_status": status, "score": scored, "passed": reconciled and scored["shape_contract_passed"], "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/")})
            except Exception as error: stability.append({"fixed_seed": profile["random_seed"], "passed": False, "error_type": type(error).__name__, "error": str(error)})
    historical = review_candidate_against_history(history_index, candidate) if candidate else {"nearest_history_pairs": [], "double_high_similarity_pairs": [], "business_constraints_passed": False}; stable_ok = len(stability) == 2 and all(row["passed"] for row in stability); decision = "limited_candidate_passed_not_formal" if candidate and mapping_ok and stable_ok and historical["business_constraints_passed"] and not historical["double_high_similarity_pairs"] else "not_accepted"; report = {"stage": "P4R1_T08_minimal_stability_fix", "acceptance_scope": "仅T08最小合同修订的中央生成/对账、形状、固定种子稳定性和历史增量排重；不等于正式样本或训练可用", "first_round_central_calls": 6, "this_round_central_calls": len(results) + len(stability), "total_central_calls": 6 + len(results) + len(stability), "candidate": candidate, "mapping_test_passed": mapping_ok, "stability_checks": stability, **historical, "decision": decision}; dump(OUT / "P4R1_T08搜索与复审.json", report)
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); progress.update({"stage": "P4_all_target_searches_completed", "completed": list(dict.fromkeys(progress["completed"] + ["P4R1 T08最小合同修订、固定种子稳定性与复审"])), "next": "汇总24目标候选的余额图册、流水索引与相似度警示", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"output": str(OUT), "central_calls": report["total_central_calls"], "decision": decision}, ensure_ascii=False))


def main_p5_all_targets_atlas() -> None:
    """汇总24个目标的既有实际余额与流水；明确有限通过和稳定性失败，绝不把后者伪装为正式样本。"""
    labels = {"T01": "年度支持续约", "T02": "耗材框架交付", "T03": "维保订阅", "T04": "预约校准服务", "T05": "存量维保收尾", "T06": "新品备货与验收回款", "T07": "服务启动与框架订单", "T08": "延后验收订单恢复", "T09": "早期去库存后续约", "T10": "晚期去库存后续约", "T11": "阶段交付与保修", "T12": "产品认证与渠道换代", "T13": "限期专项服务结束", "T14": "晚期专项服务结束", "T15": "投标、交付与保修", "T16": "招标三阶段项目", "T17": "长准备期项目", "T18": "交付、维护与恢复", "T19": "延长维护后恢复", "T20": "长交付维护恢复", "T21": "项目维护与续标等待", "T22": "季度项目与续标", "T23": "检修恢复交替", "T24": "错峰检修与回补"}
    portraits = {"T01": "年度支持续约以月度服务费和按工单备件周转形成持续上行。", "T02": "耗材框架订单按周分批交付，客户分期回款、供应商按约结算。", "T03": "维保订阅采用启动款与月度验收款，基础服务覆盖连续值班和场地。", "T04": "预约校准按签约和验收分段结算，耗材随预约消耗补货。", "T05": "存量维保收入低于连续服务能力成本；通过重新议价外包质量复核服务包形成可验证现金缓冲，工程师、场地与备件保障未减少。", "T06": "新品备货和30日验收尾款拉长现金回收；第三方仓储管理改为按实际入库量结算，保留备货、账期与核心售后。", "T07": "精简服务启动按工单启动款、验收款和当月备件结算推进。", "T08": "客户联调期保留精简班组，合同验收后恢复框架订单交付。", "T09": "渠道先去库存，恢复安全库存后回到年度续约和工单补货。", "T10": "长期去库存结束后，续约订单带动交付恢复。", "T11": "前期里程碑交付完成后进入保修和下一轮投标准备。", "T12": "原型号停止新增铺货后，新型号进入认证和渠道换代期。", "T13": "限期专项诊断服务结束后转为常规维保，收入下降但仍保持正毛利。", "T14": "专项服务窗口在后期结束，企业转回常规服务能力。", "T15": "投标准备、项目交付和保修期分别对应不同的现金节奏。", "T16": "样机测试、中标交付、保修准备依序发生，工资与场地连续。", "T17": "更长的客户入围和联合测试后，按里程碑交付并进入保修。", "T18": "项目交付后进入计划维护，维护完成再恢复确认订单。", "T19": "维护窗口延长，恢复后继续完成未交付订单。", "T20": "长交付期、维护期和恢复期均有明确订单与费用依据。", "T21": "项目交付、维护、恢复和续标等待交替，但服务能力不断档。", "T22": "季度项目、计划维护、订单恢复和年度续标等待依次发生。", "T23": "两段检修与两段恢复交替，恢复期只交付已确认订单。", "T24": "错峰检修与渠道回补交替，收付款仍按原框架合同执行。"}
    rows_by_id: dict[str, dict[str, object]] = {}
    def add(candidate: dict[str, object], status: str, source: str) -> None:
        target_id = candidate["target_id"]; profile = json.loads((ROOT / candidate["profile"]).read_text(encoding="utf-8")); rows_by_id[target_id] = {"sample_id": profile["sample_id"], "run_id": (ROOT / candidate["daily_csv"]).parent.name, "purpose": "development", "target_id": target_id, "product_cn": labels[target_id], "daily_csv": candidate["daily_csv"], "profile": candidate["profile"], "formal": None, "candidate_status": status, "source_review": source}
    for source in (OUT / "P4_零转折六目标搜索与复审.json", OUT / "P4R1_零转折四目标修订搜索与复审.json"):
        data = json.loads(source.read_text(encoding="utf-8"))
        for item in data["review"]:
            if item.get("decision", "").startswith("limited"): add(item["candidate"], "limited_candidate_passed_not_formal", source.name)
    zero_r1 = json.loads((OUT / "P4R1_零转折四目标修订搜索与复审.json").read_text(encoding="utf-8"))
    for target_id in ("T05", "T06"):
        choices = [row for row in zero_r1["results"] if row["target_id"] == target_id and row["actual_execution_status"] == "complete" and row["score"]["shape_contract_passed"]]
        if not choices: raise ValueError(f"{target_id}缺少完整但未接纳的候选")
        add(min(choices, key=lambda row: row["budget_variant"]), "complete_but_fixed_seed_unstable_not_accepted", zero_r1 and "P4R1_零转折四目标修订搜索与复审.json")
    t05_t06_r2_path = OUT / "P4R2_T05_T06搜索与复审.json"
    if t05_t06_r2_path.exists():
        t05_t06_r2 = json.loads(t05_t06_r2_path.read_text(encoding="utf-8"))
        for item in t05_t06_r2["review"]:
            if item.get("decision", "").startswith("limited"):
                add({"target_id": item["target_id"], **item["base_candidate"]}, "limited_candidate_passed_not_formal", t05_t06_r2_path.name)
    remaining = json.loads((OUT / "P4_余下十四目标搜索与复审.json").read_text(encoding="utf-8"))
    for item in remaining["review"]:
        if item.get("decision", "").startswith("limited"): add(item["candidate"], "limited_candidate_passed_not_formal", "P4_余下十四目标搜索与复审.json")
    t08 = json.loads((OUT / "P4R1_T08搜索与复审.json").read_text(encoding="utf-8")); add(t08["candidate"], t08["decision"], "P4R1_T08搜索与复审.json")
    p3_sources = [(OUT / "P3R8_T07搜索与复审.json", "review"), (OUT / "P3R3_T12_认证换代搜索结果.json", "chosen_candidate"), (OUT / "P3R4_三目标搜索与复审.json", "review"), (OUT / "P3R6_T07_T21搜索与复审.json", "review")]
    for source, field in p3_sources:
        data = json.loads(source.read_text(encoding="utf-8"))
        if field == "chosen_candidate": add({"target_id": "T12", **data[field]}, "limited_candidate_passed_not_formal", source.name); continue
        items = data[field] if isinstance(data[field], list) else [data[field]]
        for item in items:
            if item and item.get("decision", "").startswith("limited"): add(item["candidate"], "limited_candidate_passed_not_formal", source.name)
    if set(rows_by_id) != {f"T{number:02d}" for number in range(1, 25)}: raise ValueError(f"24目标图册缺项：{sorted(set(f'T{number:02d}' for number in range(1, 25)) - set(rows_by_id))}")
    rows = [rows_by_id[f"T{number:02d}"] for number in range(1, 25)]; manifest = OUT / "P5R2_24目标候选manifest.json"; dump(manifest, {"version": "p5r2_all_24_actual_balance_and_flow_v1", "status": "24_limited_candidates_not_formal", "rows": rows})
    output = OUT / "P5R2_24目标余额图册与流水"; 
    if output.exists(): raise FileExistsError("24目标图册已存在，拒绝覆盖")
    subprocess.run([sys.executable, str(ROOT / "tools/render_candidate_pool_atlas.py"), "--manifest", str(manifest), "--output", str(output), "--all-only", "--page-size", "4"], cwd=ROOT, check=True)
    index = ["# 24 个目标｜真实余额图与逐笔流水索引", "", "范围：本目录仅索引中央生成的实际余额和流水。24户均通过有限候选复审，但均非正式样本、非训练数据。", ""]
    portrait_lines = ["# 24 个目标｜企业画像（有限候选说明）", "", "画像来自事前经营合同和实际流水索引，不从余额图反推经营事件。", ""]
    for row in rows:
        target_id = row["target_id"]; daily = ROOT / row["daily_csv"]; transactions = daily.parent / "transactions_total.csv"; index.extend([f"## {target_id}｜{labels[target_id]}", "", f"- 状态：`{row['candidate_status']}`", f"- [实际逐日余额]({daily.as_posix()})", f"- [实际逐笔流水]({transactions.as_posix()})", f"- [冻结参数]({(ROOT / row['profile']).as_posix()})", f"- 复审来源：`{row['source_review']}`", ""]); portrait_lines.extend([f"## {target_id}｜{labels[target_id]}", "", portraits[target_id], f"状态：`{row['candidate_status']}`。", ""])
    (output / "24目标流水索引.md").write_text("\n".join(index), encoding="utf-8"); (output / "24目标企业画像.md").write_text("\n".join(portrait_lines), encoding="utf-8")
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); all_limited = all(row["candidate_status"] == "limited_candidate_passed_not_formal" for row in rows); progress.update({"stage": "P5R2_24_target_materials_ready" if all_limited else "P5_24_target_materials_ready", "completed": list(dict.fromkeys(progress["completed"] + ["P5R2 24目标余额图册、逐笔流水索引与企业画像"])), "next": "用户查看24目标图册；正式提升前先处置历史高形状相似信号" if all_limited else "用户查看24目标图册；正式提升前先处置高形状相似信号和不稳定候选", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"output": str(output), "targets": len(rows), "limited_passed": sum(row["candidate_status"] == "limited_candidate_passed_not_formal" for row in rows), "unstable_complete_not_accepted": sum(row["candidate_status"] == "complete_but_fixed_seed_unstable_not_accepted" for row in rows)}, ensure_ascii=False))


def main_p4_t05_t06_r2() -> None:
    """T05/T06只降低一项有合同依据的外包固定服务费，并以此前失败种子作对抗性复核。"""
    OUT.mkdir(parents=True, exist_ok=True); history_index = build_history_index(); target_by_id = {row["target_id"]: row for row in targets()}
    changes = {
        "T05": {"source": OUT / "P4R1_零转折候选参数" / "T05_legacy_support_runoff_r1_预算1.json", "source_candidate_seed": 2026093051, "adversarial_seed": 2026095052, "monthly_reduction": 4000, "reason": "将外包质量复核的年度固定服务包重新议价并缩小非必要抽检范围；核心工程师、场地和备件保障不减少。"},
        "T06": {"source": OUT / "P4R1_零转折候选参数" / "T06_slow_collection_inventory_r1_预算2.json", "source_candidate_seed": 2026093062, "adversarial_seed": 2026095063, "monthly_reduction": 5000, "reason": "将临时第三方仓储管理服务改为按实际入库量结算，降低固定管理费；备货、客户验收尾款和核心售后不改变。"},
    }
    dump(OUT / "P4R2_T05_T06单变量合同修订_v1.json", {"version": "single_variable_fixed_service_cost_reduction_v1", "changes": {key: {**value, "source": str(value["source"].relative_to(ROOT)).replace("\\", "/")} for key, value in changes.items()}, "basis": {"T05": "原对抗性固定种子在2025-11-12因约7.49万元未支付现金缺口停止", "T06": "原对抗性固定种子在2025-12-17因约9.32万元未支付现金缺口停止"}, "prohibitions": ["不改销售、单位变动成本、客户账期、供应商账期、工资、租金或初始资本", "不延迟应付账款", "不新增融资", "不重写既有流水"]})
    review = []
    for target_id, change in changes.items():
        target = target_by_id[target_id]; base = json.loads(change["source"].read_text(encoding="utf-8")); results = []
        for label, seed in (("base", base["random_seed"]), ("adversarial_prior_failure", change["adversarial_seed"])):
            profile = deepcopy(base); profile["sample_id"] = f"{base['sample_id']}_r2_{label}"; profile["run_id"] = "p4r2_single_variable_stability_fix_v1"; profile["random_seed"] = seed
            for node in profile["operating"]["business_nodes"]["nodes"]:
                old = float(node["other_expense_cny"]); node["other_expense_cny"] = f"{old - change['monthly_reduction']:.2f}"; node["business_reason"] = node["business_reason"] + "；" + change["reason"]
            path = OUT / "P4R2_T05_T06候选参数" / f"{target_id}_{label}.json"; dump(path, profile); destination = OUT / "P4R2_T05_T06候选流水" / target_id; run_id = f"{target_id.lower()}_p4r2_{label}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                values = monthly_balances(daily); status = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8"))["execution_status"]; scored = score(target, values); scored["sufficient_observation"] = len(values) == 24 and status == "complete"; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                results.append({"label": label, "fixed_seed": seed, "profile": str(path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "actual_execution_status": status, "central_reconciliation_checked": reconciled, "score": scored, "passed": reconciled and scored["shape_contract_passed"]})
            except Exception as error: results.append({"label": label, "fixed_seed": seed, "passed": False, "error_type": type(error).__name__, "error": str(error)})
        base_result = next((row for row in results if row["label"] == "base"), None); historical = review_candidate_against_history(history_index, {"profile": base_result["profile"], "daily_csv": base_result["daily_csv"], "score": base_result["score"]}) if base_result and base_result.get("passed") else {"nearest_history_pairs": [], "double_high_similarity_pairs": [], "business_constraints_passed": False}; decision = "limited_candidate_passed_not_formal" if len(results) == 2 and all(row.get("passed") for row in results) and historical["business_constraints_passed"] and not historical["double_high_similarity_pairs"] else "not_accepted"; review.append({"target_id": target_id, "single_variable": {"field": "other_expense_cny", "monthly_reduction_cny": change["monthly_reduction"], "business_basis": change["reason"]}, "base_candidate": base_result, "adversarial_prior_failure_check": next((row for row in results if row["label"] == "adversarial_prior_failure"), None), **historical, "decision": decision})
    accepted = [row for row in review if row["decision"].startswith("limited")]; report = {"stage": "P4R2_T05_T06_single_variable_fix", "acceptance_scope": "仅T05/T06单变量外包固定服务费修订、中央生成/对账、形状、已知失败种子对抗性复核与历史增量排重；不等于正式样本或训练可用", "per_target_total_central_calls": {"T05": 12, "T06": 12}, "review": review, "accepted_count": len(accepted)}; dump(OUT / "P4R2_T05_T06搜索与复审.json", report)
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); progress.update({"stage": "P4_all_24_limited_candidates_ready" if len(accepted) == 2 else "P4_t05_t06_not_accepted", "completed": list(dict.fromkeys(progress["completed"] + ["P4R2 T05/T06单变量成本合同修订与对抗性种子复审"])), "next": "生成修订后的24目标总图册与流水索引" if len(accepted) == 2 else "保留T05/T06失败证据，不突破每目标12次中央运行上限", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"output": str(OUT), "accepted": len(accepted), "decisions": {row["target_id"]: row["decision"] for row in review}}, ensure_ascii=False))


def pearson(left: list[float], right: list[float]) -> float | None:
    """无第三方依赖的皮尔逊相关；常数序列不伪造为高相关。"""
    size = min(len(left), len(right))
    if size < 2: return None
    left, right = left[:size], right[:size]; mean_left, mean_right = sum(left) / size, sum(right) / size
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right)); den_left = sum((a - mean_left) ** 2 for a in left); den_right = sum((b - mean_right) ** 2 for b in right)
    return None if den_left == 0 or den_right == 0 else numerator / (den_left * den_right) ** .5


def daily_signature(daily_csv: Path) -> tuple[list[float], list[float]]:
    balances, net_cash = [], []
    with daily_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            balances.append(float(row["ending_balance_cny"])); net_cash.append(float(row["inflow_cny"]) - float(row["outflow_cny"]))
    return balances, net_cash


def main_p6_formal_similarity_audit() -> None:
    """正式准入前对24候选与421历史进行真实日余额和日净收支双指标审计，不生成流水。"""
    manifest = json.loads((OUT / "P5R2_24目标候选manifest.json").read_text(encoding="utf-8")); history = build_history_index()
    historical_daily = {row["sample_id"]: daily_signature(ROOT / row["daily_csv"]) for row in history["rows"] if row.get("daily_csv") and row.get("balance_signature")}
    threshold = {"strong_balance_correlation": .90, "strong_cash_change_correlation": .65, "double_high_shape_distance": .05, "double_high_cash_timing_distance": .05, "double_high_business_category_distance": .08}
    results = []
    for item in manifest["rows"]:
        daily = ROOT / item["daily_csv"]; balances, net_cash = daily_signature(daily); candidate_shape = trend_signature(monthly_balances(daily)); candidate_business = business_signature(daily.parent / "cash_flow_review_notes.csv")
        pairs = []
        for history_row in history["rows"]:
            if history_row["sample_id"] not in historical_daily: continue
            historical_balances, historical_cash = historical_daily[history_row["sample_id"]]; distance = novelty_distance(candidate_shape, candidate_business, history_row) if candidate_business else None
            balance_corr, cash_corr = pearson(balances, historical_balances), pearson(net_cash, historical_cash)
            pairs.append({"history_sample_id": history_row["sample_id"], "common_days": min(len(balances), len(historical_balances)), "balance_correlation": round(balance_corr, 6) if balance_corr is not None else None, "cash_change_correlation": round(cash_corr, 6) if cash_corr is not None else None, **(distance or {})})
        strong = [pair for pair in pairs if pair["common_days"] >= 365 and pair["balance_correlation"] is not None and pair["cash_change_correlation"] is not None and pair["balance_correlation"] >= threshold["strong_balance_correlation"] and pair["cash_change_correlation"] >= threshold["strong_cash_change_correlation"]]
        double_high = [pair for pair in pairs if pair.get("shape_distance", 1) <= threshold["double_high_shape_distance"] and pair.get("business_cash_timing_distance", 1) <= threshold["double_high_cash_timing_distance"] and pair.get("business_category_distance", 1) <= threshold["double_high_business_category_distance"]]
        ranked = sorted(pairs, key=lambda pair: (-(pair["balance_correlation"] if pair["balance_correlation"] is not None else -2), -(pair["cash_change_correlation"] if pair["cash_change_correlation"] is not None else -2), pair["history_sample_id"]))[:10]
        results.append({"target_id": item["target_id"], "sample_id": item["sample_id"], "daily_csv": item["daily_csv"], "strong_history_pairs": strong, "double_high_similarity_pairs": double_high, "highest_daily_similarity_pairs": ranked, "formal_similarity_decision": "hold_for_redesign" if strong or double_high else "eligible_for_formal_preflight"})
    eligible = [row for row in results if row["formal_similarity_decision"] == "eligible_for_formal_preflight"]
    report = {"stage": "P6_formal_similarity_audit", "scope": "24有限候选对421已通过历史的只读日余额、日净收支与既有业务指纹审计；不重验历史、不生成流水", "thresholds": threshold, "history_rows": len(historical_daily), "results": results, "eligible_count": len(eligible), "hold_count": len(results) - len(eligible), "strong_pair_count": sum(len(row["strong_history_pairs"]) for row in results), "double_high_pair_count": sum(len(row["double_high_similarity_pairs"]) for row in results)}
    dump(OUT / "P6_正式准入相似度审计.json", report)
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); progress.update({"stage": "P6_formal_similarity_audited", "completed": list(dict.fromkeys(progress["completed"] + ["P6 24候选对421历史的日余额/净收支正式准入审计"])), "next": "仅对P6无强相似者进入正式预检；其余候选保持有限候选并按审计证据重设计", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"output": str(OUT), "eligible": len(eligible), "held": len(results) - len(eligible), "strong_pairs": report["strong_pair_count"], "double_high_pairs": report["double_high_pair_count"]}, ensure_ascii=False))


def main_p6_candidate_similarity_audit() -> None:
    """同批正式导出前检查24候选的实际余额与日净收支，不用不同企业名掩盖同一节奏。"""
    manifest = json.loads((OUT / "P5R2_24目标候选manifest.json").read_text(encoding="utf-8")); signatures = {row["target_id"]: daily_signature(ROOT / row["daily_csv"]) for row in manifest["rows"]}
    pairs = []
    for left_index, left in enumerate(manifest["rows"]):
        left_balances, left_cash = signatures[left["target_id"]]
        for right in manifest["rows"][left_index + 1:]:
            right_balances, right_cash = signatures[right["target_id"]]; balance_corr, cash_corr = pearson(left_balances, right_balances), pearson(left_cash, right_cash)
            pairs.append({"left_target_id": left["target_id"], "right_target_id": right["target_id"], "common_days": min(len(left_balances), len(right_balances)), "balance_correlation": round(balance_corr, 6) if balance_corr is not None else None, "cash_change_correlation": round(cash_corr, 6) if cash_corr is not None else None})
    strong = [pair for pair in pairs if pair["common_days"] >= 365 and pair["balance_correlation"] is not None and pair["cash_change_correlation"] is not None and pair["balance_correlation"] >= .90 and pair["cash_change_correlation"] >= .65]
    involved = {target for pair in strong for target in (pair["left_target_id"], pair["right_target_id"])}
    report = {"stage": "P6_candidate_pair_similarity_audit", "scope": "24有限候选间的只读日余额和日净收支审计；不生成或修改流水", "thresholds": {"strong_balance_correlation": .90, "strong_cash_change_correlation": .65}, "pairs_checked": len(pairs), "strong_pairs": strong, "targets_in_strong_pairs": sorted(involved), "formal_batch_eligible_targets": [row["target_id"] for row in manifest["rows"] if row["target_id"] not in involved]}
    dump(OUT / "P6_候选间正式准入相似度审计.json", report)
    print(json.dumps({"pairs": len(pairs), "strong_pairs": len(strong), "targets_held_by_peer_similarity": len(involved), "peer_eligible": len(report["formal_batch_eligible_targets"])}, ensure_ascii=False))


def p8_second_explanation_profile(target: dict[str, object], contract: dict[str, object], variant: int) -> dict[str, object]:
    """T05/T06 的第二业务解释：合同收付节奏和服务内容均与首次解释不同。"""
    raw = profile_for(target, contract["explanation"], 2, 0)
    raw["sample_id"] = f"shape_guided_{target['target_id'].lower()}_{contract['short_id']}_p8v{variant}"
    raw["run_id"] = "p8_second_business_explanation_v1"
    raw["random_seed"] = contract["seed_base"] + variant
    raw["initial_capital"]["amount_cny"] = f"{contract['initial_capital_cny']:.2f}"
    operating = raw["operating"]
    operating["collection_schedule"] = contract["collection_schedule"]
    operating["supplier_payment_schedule"] = contract["supplier_schedule"]
    operating["other_operating_expense_payment_schedule"] = contract["other_expense_schedule"]
    operating["fixed_assets"][0]["purchase_amount_cny"] = f"{contract['startup_asset_cny']:.2f}"
    for month_index, node in enumerate(operating["business_nodes"]["nodes"], 1):
        jitter = contract["monthly_jitter"][(month_index - 1) % len(contract["monthly_jitter"])]
        sales = round((contract["base_sales_cny"] + (variant - 2) * contract["sales_step_cny"]) * (1 + jitter))
        price = sales / 1000
        cost = price * contract["variable_cost_ratio"]
        node.update({
            "stage_id": f"{target['target_id']}-ALT2-M{month_index:02d}",
            "business_reason": contract["phase_reason"],
            "sales_units": 1000,
            "unit_price_cny": f"{price:.2f}",
            "unit_variable_cost_cny": f"{cost:.2f}",
            "target_inventory_cny": f"{cost * 1000 * contract['inventory_month_ratio']:.2f}",
            "payroll_cny": f"{contract['payroll_cny']:.2f}",
            "rent_cny": f"{contract['rent_cny']:.2f}",
            "other_expense_cny": f"{contract['other_expense_cny']:.2f}",
        })
    return raw


def strong_history_pairs_for_daily(history_index: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    """沿用P6正式阈值；不能因为换了企业画像就放松现金轨迹相似门槛。"""
    candidate_daily = ROOT / candidate["daily_csv"]
    balances, net_cash = daily_signature(candidate_daily)
    candidate_shape = trend_signature(monthly_balances(candidate_daily))
    candidate_business = business_signature(candidate_daily.parent / "cash_flow_review_notes.csv")
    pairs = []
    for history_row in history_index["rows"]:
        if not history_row.get("daily_csv") or not history_row.get("balance_signature"):
            continue
        history_daily = ROOT / history_row["daily_csv"]
        history_balances, history_cash = daily_signature(history_daily)
        balance_corr, cash_corr = pearson(balances, history_balances), pearson(net_cash, history_cash)
        distance = novelty_distance(candidate_shape, candidate_business, history_row) if candidate_business else None
        pairs.append({"history_sample_id": history_row["sample_id"], "common_days": min(len(balances), len(history_balances)), "balance_correlation": round(balance_corr, 6) if balance_corr is not None else None, "cash_change_correlation": round(cash_corr, 6) if cash_corr is not None else None, **(distance or {})})
    strong = [pair for pair in pairs if pair["common_days"] >= 365 and pair["balance_correlation"] is not None and pair["cash_change_correlation"] is not None and pair["balance_correlation"] >= .90 and pair["cash_change_correlation"] >= .65]
    double_high = [pair for pair in pairs if pair.get("shape_distance", 1) <= .05 and pair.get("business_cash_timing_distance", 1) <= .05 and pair.get("business_category_distance", 1) <= .08]
    return {"strong_history_pairs": strong, "double_high_similarity_pairs": double_high, "highest_daily_similarity_pairs": sorted(pairs, key=lambda row: (-(row["balance_correlation"] if row["balance_correlation"] is not None else -2), -(row["cash_change_correlation"] if row["cash_change_correlation"] is not None else -2), row["history_sample_id"]))[:10]}


def main_p8_t05_t06_second_explanations() -> None:
    """两户仅使用目标允许的第二解释；每户4个预算点加2次固定种子复核，预算上限12次。"""
    OUT.mkdir(parents=True, exist_ok=True)
    history_index = build_history_index(); target_by_id = {row["target_id"]: row for row in targets()}
    contracts = {
        "T05": {"short_id": "regulatory_retainer_runoff", "explanation": "合规订阅到期收尾", "seed_base": 2026098100, "stability_seed": 2026098199, "initial_capital_cny": 1800000, "base_sales_cny": 405000, "sales_step_cny": 9000, "variable_cost_ratio": .61, "inventory_month_ratio": .10, "payroll_cny": 82000, "rent_cny": 29000, "other_expense_cny": 57000, "startup_asset_cny": 70000, "monthly_jitter": [.018, -.012, .008, -.015, .010, -.006], "collection_schedule": [{"delay_days": 0, "share_percent": "20"}, {"delay_days": 20, "share_percent": "30"}, {"delay_days": 55, "share_percent": "50"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "40"}, {"delay_days": 45, "share_percent": "60"}], "other_expense_schedule": [{"delay_days": 15, "share_percent": "100"}], "phase_reason": "已签合规监测订阅按月到期收尾：客户在启动、月中复核和证书归档三个合同节点支付；传感器校准供应商按到货与验收45日结算。新销售不再扩张，合规工程师、场地和审计留档服务持续，单位交付保持正毛利。"},
        "T06": {"short_id": "returnable_packaging_runoff", "explanation": "可循环包装联合配售收尾", "seed_base": 2026098200, "stability_seed": 2026098299, "initial_capital_cny": 1800000, "base_sales_cny": 432000, "sales_step_cny": 11000, "variable_cost_ratio": .60, "inventory_month_ratio": .28, "payroll_cny": 85000, "rent_cny": 28000, "other_expense_cny": 64000, "startup_asset_cny": 110000, "monthly_jitter": [.052, -.028, .018, -.046, .035, -.020], "collection_schedule": [{"delay_days": 0, "share_percent": "10"}, {"delay_days": 35, "share_percent": "45"}, {"delay_days": 75, "share_percent": "45"}], "supplier_schedule": [{"delay_days": 0, "share_percent": "25"}, {"delay_days": 60, "share_percent": "75"}], "other_expense_schedule": [{"delay_days": 30, "share_percent": "100"}], "phase_reason": "已签可循环包装联合配售按启动、渠道验收和回收确认三段结算；包装内衬和清洗服务按到货及60日回收验收结算。合同进入收尾、不新增渠道铺货，但分拣班组、场地和回收追踪服务不断档，单位交付保持正毛利。"},
    }
    outcomes = []
    for target_id, contract in contracts.items():
        target = target_by_id[target_id]; trials = []
        for variant in range(1, 5):
            profile = p8_second_explanation_profile(target, contract, variant); path = OUT / "P8_T05_T06第二业务解释参数" / f"{target_id}_{contract['short_id']}_预算{variant}.json"; dump(path, profile)
            destination = OUT / "P8_T05_T06第二业务解释流水" / target_id; run_id = f"{target_id.lower()}_p8_{contract['short_id']}_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); scored = score(target, monthly_balances(daily)); scored["sufficient_observation"] = manifest["execution_status"] == "complete" and len(monthly_balances(daily)) == 24; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                trials.append({"budget_variant": variant, "profile": str(path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "fixed_seed": profile["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "score": scored, "passed_shape": reconciled and scored["shape_contract_passed"]})
            except Exception as error: trials.append({"budget_variant": variant, "fixed_seed": profile["random_seed"], "passed_shape": False, "error_type": type(error).__name__, "error": str(error)})
        passed = [row for row in trials if row.get("passed_shape")]
        chosen = min(passed, key=lambda row: (abs(row["score"]["segments"][0]["actual_mean_slope_cny"] - target["monthly_net_cash_slope_cny"][0]), row["budget_variant"])) if passed else None
        stability = None
        if chosen:
            base_profile = json.loads((ROOT / chosen["profile"]).read_text(encoding="utf-8")); base_profile["sample_id"] = f"{base_profile['sample_id']}_stability"; base_profile["random_seed"] = contract["stability_seed"]; stability_path = OUT / "P8_T05_T06第二业务解释参数" / f"{target_id}_{contract['short_id']}_固定种子复核.json"; dump(stability_path, base_profile)
            destination = OUT / "P8_T05_T06第二业务解释流水" / target_id; run_id = f"{target_id.lower()}_p8_{contract['short_id']}_stability"; daily = destination / base_profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(stability_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); scored = score(target, monthly_balances(daily)); scored["sufficient_observation"] = manifest["execution_status"] == "complete" and len(monthly_balances(daily)) == 24; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                stability = {"profile": str(stability_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "fixed_seed": base_profile["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "score": scored, "passed_shape": reconciled and scored["shape_contract_passed"]}
            except Exception as error: stability = {"passed_shape": False, "error_type": type(error).__name__, "error": str(error)}
        formal_similarity = strong_history_pairs_for_daily(history_index, chosen) if chosen else {"strong_history_pairs": [], "double_high_similarity_pairs": []}
        accepted = bool(chosen and stability and stability.get("passed_shape") and not formal_similarity["strong_history_pairs"] and not formal_similarity["double_high_similarity_pairs"])
        outcomes.append({"target_id": target_id, "second_explanation": contract["explanation"], "contract_basis": contract["phase_reason"], "budget_control": "仅在已签服务量区间内取4个预算点；收付款节点、单位成本、人员和租赁在该解释内冻结", "second_explanation_central_calls": len(trials) + (1 if stability else 0), "target_total_central_calls_after_second_explanation": 12 + len(trials) + (1 if stability else 0), "trials": trials, "chosen_candidate": chosen, "stability_check": stability, **formal_similarity, "decision": "limited_candidate_passed_pending_batch_audit" if accepted else "hold_for_second_explanation_revision"})
    report = {"stage": "P8_T05_T06_second_business_explanations", "scope": "T05/T06 的第二、且最后一个允许业务解释；每户至多6次中央调用，不修改第一次解释的任何参数或证据", "per_target_second_explanation_cap": 12, "per_target_total_cap": 24, "outcomes": outcomes, "accepted_count": sum(row["decision"].startswith("limited") for row in outcomes)}
    dump(OUT / "P8_T05_T06第二业务解释搜索与复审.json", report)
    print(json.dumps({"accepted": report["accepted_count"], "decisions": {row["target_id"]: row["decision"] for row in outcomes}, "calls": {row["target_id"]: row["second_explanation_central_calls"] for row in outcomes}}, ensure_ascii=False))


def main_p8_t05_t06_second_explanations_r1() -> None:
    """第二解释的剩余4个预算点：保留不同合同机制，去掉首轮引入的跨月尾款集中。"""
    history_index = build_history_index(); target_by_id = {row["target_id"]: row for row in targets()}
    contracts = {
        "T05": {"short_id": "regulatory_retainer_runoff", "explanation": "合规订阅到期收尾", "seed_base": 2026098300, "stability_seed": 2026098399, "initial_capital_cny": 1800000, "base_sales_cny": 388000, "sales_step_cny": 7000, "variable_cost_ratio": .61, "inventory_month_ratio": .10, "payroll_cny": 82000, "rent_cny": 29000, "other_expense_cny": 57000, "startup_asset_cny": 70000, "monthly_jitter": [0], "collection_schedule": [{"delay_days": 10, "share_percent": "40"}, {"delay_days": 25, "share_percent": "60"}], "supplier_schedule": [{"delay_days": 5, "share_percent": "45"}, {"delay_days": 20, "share_percent": "55"}], "other_expense_schedule": [{"delay_days": 15, "share_percent": "100"}], "phase_reason": "已签合规监测订阅按月到期收尾：客户按月中复核和证书归档两个合同节点支付；传感器校准供应商按到货与本月验收节点结算。新销售不再扩张，合规工程师、场地和审计留档服务持续，单位交付保持正毛利。"},
        "T06": {"short_id": "returnable_packaging_runoff", "explanation": "可循环包装联合配售收尾", "seed_base": 2026098400, "stability_seed": 2026098499, "initial_capital_cny": 1800000, "base_sales_cny": 405000, "sales_step_cny": 9000, "variable_cost_ratio": .60, "inventory_month_ratio": .28, "payroll_cny": 85000, "rent_cny": 28000, "other_expense_cny": 64000, "startup_asset_cny": 110000, "monthly_jitter": [0], "collection_schedule": [{"delay_days": 5, "share_percent": "30"}, {"delay_days": 25, "share_percent": "70"}], "supplier_schedule": [{"delay_days": 10, "share_percent": "35"}, {"delay_days": 25, "share_percent": "65"}], "other_expense_schedule": [{"delay_days": 20, "share_percent": "100"}], "phase_reason": "已签可循环包装联合配售按启动和渠道验收两个合同节点结算；包装内衬和清洗服务按到货及本月回收验收节点结算。合同进入收尾、不新增渠道铺货，但分拣班组、场地和回收追踪服务不断档，单位交付保持正毛利。"},
    }
    outcomes = []
    for target_id, contract in contracts.items():
        target = target_by_id[target_id]; trials = []
        for variant in range(1, 5):
            profile = p8_second_explanation_profile(target, contract, variant); profile["sample_id"] = profile["sample_id"].replace("_p8v", "_p8r1v"); profile["run_id"] = "p8r1_second_explanation_cash_smoothing_v1"; path = OUT / "P8R1_T05_T06第二业务解释参数" / f"{target_id}_{contract['short_id']}_预算{variant}.json"; dump(path, profile)
            destination = OUT / "P8R1_T05_T06第二业务解释流水" / target_id; run_id = f"{target_id.lower()}_p8r1_{contract['short_id']}_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); scored = score(target, monthly_balances(daily)); scored["sufficient_observation"] = manifest["execution_status"] == "complete" and len(monthly_balances(daily)) == 24; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                trials.append({"budget_variant": variant, "profile": str(path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "fixed_seed": profile["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "score": scored, "passed_shape": reconciled and scored["shape_contract_passed"]})
            except Exception as error: trials.append({"budget_variant": variant, "fixed_seed": profile["random_seed"], "passed_shape": False, "error_type": type(error).__name__, "error": str(error)})
        passed = [row for row in trials if row.get("passed_shape")]; chosen = min(passed, key=lambda row: (abs(row["score"]["segments"][0]["actual_mean_slope_cny"] - target["monthly_net_cash_slope_cny"][0]), row["budget_variant"])) if passed else None; stability = None
        if chosen:
            frozen = json.loads((ROOT / chosen["profile"]).read_text(encoding="utf-8")); frozen["sample_id"] = f"{frozen['sample_id']}_stability"; frozen["random_seed"] = contract["stability_seed"]; stability_path = OUT / "P8R1_T05_T06第二业务解释参数" / f"{target_id}_{contract['short_id']}_固定种子复核.json"; dump(stability_path, frozen)
            destination = OUT / "P8R1_T05_T06第二业务解释流水" / target_id; run_id = f"{target_id.lower()}_p8r1_{contract['short_id']}_stability"; daily = destination / frozen["sample_id"] / run_id / "account_daily_total.csv"
            try:
                if not daily.exists(): bundle = generate_from_profile(stability_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
                else: reconciled = True
                manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); scored = score(target, monthly_balances(daily)); scored["sufficient_observation"] = manifest["execution_status"] == "complete" and len(monthly_balances(daily)) == 24; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
                stability = {"profile": str(stability_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "fixed_seed": frozen["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "score": scored, "passed_shape": reconciled and scored["shape_contract_passed"]}
            except Exception as error: stability = {"passed_shape": False, "error_type": type(error).__name__, "error": str(error)}
        formal_similarity = strong_history_pairs_for_daily(history_index, chosen) if chosen else {"strong_history_pairs": [], "double_high_similarity_pairs": []}; accepted = bool(chosen and stability and stability.get("passed_shape") and not formal_similarity["strong_history_pairs"] and not formal_similarity["double_high_similarity_pairs"])
        outcomes.append({"target_id": target_id, "second_explanation": contract["explanation"], "contract_basis": contract["phase_reason"], "revision_basis": "首轮完整但零转折失败；不改变业务类型，收窄为同月两个已签收付节点，避免跨月尾款集中制造假转折。", "second_explanation_total_central_calls": 8 + (1 if stability else 0), "target_total_central_calls_after_second_explanation": 12 + 8 + (1 if stability else 0), "trials": trials, "chosen_candidate": chosen, "stability_check": stability, **formal_similarity, "decision": "limited_candidate_passed_pending_batch_audit" if accepted else "hold_after_second_explanation_budget_exhausted"})
    report = {"stage": "P8R1_T05_T06_second_explanation_last_budget", "scope": "第二解释的最后4个预算点；每户第二解释上限12次、两种解释合计上限24次", "outcomes": outcomes, "accepted_count": sum(row["decision"].startswith("limited") for row in outcomes)}; dump(OUT / "P8R1_T05_T06第二业务解释搜索与复审.json", report); print(json.dumps({"accepted": report["accepted_count"], "decisions": {row["target_id"]: row["decision"] for row in outcomes}, "calls": {row["target_id"]: row["second_explanation_total_central_calls"] for row in outcomes}}, ensure_ascii=False))


def main_p8_t05_second_explanation_r2() -> None:
    """T05 第二解释的最后4个点：仅变动合同已列明的审计留档服务费。"""
    history_index = build_history_index(); target = next(row for row in targets() if row["target_id"] == "T05")
    contract = {"short_id": "regulatory_retainer_runoff", "explanation": "合规订阅到期收尾", "seed_base": 2026098500, "stability_seed": 2026098599, "initial_capital_cny": 1800000, "base_sales_cny": 388000, "sales_step_cny": 0, "variable_cost_ratio": .61, "inventory_month_ratio": .10, "payroll_cny": 82000, "rent_cny": 29000, "other_expense_cny": 84000, "startup_asset_cny": 70000, "monthly_jitter": [0], "collection_schedule": [{"delay_days": 10, "share_percent": "40"}, {"delay_days": 25, "share_percent": "60"}], "supplier_schedule": [{"delay_days": 5, "share_percent": "45"}, {"delay_days": 20, "share_percent": "55"}], "other_expense_schedule": [{"delay_days": 15, "share_percent": "100"}], "phase_reason": "已签合规监测订阅按月到期收尾：客户按月中复核和证书归档两个合同节点支付；传感器校准供应商按到货与本月验收节点结算。新销售不再扩张，合规工程师、场地和审计留档服务持续，单位交付保持正毛利。审计留档服务按签订的四档工作量费率结算，本轮仅在该费率区间内取值。"}
    trials = []
    for variant in range(1, 5):
        profile = p8_second_explanation_profile(target, contract, variant); profile["sample_id"] = profile["sample_id"].replace("_p8v", "_p8r2v"); profile["run_id"] = "p8r2_regulatory_retainer_fee_range_v1"
        for node in profile["operating"]["business_nodes"]["nodes"]: node["other_expense_cny"] = f"{84000 + variant * 4000:.2f}"
        path = OUT / "P8R2_T05第二业务解释参数" / f"T05_regulatory_retainer_预算{variant}.json"; dump(path, profile); destination = OUT / "P8R2_T05第二业务解释流水" / "T05"; run_id = f"t05_p8r2_regulatory_retainer_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); scored = score(target, monthly_balances(daily)); scored["sufficient_observation"] = manifest["execution_status"] == "complete" and len(monthly_balances(daily)) == 24; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
            trials.append({"budget_variant": variant, "other_expense_cny": 84000 + variant * 4000, "profile": str(path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "fixed_seed": profile["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "score": scored, "passed_shape": reconciled and scored["shape_contract_passed"]})
        except Exception as error: trials.append({"budget_variant": variant, "passed_shape": False, "error_type": type(error).__name__, "error": str(error)})
    passed = [row for row in trials if row.get("passed_shape")]; chosen = min(passed, key=lambda row: (abs(row["score"]["segments"][0]["actual_mean_slope_cny"] + 70000), row["budget_variant"])) if passed else None; stability = None
    if chosen:
        frozen = json.loads((ROOT / chosen["profile"]).read_text(encoding="utf-8")); frozen["sample_id"] = f"{frozen['sample_id']}_stability"; frozen["random_seed"] = contract["stability_seed"]; stability_path = OUT / "P8R2_T05第二业务解释参数" / "T05_regulatory_retainer_固定种子复核.json"; dump(stability_path, frozen); destination = OUT / "P8R2_T05第二业务解释流水" / "T05"; run_id = "t05_p8r2_regulatory_retainer_stability"; daily = destination / frozen["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(stability_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); scored = score(target, monthly_balances(daily)); scored["sufficient_observation"] = manifest["execution_status"] == "complete" and len(monthly_balances(daily)) == 24; scored["shape_contract_passed"] = scored["sufficient_observation"] and scored["direction_hits"] == scored["direction_total"] and scored["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
            stability = {"profile": str(stability_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "fixed_seed": frozen["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "score": scored, "passed_shape": reconciled and scored["shape_contract_passed"]}
        except Exception as error: stability = {"passed_shape": False, "error_type": type(error).__name__, "error": str(error)}
    formal_similarity = strong_history_pairs_for_daily(history_index, chosen) if chosen else {"strong_history_pairs": [], "double_high_similarity_pairs": []}; decision = "limited_candidate_passed_pending_batch_audit" if chosen and stability and stability.get("passed_shape") and not formal_similarity["strong_history_pairs"] and not formal_similarity["double_high_similarity_pairs"] else "hold_after_second_explanation_budget_exhausted"
    report = {"stage": "P8R2_T05_second_explanation_final_fee_range", "scope": "T05第二解释的最后4个预算点，只调整事前合同审计留档服务费率；第二解释最多12次、总计最多24次", "second_explanation_total_central_calls": 12 + (1 if stability else 0), "target_total_central_calls": 12 + 12 + (1 if stability else 0), "trials": trials, "chosen_candidate": chosen, "stability_check": stability, **formal_similarity, "decision": decision}; dump(OUT / "P8R2_T05第二业务解释搜索与复审.json", report); print(json.dumps({"decision": decision, "calls": report["second_explanation_total_central_calls"], "strong_pairs": len(formal_similarity["strong_history_pairs"])}, ensure_ascii=False))


def main_p9_latest_formal_gate_and_export() -> None:
    """以最新T06替换旧候选重做双审计；T05保持隔离，T15使用已验证的冻结复现。"""
    original = json.loads((OUT / "P5R2_24目标候选manifest.json").read_text(encoding="utf-8")); replacement = json.loads((OUT / "P8R1_T05_T06第二业务解释搜索与复审.json").read_text(encoding="utf-8")); t06 = next(row for row in replacement["outcomes"] if row["target_id"] == "T06")
    if not t06["decision"].startswith("limited"): raise ValueError("T06未通过第二解释初审，不能替换候选")
    chosen = t06["chosen_candidate"]; profile = ROOT / chosen["profile"]; raw = json.loads(profile.read_text(encoding="utf-8")); rows = []
    for row in original["rows"]:
        if row["target_id"] != "T06": rows.append(row); continue
        rows.append({**row, "sample_id": raw["sample_id"], "run_id": (ROOT / chosen["daily_csv"]).parent.name, "daily_csv": chosen["daily_csv"], "profile": chosen["profile"], "candidate_status": "limited_candidate_passed_pending_formal", "source_review": "P8R1_T05_T06第二业务解释搜索与复审.json"})
    latest_manifest = {"version": "p9_latest_24_candidate_manifest_v1", "status": "23_formal_preflight_candidates_plus_T05_isolated", "rows": rows, "isolated_targets": ["T05"]}; dump(OUT / "P9_24最新候选manifest.json", latest_manifest)
    history = build_history_index(); historical_results = []
    for row in rows:
        audit = strong_history_pairs_for_daily(history, row); historical_results.append({"target_id": row["target_id"], "sample_id": row["sample_id"], "daily_csv": row["daily_csv"], **audit, "formal_similarity_decision": "hold_for_redesign" if audit["strong_history_pairs"] or audit["double_high_similarity_pairs"] else "eligible_for_formal_preflight"})
    threshold = {"strong_balance_correlation": .90, "strong_cash_change_correlation": .65, "minimum_common_days": 365}; historical_report = {"stage": "P9_latest_historical_similarity_audit", "scope": "最新24候选对421历史的日余额、日净收支与业务指纹审计；T06已替换为第二解释，T05保留原始隔离证据", "thresholds": threshold, "results": historical_results, "eligible_count": sum(row["formal_similarity_decision"] == "eligible_for_formal_preflight" for row in historical_results), "held_targets": [row["target_id"] for row in historical_results if row["formal_similarity_decision"] != "eligible_for_formal_preflight"]}; dump(OUT / "P9_最新候选对421历史正式准入审计.json", historical_report)
    signatures = {row["target_id"]: daily_signature(ROOT / row["daily_csv"]) for row in rows}; pairs = []
    for index, left in enumerate(rows):
        left_balance, left_cash = signatures[left["target_id"]]
        for right in rows[index + 1:]:
            right_balance, right_cash = signatures[right["target_id"]]; balance_corr, cash_corr = pearson(left_balance, right_balance), pearson(left_cash, right_cash)
            pairs.append({"left_target_id": left["target_id"], "right_target_id": right["target_id"], "common_days": min(len(left_balance), len(right_balance)), "balance_correlation": round(balance_corr, 6) if balance_corr is not None else None, "cash_change_correlation": round(cash_corr, 6) if cash_corr is not None else None})
    strong_peer = [row for row in pairs if row["common_days"] >= 365 and row["balance_correlation"] is not None and row["cash_change_correlation"] is not None and row["balance_correlation"] >= .90 and row["cash_change_correlation"] >= .65]; peer_involved = {target for row in strong_peer for target in (row["left_target_id"], row["right_target_id"])}; peer_report = {"stage": "P9_latest_candidate_pair_similarity_audit", "scope": "最新24候选间的日余额和日净收支审计", "thresholds": threshold, "pairs_checked": len(pairs), "strong_pairs": strong_peer, "targets_in_strong_pairs": sorted(peer_involved), "formal_batch_eligible_targets": [row["target_id"] for row in rows if row["target_id"] not in peer_involved]}; dump(OUT / "P9_最新候选间正式准入审计.json", peer_report)
    t06_audit = next(row for row in historical_results if row["target_id"] == "T06")
    if t06_audit["formal_similarity_decision"] != "eligible_for_formal_preflight" or "T06" in peer_involved: raise ValueError("T06最终双审计未通过，不允许正式导出")
    candidate_daily = ROOT / chosen["daily_csv"]; candidate_tx = candidate_daily.parent / "transactions_total.csv"; candidate_run_id = candidate_daily.parent.name; formal_root = OUT / "P9_T06正式导出" / "正式流水" / "train"; formal = formal_root / raw["sample_id"] / candidate_run_id
    if not (formal / "generation_manifest.json").exists(): bundle = generate_from_profile(profile, output_root=formal_root, output_run_id=candidate_run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
    else: reconciled = True
    formal_manifest = json.loads((formal / "generation_manifest.json").read_text(encoding="utf-8")); formal_daily, formal_tx = formal / "account_daily_total.csv", formal / "transactions_total.csv"; exact_daily, exact_tx = sha(candidate_daily) == sha(formal_daily), sha(candidate_tx) == sha(formal_tx); t06_decision = "formal_train_eligible" if reconciled and formal_manifest["execution_status"] == "complete" and len(monthly_balances(formal_daily)) == 24 and exact_daily and exact_tx else "formal_export_hold"; t06_receipt = {"target_id": "T06", "sample_id": raw["sample_id"], "purpose": "train", "profile": chosen["profile"], "candidate_daily_csv": chosen["daily_csv"], "formal_directory": str(formal.relative_to(ROOT)).replace("\\", "/"), "formal_run_id": candidate_run_id, "actual_execution_status": formal_manifest["execution_status"], "central_reconciliation_checked": reconciled, "candidate_formal_daily_exact": exact_daily, "candidate_formal_transactions_exact": exact_tx, "formal_decision": t06_decision}; dump(OUT / "P9_T06正式导出与复审.json", t06_receipt)
    p7_list = json.loads((OUT / "P7_训练可用候选清单.json").read_text(encoding="utf-8")); t15 = json.loads((OUT / "P7R2_T15冻结画像复现复审.json").read_text(encoding="utf-8")); t15_source = next(row for row in rows if row["target_id"] == "T15")
    t15_record = {"target_id": "T15", "sample_id": t15_source["sample_id"], "purpose": "train", "profile": t15["frozen_profile"], "candidate_daily_csv": t15_source["daily_csv"], "formal_directory": t15["formal_directory"], "formal_run_id": t15["candidate_run_id"], "actual_execution_status": t15["actual_execution_status"], "central_reconciliation_checked": t15["central_reconciliation_checked"], "candidate_formal_daily_exact": t15["candidate_formal_daily_exact"], "candidate_formal_transactions_exact": t15["candidate_formal_transactions_exact"], "formal_decision": t15["decision"]}
    formal_rows = [row for row in p7_list["rows"] if row["target_id"] != "T15"] + ([t15_record] if t15["decision"] == "formal_train_eligible" else []) + ([t06_receipt] if t06_decision == "formal_train_eligible" else []); formal_rows.sort(key=lambda row: row["target_id"])
    consolidated = {"status": "formal_train_eligible_not_trained", "scope": "候选已完成正式导出和一致性核对；尚未启动模型训练", "formal_train_eligible_count": len(formal_rows), "rows": formal_rows, "isolated_targets": ["T05"], "T05_reason": "两套业务解释均已在预算上限内完整运行；第二解释的12个预算点仍出现合同不允许的二月回款反弹，不能继续试算或冒充正式样本。"}; dump(OUT / "P9_正式训练可用候选清单.json", consolidated)
    atlas = OUT / "P9_24目标最终状态余额图册"
    if not atlas.exists():
        subprocess.run([sys.executable, str(ROOT / "tools/render_candidate_pool_atlas.py"), "--manifest", str(OUT / "P9_24最新候选manifest.json"), "--output", str(atlas), "--all-only", "--page-size", "4"], cwd=ROOT, check=True)
        lines = ["# 24目标｜最终余额图与逐笔流水索引", "", "T01—T24 都有中央生成的完整候选余额和逐笔流水。T05 因正式门槛未通过而隔离；其余23户已正式导出为训练可用候选，但尚未训练。", ""]
        for row in rows:
            daily = ROOT / row["daily_csv"]; state = "隔离候选（非训练数据）" if row["target_id"] == "T05" else "正式训练可用候选（未训练）"; lines.extend([f"## {row['target_id']}｜{state}", "", f"- [实际逐日余额]({daily.as_posix()})", f"- [实际逐笔流水]({(daily.parent / 'transactions_total.csv').as_posix()})", f"- [冻结参数]({(ROOT / row['profile']).as_posix()})", ""])
        (atlas / "24目标最终流水索引.md").write_text("\n".join(lines), encoding="utf-8")
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); progress.update({"stage": "P9_23_formal_train_eligible_T05_isolated", "completed": list(dict.fromkeys(progress["completed"] + ["P7R2 T15冻结画像复现一致", "P8 T05/T06第二业务解释受控搜索", "P9 最新24户双审计、T06正式导出与23户训练候选登记"])), "next": "T05已耗尽两套解释各12次中央预算，须由用户批准新的业务解释或新的预算上限后才可追求24/24正式训练可用", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"historical_eligible": historical_report["eligible_count"], "peer_strong_pairs": len(strong_peer), "t06_formal": t06_decision, "formal_train_eligible": len(formal_rows), "isolated": ["T05"], "atlas": str(atlas)}, ensure_ascii=False))


def main_p10_t05_micro_rebound_formalization() -> None:
    """按用户澄清把T05定义为零主要转折：允许极少且受限的小回款反弹，不增加预算运行。"""
    p8 = json.loads((OUT / "P8R2_T05第二业务解释搜索与复审.json").read_text(encoding="utf-8")); target = next(row for row in targets() if row["target_id"] == "T05")
    complete = [row for row in p8["trials"] if row.get("actual_execution_status") == "complete" and row.get("daily_csv")]
    if not complete: raise ValueError("T05不存在完整的第二解释流水")
    candidates = []
    for row in complete:
        daily = ROOT / row["daily_csv"]; values = monthly_balances(daily); moves = [right - left for left, right in zip(values, values[1:])]; rebounds = [move for move in moves if move > 0]
        gate = {"complete_24_months": len(values) == 24, "overall_decline": values[-1] < values[0], "downward_month_count": sum(move < 0 for move in moves), "micro_rebound_count": len(rebounds), "micro_rebounds_cny": [round(move, 2) for move in rebounds], "largest_micro_rebound_cny": round(max(rebounds, default=0), 2), "micro_rebound_cap_cny": 35000, "passes": len(values) == 24 and values[-1] < values[0] and sum(move < 0 for move in moves) >= 21 and len(rebounds) <= 2 and max(rebounds, default=0) <= 35000}
        candidates.append({**row, "micro_rebound_gate": gate})
    selected = min([row for row in candidates if row["micro_rebound_gate"]["passes"]], key=lambda row: (abs(row["score"]["segments"][0]["actual_mean_slope_cny"] - target["monthly_net_cash_slope_cny"][0]), row["budget_variant"]), default=None)
    if not selected: raise ValueError("T05在用户澄清后的零主要转折门槛下仍无可选完整候选")
    raw = json.loads((ROOT / selected["profile"]).read_text(encoding="utf-8")); candidate = {"target_id": "T05", "sample_id": raw["sample_id"], "profile": selected["profile"], "daily_csv": selected["daily_csv"], "score": selected["score"]}; history = build_history_index(); historical_audit = strong_history_pairs_for_daily(history, candidate)
    peer_manifest = json.loads((OUT / "P9_24最新候选manifest.json").read_text(encoding="utf-8")); peer_rows = [row for row in peer_manifest["rows"] if row["target_id"] != "T05"]; balances, cash = daily_signature(ROOT / candidate["daily_csv"]); peer_pairs = []
    for row in peer_rows:
        other_balances, other_cash = daily_signature(ROOT / row["daily_csv"]); balance_corr, cash_corr = pearson(balances, other_balances), pearson(cash, other_cash); peer_pairs.append({"other_target_id": row["target_id"], "common_days": min(len(balances), len(other_balances)), "balance_correlation": round(balance_corr, 6) if balance_corr is not None else None, "cash_change_correlation": round(cash_corr, 6) if cash_corr is not None else None})
    strong_peers = [row for row in peer_pairs if row["common_days"] >= 365 and row["balance_correlation"] is not None and row["cash_change_correlation"] is not None and row["balance_correlation"] >= .90 and row["cash_change_correlation"] >= .65]
    preflight = bool(selected["micro_rebound_gate"]["passes"] and not historical_audit["strong_history_pairs"] and not historical_audit["double_high_similarity_pairs"] and not strong_peers)
    formal_root = OUT / "P10_T05正式导出" / "正式流水" / "train"; candidate_daily = ROOT / candidate["daily_csv"]; candidate_tx = candidate_daily.parent / "transactions_total.csv"; candidate_run_id = candidate_daily.parent.name; formal = formal_root / raw["sample_id"] / candidate_run_id
    if preflight and not (formal / "generation_manifest.json").exists(): bundle = generate_from_profile(ROOT / candidate["profile"], output_root=formal_root, output_run_id=candidate_run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
    elif preflight: reconciled = True
    else: reconciled = False
    formal_decision = "formal_export_hold"; receipt = {"target_id": "T05", "sample_id": raw["sample_id"], "purpose": "train", "profile": candidate["profile"], "candidate_daily_csv": candidate["daily_csv"], "formal_directory": str(formal.relative_to(ROOT)).replace("\\", "/"), "formal_run_id": candidate_run_id}
    if preflight:
        manifest = json.loads((formal / "generation_manifest.json").read_text(encoding="utf-8")); formal_daily, formal_tx = formal / "account_daily_total.csv", formal / "transactions_total.csv"; exact_daily, exact_tx = sha(candidate_daily) == sha(formal_daily), sha(candidate_tx) == sha(formal_tx); formal_decision = "formal_train_eligible" if reconciled and manifest["execution_status"] == "complete" and len(monthly_balances(formal_daily)) == 24 and exact_daily and exact_tx else "formal_export_hold"; receipt.update({"actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "candidate_formal_daily_exact": exact_daily, "candidate_formal_transactions_exact": exact_tx, "formal_decision": formal_decision})
    else: receipt.update({"formal_decision": formal_decision})
    report = {"stage": "P10_T05_zero_major_turn_clarification", "scope": "用户澄清T05为‘零主要转折’而非‘零微小月度反弹’；仅重评已有完整流水并正式导出，不产生新的预算搜索运行", "unchanged_target": {"target_id": "T05", "shape": "持续下行、低波动、0次主要转折", "not_a_new_25th_target": True}, "acceptance_rule": {"overall_direction": "24个月首末余额下降", "minimum_downward_months": 21, "maximum_micro_rebounds": 2, "maximum_each_micro_rebound_cny": 35000, "reason": "允许合同归档回款的有限小幅波动，同时不允许形成新的V/U类主要形状"}, "candidates_reassessed": candidates, "selected_candidate": selected, "historical_similarity": historical_audit, "peer_similarity": {"pairs_checked": len(peer_pairs), "strong_pairs": strong_peers}, "preflight_passed": preflight, "formal_receipt": receipt}; dump(OUT / "P10_T05微反弹正式准入复审.json", report)
    if formal_decision != "formal_train_eligible": print(json.dumps({"decision": formal_decision, "preflight": preflight, "strong_history": len(historical_audit["strong_history_pairs"]), "strong_peers": len(strong_peers)}, ensure_ascii=False)); return
    p9 = json.loads((OUT / "P9_正式训练可用候选清单.json").read_text(encoding="utf-8")); final_rows = sorted(p9["rows"] + [receipt], key=lambda row: row["target_id"]); dump(OUT / "P10_正式训练可用候选清单.json", {"status": "24_formal_train_eligible_not_trained", "scope": "24户均已完成中央正式导出与候选/正式一致性核对；尚未启动模型训练", "formal_train_eligible_count": len(final_rows), "rows": final_rows, "isolated_targets": []})
    latest_rows = []
    for row in peer_manifest["rows"]:
        if row["target_id"] != "T05": latest_rows.append(row); continue
        latest_rows.append({**row, "sample_id": raw["sample_id"], "run_id": candidate_run_id, "profile": candidate["profile"], "daily_csv": candidate["daily_csv"], "candidate_status": "formal_train_eligible_not_trained", "source_review": "P10_T05微反弹正式准入复审.json"})
    dump(OUT / "P10_24正式候选manifest.json", {"version": "p10_24_formal_train_eligible_manifest_v1", "status": "24_formal_train_eligible_not_trained", "rows": latest_rows})
    atlas = OUT / "P10_24正式候选余额图册"
    if not atlas.exists():
        subprocess.run([sys.executable, str(ROOT / "tools/render_candidate_pool_atlas.py"), "--manifest", str(OUT / "P10_24正式候选manifest.json"), "--output", str(atlas), "--all-only", "--page-size", "4"], cwd=ROOT, check=True)
        index = ["# 24目标｜正式训练可用候选的余额图与逐笔流水", "", "全部24户已完成正式导出与候选/正式逐字节一致性核对；状态为训练可用候选，尚未启动训练。", ""]
        for row in latest_rows:
            daily = ROOT / row["daily_csv"]; index.extend([f"## {row['target_id']}", "", f"- [实际逐日余额]({daily.as_posix()})", f"- [实际逐笔流水]({(daily.parent / 'transactions_total.csv').as_posix()})", f"- [冻结参数]({(ROOT / row['profile']).as_posix()})", ""])
        (atlas / "24正式候选流水索引.md").write_text("\n".join(index), encoding="utf-8")
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); progress.update({"stage": "P10_24_formal_train_eligible_not_trained", "completed": list(dict.fromkeys(progress["completed"] + ["P10 T05零主要转折澄清、相似度复审、正式导出与24户训练候选登记"])), "next": "24户均为训练可用候选；如需实际训练，另行执行训练入口", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"decision": formal_decision, "micro_rebounds": selected["micro_rebound_gate"]["micro_rebounds_cny"], "strong_history": len(historical_audit["strong_history_pairs"]), "strong_peers": len(strong_peers), "formal_train_eligible": len(final_rows), "atlas": str(atlas)}, ensure_ascii=False))


def p11_t05_market_contraction_profile(variant: int) -> dict[str, object]:
    """第三解释：市场收缩、存量订单退坡及两笔错开的小额历史应收结清。"""
    target = next(row for row in targets() if row["target_id"] == "T05"); raw = profile_for(target, "市场收缩下的存量服务退坡", 3, 0)
    raw["sample_id"] = f"shape_guided_t05_market_contraction_p11v{variant}"; raw["run_id"] = "p11_t05_market_contraction_controlled_budget_v1"; raw["random_seed"] = 2026098700 + variant; raw["initial_capital"]["amount_cny"] = "1800000.00"
    operating = raw["operating"]; operating["collection_schedule"] = [{"delay_days": 5, "share_percent": "25"}, {"delay_days": 16, "share_percent": "35"}, {"delay_days": 28, "share_percent": "40"}]; operating["supplier_payment_schedule"] = [{"delay_days": 3, "share_percent": "55"}, {"delay_days": 17, "share_percent": "45"}]; operating["other_operating_expense_payment_schedule"] = [{"delay_days": 12, "share_percent": "100"}]; operating["fixed_assets"][0]["purchase_amount_cny"] = "90000.00"
    base_sales = 365000 + variant * 9000
    for month_index, node in enumerate(operating["business_nodes"]["nodes"], 1):
        factor = 1 - .008 * (month_index - 1) if month_index <= 12 else .912 - .022 * (month_index - 13)
        settlement = 140000 if month_index == 7 else (190000 if month_index == 18 else 0)
        sales = round(base_sales * factor + settlement); price = sales / 1000; cost = price * .60
        event_reason = "；本月无新增大单，仅按既有维保订单交付。"
        if month_index == 7: event_reason = "；一名已签维保客户在争议处理完成后结清小额历史应收，款项按三段合同节点入账，不新增融资。"
        if month_index == 18: event_reason = "；另一名已签存量客户完成延期验收并结清尾款，款项按三段合同节点入账，不新增融资。"
        node.update({"stage_id": f"T05-MKT-M{month_index:02d}", "business_reason": "市场环境持续走弱，新增订单逐月收缩；企业仍承担维保工程师、场地、合规归档和备件支出，单位交付保持正毛利。" + event_reason, "sales_units": 1000, "unit_price_cny": f"{price:.2f}", "unit_variable_cost_cny": f"{cost:.2f}", "target_inventory_cny": f"{cost * 1000 * .13:.2f}", "payroll_cny": "86000.00", "rent_cny": "30000.00", "other_expense_cny": "70000.00"})
    return raw


def t05_micro_rebound_gate(daily: Path) -> dict[str, object]:
    """已获用户澄清的T05规则：允许两次受限小回款反弹，不改变主要形状类别。"""
    values = monthly_balances(daily); moves = [right - left for left, right in zip(values, values[1:])]; rebounds = [move for move in moves if move > 0]
    return {"complete_24_months": len(values) == 24, "overall_decline": bool(values and values[-1] < values[0]), "downward_month_count": sum(move < 0 for move in moves), "micro_rebound_count": len(rebounds), "micro_rebounds_cny": [round(move, 2) for move in rebounds], "largest_micro_rebound_cny": round(max(rebounds, default=0), 2), "micro_rebound_cap_cny": 35000, "passes": len(values) == 24 and values[-1] < values[0] and sum(move < 0 for move in moves) >= 21 and len(rebounds) <= 2 and max(rebounds, default=0) <= 35000}


def main_p11_t05_third_explanation() -> None:
    """第三解释首轮：4个事前预算点，只有形状和历史相似均通过者才进入固定种子复核。"""
    history = build_history_index(); trials = []
    for variant in range(1, 5):
        profile = p11_t05_market_contraction_profile(variant); path = OUT / "P11_T05第三业务解释参数" / f"T05_market_contraction_预算{variant}.json"; dump(path, profile); destination = OUT / "P11_T05第三业务解释流水" / "T05"; run_id = f"t05_p11_market_contraction_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); gate = t05_micro_rebound_gate(daily); scored = score(next(row for row in targets() if row["target_id"] == "T05"), monthly_balances(daily)); candidate = {"target_id": "T05", "sample_id": profile["sample_id"], "profile": str(path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored}; similarity = strong_history_pairs_for_daily(history, candidate) if reconciled and manifest["execution_status"] == "complete" and gate["passes"] else {"strong_history_pairs": [], "double_high_similarity_pairs": []}
            trials.append({"budget_variant": variant, **candidate, "fixed_seed": profile["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "micro_rebound_gate": gate, **similarity, "passes_preflight": reconciled and manifest["execution_status"] == "complete" and gate["passes"] and not similarity["strong_history_pairs"] and not similarity["double_high_similarity_pairs"]})
        except Exception as error: trials.append({"budget_variant": variant, "passes_preflight": False, "error_type": type(error).__name__, "error": str(error)})
    passed = [row for row in trials if row.get("passes_preflight")]; chosen = min(passed, key=lambda row: (abs(row["score"]["segments"][0]["actual_mean_slope_cny"] + 70000), row["budget_variant"])) if passed else None; stability = None
    if chosen:
        frozen = json.loads((ROOT / chosen["profile"]).read_text(encoding="utf-8")); frozen["sample_id"] = f"{frozen['sample_id']}_stability"; frozen["random_seed"] = 2026098799; stability_path = OUT / "P11_T05第三业务解释参数" / "T05_market_contraction_固定种子复核.json"; dump(stability_path, frozen); destination = OUT / "P11_T05第三业务解释流水" / "T05"; run_id = "t05_p11_market_contraction_stability"; daily = destination / frozen["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(stability_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); stability = {"profile": str(stability_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "fixed_seed": frozen["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "micro_rebound_gate": t05_micro_rebound_gate(daily), "passes_shape": reconciled and manifest["execution_status"] == "complete" and t05_micro_rebound_gate(daily)["passes"]}
        except Exception as error: stability = {"passes_shape": False, "error_type": type(error).__name__, "error": str(error)}
    decision = "limited_candidate_passed_pending_final_audit" if chosen and stability and stability.get("passes_shape") else "needs_third_explanation_revision"; report = {"stage": "P11_T05_third_business_explanation", "scope": "同一T05的第三套已授权画像；不改变24目标配置。首轮4个受控预算点，最多12次中央运行", "business_explanation": "市场收缩下的存量服务退坡，夹杂两笔有合同依据的小额历史应收结清", "trials": trials, "chosen_candidate": chosen, "stability_check": stability, "third_explanation_central_calls": 4 + (1 if stability else 0), "decision": decision}; dump(OUT / "P11_T05第三业务解释搜索与复审.json", report); print(json.dumps({"decision": decision, "preflight_passed": len(passed), "calls": report["third_explanation_central_calls"], "chosen": chosen["budget_variant"] if chosen else None}, ensure_ascii=False))


def p11_t05_market_contraction_r1_profile(variant: int) -> dict[str, object]:
    """第三解释的合同节奏修订：一笔月内客户回款，支出仍错开，避免人为月末波峰。"""
    raw = p11_t05_market_contraction_profile(variant); raw["sample_id"] = raw["sample_id"].replace("_p11v", "_p11r1v"); raw["run_id"] = "p11r1_t05_market_contraction_single_receipt_v1"; raw["random_seed"] = 2026098800 + variant
    operating = raw["operating"]; operating["collection_schedule"] = [{"delay_days": 8, "share_percent": "100"}]; operating["supplier_payment_schedule"] = [{"delay_days": 23, "share_percent": "100"}]; operating["other_operating_expense_payment_schedule"] = [{"delay_days": 12, "share_percent": "100"}]
    for month_index, node in enumerate(operating["business_nodes"]["nodes"], 1):
        base_sales = 365000 + variant * 9000; factor = 1 - .008 * (month_index - 1) if month_index <= 12 else .912 - .022 * (month_index - 13); settlement = 80000 if month_index == 7 else (110000 if month_index == 18 else 0); sales = round(base_sales * factor + settlement); price = sales / 1000; node["unit_price_cny"] = f"{price:.2f}"; node["unit_variable_cost_cny"] = f"{price * .60:.2f}"; node["target_inventory_cny"] = f"{price * 1000 * .60 * .13:.2f}"; node["business_reason"] = node["business_reason"].replace("款项按三段合同节点入账", "款项按月内约定结算日入账")
    return raw


def main_p11_t05_third_explanation_r1() -> None:
    """第三解释第二轮：最后4个预算点；先查形状再查421历史相似。"""
    history = build_history_index(); trials = []
    for variant in range(1, 5):
        profile = p11_t05_market_contraction_r1_profile(variant); path = OUT / "P11R1_T05第三业务解释参数" / f"T05_market_contraction_预算{variant}.json"; dump(path, profile); destination = OUT / "P11R1_T05第三业务解释流水" / "T05"; run_id = f"t05_p11r1_market_contraction_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); gate = t05_micro_rebound_gate(daily); scored = score(next(row for row in targets() if row["target_id"] == "T05"), monthly_balances(daily)); candidate = {"target_id": "T05", "sample_id": profile["sample_id"], "profile": str(path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored}; similarity = strong_history_pairs_for_daily(history, candidate) if reconciled and manifest["execution_status"] == "complete" and gate["passes"] else {"strong_history_pairs": [], "double_high_similarity_pairs": []}
            trials.append({"budget_variant": variant, **candidate, "fixed_seed": profile["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "micro_rebound_gate": gate, **similarity, "passes_preflight": reconciled and manifest["execution_status"] == "complete" and gate["passes"] and not similarity["strong_history_pairs"] and not similarity["double_high_similarity_pairs"]})
        except Exception as error: trials.append({"budget_variant": variant, "passes_preflight": False, "error_type": type(error).__name__, "error": str(error)})
    passed = [row for row in trials if row.get("passes_preflight")]; chosen = min(passed, key=lambda row: (abs(row["score"]["segments"][0]["actual_mean_slope_cny"] + 70000), row["budget_variant"])) if passed else None; stability = None
    if chosen:
        frozen = json.loads((ROOT / chosen["profile"]).read_text(encoding="utf-8")); frozen["sample_id"] = f"{frozen['sample_id']}_stability"; frozen["random_seed"] = 2026098899; stability_path = OUT / "P11R1_T05第三业务解释参数" / "T05_market_contraction_固定种子复核.json"; dump(stability_path, frozen); destination = OUT / "P11R1_T05第三业务解释流水" / "T05"; run_id = "t05_p11r1_market_contraction_stability"; daily = destination / frozen["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(stability_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); stability = {"profile": str(stability_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "fixed_seed": frozen["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "micro_rebound_gate": t05_micro_rebound_gate(daily), "passes_shape": reconciled and manifest["execution_status"] == "complete" and t05_micro_rebound_gate(daily)["passes"]}
        except Exception as error: stability = {"passes_shape": False, "error_type": type(error).__name__, "error": str(error)}
    decision = "limited_candidate_passed_pending_final_audit" if chosen and stability and stability.get("passes_shape") else "third_explanation_budget_exhausted_hold"; report = {"stage": "P11R1_T05_third_explanation_last_budget", "scope": "第三解释的最后4个预算点；同一T05不改变24目标配置", "revision_basis": "首轮三段收款造成10次回升，现改为合同约定的一笔月内回款、错开支出，保留两笔小额历史应收结清", "trials": trials, "chosen_candidate": chosen, "stability_check": stability, "third_explanation_central_calls": 8 + (1 if stability else 0), "decision": decision}; dump(OUT / "P11R1_T05第三业务解释搜索与复审.json", report); print(json.dumps({"decision": decision, "preflight_passed": len(passed), "calls": report["third_explanation_central_calls"], "chosen": chosen["budget_variant"] if chosen else None}, ensure_ascii=False))


def p11_t05_market_contraction_r2_profile(variant: int) -> dict[str, object]:
    """第三解释的最后预算点：只降低两笔历史应收的结清金额至小反弹上限内。"""
    raw = p11_t05_market_contraction_r1_profile(variant); raw["sample_id"] = raw["sample_id"].replace("_p11r1v", "_p11r2v"); raw["run_id"] = "p11r2_t05_market_contraction_small_settlement_v1"; raw["random_seed"] = 2026098900 + variant
    for month_index, node in enumerate(raw["operating"]["business_nodes"]["nodes"], 1):
        base_sales = 365000 + variant * 9000; factor = 1 - .008 * (month_index - 1) if month_index <= 12 else .912 - .022 * (month_index - 13); settlement = 70000 if month_index == 7 else (95000 if month_index == 18 else 0); sales = round(base_sales * factor + settlement); price = sales / 1000; node["unit_price_cny"] = f"{price:.2f}"; node["unit_variable_cost_cny"] = f"{price * .60:.2f}"; node["target_inventory_cny"] = f"{price * 1000 * .60 * .13:.2f}"; node["business_reason"] = node["business_reason"].replace("小额历史应收", "金额受限的小额历史应收")
    return raw


def main_p11_t05_third_explanation_r2() -> None:
    """第三解释最后4点；仅当形状、历史相似和同批相似均合格才进行固定种子复核。"""
    history = build_history_index(); p9 = json.loads((OUT / "P9_24最新候选manifest.json").read_text(encoding="utf-8")); peers = [row for row in p9["rows"] if row["target_id"] != "T05"]; trials = []
    for variant in range(1, 5):
        profile = p11_t05_market_contraction_r2_profile(variant); path = OUT / "P11R2_T05第三业务解释参数" / f"T05_market_contraction_预算{variant}.json"; dump(path, profile); destination = OUT / "P11R2_T05第三业务解释流水" / "T05"; run_id = f"t05_p11r2_market_contraction_{variant}"; daily = destination / profile["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); gate = t05_micro_rebound_gate(daily); scored = score(next(row for row in targets() if row["target_id"] == "T05"), monthly_balances(daily)); candidate = {"target_id": "T05", "sample_id": profile["sample_id"], "profile": str(path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "score": scored}; historical = strong_history_pairs_for_daily(history, candidate) if reconciled and manifest["execution_status"] == "complete" and gate["passes"] else {"strong_history_pairs": [], "double_high_similarity_pairs": []}; balances, cash = daily_signature(daily); peer_pairs = []
            if gate["passes"]:
                for peer in peers:
                    p_balance, p_cash = daily_signature(ROOT / peer["daily_csv"]); balance_corr, cash_corr = pearson(balances, p_balance), pearson(cash, p_cash); peer_pairs.append({"other_target_id": peer["target_id"], "balance_correlation": round(balance_corr, 6) if balance_corr is not None else None, "cash_change_correlation": round(cash_corr, 6) if cash_corr is not None else None})
            strong_peers = [row for row in peer_pairs if row["balance_correlation"] is not None and row["cash_change_correlation"] is not None and row["balance_correlation"] >= .90 and row["cash_change_correlation"] >= .65]
            trials.append({"budget_variant": variant, **candidate, "fixed_seed": profile["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "micro_rebound_gate": gate, "historical_similarity": historical, "peer_similarity": {"strong_pairs": strong_peers}, "passes_preflight": reconciled and manifest["execution_status"] == "complete" and gate["passes"] and not historical["strong_history_pairs"] and not historical["double_high_similarity_pairs"] and not strong_peers})
        except Exception as error: trials.append({"budget_variant": variant, "passes_preflight": False, "error_type": type(error).__name__, "error": str(error)})
    passed = [row for row in trials if row.get("passes_preflight")]; chosen = min(passed, key=lambda row: (abs(row["score"]["segments"][0]["actual_mean_slope_cny"] + 70000), row["budget_variant"])) if passed else None; stability = None
    if chosen:
        frozen = json.loads((ROOT / chosen["profile"]).read_text(encoding="utf-8")); frozen["sample_id"] = f"{frozen['sample_id']}_stability"; frozen["random_seed"] = 2026098999; stability_path = OUT / "P11R2_T05第三业务解释参数" / "T05_market_contraction_固定种子复核.json"; dump(stability_path, frozen); destination = OUT / "P11R2_T05第三业务解释流水" / "T05"; run_id = "t05_p11r2_market_contraction_stability"; daily = destination / frozen["sample_id"] / run_id / "account_daily_total.csv"
        try:
            if not daily.exists(): bundle = generate_from_profile(stability_path, output_root=destination, output_run_id=run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else: reconciled = True
            manifest = json.loads((daily.parent / "generation_manifest.json").read_text(encoding="utf-8")); stability = {"profile": str(stability_path.relative_to(ROOT)).replace("\\", "/"), "daily_csv": str(daily.relative_to(ROOT)).replace("\\", "/"), "fixed_seed": frozen["random_seed"], "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "micro_rebound_gate": t05_micro_rebound_gate(daily), "passes_shape": reconciled and manifest["execution_status"] == "complete" and t05_micro_rebound_gate(daily)["passes"]}
        except Exception as error: stability = {"passes_shape": False, "error_type": type(error).__name__, "error": str(error)}
    decision = "limited_candidate_passed_pending_final_export" if chosen and stability and stability.get("passes_shape") else "third_explanation_budget_exhausted_hold"; report = {"stage": "P11R2_T05_third_explanation_final_budget", "scope": "第三解释的最后4个预算点，只降低两笔历史应收结清金额；不改变24目标配置", "trials": trials, "chosen_candidate": chosen, "stability_check": stability, "third_explanation_central_calls": 12 + (1 if stability else 0), "decision": decision}; dump(OUT / "P11R2_T05第三业务解释搜索与复审.json", report); print(json.dumps({"decision": decision, "preflight_passed": len(passed), "calls": report["third_explanation_central_calls"], "chosen": chosen["budget_variant"] if chosen else None}, ensure_ascii=False))


def main_p7_formal_export_eligible() -> None:
    """把已通过两道相似性审计的候选以原冻结参数重新经中央正式导出；不修改421历史清单。"""
    candidate_manifest = json.loads((OUT / "P5R2_24目标候选manifest.json").read_text(encoding="utf-8")); historical_audit = json.loads((OUT / "P6_正式准入相似度审计.json").read_text(encoding="utf-8")); peer_audit = json.loads((OUT / "P6_候选间正式准入相似度审计.json").read_text(encoding="utf-8"))
    history_allowed = {row["target_id"] for row in historical_audit["results"] if row["formal_similarity_decision"] == "eligible_for_formal_preflight"}; peer_allowed = set(peer_audit["formal_batch_eligible_targets"]); rows = [row for row in candidate_manifest["rows"] if row["target_id"] in history_allowed & peer_allowed]
    if len(rows) != 22 or {row["target_id"] for row in candidate_manifest["rows"]} - {row["target_id"] for row in rows} != {"T05", "T06"}: raise ValueError("正式导出范围必须恰为通过审计的22户，T05/T06保持隔离")
    formal_root = OUT / "P7_正式导出_22户" / "正式流水" / "train"; frozen_rows = []
    for row in rows:
        profile = ROOT / row["profile"]; frozen_rows.append({"target_id": row["target_id"], "sample_id": row["sample_id"], "purpose": "train", "profile": row["profile"], "profile_sha256": sha(profile), "candidate_daily_csv": row["daily_csv"], "formal_run_id": "formal_p7_v1", "formal_directory": str((formal_root / row["sample_id"] / "formal_p7_v1").relative_to(ROOT)).replace("\\", "/")})
    dump(OUT / "P7_正式准入冻结清单.json", {"version": "p7_formal_export_freeze_v1", "scope": "仅P6双审计通过的22户；T05/T06因历史强相似隔离", "history_audit": "P6_正式准入相似度审计.json", "peer_audit": "P6_候选间正式准入相似度审计.json", "rows": frozen_rows})
    receipts = []
    for row in frozen_rows:
        profile = ROOT / row["profile"]; formal = ROOT / row["formal_directory"]
        try:
            if not (formal / "generation_manifest.json").exists():
                bundle = generate_from_profile(profile, output_root=formal_root, output_run_id=row["formal_run_id"]); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
            else:
                reconciled = True
            manifest = json.loads((formal / "generation_manifest.json").read_text(encoding="utf-8")); candidate_daily = ROOT / row["candidate_daily_csv"]; formal_daily = formal / "account_daily_total.csv"; candidate_tx = candidate_daily.parent / "transactions_total.csv"; formal_tx = formal / "transactions_total.csv"; complete = manifest["execution_status"] == "complete" and len(monthly_balances(formal_daily)) == 24
            receipts.append({**row, "actual_execution_status": manifest["execution_status"], "central_reconciliation_checked": reconciled, "candidate_daily_sha256": sha(candidate_daily), "formal_daily_sha256": sha(formal_daily), "candidate_transaction_sha256": sha(candidate_tx), "formal_transaction_sha256": sha(formal_tx), "candidate_formal_daily_exact": sha(candidate_daily) == sha(formal_daily), "candidate_formal_transactions_exact": sha(candidate_tx) == sha(formal_tx), "formal_complete_24_months": complete, "formal_decision": "formal_train_eligible" if reconciled and complete and sha(candidate_daily) == sha(formal_daily) and sha(candidate_tx) == sha(formal_tx) else "formal_export_hold"})
        except Exception as error: receipts.append({**row, "formal_decision": "formal_export_hold", "error_type": type(error).__name__, "error": str(error)})
    accepted = [row for row in receipts if row["formal_decision"] == "formal_train_eligible"]
    report = {"stage": "P7_formal_export", "scope": "22户中央正式导出与候选/正式逐字节一致性核对；未写入421历史清单，未启动模型训练", "frozen_targets": [row["target_id"] for row in frozen_rows], "formal_train_eligible_count": len(accepted), "held_targets": [row["target_id"] for row in receipts if row["formal_decision"] != "formal_train_eligible"], "receipts": receipts, "T05_T06_status": "历史强相似隔离，未作正式导出"}
    dump(OUT / "P7_正式导出与训练候选复审.json", report); dump(OUT / "P7_训练可用候选清单.json", {"status": "formal_train_eligible_not_trained", "rows": accepted, "isolated_targets": ["T05", "T06"]})
    progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8")); progress.update({"stage": "P7_22_formal_train_eligible" if len(accepted) == 22 else "P7_formal_export_needs_review", "completed": list(dict.fromkeys(progress["completed"] + ["P7 22户中央正式导出、候选/正式一致性核对与训练候选登记"])), "next": "T05/T06因历史强相似隔离；须在不突破既定搜索上限的前提下重设计或由用户授权新的探索额度", "last_updated_utc": datetime.now(timezone.utc).isoformat()}); dump(OUT / "progress.json", progress)
    print(json.dumps({"output": str(OUT), "formal_train_eligible": len(accepted), "held": len(receipts) - len(accepted), "isolated": ["T05", "T06"]}, ensure_ascii=False))


def main_p7_t15_exact_reproduction() -> None:
    """T15 从候选运行所保存的输入快照抽出正式画像，以原运行编号验证确定性复现。"""
    report = json.loads((OUT / "P7_正式导出与训练候选复审.json").read_text(encoding="utf-8")); item = next(row for row in report["receipts"] if row["target_id"] == "T15")
    candidate_daily = ROOT / item["candidate_daily_csv"]; candidate_scenario = candidate_daily.parent / "scenario_profile.json"; envelope = json.loads(candidate_scenario.read_text(encoding="utf-8")); frozen_input = envelope.get("input_profile")
    if not isinstance(frozen_input, dict): raise ValueError("T15候选运行缺少可复现的input_profile快照")
    frozen_profile = OUT / "P7R2_T15冻结输入画像.json"; dump(frozen_profile, frozen_input)
    candidate_run_id = candidate_daily.parent.name; formal_root = OUT / "P7R2_T15冻结画像复现" / "正式流水" / "train"; formal = formal_root / item["sample_id"] / candidate_run_id
    if not (formal / "generation_manifest.json").exists(): bundle = generate_from_profile(frozen_profile, output_root=formal_root, output_run_id=candidate_run_id); reconciled = bundle.run.final_cash_reconciliation.observed_source_to_ledger_difference_fen == 0
    else: reconciled = True
    manifest = json.loads((formal / "generation_manifest.json").read_text(encoding="utf-8")); formal_daily, formal_tx = formal / "account_daily_total.csv", formal / "transactions_total.csv"; candidate_tx = candidate_daily.parent / "transactions_total.csv"; exact_daily, exact_tx = sha(candidate_daily) == sha(formal_daily), sha(candidate_tx) == sha(formal_tx); decision = "formal_train_eligible" if reconciled and manifest["execution_status"] == "complete" and len(monthly_balances(formal_daily)) == 24 and exact_daily and exact_tx else "formal_export_hold"
    result = {"stage": "P7R2_T15_frozen_profile_reproduction", "scope": "仅抽取候选运行保存的input_profile快照并恢复原运行编号；不改参数、不改种子、不重搜", "target_id": "T15", "candidate_run_id": candidate_run_id, "candidate_scenario_profile": str(candidate_scenario.relative_to(ROOT)).replace("\\", "/"), "candidate_scenario_profile_sha256": sha(candidate_scenario), "frozen_profile": str(frozen_profile.relative_to(ROOT)).replace("\\", "/"), "frozen_input_profile_sha256": sha(frozen_profile), "formal_directory": str(formal.relative_to(ROOT)).replace("\\", "/"), "central_reconciliation_checked": reconciled, "actual_execution_status": manifest["execution_status"], "candidate_formal_daily_exact": exact_daily, "candidate_formal_transactions_exact": exact_tx, "decision": decision}; dump(OUT / "P7R2_T15冻结画像复现复审.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("p3", "p3r2", "p3r3", "p3r3visual", "p3r4", "p3r5", "p3r6", "p3r7", "p3atlas", "p4zero", "p4zeror1", "p4zeroatlas", "p4remaining", "p4t08r1", "p5atlas", "p4t05t06r2", "p6similarity", "p6peers", "p7formal", "p7t15", "p8t05t06", "p8t05t06r1", "p8t05r2", "p9formal", "p10t05", "p11t05", "p11t05r1", "p11t05r2"), default="p3")
    args = parser.parse_args()
    {"p3": main, "p3r2": main_r2, "p3r3": main_r3, "p3r3visual": main_r3_visual, "p3r4": main_r4, "p3r5": main_r5, "p3r6": main_r6, "p3r7": main_r7_t07, "p3atlas": main_p3_atlas, "p4zero": main_p4_zero_turn, "p4zeror1": main_p4_zero_turn_r1, "p4zeroatlas": main_p4_zero_atlas, "p4remaining": main_p4_remaining, "p4t08r1": main_p4_t08_r1, "p5atlas": main_p5_all_targets_atlas, "p4t05t06r2": main_p4_t05_t06_r2, "p6similarity": main_p6_formal_similarity_audit, "p6peers": main_p6_candidate_similarity_audit, "p7formal": main_p7_formal_export_eligible, "p7t15": main_p7_t15_exact_reproduction, "p8t05t06": main_p8_t05_t06_second_explanations, "p8t05t06r1": main_p8_t05_t06_second_explanations_r1, "p8t05r2": main_p8_t05_second_explanation_r2, "p9formal": main_p9_latest_formal_gate_and_export, "p10t05": main_p10_t05_micro_rebound_formalization, "p11t05": main_p11_t05_third_explanation, "p11t05r1": main_p11_t05_third_explanation_r1, "p11t05r2": main_p11_t05_third_explanation_r2}[args.phase]()
