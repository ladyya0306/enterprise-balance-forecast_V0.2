"""合同预算先排重、中央精确执行、实际流水再排重的可恢复能力验收入口。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import subprocess
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from random import Random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from syn_b1.formal_profile import default_profile, resolve_profile
from syn_b1.formal_generator import generate_from_profile
from run_shape_guided_budget_search_20260913 import monthly_balances, trend_signature, t05_micro_rebound_gate

HISTORY = ROOT / "data/synthetic_production/coverage102_v1/现行421户历史清单_20260912_v1/现行历史manifest_421.json"
OLD = HISTORY.parent.parent / "形状引导预算搜索_20260913_v1"
OUTPUT = HISTORY.parent.parent / "合同流水整改_20260913_v1"
MONTHS = [f"{2024 + i // 12}-{1 + i % 12:02d}" for i in range(24)]
DAYS = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(731)]
DAY_INDEX = {d: i for i, d in enumerate(DAYS)}
RULE = {"version": "contract_novelty_v1", "balance_corr": .90, "operating_cash_corr": .65,
        "monthly_budget_distance": .00015, "minimum_common_days": 365,
        "basis": "经营节奏相关只使用人工核对表标识的非融资收支；原含注资相关仍单列保留。形状相似可接受，形状及经营同时相似拒绝。"}
# Migration bridge for the two receipts written before candidate_id was added.
# It is deliberately exact-ID only, never a family-wide exemption.
LEGACY_FAILED_CANDIDATE_IDS={"t05_exit_milestone_r00":"t05_exit_milestone_candidate",
                             "t05_exit_milestone_r01":"t05_exit_milestone_candidate"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_candidate_revision(candidate_id, sample_id, candidate_by_id=None):
    """Exact candidate identity only; family equality is never sufficient."""
    source = (candidate_by_id or {}).get(sample_id) or LEGACY_FAILED_CANDIDATE_IDS.get(sample_id)
    return bool(candidate_id) and source == candidate_id


def valid_failed_parent(parent_row, candidate_id, target_id, family_id):
    observed = parent_row.get("candidate_id") or LEGACY_FAILED_CANDIDATE_IDS.get(parent_row.get("sample_id"))
    return (parent_row.get("formal_admission") is False
            and parent_row.get("status") in {"generated_pending_full_review", "generated_needs_revision"}
            and observed == candidate_id and parent_row.get("target_id") == target_id
            and parent_row.get("family_id") == family_id)


def save(path, value):
    # 每份回执完整落盘；同名预算如内容不同必须另版。
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != data:
        raise FileExistsError(f"拒绝覆盖已有版本：{path}")
    path.write_text(data, encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def money(fen):
    return f"{fen // 100}.{fen % 100:02d}"


def correlation(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def unit(values):
    a = np.asarray(values, dtype=float)
    return a / max(1, np.abs(a).sum())


def signature(directory):
    # 使用真实日期交集；停止后的日期保留缺失，不能补零或按行错位比较。
    balance = np.full(731, np.nan); cash = np.full(731, np.nan)
    with (directory / "account_daily_total.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["calendar_date"] in DAY_INDEX:
                i = DAY_INDEX[row["calendar_date"]]
                balance[i] = float(row["ending_balance_cny"])
                cash[i] = float(row["inflow_cny"]) - float(row["outflow_cny"])
    operating = np.where(np.isfinite(cash), 0.0, np.nan)
    with (directory / "cash_flow_review_notes.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            i = DAY_INDEX.get(row["booking_datetime"][:10])
            # 仅删除有明确融资分类的款项；工资、采购、税费和设备均保留。
            if i is not None and not any(word in row["cash_flow_type_cn"] for word in ("股东", "银行借款", "贷款", "借款本金", "利息")):
                operating[i] += float(row["amount_cny"]) * (1 if row["direction_cn"] == "流入" else -1)
    return {"balance": balance, "cash": cash, "operating": operating}


def compare(a, b):
    common = int(np.sum(np.isfinite(a["balance"]) & np.isfinite(b["balance"])))
    bc = correlation(a["balance"], b["balance"])
    raw = correlation(a["cash"], b["cash"])
    oc = correlation(a["operating"], b["operating"])
    return {"common_days": common, "balance_correlation": bc, "raw_cash_correlation": raw,
            "operating_cash_correlation": oc,
            "legacy_strong": common >= 365 and bc is not None and raw is not None and bc >= .90 and raw >= .65,
            "operating_strong": common >= 365 and bc is not None and oc is not None and bc >= .90 and oc >= .65}


def budget_vector(raw):
    # 只读已保存的确认预算；无法恢复的旧字段明确记缺失，不猜历史预算。
    plan = raw.get("operating", {}).get("business_nodes", {})
    nodes = {n["month"]: n for n in plan.get("nodes", [])}
    if any(m not in nodes for m in MONTHS):
        return None
    series = []
    for field in ("sales", "variable_cost", "payroll", "rent", "other_expense"):
        vals = []
        for m in MONTHS:
            n = nodes[m]
            if "sales_units" in n:
                if field in ("sales", "variable_cost"):
                    price = n["unit_price_cny" if field == "sales" else "unit_variable_cost_cny"]
                    vals.append(n["sales_units"] * round(float(price) * 100))
                else:
                    vals.append(round(float(n[field + "_cny"]) * 100))
            else:
                vals.append(n[field + "_fen"])
        series.extend(unit(vals).tolist())
    return np.asarray(series)


def history_index():
    manifest = read(HISTORY)
    rows = []
    for row in manifest["rows"]:
        directory = ROOT / row["formal"]
        envelope = read(directory / "scenario_profile.json")
        raw = envelope["input_profile"]
        rows.append({"id": row["sample_id"], "directory": directory, "signature": signature(directory),
                     "budget": budget_vector(raw), "shape": trend_signature(monthly_balances(directory / "account_daily_total.csv")),
                     "daily_sha": digest(directory / "account_daily_total.csv"), "profile_sha": digest(directory / "scenario_profile.json")})
    return rows


def make_budget(index):
    # 固定主种子产生事前虚构客户合同，不根据实际余额或历史通过结果重抽。
    rng = Random(713000 + index)
    p = default_profile(f"contract_diversity_t05_{index:04d}", "contract_repair_v1")
    p.update(schema_version="syn_b1_formal_profile_2_0", start_date=DAYS[0], end_date=DAYS[-1], random_seed=714000 + index)
    p["calendar"]["prediction_cutoff"] = DAYS[-1]
    p["initial_capital"]["amount_cny"] = "1800000.00"
    op = p["operating"]
    for key in ("base_monthly_sales_cny", "sales_monthly_growth_percent", "sales_month_weights_percent", "gross_margin_percent", "monthly_payroll_cny", "monthly_rent_and_utilities_cny", "monthly_other_operating_expense_cny"):
        op.pop(key)
    for key in ("collection_schedule", "supplier_payment_schedule", "payroll_payment_schedule", "rent_and_utilities_payment_schedule", "other_operating_expense_payment_schedule"):
        op[key] = [{"delay_days": 0, "share_percent": "100"}]
    op["fixed_assets"] = [{"event_id": "EQ01", "asset_type": "equipment", "purchase_month": MONTHS[0], "purchase_amount_cny": "90000.00", "payment_schedule": [{"delay_days": 0, "share_percent": "100"}], "ready_for_use_date": DAYS[0], "useful_life_months": 60, "residual_value_percent": "5"}]
    client_count = rng.randint(6, 10)
    clients = [{"id": f"C{i:02d}", "initial_units": rng.randint(63, 115), "attrition_per_month": rng.choice([1, 2, 3]),
                "attrition_start": rng.randint(2, 8), "billing_day": rng.randint(3, 24)} for i in range(client_count)]
    # 每家客户保留独立订单数量和收缩起点，绝不整体同比缩放已有销售序列。
    price = rng.choice([42000, 45000, 48000]); cost = rng.choice([19000, 21000, 22500])
    payroll, rent, other = rng.randint(73000, 86000) * 100, rng.randint(23000, 29000) * 100, rng.randint(33000, 47000) * 100
    payroll_day, rent_day = rng.randint(6, 24), rng.randint(2, 22)
    late_sources = sorted(rng.sample(range(2, 14), 2))
    late_durations = [rng.randint(2, 4), rng.randint(3, 5)]
    delayed_client = rng.randrange(client_count)
    events = []; nodes = []; customer_orders = []
    def add(category, month, when, amount, suffix, note, asset=""):
        if amount:
            events.append(dict(event_id=f"{month}-{category}-{suffix}", category=category, recognition_month=month,
                               asset_id=asset, due_date=when, amount_fen=amount, note_cn=note))
    inventory_before = 0
    for m, month in enumerate(MONTHS):
        units = [max(12, c["initial_units"] - max(0, m - c["attrition_start"]) * c["attrition_per_month"]) for c in clients]
        total = sum(units); material = total * cost; inventory = material // 8
        node = dict(month=month, stage_id=f"declining-demand-{m+1}", business_reason="客户框架订单逐户收缩；工资、租金、基本采购连续。延期验收款引用原交付月，不重复确认销售或成本。",
                    sales_units=total, unit_price_cny=money(price), unit_variable_cost_cny=money(cost),
                    target_inventory_cny=money(inventory), payroll_cny=money(payroll), rent_cny=money(rent), other_expense_cny=money(other))
        nodes.append(node)
        for i, (client, quantity) in enumerate(zip(clients, units)):
            amount = quantity * price; late = 0
            if i == delayed_client and m in late_sources:
                # 部分尾款在后续验收月份结清，原月销售只确认一次。
                late = min(3500000, amount // 2)
                due_m = m + late_durations[late_sources.index(m)]
                add("collection", month, MONTHS[due_m] + f"-{client['billing_day']:02d}", late, client["id"] + "-tail", "原交付合同延期验收尾款结清")
            add("collection", month, month + f"-{client['billing_day']:02d}", amount - late, client["id"], "本月已交付服务合同回款")
            customer_orders.append(dict(month=month, customer_id=client["id"], units=quantity, price_fen=price, deferred_fen=late))
        purchase = material + inventory - inventory_before
        # 采购三至五个真实批次，月份与客户结构无共同模板。
        supplier_count = rng.randint(3, 5); weights = [rng.randint(10, 30) for _ in range(supplier_count)]
        allocations = [purchase * w // sum(weights) for w in weights]; allocations[-1] += purchase - sum(allocations)
        for i, amount in enumerate(allocations):
            add("supplier_payment", month, month + f"-{rng.randint(2, 27):02d}", amount, f"S{i}", "已签本月到货采购合同付款")
        add("payroll_payment", month, month + f"-{payroll_day:02d}", payroll, "team", "在岗服务班组工资")
        add("rent_payment", month, month + f"-{rent_day:02d}", rent, "lease", "固定场地租赁合同付款")
        add("other_expense_payment", month, month + f"-{rng.randint(3, 26):02d}", other, "support", "本月持续合规和外部支持服务费")
        inventory_before = inventory
    add("fixed_asset_payment", MONTHS[0], "2024-01-03", 9000000, "equipment", "开工设备购买款", "EQ01")
    op["business_nodes"] = dict(version="physical_contract_business_v1", nodes=nodes,
                                asset_capacities=[dict(event_id="EQ01", monthly_units=2000)], settlements=events,
                                capacity_basis="自有设备已投用，月产能2000等效服务件；采购及人员独立签约。")
    basis = dict(sample_id=p["sample_id"], family_id=p["sample_id"], clients=clients, customer_orders=customer_orders,
                 recognition_rule="本期成立；应收来自本期早月已确认交付；延期收回不重复产生销售、成本或采购。",
                 mechanism="市场收缩下多客户存量服务与两项延期验收", seed=713000 + index)
    return p, basis


def planned_signature(raw):
    # 前置计划只统计明确合同金额日期，不调用中央流水，也不当成实际税后余额。
    cash = np.zeros(731)
    for e in raw["operating"]["business_nodes"]["settlements"]:
        if e["due_date"] in DAY_INDEX:
            cash[DAY_INDEX[e["due_date"]]] += e["amount_fen"] / 100 * (1 if e["category"] == "collection" else -1)
    return dict(balance=np.cumsum(cash), cash=cash.copy(), operating=cash)


def calibration():
    # 已知复制、同比缩放必须拒绝；只有同日注资相同不能当成相同经营。
    rng = np.random.default_rng(43)
    a = rng.normal(-100, 500, 731); b = rng.normal(-100, 500, 731)
    sa = dict(balance=1800000 + np.cumsum(a), operating=a, cash=a.copy())
    sb = dict(balance=1800000 + np.cumsum(b), operating=b, cash=b.copy())
    sa["cash"][0] += 1800000; sb["cash"][0] += 1800000
    copy_result = compare(sa, sa)
    scaled = compare(sa, {k: v * 2 for k, v in sa.items()})
    funding_only = compare(sa, sb)
    assert copy_result["operating_strong"] and scaled["operating_strong"]
    assert funding_only["legacy_strong"] and not funding_only["operating_strong"]
    return dict(exact_copy=copy_result, scaled_copy=scaled, same_capital_different_operations=funding_only)


def audit_export(directory, raw):
    # 从公开流水独立逐笔核账，并按自然日重新汇总；不只相信生成器布尔标记。
    def fen(value):
        return int(Decimal(value) * 100)
    with (directory / "transactions_total.csv").open(encoding="utf-8-sig", newline="") as f:
        tx = list(csv.DictReader(f))
    daily = defaultdict(lambda: [0, 0, 0])
    balance = 0
    for row in tx:
        debit, credit = fen(row["debit_cny"]), fen(row["credit_cny"])
        assert (debit > 0) != (credit > 0)
        balance += credit - debit
        assert balance >= 0 and balance == fen(row["post_transaction_balance_cny"])
        d = daily[row["booking_datetime"][:10]]
        d[0] += credit; d[1] += debit; d[2] += 1
    with (directory / "account_daily_total.csv").open(encoding="utf-8-sig", newline="") as f:
        days = list(csv.DictReader(f))
    assert [r["calendar_date"] for r in days] == DAYS
    balance = 0
    for row in days:
        inflow, outflow, count = daily[row["calendar_date"]]
        balance += inflow - outflow
        assert (fen(row["inflow_cny"]), fen(row["outflow_cny"]), int(row["transaction_count"]), fen(row["ending_balance_cny"])) == (inflow, outflow, count, balance)
    truth = read(directory / "_restricted/scenario_truth.json")
    candidates = truth["transaction_plan"]["candidates"]
    contracts = raw["operating"]["business_nodes"]["settlements"]
    expected = {e["event_id"]: (e["due_date"], e["amount_fen"]) for e in contracts if e["due_date"] <= DAYS[-1]}
    observed = {e["business_order_id"]: (e["booking_datetime"][:10], e["amount_fen"]) for e in candidates if e.get("business_order_id")}
    assert expected == observed
    for row in truth["operating_cycle"]["monthly_rows"]:
        receivable = sum(e["amount_fen"] for e in contracts if e["category"] == "collection" and e["recognition_month"] <= row["month"] < e["due_date"][:7])
        assert row["ending_accounts_receivable_fen"] == receivable
    quality = read(directory / "quality_report.json")
    assert quality["future_training_data_quality_ready"] and quality["checks"]["final_cash_reconciled"]
    return dict(days=len(days), transactions=len(tx), closing_balance_cny=balance / 100, contract_dates_and_amounts_exact=True, independent_ledger_reconciled=True, receivables_reconciled=True)


def finalize():
    # 批量候选只做工程验收；按事前顺序挑首份合格T05完成正式复现。
    summary = read(OUTPUT / "能力验收汇总.json")
    old23 = read(OLD / "P9_正式训练可用候选清单.json")
    verified = []
    for receipt in summary["receipts"]:
        raw = read(receipt["profile"])
        verified.append(dict(sample_id=receipt["sample_id"], **audit_export(Path(receipt["directory"]), raw)))
    save(OUTPUT / "12户独立账务与合同复核.json", verified)
    candidates = [r for r in summary["receipts"] if r["passed"]]
    selected = None; peer_audits = []
    for r in candidates:
        sig = signature(Path(r["directory"]))
        pairs = [dict(target_id=p["target_id"], **compare(sig, signature(ROOT / p["formal_directory"]))) for p in old23["rows"]]
        peer_audits.append(dict(sample_id=r["sample_id"], pairs=pairs))
        if not any(p["operating_strong"] for p in pairs):
            selected = r
            break
    save(OUTPUT / "与既有23目标比较.json", peer_audits)
    if selected is None:
        raise ValueError("没有通过既有23目标排重的T05，保留候选继续诊断")
    raw = read(selected["profile"]); candidate_dir = Path(selected["directory"])
    stability = []
    for seed in (771901, 771902):
        probe = deepcopy(raw); probe["random_seed"] = seed
        path = OUTPUT / "复现参数" / f"seed_{seed}.json"; save(path, probe)
        destination = OUTPUT / f"固定种子{seed}"; directory = destination / probe["sample_id"] / probe["run_id"]
        if not directory.exists():
            generate_from_profile(path, output_root=destination)
        audit = audit_export(directory, probe)
        assert t05_micro_rebound_gate(directory / "account_daily_total.csv")["passes"]
        # 同合同换种子不能改变普通收付款日期金额；交易编号允许变化。
        assert np.array_equal(signature(directory)["balance"], signature(candidate_dir)["balance"])
        stability.append(dict(seed=seed, **audit, same_daily_cash=True))
    destination = OUTPUT / "正式流水" / "train"
    directory = destination / raw["sample_id"] / raw["run_id"]
    if not directory.exists():
        generate_from_profile(selected["profile"], output_root=destination)
    final_audit = audit_export(directory, raw)
    exact = {f: digest(directory / f) == digest(candidate_dir / f) for f in ("account_daily_total.csv", "transactions_total.csv", "cash_flow_review_notes.csv")}
    assert all(exact.values())
    historical = read(OUTPUT / "历史形状与预算索引.json")
    # 最终只读指纹检查保护历史，不重新生成或重验历史账务。
    by_id = {r["sample_id"]: r for r in read(HISTORY)["rows"]}
    for row in historical["rows"]:
        d = ROOT / by_id[row["id"]]["formal"]
        assert digest(d / "account_daily_total.csv") == row["daily_sha"]
        assert digest(d / "scenario_profile.json") == row["profile_sha"]
    receipt = dict(target_id="T05", sample_id=raw["sample_id"], family_id=raw["sample_id"], purpose="train", profile=selected["profile"],
                   formal_directory=str(directory.relative_to(ROOT)), candidate_daily_csv=str(candidate_dir / "account_daily_total.csv"),
                   formal_decision="formal_train_eligible", training_started=False, admission_rule=RULE, **final_audit,
                   exact_reproduction=exact, stability=stability, historical_fingerprints_unchanged=421)
    save(OUTPUT / "T05正式验收.json", receipt)
    save(OUTPUT / "24目标正式样本清单.json", dict(status="formal_train_eligible_not_trained", rows=sorted(old23["rows"] + [receipt], key=lambda r:r["target_id"]),
                                                new_scope="本轮新增T05；另外23户引用P9既有结论，不重新审查", training_started=False))
    labels = {r["target_id"]: r["product_cn"] for r in read(OLD / "P9_24最新候选manifest.json")["rows"]}
    labels.update(T05="市场收缩与延期验收尾款", T06="可循环包装联合配售收尾")
    rows = []
    for row in sorted(old23["rows"] + [receipt], key=lambda r:r["target_id"]):
        d = ROOT / row["formal_directory"]
        rows.append(dict(target_id=row["target_id"], sample_id=row["sample_id"], run_id=d.name, purpose="train", product_cn=labels[row["target_id"]],
                         formal=str(d), daily_csv=str(d / "account_daily_total.csv"), profile=str(ROOT / row["profile"]), candidate_status="formal_train_eligible_not_trained"))
    save(OUTPUT / "24目标绘图manifest.json", dict(rows=rows))
    atlas = OUTPUT / "24目标正式余额图册"
    if not atlas.exists():
        subprocess.run([sys.executable, str(ROOT / "tools/render_candidate_pool_atlas.py"), "--manifest", str(OUTPUT / "24目标绘图manifest.json"), "--output", str(atlas), "--all-only", "--page-size", "4"], check=True, cwd=ROOT)
    from syn_b1.profile_document import write_user_profile
    yaml = OUTPUT / "T05完整参数.yaml"
    if not yaml.exists():
        write_user_profile(yaml, raw)
    source_files = ["src/syn_b1/business_nodes.py", "src/syn_b1/contract_business.py", "src/syn_b1/formal_profile.py", "src/syn_b1/formal_export.py", "src/syn_b1/profile_document.py", "src/syn_b1/operating_cycle.py", "src/syn_b1/transaction_planner.py", "tools/shape_contract_search.py"]
    save(OUTPUT / "整改源代码指纹.json", {f:digest(ROOT / f) for f in source_files})
    print(json.dumps(dict(T05="formal_train_eligible", targets=24, ledger=final_audit, stability_seeds=2, exact=exact, training_started=False)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-count", type=int, default=320)
    parser.add_argument("--central-limit", type=int, default=12)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--mechanism-config", type=Path, help="独立经营机制配置；只做有界探针，不正式入库")
    parser.add_argument("--expansion-config", type=Path, help="24目标通用实体合同队列")
    parser.add_argument("--list-targets", action="store_true", help="列出原24目标，不生成预算或流水")
    parser.add_argument("--test-t05-config", type=Path, help="有界T05实体闭环能力验收")
    parser.add_argument("--test-target24-config", type=Path, help="24目标预算响应及近重复负对照")
    parser.add_argument("--autonomous-target-config", type=Path, help="v2：24目标合同预算的受控自动扩展")
    parser.add_argument("--single-stage-ols-d-audit", type=Path, help="对中央流水目录执行单阶段OLS+D验收")
    args = parser.parse_args()
    if sum(bool(x) for x in (args.finalize, args.mechanism_config, args.expansion_config, args.list_targets, args.test_t05_config, args.test_target24_config, args.autonomous_target_config, args.single_stage_ols_d_audit)) > 1:
        raise ValueError("运行模式互斥，不能混合新队列和旧正式导出")
    if args.test_target24_config:
        return test_target24_response(args.test_target24_config)
    if args.autonomous_target_config:
        return run_autonomous_target_expansion(args.autonomous_target_config)
    if args.single_stage_ols_d_audit:
        print(json.dumps(single_stage_ols_d_assessment(args.single_stage_ols_d_audit, slope_reference_cny_per_month=-70000,
              initial_capital_cny=4000000.0), ensure_ascii=False))
        return
    if args.test_t05_config:
        return test_t05_entities(args.test_t05_config)
    if args.list_targets:
        from run_shape_guided_budget_search_20260913 import targets
        print(json.dumps(dict(targets=targets(), note="目标目录不是机制就绪或样本数量承诺"), ensure_ascii=False))
        return
    if args.expansion_config:
        return run_entity_expansion(args.expansion_config)
    if args.mechanism_config:
        if args.finalize:
            raise ValueError("机制探针不能使用旧T05正式登记入口")
        return run_mechanism_pilot(args.mechanism_config)
    if args.finalize:
        return finalize()
    if not 1 <= args.budget_count <= 400 or not 1 <= args.central_limit <= 16:
        raise ValueError("本次能力验收限400份便宜预算、16次中央候选；不自动量产")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history_sha = digest(HISTORY)
    save(OUTPUT / "规则与预算冻结.json", dict(rule=RULE, budget_count=args.budget_count, central_limit=args.central_limit,
                                            stability_seeds=[771901, 771902], history_sha=history_sha, calibration=calibration()))
    history = history_index()
    save(OUTPUT / "历史形状与预算索引.json", dict(history_count=len(history), shape_counts=dict(Counter(("上行" if h["shape"]["mean_monthly_change_cny"] > 0 else "下行") + f"_转折{h['shape']['major_turn_count']}" for h in history)),
         budget_available=sum(h["budget"] is not None for h in history), rows=[{k:v for k,v in h.items() if k in ("id", "shape", "daily_sha", "profile_sha")} for h in history]))
    # 复查上一轮T05拒绝原因，保留原含注资分数和修正后分数，绝不重生历史。
    old = read(OLD / "P11R2_T05第三业务解释搜索与复审.json")
    diagnoses = []
    for row in old["trials"]:
        sig = signature((ROOT / row["daily_csv"]).parent)
        pairs = [dict(id=h["id"], **compare(sig, h["signature"])) for h in history]
        diagnoses.append(dict(variant=row["budget_variant"], legacy_strong=sum(p["legacy_strong"] for p in pairs), operating_strong=sum(p["operating_strong"] for p in pairs), pairs=[p for p in pairs if p["legacy_strong"] or p["operating_strong"]]))
    save(OUTPUT / "T05旧拒绝原因复核.json", diagnoses)
    print(json.dumps(dict(stage="history_indexed", historical_budgets=sum(h["budget"] is not None for h in history), old_gate_counts=[(d["legacy_strong"], d["operating_strong"]) for d in diagnoses])), flush=True)
    queue = []; budget_results = []; budget_signatures = []
    for i in range(args.budget_count):
        raw, basis = make_budget(i)
        # 同一数值模板和整体同比缩放前置拒绝；形状本身允许有重叠。
        vector = budget_vector(raw)
        hist_dist = min((float(np.abs(vector - h["budget"]).mean()) for h in history if h["budget"] is not None), default=None)
        peer_dist = min((float(np.abs(vector - v).mean()) for v in budget_signatures), default=None)
        sig = planned_signature(raw)
        # 先测经营收支，明显不同则无需再算累计趋势，减少前置筛查成本。
        plan_strong = []
        for h in history:
            oc = correlation(sig["operating"], h["signature"]["operating"])
            if oc is not None and oc >= .65 and compare(sig, h["signature"])["operating_strong"]:
                plan_strong.append(h["id"])
        resolve_profile(raw)
        # 便宜可行性界只使用合同收付；预留税费和日内支出缓冲。
        planned_min = 1800000 + float(np.min(sig["balance"]))
        passed = hist_dist is not None and hist_dist > RULE["monthly_budget_distance"] and (peer_dist is None or peer_dist > RULE["monthly_budget_distance"]) and not plan_strong and planned_min > 180000 and sig["balance"][-1] < -200000
        result = dict(index=i, sample_id=raw["sample_id"], nearest_history_budget_distance=hist_dist, nearest_admitted_budget_distance=peer_dist,
                      planned_strong_history=plan_strong, estimated_min_cash_before_tax=planned_min, preflight_passed=passed)
        budget_results.append(result)
        if passed:
            path = OUTPUT / "预算参数" / (raw["sample_id"] + ".json")
            save(path, raw); save(OUTPUT / "业务合同" / (raw["sample_id"] + ".json"), basis)
            budget_signatures.append(vector); queue.append((raw, path, result))
    save(OUTPUT / "预算前置筛查.json", dict(evaluated=len(budget_results), admitted=len(queue), results=budget_results))
    print(json.dumps(dict(stage="budgets_screened", evaluated=len(budget_results), admitted=len(queue))), flush=True)
    receipts = []; admitted = []
    for raw, path, preflight in queue[:args.central_limit]:
        destination = OUTPUT / "中央验证流水"; directory = destination / raw["sample_id"] / raw["run_id"]
        if not directory.exists():
            generate_from_profile(path, output_root=destination)
        sig = signature(directory)
        history_pairs = [dict(id=h["id"], **compare(sig, h["signature"])) for h in history]
        peer_pairs = [dict(id=a["sample_id"], **compare(sig, a["signature"])) for a in admitted]
        gate = t05_micro_rebound_gate(directory / "account_daily_total.csv")
        manifest = read(directory / "generation_manifest.json")
        strong = [p for p in history_pairs + peer_pairs if p["operating_strong"]]
        passed = gate["passes"] and manifest["execution_status"] == "complete" and not strong
        receipt = dict(sample_id=raw["sample_id"], profile=str(path), directory=str(directory), shape=gate, status=manifest["execution_status"], strong_pairs=strong,
                       legacy_strong_history=sum(p["legacy_strong"] for p in history_pairs), passed=passed)
        receipts.append(receipt)
        if passed:
            admitted.append(dict(sample_id=raw["sample_id"], signature=sig, raw=raw, path=path, directory=directory))
        save(OUTPUT / "回执" / (raw["sample_id"] + ".json"), receipt)
        print(json.dumps(dict(stage="central_checked", sample=raw["sample_id"], passed=passed, shape=gate["passes"], strong=len(strong))), flush=True)
    assert digest(HISTORY) == history_sha
    save(OUTPUT / "能力验收汇总.json", dict(budget_evaluated=len(budget_results), budget_admitted=len(queue), central_calls=len(receipts), admitted=len(admitted),
                                           receipts=receipts, rule=RULE, scope="T05合同整改能力验证；预算候选数不等于正式通过数，尚不能宣称数百独立企业已验收", historical_manifest_unchanged=True))


def run_mechanism_pilot(config_path):
    # 复用同一历史索引、预算比较、中央入口、独立账务核验和图册工具。
    from syn_b1.mechanism_budget import mechanism_budget, causal_evidence
    config = read(config_path)
    output = (ROOT / config["output"]).resolve()
    if not output.is_relative_to(HISTORY.parent.parent.resolve()) or output == OUTPUT.resolve():
        raise ValueError("机制探针必须使用独立生产子目录，不能覆盖上一轮验收")
    count, limit = config["budget_per_mechanism"], config["central_per_mechanism"]
    if not 1 <= count <= 8 or not 1 <= limit <= 2 or len(config["mechanisms"]) > 4:
        raise ValueError("本轮最多4机制、每机制8预算和2次中央调用")
    save(output / "事前冻结.json", dict(config=config, rule=RULE, baseline_manifest_sha=digest(HISTORY)))
    history = history_index()
    # 比较范围包括421历史、24目标及上一轮已通过工程候选；去掉重复身份。
    known_ids = {h["id"] for h in history}
    extra = read(OUTPUT / "24目标正式样本清单.json")["rows"]
    extra += [dict(sample_id=r["sample_id"], formal_directory=r["directory"], profile=r["profile"]) for r in read(OUTPUT / "能力验收汇总.json")["receipts"] if r["passed"]]
    for row in extra:
        if row["sample_id"] not in known_ids:
            directory = ROOT / row["formal_directory"]
            raw = read(ROOT / row["profile"])
            history.append(dict(id=row["sample_id"], signature=signature(directory), budget=budget_vector(raw), directory=directory,
                                daily_sha=digest(directory / "account_daily_total.csv")))
            known_ids.add(row["sample_id"])
    save(output / "比较基线.json", [dict(id=h["id"], daily_sha=h["daily_sha"]) for h in history])
    budgets, receipts, selected_vectors, admitted, atlas_rows = [], [], [], [], []
    for spec in config["mechanisms"]:
        queue = []
        for i in range(count):
            raw, basis = mechanism_budget(spec, i, config)
            counter, _ = mechanism_budget(spec, i, config, counterfactual=True)
            resolve_profile(raw); resolve_profile(counter)
            evidence = causal_evidence(raw, counter)
            vector, planned = budget_vector(raw), planned_signature(raw)
            nearest = min(float(np.abs(vector-h["budget"]).mean()) for h in history if h["budget"] is not None)
            peer = min((float(np.abs(vector-v).mean()) for v in selected_vectors), default=1.0)
            minimum = float(config["initial_capital_cny"]) + float(planned["balance"].min())
            reasons = []
            if nearest <= config["budget_distance"]: reasons.append("历史或已通过预算近重复")
            if peer <= config["budget_distance"]: reasons.append("本轮预算近重复")
            if minimum <= 180000: reasons.append("税前资金缓冲不足")
            if planned["balance"][-1] >= -200000: reasons.append("未形成T05所需净下降")
            sid = raw["sample_id"]
            path = output / "预算参数" / f"{sid}.json"
            # 拒绝预算同样留存，不藏掉失败后重新抽样。
            save(path, raw); save(output / "经营依据" / f"{sid}.json", basis)
            record = dict(sample_id=sid, mechanism_id=spec["id"], nearest_baseline_budget_distance=nearest,
                          nearest_peer_budget_distance=peer, counterfactual=evidence, rejection_reasons=reasons)
            budgets.append(record)
            save(output / "预算回执" / f"{sid}.json", record)
            if not reasons:
                selected_vectors.append(vector); queue.append((raw, path, basis))
        for raw, path, basis in queue[:limit]:
            sid = raw["sample_id"]
            directory = output / "中央流水" / sid / raw["run_id"]
            if not directory.exists():
                generate_from_profile(path, output_root=output / "中央流水")
            complete = read(directory / "generation_manifest.json")["execution_status"] == "complete"
            ledger = audit_export(directory, raw) if complete else dict(complete=False, reason="中央真实停止，保留原件")
            sig = signature(directory)
            pairs = [dict(id=h["id"], **compare(sig, h["signature"])) for h in history + admitted]
            gate = t05_micro_rebound_gate(directory / "account_daily_total.csv")
            strong = [p for p in pairs if p["operating_strong"]]
            passed = complete and gate["passes"] and not strong
            receipt = dict(sample_id=sid, mechanism_id=spec["id"], family_id=basis["family_id"], profile=str(path),
                           directory=str(directory), complete=complete, ledger=ledger, shape=gate, strong_pairs=strong,
                           legacy_strong_count=sum(p["legacy_strong"] for p in pairs), passed=passed, formal_admission=False)
            save(output / "流水回执" / f"{sid}.json", receipt); receipts.append(receipt)
            if passed: admitted.append(dict(id=sid, signature=sig))
            atlas_rows.append(dict(sample_id=sid, run_id=raw["run_id"], target_id="T05", purpose="train", product_cn=spec["name"],
                                   daily_csv=str(directory / "account_daily_total.csv"), candidate_status="工程验证通过_未正式准入" if passed else "未通过_保留原因"))
            print(json.dumps(dict(mechanism=spec["id"], sample=sid, complete=complete, passed=passed, strong=len(strong))), flush=True)
    # 最后只查指纹，原有审批和历史流水都没有重跑。
    for h in history:
        assert digest(h["directory"] / "account_daily_total.csv") == h["daily_sha"]
    save(output / "图册清单.json", dict(rows=atlas_rows))
    if atlas_rows and not (output / "余额图册").exists():
        subprocess.run([sys.executable, str(ROOT / "tools/render_candidate_pool_atlas.py"), "--manifest", str(output / "图册清单.json"), "--output", str(output / "余额图册"), "--all-only", "--page-size", "4"], check=True, cwd=ROOT)
    summary = dict(baseline_count=len(history), budgets=len(budgets), budget_admitted=sum(not b["rejection_reasons"] for b in budgets),
                   central_calls=len(receipts), passed=sum(r["passed"] for r in receipts), formal_admissions=0, training_started=False,
                   mechanisms=[dict(id=s["id"], name=s["name"], budget_admitted=sum(b["mechanism_id"] == s["id"] and not b["rejection_reasons"] for b in budgets),
                                    central_calls=sum(r["mechanism_id"] == s["id"] for r in receipts), passed=sum(r["mechanism_id"] == s["id"] and r["passed"] for r in receipts)) for s in config["mechanisms"]],
                   limitation="工程探针不是数百户能力证明；同机制归同一家族，实际机制新颖性仍须与历史业务语义复核", receipts=receipts)
    save(output / "机制验证汇总.json", summary)
    save(output / "源码指纹.json", {f:digest(ROOT / f) for f in ("src/syn_b1/mechanism_budget.py", "tools/shape_contract_search.py", "src/syn_b1/contract_business.py")})
    print(json.dumps({k:v for k,v in summary.items() if k != "receipts"}, ensure_ascii=False), flush=True)


def target_screen(target_id, values):
    # 复用原目标定义与分段诊断；通用诊断不冒充完整形状准入。
    from run_shape_guided_budget_search_20260913 import targets, score
    target = next((t for t in targets() if t["target_id"] == target_id), None)
    if target is None:
        raise ValueError("目标必须属于原T01至T24")
    if len(values) != 24:
        return dict(screen_passed=False, reason="不足24个月", formal_shape_eligible=False)
    result = score(target, values)
    passed = result["direction_hits"] == result["direction_total"] and result["shape_signature"]["major_turn_count"] <= target["major_turn_limit"]
    return dict(target_id=target_id, screen_passed=passed, diagnostics=result, formal_shape_eligible=False,
                remaining_checks="该筛查未完整区分U/V宽深、近平斜率、低中波动及转折时点容差，不自动正式准入")


def single_stage_daily_x(rows: list[dict]) -> np.ndarray:
    """Calendar-day coordinates whose month ends are exactly 0..23."""
    month_end = [index for index, row in enumerate(rows) if index + 1 == len(rows) or rows[index + 1]["calendar_date"][5:7] != row["calendar_date"][5:7]]
    month_lengths = [end - (month_end[i - 1] + 1 if i else 0) + 1 for i, end in enumerate(month_end)]
    return np.asarray([12 * (int(row["calendar_date"][:4]) - 2024) + int(row["calendar_date"][5:7]) - 1
                       - 1 + (int(row["calendar_date"][8:]) - 1) / max(1, month_lengths[12 * (int(row["calendar_date"][:4]) - 2024) + int(row["calendar_date"][5:7]) - 1] - 1)
                       for row in rows], dtype=float)


def single_stage_ols_d_assessment(directory: Path, *, slope_reference_cny_per_month: float,
                                  initial_capital_cny: float, dispersion_reference=(.01, .04),
                                  tolerance: float = .20) -> dict:
    """Audit one no-turn target from actual complete daily observations only."""
    rows = list(csv.DictReader((directory / "account_daily_total.csv").open(encoding="utf-8-sig", newline="")))
    observed_dates = [row["calendar_date"] for row in rows]
    if observed_dates != DAYS:
        return dict(passed=False, complete_observation=False,
                    reason="实际观察期必须恰为2024-01-01至2025-12-31的731完整日；不得把未来或停止后的缺失标签纳入验收")
    balance = np.asarray([float(row["ending_balance_cny"]) for row in rows], dtype=float)
    # The frozen slope convention is the 24 actual month-end balances at
    # x=0..23.  Daily points are not allowed to alter its sampling weights.
    month_end = [index for index, row in enumerate(rows) if index + 1 == len(rows) or rows[index + 1]["calendar_date"][5:7] != row["calendar_date"][5:7]]
    if len(month_end) != 24:
        return dict(passed=False, complete_observation=False, reason="无法取得24个实际月末余额")
    month_balance = balance[month_end]
    month_x = np.arange(24, dtype=float)
    intercept, per_month = np.linalg.lstsq(np.column_stack((np.ones_like(month_x), month_x)), month_balance, rcond=None)[0]
    lower, upper = sorted((slope_reference_cny_per_month * (1 - tolerance), slope_reference_cny_per_month * (1 + tolerance)))
    # D uses the same month-end OLS line, evaluated over the daily position in
    # each calendar month.  It never falls back to a separate daily OLS or a
    # piecewise line through realised month ends.
    daily_x = single_stage_daily_x(rows)
    residual = balance - (intercept + per_month * daily_x)
    d = float(np.quantile(np.abs(residual - np.median(residual)), .90) / initial_capital_cny)
    d_lower, d_upper = (dispersion_reference[0] * (1 - tolerance), dispersion_reference[1] * (1 + tolerance))
    slope_passed, d_passed = bool(lower <= per_month <= upper), bool(d_lower <= d <= d_upper)
    return dict(acceptance_version="single_stage_ols_d_v1", complete_observation=True,
                observation_start=observed_dates[0], observation_end=observed_dates[-1], days=len(rows),
                ols_slope_cny_per_month=round(per_month, 2), slope_reference_cny_per_month=slope_reference_cny_per_month,
                slope_band_cny_per_month=[round(lower, 2), round(upper, 2)], slope_passed=slope_passed,
                D=round(d, 8), dispersion_reference=list(dispersion_reference), D_band=[round(d_lower, 8), round(d_upper, 8)],
                dispersion_passed=d_passed, passed=bool(slope_passed and d_passed),
                method="24_actual_month_end_OLS_x0_to23; D=p90(abs(daily_balance-same_OLS_month_trend-median_residual))/initial_capital")


def run_entity_expansion(config_path):
    # 单一通用调度入口：实体编译与目标筛查分离，未知范围失败关闭。
    from syn_b1.entity_contracts import compile_entity_profile
    from run_shape_guided_budget_search_20260913 import targets
    config = read(config_path)
    if set(config) != {"version", "output", "max_budgets", "max_central_calls", "candidates"} or config["version"] != "entity_expansion_v1":
        raise ValueError("通用扩展配置字段不匹配")
    candidates = config["candidates"]
    if type(config["max_budgets"]) is not int or type(config["max_central_calls"]) is not int or not 0 <= len(candidates) <= config["max_budgets"] <= 64 or not 0 <= config["max_central_calls"] <= 12:
        raise ValueError("单批最多64预算、12次中央调用；允许0次中央作仅前筛运行")
    output = (ROOT / config["output"]).resolve()
    if not output.is_relative_to(HISTORY.parent.parent.resolve()) or output in {HISTORY.parent.parent.resolve(), OUTPUT.resolve()}:
        raise ValueError("必须指定独立的版本子目录")
    if output.exists() and not (output / "输入冻结.json").exists():
        raise ValueError("已有非本协议目录不能作为扩展输出，防止污染历史")
    allowed = {t["target_id"] for t in targets()}
    loaded = []; ids = set()
    allowed_candidate_keys={"target_id", "family_id", "mechanism_id", "profile", "entity_book", "candidate_id", "parent_revision"}
    for item in candidates:
        if not {"target_id", "family_id", "mechanism_id", "profile", "entity_book"}.issubset(item) or not set(item).issubset(allowed_candidate_keys) or item["target_id"] not in allowed or not item["family_id"] or not item["mechanism_id"]:
            raise ValueError("候选缺失已声明目标、业务家族、机制或实体文件")
        if ("candidate_id" in item) != ("parent_revision" in item):
            raise ValueError("失败修订必须同时声明候选身份和父回执")
        if "parent_revision" in item:
            parent=item["parent_revision"]
            if set(parent) != {"sample_id", "receipt", "receipt_sha256"}:
                raise ValueError("父回执字段不完整")
            receipt=ROOT/parent["receipt"]
            if not receipt.exists() or digest(receipt) != parent["receipt_sha256"]:
                raise ValueError("父回执不存在或指纹不匹配")
            parent_row=read(receipt)
            parent_candidate=parent_row.get("candidate_id") or LEGACY_FAILED_CANDIDATE_IDS.get(parent_row.get("sample_id"))
            if (parent_row.get("sample_id") != parent["sample_id"] or parent_row.get("formal_admission") is not False
                    or parent_row.get("status") not in {"generated_pending_full_review", "generated_needs_revision"}
                    or parent_candidate != item["candidate_id"] or parent_row.get("target_id") != item["target_id"]
                    or parent_row.get("family_id") != item["family_id"]):
                raise ValueError("父回执不是同一未准入失败候选")
        raw, book = read(ROOT / item["profile"]), read(ROOT / item["entity_book"])
        if (raw["start_date"], raw["end_date"]) != (DAYS[0], DAYS[-1]):
            raise ValueError("本版通用入口仍限定2024-01至2025-12；其他观察期需另版协议")
        if raw["sample_id"] in ids: raise ValueError("队列样本身份重复")
        ids.add(raw["sample_id"])
        loaded.append((item, raw, book))
    # 输入内容整体冻结，续跑不能偷偷换预算、实体或程序。
    source = {f:digest(ROOT/f) for f in ("src/syn_b1/entity_contracts.py", "tools/shape_contract_search.py", "src/syn_b1/contract_business.py")}
    save(output / "输入冻结.json", dict(config=config, inputs=loaded, source=source, rule=RULE))
    if not loaded:
        save(output / "扩展汇总.json", dict(budgets=0, central_calls=0, formal_admissions=0, status="empty_template_not_executed"))
        return
    # Keep the central-batch actual pool (481 unique records) intact, then
    # only add later registries below.  Do not replace it with the diagnostic
    # 440 subset, which omits some stopped historical records.
    from run_mechanism_matrix_central_batch import actual_baseline
    history = actual_baseline()
    for item in history:
        if item.get("directory"):
            profile_path = item["directory"] / "scenario_profile.json"
            item["budget"] = budget_vector(read(profile_path)["input_profile"]) if profile_path.exists() else None
    diagnostic_population = ROOT / "data/diagnostics/accepted_family_holdout_20260913_v3/population.json"
    if diagnostic_population.exists():
        known_ids = {item["id"] for item in history}
        known_dirs = {str(item["directory"].resolve()) for item in history if item.get("directory")}
        for row in read(diagnostic_population)["included"]:
            directory = Path(row["directory"])
            if row["sample_id"] not in known_ids and str(directory.resolve()) not in known_dirs:
                history.append(dict(id=row["sample_id"], directory=directory, signature=signature(directory),
                                    budget=budget_vector(read(directory / "scenario_profile.json")["input_profile"])))
                known_ids.add(row["sample_id"]); known_dirs.add(str(directory.resolve()))
    # 既有目标、工程通过候选也进入累计基线，不只比较最早421户。
    extra = read(OUTPUT / "24目标正式样本清单.json")["rows"]
    extra += [dict(sample_id=r["sample_id"], formal_directory=r["directory"], profile=r["profile"]) for r in read(OUTPUT / "能力验收汇总.json")["receipts"] if r["passed"]]
    pilot = OUTPUT.parent / "四类经营机制验证_20260913_v1/机制验证汇总.json"
    if pilot.exists():
        extra += [dict(sample_id=r["sample_id"], formal_directory=r["directory"], profile=r["profile"]) for r in read(pilot)["receipts"] if r["passed"]]
    known = {h["id"] for h in history}
    for row in extra:
        if row["sample_id"] not in known:
            directory = ROOT / row["formal_directory"]
            history.append(dict(id=row["sample_id"], signature=signature(directory), budget=budget_vector(read(ROOT/row["profile"]))))
            known.add(row["sample_id"])
    # 后续批次自动读取此前本入口的排重登记；登记不代表正式准入。
    registry = OUTPUT.parent / "通用实体扩展排重登记"
    batch_key = hashlib.sha256(str(output).encode("utf-8")).hexdigest()
    if registry.exists():
        for registered in sorted(registry.glob("*.json")):
            if registered.stem == batch_key: continue
            for row in read(registered)["rows"]:
                if digest(Path(row["daily_csv"])) != row["daily_sha"] or digest(Path(row["profile"])) != row["profile_sha"]:
                    raise ValueError("已登记扩展输入或流水发生变化，不能复用旧排重结论")
                if row["sample_id"] not in known:
                    history.append(dict(id=row["sample_id"], candidate_id=row.get("candidate_id"),
                                        signature=signature(Path(row["daily_csv"]).parent), budget=budget_vector(read(row["profile"]))))
                    known.add(row["sample_id"])
    # Persisted registry parent links are revalidated before being used for a
    # recursive exemption; family/candidate equality alone never creates one.
    recorded_parents = {}
    if registry.exists():
        for registered in registry.glob("*.json"):
            for entry in read(registered).get("rows", []):
                recorded_parents[entry["sample_id"]] = entry.get("parent_revision")
    def failed_ancestor_chain(sample_id, candidate_id):
        chain = set()
        while sample_id:
            parent = recorded_parents.get(sample_id)
            if not parent or set(parent) != {"sample_id", "receipt", "receipt_sha256"}:
                break
            receipt = ROOT / parent["receipt"]
            if not receipt.exists() or digest(receipt) != parent["receipt_sha256"]:
                break
            parent_row = read(receipt)
            if not valid_failed_parent(parent_row, candidate_id, parent_row.get("target_id"), parent_row.get("family_id")):
                break
            sample_id = parent["sample_id"]; chain.add(sample_id)
        return chain
    lineage_by_id={row["id"]: row.get("candidate_id") or LEGACY_FAILED_CANDIDATE_IDS.get(row["id"]) for row in history}
    vectors, planned_peers, actual_peers, records, charts = [], [], [], [], []
    calls = 0
    for item, raw, book in loaded:
        sid = raw["sample_id"]
        checkpoint = output / "回执" / f"{sid}.json"
        # 已完成回执作为运行缓存；冻结输入确保不能以续跑改种子。
        if checkpoint.exists():
            row = read(checkpoint); records.append(row)
            compiled = read(output / "编译参数" / f"{sid}.json") if row.get("compiled") else None
        else:
            try:
                compiled, lineage = compile_entity_profile(raw, book)
            except (ValueError, KeyError, TypeError) as exc:
                row = dict(sample_id=sid, target_id=item["target_id"], compiled=False, status="entity_validation_failed", reason=str(exc), formal_admission=False)
                save(checkpoint, row); records.append(row); continue
            path = output / "编译参数" / f"{sid}.json"
            save(path, compiled); save(output / "实体追溯" / f"{sid}.json", lineage)
            vector, planned = budget_vector(compiled), planned_signature(compiled)
            reasons = []
            parent_id=item.get("parent_revision", {}).get("sample_id")
            candidate_id=item.get("candidate_id")
            # Only hash-verified failed ancestry is exempt.  The two legacy
            # IDs are an explicit migration bridge, not a family exemption;
            # anything later registered with the same candidate_id remains in
            # the comparison baseline.
            verified_ancestors = ({parent_id} | failed_ancestor_chain(parent_id, candidate_id) | {sid for sid, cid in LEGACY_FAILED_CANDIDATE_IDS.items() if cid == candidate_id}) if candidate_id else set()
            def same_candidate(sample_id): return sample_id in verified_ancestors
            distance = min((float(np.abs(vector-h["budget"]).mean()) for h in history if h["budget"] is not None and not same_candidate(h["id"])), default=0)
            if distance <= RULE["monthly_budget_distance"]: reasons.append("baseline_budget_near_copy")
            if any(float(np.abs(vector-v).mean()) <= RULE["monthly_budget_distance"] for v in vectors): reasons.append("peer_budget_near_copy")
            matches = [h["id"] for h in history + planned_peers if compare(planned, h["signature"])["operating_strong"]]
            same_revision=[match for match in matches if same_candidate(match)]
            external=[match for match in matches if not same_candidate(match)]
            if external: reasons.append("planned_contract_rhythm_strong")
            row = dict(sample_id=sid, candidate_id=candidate_id, target_id=item["target_id"], family_id=item["family_id"], mechanism_id=item["mechanism_id"], compiled=True,
                       preflight_reasons=reasons, rhythm_matches=external, same_candidate_revision_matches=same_revision, nearest_budget_distance=distance, formal_admission=False,
                       status="preflight_rejected" if reasons else "preflight_passed")
            if not reasons and calls < config["max_central_calls"]:
                directory = output / "中央流水" / sid / compiled["run_id"]
                # 调用前留记录；已有残缺目录不重新生成覆盖。
                save(output / "调用记录" / f"{sid}.json", dict(profile_sha=digest(path), directory=str(directory)))
                if not directory.exists(): generate_from_profile(path, output_root=output / "中央流水")
                manifest = read(directory / "generation_manifest.json")
                complete = manifest["execution_status"] == "complete"
                row.update(directory=str(directory), complete=complete, shape=target_screen(item["target_id"], monthly_balances(directory / "account_daily_total.csv")))
                if item["target_id"] == "T05" and complete:
                    row["single_stage_acceptance"] = single_stage_ols_d_assessment(
                        directory, slope_reference_cny_per_month=-70000,
                        initial_capital_cny=float(raw["initial_capital"]["amount_cny"]))
                if complete: row["ledger"] = audit_export(directory, compiled)
                sig = signature(directory)
                actual_matches=[h["id"] for h in history + actual_peers if compare(sig, h["signature"])["operating_strong"]]
                row["same_candidate_actual_revision_matches"]=[match for match in actual_matches if same_candidate(match)]
                row["actual_strong_matches"]=[match for match in actual_matches if not same_candidate(match)]
                row["status"] = "generated_pending_full_review" if row.get("single_stage_acceptance", {"passed": True})["passed"] else "generated_needs_revision"
            elif not reasons:
                row["status"] = "central_budget_exhausted"
            save(checkpoint, row); records.append(row)
        if compiled and not row.get("preflight_reasons"):
            vectors.append(budget_vector(compiled)); planned_peers.append(dict(id=sid, signature=planned_signature(compiled)))
        if row.get("directory"):
            calls += 1
            actual_peers.append(dict(id=sid, signature=signature(Path(row["directory"]))))
            charts.append(dict(sample_id=sid, target_id=item["target_id"], purpose="train", product_cn=item["mechanism_id"], daily_csv=str(Path(row["directory"]) / "account_daily_total.csv")))
    save(output / "绘图清单.json", dict(rows=charts))
    if charts and not (output / "余额图册").exists():
        subprocess.run([sys.executable, str(ROOT/"tools/render_candidate_pool_atlas.py"), "--manifest", str(output/"绘图清单.json"), "--output", str(output/"余额图册"), "--all-only", "--page-size", "4"], check=True, cwd=ROOT)
    save(output / "扩展汇总.json", dict(budgets=len(records), central_calls=calls, baseline_count=len(history), formal_admissions=0,
         training_started=False, target_count=len({i["target_id"] for i,_,_ in loaded}), records=records,
         limitation="通用编译与前筛已接入24目标目录；严格形状验收、训练开发家族划分及所有目标机制覆盖仍需完善，不自动正式准入"))
    save(registry / f"{batch_key}.json", dict(scope="排重索引，不是正式或训练准入", rows=[dict(sample_id=r["sample_id"], candidate_id=r.get("candidate_id"),
         parent_revision=next((item.get("parent_revision") for item, raw, book in loaded if raw["sample_id"] == r["sample_id"]), None),
         family_id=r["family_id"], mechanism_id=r["mechanism_id"], profile=str(output/"编译参数"/f"{r['sample_id']}.json"),
         profile_sha=digest(output/"编译参数"/f"{r['sample_id']}.json"), daily_csv=str(Path(r["directory"])/"account_daily_total.csv"),
         daily_sha=digest(Path(r["directory"])/"account_daily_total.csv")) for r in records if r.get("directory")]))


def test_t05_entities(config_path):
    # 同一入口复用实体编译、实际账务核验和既有波段检测，不能换判定器挑结果。
    from syn_b1.mechanism_budget import replenishment_entity_budget
    from curve_repetition_check import check, POLICY
    config = read(config_path)
    if not 1 <= config["candidates"] <= 4 or not 0 <= config["max_central_calls"] <= 4 or len(config["stability_seeds"]) != 2:
        raise ValueError("本能力试验上限4候选、4次中央、2个预定复现种子")
    output = (ROOT / config["output"]).resolve()
    if not output.is_relative_to(HISTORY.parent.parent.resolve()) or output == HISTORY.parent.parent.resolve():
        raise ValueError("验收结果必须在独立生产子目录")
    save(output/"验收事前冻结.json", dict(config=config, repetition_policy=POLICY, similarity_policy=RULE))
    queue = []
    for i in range(config["candidates"]):
        raw, book, basis = replenishment_entity_budget(i, config["seed_base"])
        sid = raw["sample_id"]
        path, entities = output/"基础参数"/f"{sid}.json", output/"实体合同"/f"{sid}.json"
        save(path, raw); save(entities, book); save(output/"业务解释"/f"{sid}.json", basis)
        queue.append(dict(target_id="T05", family_id=basis["family_id"], mechanism_id="installed_base_consumption_replenishment", profile=str(path), entity_book=str(entities)))
    run_config = output/"通用队列配置.json"
    save(run_config, dict(version="entity_expansion_v1", output=str(output/"通用队列"), max_budgets=config["candidates"], max_central_calls=config["max_central_calls"], candidates=queue))
    run_entity_expansion(run_config)
    result = read(output/"通用队列/扩展汇总.json")
    audited = []

    def assess(directory):
        with (directory/"account_daily_total.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        wave = check(rows)
        micro = t05_micro_rebound_gate(directory/"account_daily_total.csv")
        monthly = monthly_balances(directory/"account_daily_total.csv")
        mean = float(np.mean(np.diff(monthly))) if len(monthly) > 1 else 0.0
        residual_ranges = []
        for month in MONTHS:
            values = np.array([float(r["ending_balance_cny"]) for r in rows if r["calendar_date"].startswith(month)])
            if len(values) > 1:
                residual_ranges.append(float(np.ptp(values - np.linspace(values[0], values[-1], len(values)))))
        low = max(residual_ranges, default=float("inf")) / 1800000
        slope_ok = config["monthly_slope_cny_range"][0] <= mean <= config["monthly_slope_cny_range"][1]
        passed = wave["passed"] and wave["sufficient_observation"] and micro["passes"] and slope_ok and low <= config["max_monthly_detrended_range_capital_share"]
        return dict(repetition=wave, t05_micro=micro, mean_monthly_change_cny=mean, slope_passed=slope_ok, maximum_monthly_residual_capital_share=low, shape_and_repetition_passed=passed)

    selected = None
    for row in result["records"]:
        receipt = dict(sample_id=row["sample_id"], generation_status=row["status"], accepted=False)
        if row.get("directory"):
            directory = Path(row["directory"])
            receipt.update(assess(directory))
            receipt.update(actual_strong_matches=row["actual_strong_matches"], ledger=row.get("ledger"))
            receipt["accepted"] = row["complete"] and receipt["shape_and_repetition_passed"] and not row["actual_strong_matches"]
            if receipt["accepted"] and selected is None: selected = row
        else:
            receipt["preflight_reasons"] = row.get("preflight_reasons", [row.get("reason")])
        save(output/"严格复核"/f"{row['sample_id']}.json", receipt); audited.append(receipt)
    stability = []
    if selected:
        raw = read(output/"通用队列/编译参数"/f"{selected['sample_id']}.json")
        for seed in config["stability_seeds"]:
            frozen = deepcopy(raw); frozen["random_seed"] = seed
            path = output/"复现参数"/f"{seed}.json"; save(path, frozen)
            destination = output/"固定种子"/str(seed)
            directory = destination/raw["sample_id"]/raw["run_id"]
            if not directory.exists(): generate_from_profile(path, output_root=destination)
            ledger = audit_export(directory, frozen)
            same = np.array_equal(signature(directory)["balance"], signature(Path(selected["directory"]))["balance"])
            stability.append(dict(seed=seed, ledger=ledger, same_daily_cash=same, **assess(directory)))
    passed = selected is not None and len(stability) == 2 and all(r["same_daily_cash"] and r["shape_and_repetition_passed"] for r in stability)
    save(output/"T05能力验收结论.json", dict(passed=passed, central_candidate_calls=result["central_calls"], stability_calls=len(stability),
         selected=selected["sample_id"] if passed else None, candidates=audited, stability=stability, formal_admissions=0, training_started=False,
         conclusion="本轮数值与复现门通过，仍须核对业务与训练特征隔离后正式登记" if passed else "本轮未证明可直接作为样本；保留失败证据，不能宣称生成器整体不具备能力"))
    print(json.dumps(dict(passed=passed, central_calls=result["central_calls"], stability_calls=len(stability), audited=len(audited)), ensure_ascii=False))


def controlled_budget_change(raw, *, fee_increment_fen=0, price_increase_percent=Decimal(0)):
    # 一次只改一个经济控制变量，保持身份以外的日期、种子和其他输入不变。
    if bool(fee_increment_fen) == bool(price_increase_percent):
        raise ValueError("响应试验必须且只能启用一种变量")
    changed = deepcopy(raw)
    plan = changed["operating"]["business_nodes"]
    for node in plan["nodes"]:
        if fee_increment_fen:
            node["other_expense_cny"] = f"{Decimal(node['other_expense_cny']) + Decimal(fee_increment_fen)/100:.2f}"
        else:
            node["unit_price_cny"] = f"{(Decimal(node['unit_price_cny']) * (1 + price_increase_percent/100)).quantize(Decimal('.01')):.2f}"
    # 显式合同的来源总额同步修改，账期和事件编号绝不改变。
    if "settlements" in plan:
        category = "other_expense_payment" if fee_increment_fen else "collection"
        for node in plan["nodes"]:
            events = [e for e in plan["settlements"] if e["category"] == category and e["recognition_month"] == node["month"]]
            total = sum(e["amount_fen"] for e in events)
            target = int(Decimal(node["other_expense_cny"])*100) if fee_increment_fen else node["sales_units"] * int(Decimal(node["unit_price_cny"])*100)
            if total <= 0: raise ValueError("控制变量缺少对应合同来源")
            assigned = 0
            for i, event in enumerate(events):
                value = target - assigned if i == len(events)-1 else target * event["amount_fen"] // total
                event["amount_fen"] = value; assigned += value
    resolve_profile(changed)
    return changed


def verify_other_cash_mapping(truth):
    # 中央把其他运营支出标为other_declared_event，来源后缀other才对应本检查。
    paid = defaultdict(int)
    for event in truth["transaction_plan"]["candidates"]:
        if event["category"] == "other_declared_event" and event["source_item_id"].endswith(":other"):
            paid[event["booking_datetime"][:7]] += event["amount_fen"]
    return all(paid[row["month"]] == row["other_operating_cash_payment_fen"] for row in truth["operating_cycle"]["monthly_rows"])


def test_target24_response(config_path):
    # 原24户只读作参考；中央调用仅生成已声明的新增工程探针。
    from curve_repetition_check import check, POLICY
    config = read(config_path)
    if config["max_response_calls"] != 24:
        raise ValueError("本次明确冻结为24个工程响应探针")
    output = (ROOT/config["output"]).resolve()
    if not output.is_relative_to(OUTPUT.parent.resolve()) or output in {OUTPUT.resolve(), OUTPUT.parent.resolve()}:
        raise ValueError("必须使用独立结果子目录")
    parents = read(OUTPUT/"24目标正式样本清单.json")["rows"]
    if {r["target_id"] for r in parents} != {f"T{i:02d}" for i in range(1,25)}:
        raise ValueError("原24目标参考清单不完整")
    loaded = [(r, read(ROOT/r["profile"])) for r in parents]
    protection = {str(ROOT/r["formal_directory"]): digest(ROOT/r["formal_directory"]/"account_daily_total.csv") for r in parents}
    save(output/"事前冻结.json", dict(config=config, input_profiles=loaded, parent_daily_hashes=protection, repetition_policy=POLICY, similarity_rule=RULE))
    fee = int(Decimal(config["monthly_service_fee_increment_cny"])*100)
    results, charts = [], []
    for parent, raw in loaded:
        tid = parent["target_id"]
        path = output/"响应预算"/f"{tid}.json"
        probe = controlled_budget_change(raw, fee_increment_fen=fee)
        probe.update(sample_id=f"response24_{tid.lower()}_fee", run_id=config["version"])
        save(path, probe)
        checkpoint = output/"回执"/f"{tid}.json"
        if checkpoint.exists():
            result = read(checkpoint)
        else:
            destination = output/"工程流水"
            directory = destination/probe["sample_id"]/probe["run_id"]
            if not directory.exists(): generate_from_profile(path, output_root=destination)
            manifest = read(directory/"generation_manifest.json")
            truth = read(directory/"_restricted/scenario_truth.json")
            actual_nodes = truth["operating_cycle"]["monthly_rows"]
            input_nodes = probe["operating"]["business_nodes"]["nodes"]
            matched = len(actual_nodes) == 24 and all(row["other_operating_expense_fen"] == int(Decimal(node["other_expense_cny"])*100) for row,node in zip(actual_nodes,input_nodes))
            # 各月实际支付与中央经营模块到期金额独立对齐，不能只看输入映射。
            cash_mapped = verify_other_cash_mapping(truth)
            with (directory/"transactions_total.csv").open(encoding="utf-8-sig", newline="") as handle: transactions = list(csv.DictReader(handle))
            cash = 0; ledger_ok = True
            for tx in transactions:
                cash += int(Decimal(tx["credit_cny"])*100) - int(Decimal(tx["debit_cny"])*100)
                ledger_ok &= cash == int(Decimal(tx["post_transaction_balance_cny"])*100) and cash >= 0
            with (directory/"account_daily_total.csv").open(encoding="utf-8-sig", newline="") as handle: days = list(csv.DictReader(handle))
            quality = read(directory/"quality_report.json")
            ledger_ok &= quality["future_training_data_quality_ready"] and quality["checks"]["final_cash_reconciled"]
            shape = target_screen(tid, monthly_balances(directory/"account_daily_total.csv"))
            if tid == "T05": shape["screen_passed"] = t05_micro_rebound_gate(directory/"account_daily_total.csv")["passes"]
            repetition = check(days)
            parent_pair = compare(signature(directory), signature(ROOT/parent["formal_directory"]))
            # 单纯提价的另一个预算用于负对照，近重复者不调用中央凑数量。
            price_probe = controlled_budget_change(raw, price_increase_percent=Decimal(config["price_increase_percent"]))
            save(output/"提价负对照预算"/f"{tid}.json", price_probe)
            budget_distance = float(np.abs(budget_vector(price_probe)-budget_vector(raw)).mean())
            result = dict(target_id=tid, sample_id=probe["sample_id"], family_id=parent["sample_id"], directory=str(directory),
                          complete=manifest["execution_status"] == "complete", monthly_fee_mapping=matched, cash_payment_mapping=cash_mapped,
                          ledger_reconciled=bool(ledger_ok), transaction_count=len(transactions), shape=shape, repetition=repetition,
                          similarity_to_parent=parent_pair, price_only_budget_distance=budget_distance,
                          price_only_negative_control_rejected=budget_distance <= RULE["monthly_budget_distance"],
                          formal_admission=False, scope="响应正对照与近重复负对照；不声称独立经营机制或新样本")
            save(checkpoint,result)
        results.append(result)
        charts.append(dict(target_id=tid,sample_id=probe["sample_id"],purpose="train",product_cn=f"{tid}费用响应探针_非新样本",daily_csv=str(Path(result["directory"])/"account_daily_total.csv")))
        print(json.dumps(dict(target=tid,complete=result["complete"],mapped=result["monthly_fee_mapping"],cash_mapped=result["cash_payment_mapping"],wave=result["repetition"]["passed"])),flush=True)
    # 对首版分类名称错误只重做只读核对，旧回执保留，不重新生成24户。
    corrections = []
    for result in results:
        truth = read(Path(result["directory"])/"_restricted/scenario_truth.json")
        corrected = verify_other_cash_mapping(truth)
        corrections.append(dict(target_id=result["target_id"], original_cash_payment_mapping=result["cash_payment_mapping"],
                                corrected_cash_payment_mapping=corrected, reason="按中央实际分类other_declared_event及来源:other核对，非改流水"))
        result["cash_payment_mapping"] = corrected
    save(output/"支付分类核对纠正.json", corrections)
    for directory, expected in protection.items(): assert digest(Path(directory)/"account_daily_total.csv") == expected
    save(output/"绘图清单.json",dict(rows=charts))
    if not (output/"余额图册").exists():
        subprocess.run([sys.executable,str(ROOT/"tools/render_candidate_pool_atlas.py"),"--manifest",str(output/"绘图清单.json"),"--output",str(output/"余额图册"),"--all-only","--page-size","4"],check=True,cwd=ROOT)
    summary = dict(targets=24,central_response_calls=len(results),negative_control_budgets=len(results),formal_admissions=0,
                   response_passed=sum(r["monthly_fee_mapping"] and r["cash_payment_mapping"] and r["ledger_reconciled"] and r["complete"] for r in results),
                   shape_screen_passed=sum(r["shape"]["screen_passed"] for r in results),no_repetition=sum(r["repetition"]["passed"] for r in results),
                   strong_to_parent=sum(r["similarity_to_parent"]["operating_strong"] for r in results),price_only_rejected=sum(r["price_only_negative_control_rejected"] for r in results),
                   rows=results,training_started=False,conclusion="验证单一费用变量在24目标上的传导；不能据此认定独立多样化扩展已成功")
    save(output/"24目标响应汇总_复核版.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "rows"},ensure_ascii=False))


def _daily_balance_values(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    return (np.arange(len(rows), dtype=float), np.asarray([float(row["ending_balance_cny"]) for row in rows], dtype=float))


def _dispersion(directory: Path, initial_capital_cny: float) -> dict:
    """D is a diagnostic of real daily balances, never an input to the ledger."""
    days, balances = _daily_balance_values(directory / "account_daily_total.csv")
    monthly = monthly_balances(directory / "account_daily_total.csv")
    # The actual month-end points define the piecewise-free reference line.  This
    # deliberately retains settlement spikes in the residual instead of hiding
    # them behind a smoothed balance curve.
    month_rows = list(csv.DictReader((directory / "account_daily_total.csv").open(encoding="utf-8-sig", newline="")))
    end_x = [i for i, row in enumerate(month_rows) if row["calendar_date"][8:] in {"28", "29", "30", "31"} and (i + 1 == len(month_rows) or month_rows[i + 1]["calendar_date"][5:7] != row["calendar_date"][5:7])]
    if len(end_x) != 24 or len(monthly) != 24:
        return {"passed": False, "reason": "月末观测不足24个月"}
    trend = np.interp(days, [0] + end_x, [balances[0]] + monthly)
    residual = balances - trend
    centred = residual - np.median(residual)
    p90 = float(np.quantile(np.abs(centred), .90))
    return {"D": p90 / initial_capital_cny, "p90_residual_cny": round(p90, 2),
            "max_abs_residual_cny": round(float(np.max(np.abs(centred))), 2)}


def _target_assessment(target: dict, directory: Path, initial_capital_cny: float) -> dict:
    """One frozen, deliberately moderate acceptance rule for all 24 targets."""
    from curve_repetition_check import check
    values = monthly_balances(directory / "account_daily_total.csv")
    basic = target_screen(target["target_id"], values)
    segments = basic.get("diagnostics", {}).get("segments", [])
    amplitude = []
    for row in segments:
        expected, actual = row["expected_slope_cny"], row["actual_mean_slope_cny"]
        if target["target_id"] in {"T03", "T04"}:
            # C0 is 360万元 here, so the documented 0.6%/month tolerance is 2.16万元。
            within = abs(actual) <= initial_capital_cny * .006
        else:
            within = abs(abs(actual) - abs(expected)) <= abs(expected) * .20
        amplitude.append(dict(**row, amplitude_within_20pct=within))
    directions = all(row["direction_match"] for row in amplitude)
    amplitudes = all(row["amplitude_within_20pct"] for row in amplitude)
    # Raw month-to-month signs count normal settlement noise as a turn.  An
    # extra major turn must oppose its declared stage for three whole months and
    # move at least 35% of the declared monthly magnitude.
    changes = np.diff(np.asarray(values, dtype=float))
    cuts = [0] + target["breakpoint_months"] + [24]
    sustained_extra = []
    for index, expected in enumerate(target["monthly_net_cash_slope_cny"]):
        part = changes[cuts[index]:cuts[index + 1] - 1]
        threshold = max(15000.0, abs(expected) * .35)
        opposing = [value * (1 if expected >= 0 else -1) < -threshold for value in part]
        if any(all(opposing[start:start + 3]) for start in range(max(0, len(opposing) - 2))):
            sustained_extra.append(index + 1)
    # A near-platform target is accepted on its absolute platform band.  A
    # three-month settlement drift inside that band is explicitly a local
    # fluctuation, not a new main V/U turning point.
    turns = True if target["target_id"] in {"T03", "T04"} else not sustained_extra
    d = _dispersion(directory, initial_capital_cny)
    low = target["volatility"] == "低"
    d_pass = d.get("D") is not None and ((d["D"] <= .048) if low else (.032 <= d["D"] <= .12))
    rows = list(csv.DictReader((directory / "account_daily_total.csv").open(encoding="utf-8-sig", newline="")))
    repetition = check(rows)
    return dict(monthly_balances_cny=values, segments=amplitude, directions_pass=directions,
                amplitude_pass=amplitudes, raw_monthly_sign_turns=basic["diagnostics"]["shape_signature"]["major_turn_count"],
                sustained_extra_turn_stages=sustained_extra,
                major_turns_pass=turns, dispersion=d, dispersion_pass=d_pass, repetition=repetition,
                shape_pass=directions and amplitudes and turns)


def _ledger_assessment(directory: Path) -> dict:
    manifest = read(directory / "generation_manifest.json")
    quality = read(directory / "quality_report.json")
    cash = 0; rows = 0; exact = True
    with (directory / "transactions_total.csv").open(encoding="utf-8-sig", newline="") as handle:
        for tx in csv.DictReader(handle):
            rows += 1
            cash += int(Decimal(tx["credit_cny"]) * 100) - int(Decimal(tx["debit_cny"]) * 100)
            exact &= cash == int(Decimal(tx["post_transaction_balance_cny"]) * 100) and cash >= 0
    return dict(complete=manifest["execution_status"] == "complete", transaction_count=rows,
                running_balance_exact=bool(exact), quality_ready=bool(quality["future_training_data_quality_ready"]),
                final_cash_reconciled=bool(quality["checks"]["final_cash_reconciled"]),
                passed=bool(exact and manifest["execution_status"] == "complete" and quality["future_training_data_quality_ready"] and quality["checks"]["final_cash_reconciled"]))


def _adjust_stage_orders(previous: tuple[float, ...], assessment: dict) -> tuple[float, ...]:
    """The sole retry control: customer-contract order volume per declared stage."""
    result = []
    for multiplier, segment in zip(previous, assessment["segments"]):
        expected, actual = segment["expected_slope_cny"], segment["actual_mean_slope_cny"]
        if expected == 0:
            change = -actual / 180000.0
        else:
            # Empirical contract-cash sensitivity is deliberately damped: a
            # small order-volume correction should not jump across the target
            # because contract units are integral and settlement is lumpy.
            change = (expected - actual) / 200000.0
        change = min(.12, max(-.12, change))
        result.append(round(min(1.45, max(.65, multiplier * (1 + change))), 4))
    return tuple(result)


def _load_cumulative_baseline() -> list[dict]:
    base = history_index()
    known = {str(row["directory"].resolve()) for row in base}
    manifest = read(OUTPUT / "24目标正式样本清单.json")
    for row in manifest["rows"]:
        directory = (ROOT / row["formal_directory"]).resolve()
        if str(directory) not in known:
            envelope = read(directory / "scenario_profile.json")
            base.append(dict(id=row["sample_id"], directory=directory, signature=signature(directory),
                             budget=budget_vector(envelope["input_profile"])))
            known.add(str(directory))
    return base


def _make_stability_profile(raw: dict, seed: int) -> dict:
    replica = deepcopy(raw)
    replica["sample_id"] = f"{raw['sample_id']}_stability_{seed}"
    replica["run_id"] = f"{raw['run_id']}_stability_{seed}"
    replica["random_seed"] = seed
    return replica


def run_autonomous_target_expansion(config_path: Path):
    """Execute the v2 controlled cycle without changing accepted predecessors.

    A retry records the failed immutable candidate and changes only declared
    stage order volume.  Names, dates, prices, costs and seeds are held fixed.
    """
    from syn_b1.autonomous_target_factory import build_target_profile
    from run_shape_guided_budget_search_20260913 import targets
    config = read(config_path)
    required = {"version", "output", "seed_base", "max_revisions", "stability_seeds", "excluded_existing_targets", "tolerance"}
    optional = {"target_filter", "carry_forward_acceptance"}
    if not required.issubset(config) or not set(config).issubset(required | optional) or config["version"] != "autonomous_target_expansion_v2":
        raise ValueError("v2自动扩展配置字段或版本不匹配")
    if config["tolerance"] != .20 or len(config["stability_seeds"]) != 2 or not 1 <= config["max_revisions"] <= 5:
        raise ValueError("本协议固定20%形状容差、两个复现种子、每目标最多5个版本")
    output = (ROOT / config["output"]).resolve()
    if not output.is_relative_to(HISTORY.parent.parent.resolve()) or output == OUTPUT.resolve():
        raise ValueError("输出必须是合同流水整改目录下的独立版本子目录")
    all_targets = targets(); by_id = {t["target_id"]: t for t in all_targets}
    if not set(config["excluded_existing_targets"]).issubset(by_id):
        raise ValueError("排除目标必须来自原24目标")
    source_files = ["src/syn_b1/autonomous_target_factory.py", "tools/shape_contract_search.py", "src/syn_b1/contract_business.py", "docs/24目标受控扩展方案书_20260913_v2.md", "docs/Terra自主循环执行说明书_20260913_v2.md"]
    frozen = dict(config=config, rule=RULE, source={f: digest(ROOT / f) for f in source_files},
                  note="20%仅用于回归形状、阶段幅度和离散度；账务、合同、重复和相似度不放宽。")
    save(output / "运行冻结.json", frozen)
    baseline = _load_cumulative_baseline()
    accepted = []
    if config.get("carry_forward_acceptance"):
        carried = read(ROOT / config["carry_forward_acceptance"])["rows"]
        for row in carried:
            if row.get("status") == "formal_train_eligible_not_trained":
                raw = read(Path(row["profile"]))
                baseline.append(dict(id=row["sample_id"], directory=Path(row["directory"]), signature=signature(Path(row["directory"])), budget=budget_vector(raw)))
                accepted.append(row)
            elif row.get("status") == "existing_user_accepted_not_regenerated":
                accepted.append(row)
    selected_ids = config.get("target_filter", [t["target_id"] for t in all_targets])
    if not selected_ids or not set(selected_ids).issubset(by_id):
        raise ValueError("target_filter必须是非空的原24目标子集")
    for target in (by_id[target_id] for target_id in selected_ids):
        target_id = target["target_id"]
        if target_id in config["excluded_existing_targets"]:
            accepted.append(dict(target_id=target_id, status="existing_user_accepted_not_regenerated", new_generation=False,
                                 reason="用户认可T05及既有代表保持原审批，不因扩展任务重做"))
            continue
        registered_path = output / "正式登记候选" / f"{target_id}.json"
        if registered_path.exists():
            final = read(registered_path)
            accepted.append(final)
            if final.get("status") == "formal_train_eligible_not_trained":
                raw = read(Path(final["profile"]))
                baseline.append(dict(id=final["sample_id"], directory=Path(final["directory"]),
                                     signature=signature(Path(final["directory"])), budget=budget_vector(raw)))
            continue
        adjustments = tuple(1.0 for _ in target["monthly_net_cash_slope_cny"])
        attempts = []
        chosen = None
        for revision in range(config["max_revisions"]):
            receipt_path = output / "回执" / target_id / f"r{revision:02d}.json"
            # Safe restart: immutable receipts are evidence, not cache files to
            # overwrite.  Carry their single-variable calibration forward.
            if receipt_path.exists():
                prior = read(receipt_path); attempts.append(prior)
                if prior["status"] == "passed_pending_stability":
                    primary_raw = read(Path(prior["profile"]))
                    chosen = (primary_raw, read(Path(prior["entity_contract"])), prior, signature(Path(prior["directory"])))
                    break
                if prior.get("assessment"):
                    adjustments = _adjust_stage_orders(adjustments, prior["assessment"])
                continue
            raw, basis = build_target_profile(target, revision, seed_base=config["seed_base"], stage_order_adjustment=adjustments)
            raw["sample_id"] = f"v2_{target_id.lower()}_contract_r{revision:02d}"
            raw["run_id"] = "autonomous_target_expansion_v2"
            profile_path = output / "预算" / target_id / f"r{revision:02d}.json"
            basis_path = output / "实体合同" / target_id / f"r{revision:02d}.json"
            save(profile_path, raw); save(basis_path, basis)
            planned = budget_vector(raw)
            distances = [float(np.abs(planned - row["budget"]).mean()) for row in baseline if row.get("budget") is not None]
            preflight = dict(nearest_budget_distance=min(distances) if distances else None,
                             near_copy_warning=bool(distances and min(distances) <= RULE["monthly_budget_distance"]),
                             business_variable_group="stage_order_volume", stage_order_adjustment=list(adjustments))
            directory = output / "候选流水" / raw["sample_id"] / raw["run_id"]
            try:
                if not directory.exists():
                    generate_from_profile(profile_path, output_root=output / "候选流水")
                ledger = _ledger_assessment(directory)
                assessment = _target_assessment(target, directory, 3600000.0)
                sig = signature(directory)
                pairs = [dict(id=row["id"], **compare(sig, row["signature"])) for row in baseline]
                strong = [pair for pair in pairs if pair["operating_strong"]]
                passed = ledger["passed"] and assessment["shape_pass"] and assessment["dispersion_pass"] and assessment["repetition"]["passed"] and not strong
                reason = []
                if not ledger["passed"]: reason.append("账务或中央生成未通过")
                if not assessment["shape_pass"]: reason.append("主要形态或阶段幅度不在20%容差内")
                if not assessment["dispersion_pass"]: reason.append("离散程度不在预先标签的20%容差范围内")
                if not assessment["repetition"]["passed"]: reason.append("图内重复波段")
                if strong: reason.append("累计基线余额与非融资经营收支双高相似")
                receipt = dict(target_id=target_id, revision=revision, profile=str(profile_path), entity_contract=str(basis_path), directory=str(directory),
                               preflight=preflight, ledger=ledger, assessment=assessment, baseline_double_high=strong,
                               status="passed_pending_stability" if passed else "rejected", rejection_reasons=reason,
                               next_variable_group="none" if passed else "stage_order_volume")
                save(receipt_path, receipt)
                attempts.append(receipt)
                if passed:
                    chosen = (raw, basis, receipt, sig)
                    break
                adjustments = _adjust_stage_orders(adjustments, assessment)
            except Exception as error:
                receipt = dict(target_id=target_id, revision=revision, profile=str(profile_path), entity_contract=str(basis_path),
                               status="technical_or_contract_failure", error_type=type(error).__name__, error=str(error),
                               next_variable_group="stage_order_volume")
                save(receipt_path, receipt)
                attempts.append(receipt)
        if chosen is None:
            accepted.append(dict(target_id=target_id, status="not_accepted_after_bounded_revisions", attempts=len(attempts), rows=attempts))
            continue
        raw, basis, receipt, primary_sig = chosen
        stability = []
        for seed in config["stability_seeds"]:
            replica = _make_stability_profile(raw, seed)
            replica_path = output / "复现预算" / target_id / f"seed_{seed}.json"
            save(replica_path, replica)
            directory = output / "复现流水" / replica["sample_id"] / replica["run_id"]
            if not directory.exists(): generate_from_profile(replica_path, output_root=output / "复现流水")
            ledger = _ledger_assessment(directory); assessment = _target_assessment(target, directory, 3600000.0)
            stability.append(dict(seed=seed, profile=str(replica_path), directory=str(directory), ledger=ledger, assessment=assessment,
                                  balance_correlation_to_primary=compare(signature(directory), primary_sig)["balance_correlation"],
                                  passed=ledger["passed"] and assessment["shape_pass"] and assessment["dispersion_pass"] and assessment["repetition"]["passed"]))
        stable = all(row["passed"] for row in stability)
        final = dict(target_id=target_id, sample_id=raw["sample_id"], family_id=basis["family_id"], mechanism_id=basis["mechanism_id"],
                     product_cn=basis["business_reason"], profile=str(output / "预算" / target_id / f"r{receipt['revision']:02d}.json"),
                     entity_contract=str(output / "实体合同" / target_id / f"r{receipt['revision']:02d}.json"), directory=receipt["directory"],
                     stability=stability, status="formal_train_eligible_not_trained" if stable else "rejected_stability", attempts=len(attempts),
                     no_training=True)
        save(output / "正式登记候选" / f"{target_id}.json", final)
        if stable:
            accepted.append(final)
            baseline.append(dict(id=raw["sample_id"], directory=Path(receipt["directory"]), signature=primary_sig, budget=budget_vector(raw)))
        else:
            accepted.append(dict(target_id=target_id, status="rejected_stability", attempts=len(attempts), stability=stability))
    rows = accepted
    atlas = [dict(target_id=row["target_id"], sample_id=row["sample_id"], purpose="train", product_cn=row["product_cn"], daily_csv=str(Path(row["directory"]) / "account_daily_total.csv"), candidate_status=row["status"])
             for row in rows if row.get("status") == "formal_train_eligible_not_trained"]
    save(output / "实际图册清单.json", dict(rows=atlas))
    if atlas:
        subprocess.run([sys.executable, str(ROOT / "tools/render_candidate_pool_atlas.py"), "--manifest", str(output / "实际图册清单.json"), "--output", str(output / "实际余额图册"), "--all-only", "--page-size", "4"], check=True, cwd=ROOT)
    summary = dict(version=config["version"], targets=24, existing_not_regenerated=config["excluded_existing_targets"],
                   new_formal_eligible=sum(row.get("status") == "formal_train_eligible_not_trained" for row in rows),
                   completed_coverage=sum(row.get("status") in {"formal_train_eligible_not_trained", "existing_user_accepted_not_regenerated"} for row in rows),
                   rows=rows, no_training=True, final_exam_generated=False,
                   conclusion="仅通过全部账务、形态、离散、图内重复、累计相似度和双种子复现者登记；失败保留原预算及原因。")
    save(output / "验收清单.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
