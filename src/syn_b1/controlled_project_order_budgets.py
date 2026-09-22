"""Non-periodic project-order budgets for the controlled T06 search lane.

The module stops at compiled contracts and a *planned* cash diagnostic.  It
does not call the central generator and its target is only a search loss after
the project portfolio has been fixed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
from json import dumps
from typing import Any

import numpy as np

from syn_b1.controlled_business_budgets import MONTHS, _money, _template
from syn_b1.mechanism_event_planner import plan_mechanism


TARGET_ID = "T06"
PROJECT_PRICE_FEN = 520_000
PROJECT_COST_FEN = 286_000
def _project(contract_id: str, customer_id: str, start: int, end: int, credit_days: int,
             milestones: tuple[tuple[int, int, int], ...], purpose: str) -> dict[str, Any]:
    """One signed task with its real work-package milestone quantities."""
    if not milestones or any(not start <= month <= end for month, _, _ in milestones):
        raise ValueError("项目里程碑必须在项目工期内")
    return {"contract_id": contract_id, "customer_id": customer_id,
            "start_month": MONTHS[start], "end_month": MONTHS[end],
            "monthly_units": max(units for _, _, units in milestones), "delivery_day": milestones[0][1],
            "credit_days": credit_days, "project_purpose": purpose,
            "delivery_plan": [dict(month=MONTHS[month], delivery_day=day, units=units)
                              for month, day, units in milestones]}


# The variants change customer/task composition, term and work-package mix;
# they are not translated copies of the same calendar or scaled cash amounts.
PORTFOLIOS: tuple[tuple[dict[str, Any], ...], ...] = (
    (
        _project("retrofit-a", "hospital_group_a", 0, 5, 30, ((0, 9, 31), (5, 12, 8)), "病区设备改造的主设备与验收包"),
        _project("energy-b", "energy_service_b", 1, 9, 45, ((1, 21, 16), (9, 15, 12)), "园区能耗改造的两期交付"),
        _project("lab-c", "laboratory_c", 3, 12, 21, ((3, 14, 16), (12, 10, 10)), "实验室安全系统主控与验收"),
        _project("archive-d", "archive_center_d", 8, 17, 30, ((8, 7, 16), (17, 9, 8)), "档案中心迁移主批与收尾"),
    ),
    (
        _project("metro-e", "metro_operator_e", 0, 7, 45, ((0, 6, 5), (3, 20, 3), (5, 13, 4), (7, 25, 2)), "轨交站点巡检改造"),
        _project("factory-f", "factory_f", 2, 10, 30, ((2, 12, 4), (4, 23, 4), (8, 9, 3), (10, 18, 2)), "产线追溯系统改造"),
        _project("school-g", "school_group_g", 6, 14, 21, ((6, 16, 3), (9, 6, 4), (12, 20, 3), (14, 11, 2)), "学校网络分校区升级"),
        _project("water-h", "water_utility_h", 11, 19, 45, ((11, 8, 3), (13, 24, 3), (16, 14, 3), (19, 7, 2)), "水厂仪表改造"),
    ),
    (
        _project("retail-i", "retail_chain_i", 0, 4, 21, ((0, 18, 4), (1, 8, 3), (3, 22, 4), (4, 10, 2)), "门店设备更新"),
        _project("port-j", "port_logistics_j", 3, 11, 60, ((3, 7, 5), (6, 18, 3), (9, 12, 4), (11, 24, 2)), "仓储分拣线优化"),
        _project("property-k", "property_group_k", 7, 15, 30, ((7, 13, 3), (10, 21, 4), (13, 9, 3), (15, 17, 2)), "物业消防联动整改"),
        _project("research-l", "research_center_l", 10, 18, 45, ((10, 6, 3), (12, 19, 3), (16, 11, 3), (18, 23, 2)), "研发园区实验环境改造"),
    ),
)


def _identity(index: int) -> tuple[str, int]:
    if type(index) is not int or index < 0:
        raise ValueError("候选索引必须为非负整数")
    return f"controlled_t06_project_{index:03d}", 946_000 + index


def _specification(index: int, seed: int) -> dict[str, Any]:
    portfolio = deepcopy(PORTFOLIOS[index % len(PORTFOLIOS)])
    # Vary actual scope types rather than calendar decoration: the last job
    # optionally ends after a smaller warranty work package, while each
    # customer and milestone remains independently evidenced.
    if (index // len(PORTFOLIOS)) % 2:
        last = portfolio[-1]
        last["delivery_plan"] = last["delivery_plan"][:-1]
        last["end_month"] = last["delivery_plan"][-1]["month"]
    return {"mechanism_id": "nonperiodic_project_runoff", "candidate_seed": seed,
            "unit_price_fen": PROJECT_PRICE_FEN, "unit_cost_fen": PROJECT_COST_FEN,
            "supplier_credit_days": (14, 28, 35)[index % 3], "stock_cover_months": (1, 2, 3)[index % 3],
            "customers": portfolio,
            "positions": [
                {"id": "project_core", "start_month": "2024-01", "end_month": "2025-12", "monthly_salary_fen": 3_400_000,
                 "monthly_capacity_units": 80, "pay_day": 7},
                {"id": "warranty_retention", "start_month": "2024-01", "end_month": "2025-12", "monthly_salary_fen": 1_700_000,
                 "monthly_capacity_units": 40, "pay_day": 16},
            ],
            "reason": "多个非等长客户改造任务按独立工期和里程碑交付；客户项目完成后，质保和项目团队、场地及支持合同仍持续。",
            "runoff_commitment": "客户退出后固定岗位、场地和技术支持合同完整延续至观察期末。",
            "capital_coverage_months": 18,
            "capital_commitment_reason": "事前签订的18个月项目团队、质保、场地和支持合同覆盖；不允许事后补资。"}


def _capital_fen(specification: dict[str, Any]) -> int:
    payroll = sum(row["monthly_salary_fen"] for row in specification["positions"])
    project_material = sum(milestone["units"] for customer in specification["customers"]
                           for milestone in customer["delivery_plan"]) * specification["unit_cost_fen"]
    monthly_contract_cost = payroll + 3_000_000 + project_material / 24
    # Eighteen months of signed payroll/rent/support/project-material burden
    # plus real equipment capex.  It is set before the portfolio is compiled
    # and is not a rescue injection.
    return int(round(6_000_000 + monthly_contract_cost * specification["capital_coverage_months"]))


def _daily_rows(raw: dict, capital_fen: int) -> list[dict[str, str]]:
    start, end = date(2024, 1, 1), date(2025, 12, 31)
    events: dict[str, list[int]] = {}
    for event in raw["operating"]["business_nodes"]["settlements"]:
        when = event["due_date"]
        if start.isoformat() <= when <= end.isoformat():
            events.setdefault(when, [0, 0])
            slot = 0 if event["category"] == "collection" else 1
            events[when][slot] += event["amount_fen"]
    cash = capital_fen
    rows = []
    for offset in range((end - start).days + 1):
        when = (start + timedelta(days=offset)).isoformat(); incoming, outgoing = events.get(when, [0, 0])
        cash += incoming - outgoing
        rows.append({"calendar_date": when, "inflow_cny": f"{incoming / 100:.2f}", "outflow_cny": f"{outgoing / 100:.2f}",
                     "ending_balance_cny": f"{cash / 100:.2f}", "transaction_count": str(bool(incoming) + bool(outgoing))})
    return rows


def _preflight(rows: list[dict[str, str]], capital_fen: int, target: dict[str, Any]) -> dict[str, Any]:
    """Exact no-turn OLS/D coordinate and frozen repetition rule on plan events."""
    from curve_repetition_check import check as repetition_check
    from shape_contract_search import single_stage_daily_x
    import statistics

    x_daily = single_stage_daily_x(rows)
    month_end = [i for i, row in enumerate(rows) if i + 1 == len(rows) or rows[i + 1]["calendar_date"][5:7] != row["calendar_date"][5:7]]
    y_end = np.asarray([float(rows[i]["ending_balance_cny"]) for i in month_end], dtype=float)
    x_end = np.arange(24, dtype=float)
    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(24), x_end]), y_end, rcond=None)[0]
    balances = np.asarray([float(row["ending_balance_cny"]) for row in rows], dtype=float)
    residual = balances - (intercept + slope * x_daily)
    d_value = float(np.quantile(np.abs(residual - np.median(residual)), .90) / (capital_fen / 100))
    expected = float(target["monthly_net_cash_slope_cny"][0])
    slope_pass = abs(slope - expected) <= abs(expected) * .20
    d_pass = .04 * .8 <= d_value <= .10 * 1.2
    capital_pass = bool(balances.min() >= 0)
    repetition = repetition_check(rows)
    passed = bool(slope_pass and d_pass and capital_pass and repetition["passed"])
    reasons = ([name for name, value in (("planned_slope_outside_20pct", slope_pass), ("planned_D_outside_20pct", d_pass),
                                         ("planned_capital_insufficient", capital_pass), ("planned_repeating_waveform", repetition["passed"])) if not value])
    return {"acceptance_version": "planned_single_stage_ols_d_v1_not_formal", "OLS_method": "24_month_end_x0_to23; daily_x=single_stage_daily_x",
            "planned_ols_slope_cny_per_month": round(float(slope), 2), "slope_reference_cny_per_month": expected,
            "slope_band_cny_per_month": [round(expected * 1.2, 2), round(expected * .8, 2)], "planned_D": round(d_value, 8),
            "D_reference": [.04, .10], "D_band": [.032, .12], "minimum_planned_balance_cny": round(float(balances.min()), 2),
            "slope_passed": bool(slope_pass), "D_passed": bool(d_pass), "capital_passed": capital_pass, "reasons": reasons,
            "curve_repetition": repetition,
            "passed": passed,
            "scope": "仅由冻结合同事件派生的计划诊断；实际中央流水仍须重新验收。"}


def _fingerprint(specification: dict[str, Any]) -> str:
    facts = {"customers": [(row["customer_id"], row["start_month"], row["end_month"], row["credit_days"],
                              tuple((m["month"], m["units"]) for m in row["delivery_plan"])) for row in specification["customers"]],
             "supplier_credit_days": specification["supplier_credit_days"], "stock_cover_months": specification["stock_cover_months"],
             "positions": [(row["monthly_capacity_units"], row["monthly_salary_fen"]) for row in specification["positions"]]}
    return sha256(dumps(facts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _planned_signature_evidence(raw: dict) -> dict[str, Any]:
    """Compact evidence from the common planned-signature implementation."""
    from shape_contract_search import planned_signature
    signature = planned_signature(raw)
    cash = signature["cash"]
    return {"days": int(len(cash)), "contract_net_cash_cny": round(float(cash.sum()), 2),
            "nonzero_contract_event_days": int(np.count_nonzero(cash)),
            "method": "shape_contract_search.planned_signature; no initial-capital balance target"}


def build_candidate(target: dict[str, Any], candidate_index: int, revision: int = 0, base_spec: dict[str, Any] | None = None):
    """Compile one project portfolio and return the standard provider tuple."""
    if target.get("target_id") != TARGET_ID:
        raise ValueError("project-order provider only serves T06")
    if type(revision) is not int or revision < 0:
        raise ValueError("revision must be a non-negative integer")
    candidate_id, seed = _identity(candidate_index)
    specification = deepcopy(base_spec) if base_spec is not None else _specification(candidate_index, seed)
    if specification.get("candidate_seed") != seed:
        raise ValueError("base specification does not belong to this frozen candidate")
    change = {"revision": revision, "variable_group": "none", "changed_fields": [], "economic_reason": "初始独立项目组合"}
    if revision:
        # One documented work-package change, within a 15% signed quantity
        # variation clause; dates, customer identities and funding remain frozen.
        milestones = specification["customers"][-1]["delivery_plan"]
        milestones[-1]["units"] = max(1, round(milestones[-1]["units"] * (1.10 if revision % 2 else .90)))
        change = {"revision": revision, "variable_group": "final_project_work_package", "changed_fields": ["customers[-1].delivery_plan[-1].units"],
                  "economic_reason": "末项已签项目工作包在数量变更条款内调整；非余额或日期修订。"}
    capital_fen = _capital_fen(specification)
    raw, facts = plan_mechanism(_template(candidate_id, seed, capital_fen), specification)
    rows = _daily_rows(raw, capital_fen)
    preflight = _preflight(rows, capital_fen, target)
    structural = {"project_terms": [(row["customer_id"], row["start_month"], row["end_month"], len(row["delivery_plan"])) for row in specification["customers"]],
                  "retained_positions": [(row["id"], row["end_month"]) for row in specification["positions"]],
                  "stock_cover_months": specification["stock_cover_months"]}
    budget_response = {"preflight": {"passed": preflight["passed"], "reasons": preflight["reasons"],
                                         "measurement": "project_order_contract_plan_only"},
                       "target_screen_only": {"all_directions_match": bool(preflight["planned_ols_slope_cny_per_month"] < 0),
                                              "stages": [{"expected_slope_cny_per_month": target["monthly_net_cash_slope_cny"][0],
                                                          "planned_slope_cny_per_month": preflight["planned_ols_slope_cny_per_month"],
                                                          "direction_match": bool(preflight["planned_ols_slope_cny_per_month"] < 0)}]}}
    spec = {**specification, "candidate_id": candidate_id, "family_id": "controlled_family_nonperiodic_project_runoff", "target_id": TARGET_ID,
            "seed": seed, "initial_capital_fen": capital_fen, "economic_fingerprint": _fingerprint(specification),
            "structural_features": structural, "target_used_only_for_search_loss": True, "target_used_to_construct_budget": False,
            "preflight": preflight, "planned_signature_evidence": _planned_signature_evidence(raw),
            "budget_response": budget_response, "capital_basis": specification["capital_commitment_reason"]}
    facts.update(candidate_id=candidate_id, family_id=spec["family_id"], mechanism_id=specification["mechanism_id"],
                 inventory_identity_exact=all(row["ending_fen"] >= 0 for row in facts["inventory"]), target_balance_used=False)
    return raw, facts, spec, change


def search_candidates(target: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    """Bounded independent portfolio search; only planned passes are returned."""
    if not 1 <= limit <= 24:
        raise ValueError("project portfolio search limit must be 1..24")
    results = []
    for index in range(limit):
        raw, facts, spec, change = build_candidate(target, index)
        if spec["preflight"]["passed"]:
            results.append({"candidate_index": index, "raw": raw, "facts": facts, "spec": spec, "change": change,
                            "preflight_status": "planned_contract_OLS_D_capital_repetition_pass_not_formal"})
    return results
