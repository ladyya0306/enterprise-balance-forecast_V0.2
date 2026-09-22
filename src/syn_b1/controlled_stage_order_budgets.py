"""Contract-first, multi-stage project-order budgets for controlled expansion.

Targets choose a predeclared project mechanism only.  They never calculate an
order value, collection amount, or balance.  Customer task packages, their
acceptance dates, supplier stock and fixed-term roles are compiled by the
normal entity planner before the inexpensive shape screen runs.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from json import dumps

from syn_b1.controlled_business_budgets import (
    MONTHS, _budget_response, _capital_for, _core_metrics, _planned_repetition, _template,
)
from syn_b1.mechanism_event_planner import plan_mechanism


STAGE_MECHANISMS = {
    "T07": "acceptance_restart_projects",
    "T11": "completion_runoff_projects",
    "T15": "mobilise_deliver_demobilise",
    "T18": "deliver_pause_recontract",
}


def _identity(target_id: str, candidate_index: int) -> tuple[str, int]:
    if target_id not in STAGE_MECHANISMS or type(candidate_index) is not int or not 0 <= candidate_index < 96:
        raise ValueError("阶段项目预算只支持T07、T11、T15、T18及0至95索引")
    return f"stage_order_{target_id.lower()}_{candidate_index:03d}", 967_000 + int(target_id[1:]) * 1_000 + candidate_index


def _task_plan(start: int, end: int, total_monthly_units: int, customer_ordinal: int, variant: int) -> list[dict]:
    """Create irregular, explicitly accepted project tasks within one contract."""
    # Package counts and dates are agreed task facts.  Their non-periodic
    # sequence is intentionally not a uniform monthly payment template.
    counts = (1, 2, 1, 3, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 3, 2, 1, 3, 2, 1, 2, 3, 1, 2)
    # Task classes have distinct acceptance windows.  This is preserved in
    # the contract plan as a project-task fact, rather than shifting an
    # otherwise identical invoice date after observing a curve.
    day_patterns = {
        1: ((4,), (11,), (18,), (26,)),
        2: ((3, 16), (7, 24), (5, 27), (10, 21)),
        3: ((2, 12, 25), (5, 16, 27), (3, 18, 24), (8, 14, 26)),
    }
    demand = (
        (94, 111, 87, 118, 96, 108, 91, 121, 89, 114, 98, 106, 85, 119, 93, 112, 88, 116, 97, 109, 90, 122, 95, 105),
        (108, 89, 116, 95, 121, 86, 110, 92, 118, 88, 113, 97, 124, 84, 107, 94, 120, 90, 115, 87, 123, 93, 111, 98),
        (91, 120, 86, 113, 98, 117, 89, 122, 94, 108, 85, 119, 96, 114, 88, 123, 92, 110, 99, 116, 87, 121, 95, 106),
    )[variant % 3]
    rows = []
    for month in range(start, end + 1):
        units = max(1, total_monthly_units * demand[(month + customer_ordinal * 5) % 24] // 100)
        count = counts[(month + customer_ordinal * 7 + variant) % 24]
        pieces = [max(1, units // count) for _ in range(count)]
        pieces[-1] += units - sum(pieces)
        pattern = day_patterns[count][(month + customer_ordinal * 3 + variant) % len(day_patterns[count])]
        rows.extend(dict(month=MONTHS[month], delivery_day=day, units=piece)
                    for day, piece in zip(pattern, pieces, strict=True))
    return rows


def _customer(contract_id: str, ordinal: int, start: int, end: int, units: int, variant: int) -> dict:
    return dict(contract_id=contract_id, customer_id=f"stage_project_customer_{ordinal + 1}",
                start_month=MONTHS[start], end_month=MONTHS[end], monthly_units=units, delivery_day=12,
                credit_days=(0, 14, 28)[(ordinal + variant) % 3],
                delivery_plan=_task_plan(start, end, units, ordinal, variant),
                quantity_variation_clause_percent=15,
                task_clause="不等长装机/验收任务包；每项交付及账期由客户项目合同分别冻结")


def _spec(target_id: str, index: int, seed: int) -> dict:
    variant = index % 3
    # These are catalogue project volumes, selected as whole contract options;
    # no target slope is used to solve for a delivery quantity.
    catalogue = {
        "T07": (72, 78, 84), "T11": (100, 81, 83),
        "T15": (72, 78, 84), "T18": (72, 78, 84),
    }
    options = catalogue[target_id][variant]
    common = dict(mechanism_id=STAGE_MECHANISMS[target_id], unit_price_fen=680_000, unit_cost_fen=388_000,
                  supplier_credit_days=(7, 21, 35)[(index // 3) % 3], stock_cover_months=(1, 2, 3)[(index // 9) % 3],
                  candidate_seed=seed, capital_coverage_months=18,
                  capital_commitment_reason="项目验收、质保岗位、场地及技术支持合同的18个月事前覆盖",
                  reason="客户项目合同、采购库存和固定期限岗位在生成前冻结；余额目标不参与金额计算")
    if target_id == "T07":
        common.update(customers=[_customer("preparation-validation", 2, 0, 6, 7, variant),
                                 _customer("restart-acceptance-a", 0, 7, 23, options, variant),
                                 _customer("restart-acceptance-b", 1, 8, 23, options // 3, variant)],
                      positions=[dict(id="retained_warranty", start_month="2024-01", end_month="2025-12", monthly_salary_fen=7_500_000,
                                      monthly_capacity_units=120, pay_day=7),
                                 dict(id="preparation_role", start_month="2024-01", end_month="2024-07", monthly_salary_fen=3_000_000,
                                      monthly_capacity_units=50, pay_day=21),
                                 dict(id="acceptance_delivery", start_month="2024-08", end_month="2025-12", monthly_salary_fen=3_000_000,
                                      monthly_capacity_units=160, pay_day=14)],
                      causal_event="前期筹备与质保岗位持续支出；客户验收通过后，两份已中标项目按不等长任务包签入并交付。")
    elif target_id == "T11":
        common.update(customers=[_customer("completion-a", 0, 0, 6, options, variant),
                                 _customer("completion-b", 1, 1, 6, options // 3, variant),
                                 _customer("warranty-service", 2, 7, 23, 40, variant)],
                      positions=[dict(id="retained_warranty", start_month="2024-01", end_month="2025-12", monthly_salary_fen=20_200_000,
                                      monthly_capacity_units=120, pay_day=7),
                                 dict(id="completion_project_role", start_month="2024-01", end_month="2024-07", monthly_salary_fen=3_000_000,
                                      monthly_capacity_units=160, pay_day=14)],
                      causal_event="两份项目在验收期完成后到期退出；项目岗位随合同到期，质保和场地合同继续履约。")
    elif target_id == "T15":
        common.update(customers=[_customer("mobilise-a", 0, 6, 15, options, variant),
                                 _customer("mobilise-b", 1, 7, 15, options // 3, variant)],
                      positions=[dict(id="base_warranty", start_month="2024-01", end_month="2025-12", monthly_salary_fen=4_000_000,
                                      monthly_capacity_units=100, pay_day=7),
                                 dict(id="pre_mobilisation", start_month="2024-01", end_month="2024-06", monthly_salary_fen=5_000_000,
                                      monthly_capacity_units=70, pay_day=14),
                                 dict(id="delivery_team", start_month="2024-07", end_month="2025-04", monthly_salary_fen=2_000_000,
                                      monthly_capacity_units=130, pay_day=21)],
                      causal_event="筹备团队先占用资金，获批后客户项目集中交付，项目结束后交付岗位到期而只保留质保班组。")
    else:  # T18
        common.update(customers=[_customer("initial-delivery", 0, 0, 5, options - 18, variant),
                                 _customer("recontract-a", 1, 16, 23, options - 18, variant),
                                 _customer("recontract-b", 2, 18, 23, max(12, options // 4), variant)],
                      positions=[dict(id="base_delivery", start_month="2024-01", end_month="2025-12", monthly_salary_fen=6_000_000,
                                      monthly_capacity_units=120, pay_day=7),
                                 dict(id="pause_integration", start_month="2024-07", end_month="2025-04", monthly_salary_fen=2_000_000,
                                      monthly_capacity_units=80, pay_day=14)],
                      causal_event="初始项目交付后进入整合停单期；整合岗位到期前，新客户分别签入两份续约项目恢复交付。")
    common["grid_coordinates"] = dict(project_volume_catalogue=variant, supplier_term_variant=(index // 3) % 3,
                                       inventory_cover_variant=(index // 9) % 3)
    return common


def _economic_features(spec: dict) -> dict:
    return dict(mechanism_id=spec["mechanism_id"], unit_price_fen=spec["unit_price_fen"], unit_cost_fen=spec["unit_cost_fen"],
                supplier_credit_days=spec["supplier_credit_days"], stock_cover_months=spec["stock_cover_months"],
                projects=sorted((c["start_month"], c["end_month"], c["credit_days"],
                                 tuple((m["month"], m["delivery_day"], m["units"]) for m in c["delivery_plan"])) for c in spec["customers"]),
                roles=sorted((p["start_month"], p["end_month"], p["monthly_salary_fen"], p["monthly_capacity_units"]) for p in spec["positions"]))


def _revision(spec: dict, revision: int) -> tuple[dict, dict]:
    if type(revision) is not int or revision not in {0, 1, 2}:
        raise ValueError("阶段项目修订仅允许0至2，且每次只调整一组项目任务量")
    revised = deepcopy(spec)
    if revision == 0:
        return revised, dict(revision=0, variable_group="none", changed_fields=[], economic_reason="初始冻结项目合同")
    multiplier = (90, 110)[revision - 1]
    for customer in revised["customers"]:
        customer["monthly_units"] = max(1, customer["monthly_units"] * multiplier // 100)
        for task in customer["delivery_plan"]:
            task["units"] = max(1, task["units"] * multiplier // 100)
    return revised, dict(revision=revision, variable_group="project_task_volume", changed_fields=["customers[].delivery_plan[].units"],
                         multiplier_percent=multiplier, economic_reason="仅按已签项目任务量变更条款调整；客户期限、岗位、账期和资金冻结")


def build_candidate(target: dict | str, candidate_index: int, revision: int, base_spec: dict | None = None):
    target_id = target if isinstance(target, str) else target.get("target_id")
    candidate_id, seed = _identity(target_id, candidate_index)
    spec = deepcopy(base_spec) if base_spec is not None else _spec(target_id, candidate_index, seed)
    if spec.get("candidate_seed", seed) != seed or spec.get("mechanism_id") != STAGE_MECHANISMS[target_id]:
        raise ValueError("恢复规格与冻结候选身份或阶段机制不一致")
    revised, change = _revision(spec, revision)
    raw, facts = plan_mechanism(_template(candidate_id, seed, _capital_for(revised)), revised)
    economic = _economic_features(revised)
    fingerprint = sha256(dumps(economic, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    response = _budget_response(raw, target)
    result = dict(revised, candidate_id=candidate_id, target_id=target_id, seed=seed,
                  family_id=f"controlled_stage_family_{revised['mechanism_id']}", target_used_only_for_mechanism_selection=True,
                  economic_features=economic, economic_fingerprint=fingerprint, core_metrics=_core_metrics(target),
                  budget_response=response, capital_basis="18个月已签项目、岗位、场地及技术支持合同支出覆盖；不允许中途补资")
    facts.update(candidate_id=candidate_id, mechanism_id=revised["mechanism_id"], family_id=result["family_id"],
                 economic_fingerprint=fingerprint, inventory_identity_exact=all(row["ending_fen"] >= 0 for row in facts["inventory"]))
    return raw, facts, result, change


def search_budget_candidates(target: dict, *, limit: int = 96) -> list[dict]:
    if type(limit) is not int or not 1 <= limit <= 96:
        raise ValueError("阶段项目预算搜索上限必须为1至96")
    declared = _core_metrics(target)["stages"]
    rows = []
    for index in range(limit):
        raw, facts, spec, _ = build_candidate(target, index, 0)
        response, screen = spec["budget_response"], spec["budget_response"].get("target_screen_only", {})
        stages, stage_d = screen.get("stages", []), response.get("planned_stage_D", [])
        amplitude = response.get("target_screen_only", {}).get("hinge_passed", False)
        slopes_ok = len(stages) == len(declared) and all(
            abs(actual["planned_slope_cny_per_month"] - expected["slope_cny_per_month"]) <= abs(expected["slope_cny_per_month"]) * .20
            for actual, expected in zip(stages, declared, strict=True))
        d_ok = len(stage_d) == len(declared) and all(low * .8 <= actual <= high * 1.2
                                                          for actual, (low, high) in zip(stage_d, (x["D_reference"] for x in declared), strict=True))
        capital_ok = response["planned_capital_adequate"]
        if not (screen.get("all_directions_match") and slopes_ok and d_ok and capital_ok and amplitude):
            continue
        repetition = _planned_repetition(raw)
        if not repetition["passed"]:
            continue
        response["preflight"] = dict(passed=True, reasons=[], planned_slope_cny_per_month=[x["planned_slope_cny_per_month"] for x in stages],
                                       planned_D=response["planned_D"], repetition=repetition)
        rows.append(dict(candidate_index=index, candidate_id=spec["candidate_id"], family_id=spec["family_id"], mechanism_id=spec["mechanism_id"],
                         economic_fingerprint=spec["economic_fingerprint"], budget_response=response, planned_repetition=repetition,
                         preflight_status="planned_multistage_budget_direction_amplitude_D_repetition_pass_not_formal"))
    return rows
