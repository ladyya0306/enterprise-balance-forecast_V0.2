"""Deterministic, contract-first budgets for controlled expansion candidates.

This module deliberately stops before the central ledger.  A target selects a
plausible *business mechanism*, but no target slope or target balance is used
to compute a contract amount.  The returned profile has already passed the
entity compiler, so callers can cheaply reject invalid budgets before asking
the central generator to create any cashflow.
"""
from __future__ import annotations

from copy import deepcopy
# 导入月末天数工具，用于校验并迁移显式合同日期。
from calendar import monthrange
from datetime import date, timedelta
from hashlib import sha256
from json import dumps
from statistics import mean

from syn_b1.formal_profile import default_profile
from syn_b1.mechanism_event_planner import plan_mechanism


MONTHS = tuple(f"{2024 + i // 12}-{i % 12 + 1:02d}" for i in range(24))


# 校验受控预算的显式24月日期合同。
def _schedule(start_date: str, end_date: str, prediction_cutoff: str) -> tuple[date, date]:
    """校验受控两年合同期；旧调用不传参数时仍使用原期间。"""
    # 把输入字符串解析为可比较的日期对象。
    start, end, cutoff = (date.fromisoformat(value) for value in (start_date, end_date, prediction_cutoff))
    # 拒绝半月合同期，保证节点和月度预算完整。
    if start.day != 1 or end.day != monthrange(end.year, end.month)[1]:
        raise ValueError("经营期间必须从月初开始并在月末结束")
    # 截止日必须与合同期末日一致。
    if cutoff != end:
        raise ValueError("prediction_cutoff必须等于经营期间截止日")
    # 计算闭区间合同月数。
    month_count = (end.year - start.year) * 12 + end.month - start.month + 1
    # 受控成熟机制只接受原有24月结构。
    if month_count != len(MONTHS):
        raise ValueError("受控机制预算目前要求连续24个月经营期间")
    # 返回已校验的合同边界。
    return start, end


# 递归按月偏移迁移旧合同事实到显式日期合同。
def _rebase_schedule(value, start: date):
    """将冻结的24月合同事实按月序列移至显式经营期间。"""
    # 字典逐字段迁移，保留字段结构。
    if isinstance(value, dict):
        return {key: _rebase_schedule(item, start) for key, item in value.items()}
    # 列表逐项迁移，保留事件顺序。
    if isinstance(value, list):
        return [_rebase_schedule(item, start) for item in value]
    # 元组逐项迁移，保留不可变特征结构。
    if isinstance(value, tuple):
        return tuple(_rebase_schedule(item, start) for item in value)
    # 非字符串不是日期事实，原样返回。
    if not isinstance(value, str):
        return value
    # 仅解析精确的月份或日期字符串，避免误改身份和业务说明。
    try:
        # 迁移合同月份字段。
        if len(value) == 7 and value[4] == "-":
            # 将月份补全为首日以计算相对月偏移。
            source = date.fromisoformat(value + "-01")
            # 仅处理冻结源合同的24个月份。
            if date(2024, 1, 1) <= source <= date(2025, 12, 1):
                # 计算源月份在24月合同内的零基位置。
                index = (source.year - 2024) * 12 + source.month - 1
                # 将位置映射为新合同中的目标月份。
                return f"{start.year + (start.month - 1 + index) // 12}-{(start.month - 1 + index) % 12 + 1:02d}"
        # 迁移具体到期日和投用日字段。
        if len(value) == 10 and value[4] == "-" and value[7] == "-":
            # 解析源自然日以保留日序和月底裁剪规则。
            source = date.fromisoformat(value)
            # 期末交付的合同回款可跨到下一自然年；它仍属于本24月合同
            # 的同一月序列，不能在迁移后落回旧年份。
            # 覆盖期末交付后最多一季的原合同尾款。
            if date(2024, 1, 1) <= source <= date(2026, 3, 31):
                # 计算日期所属源月份的相对位置。
                index = (source.year - 2024) * 12 + source.month - 1
                # 计算目标年份和月份。
                year, month = start.year + (start.month - 1 + index) // 12, (start.month - 1 + index) % 12 + 1
                # 月末较短时裁剪日号，保留合法日期。
                return date(year, month, min(source.day, monthrange(year, month)[1])).isoformat()
    # 非ISO日期字符串不属于可迁移日期事实。
    except ValueError:
        # 保留原字符串，避免错误修改业务文本。
        pass
    # 未命中日期合同的值保持不变。
    return value

# These are causal choices, rather than four aliases of a monthly sales curve.
TARGET_MECHANISMS = {
    "T01": "customer_renewal", "T02": "inventory_replenishment",
    "T03": "customer_renewal", "T04": "inventory_replenishment",
    "T05": "customer_renewal", "T06": "customer_renewal",
    "T07": "maintenance_restart", "T08": "maintenance_restart",
    "T09": "inventory_replenishment", "T10": "inventory_replenishment",
    "T11": "tender_delivery", "T12": "tender_delivery",
    "T13": "customer_renewal", "T14": "customer_renewal",
    "T15": "tender_delivery", "T16": "customer_renewal",
    "T17": "inventory_replenishment", "T18": "tender_delivery",
    "T19": "maintenance_restart", "T20": "tender_delivery",
    "T21": "tender_delivery", "T22": "tender_delivery",
    "T23": "maintenance_restart", "T24": "inventory_replenishment",
}


def _money(fen: int) -> str:
    return f"{fen / 100:.2f}"


# 将任意合同月份映射为连续整数坐标。
def _month_position(value: str) -> int:
    """返回连续月坐标，供日期参数化后的合同期限比较复用。"""
    # 拆解年和月以构造绝对月序号。
    year, month = map(int, value.split("-"))
    # 返回可跨年份比较的零基月坐标。
    return year * 12 + month - 1


def _candidate_identity(target_id: str, candidate_index: int) -> tuple[str, int]:
    if target_id not in TARGET_MECHANISMS or type(candidate_index) is not int or candidate_index < 0:
        raise ValueError("目标必须是T01至T24，候选索引必须为非负整数")
    candidate_id = f"controlled_{target_id.lower()}_{candidate_index:03d}"
    # Stable across revisions.  It is not a search seed that may be swapped
    # after observing a central run.
    return candidate_id, 916_000 + int(target_id[1:]) * 1_000 + candidate_index


def _template(candidate_id: str, seed: int, capital_fen: int) -> dict:
    raw = default_profile(candidate_id, "controlled_business_budget_v1")
    raw.update(schema_version="syn_b1_formal_profile_2_0", start_date="2024-01-01", end_date="2025-12-31", random_seed=seed)
    raw["calendar"]["prediction_cutoff"] = "2025-12-31"
    raw["initial_capital"]["amount_cny"] = _money(capital_fen)
    op = raw["operating"]
    for key in ("base_monthly_sales_cny", "sales_monthly_growth_percent", "sales_month_weights_percent", "gross_margin_percent",
                "monthly_payroll_cny", "monthly_rent_and_utilities_cny", "monthly_other_operating_expense_cny"):
        op.pop(key, None)
    # The equipment and its payment are a real pre-existing capital commitment;
    # they establish capacity and are never added as an emergency cash rescue.
    op["fixed_assets"] = [{"event_id": "EQ01", "asset_type": "equipment", "purchase_month": "2024-01",
                            "purchase_amount_cny": "60000.00", "payment_schedule": [{"delay_days": 0, "share_percent": "100"}],
                            "ready_for_use_date": "2024-01-01", "useful_life_months": 60, "residual_value_percent": "5"}]
    nodes = [dict(month=m, stage_id="frozen-budget", business_reason="事前冻结的合同经营计划", sales_units=0,
                  unit_price_cny="0.00", unit_variable_cost_cny="0.00", target_inventory_cny="0.00",
                  payroll_cny="0.00", rent_cny="18000.00", other_expense_cny="12000.00") for m in MONTHS]
    continuing = []
    for month in MONTHS:
        # These continuing contracts are carried through the entity compiler;
        # it adds payroll, purchase and collection events from the real book.
        continuing.extend((dict(event_id=f"lease-{month}", category="rent_payment", recognition_month=month, asset_id="",
                                due_date=f"{month}-02", amount_fen=1_800_000, note_cn="已签场地租约"),
                           dict(event_id=f"support-{month}", category="other_expense_payment", recognition_month=month, asset_id="",
                                due_date=f"{month}-24", amount_fen=1_200_000, note_cn="已签技术支持服务合同")))
    op["business_nodes"] = dict(version="physical_contract_business_v1", nodes=nodes,
                                 settlements=[dict(event_id="EQ01", category="fixed_asset_payment", recognition_month="2024-01",
                                                   asset_id="EQ01", due_date="2024-01-03", amount_fen=6_000_000, note_cn="已签生产设备采购合同"), *continuing],
                                 asset_capacities=[dict(event_id="EQ01", monthly_units=2_000)],
                                 capacity_basis="已签设备月产能2000件；实际交付另受岗位期限约束。")
    return raw


def _milestones(start: int, end: int, units: int, day: int, cadence: int = 1) -> list[dict]:
    return [dict(month=MONTHS[i], delivery_day=day, units=units) for i in range(start, end + 1, cadence)]


def _mechanism_spec(target_id: str, candidate_index: int, seed: int) -> dict:
    mechanism_id = TARGET_MECHANISMS[target_id]
    variant = candidate_index % 3
    price = (520_000, 680_000, 840_000)[variant]
    cost = (286_000, 388_000, 470_000)[variant]
    # All routes have positive unit contribution; cash pressure, when present,
    # is caused by signed stock, equipment and continuing commitments.
    common = dict(mechanism_id=mechanism_id, unit_price_fen=price, unit_cost_fen=cost, supplier_credit_days=(7, 21, 35)[variant],
                  reason="事前签订的客户、供应商、岗位及场地合同；不由余额目标反推", candidate_seed=seed)
    if mechanism_id == "customer_renewal":
        customers = []
        for i, (start, end, qty) in enumerate(((0, 5, 22 + variant), (0, 11, 18), (3, 17, 20 + variant), (8, 23, 24)), 1):
            # The first contract expires without renewal; the later ones have
            # explicit renewal contracts, so no order can exist after expiry.
            customers.append(dict(contract_id=f"renew-{i}", customer_id=f"renewal_customer_{i}", start_month=MONTHS[start], end_month=MONTHS[end],
                                  monthly_units=qty, delivery_day=4 + i * 4, credit_days=(7, 21, 35, 45)[i - 1],
                                  delivery_plan=_milestones(start, end, qty, 4 + i * 4)))
        common.update(stock_cover_months=1, customers=customers,
                      positions=[dict(id="renewal_core", start_month="2024-01", end_month="2025-12", monthly_salary_fen=3_400_000,
                                      monthly_capacity_units=180, pay_day=7),
                                 dict(id="renewal_success", start_month="2024-04", end_month="2025-12", monthly_salary_fen=1_700_000,
                                      monthly_capacity_units=80, pay_day=14)],
                      causal_event="客户分别到期：一户退出，后续客户以独立续签合同接替。")
    elif mechanism_id == "inventory_replenishment":
        customers = []
        for i, (start, end, qty, cadence) in enumerate(((0, 23, 24 + variant, 1), (0, 11, 30, 1), (7, 23, 38, 2)), 1):
            customers.append(dict(contract_id=f"stock-{i}", customer_id=f"stock_customer_{i}", start_month=MONTHS[start], end_month=MONTHS[end],
                                  monthly_units=qty, delivery_day=5 + i * 5, credit_days=(14, 30, 45)[i - 1],
                                  delivery_plan=_milestones(start, end, qty, 5 + i * 5, cadence)))
        common.update(stock_cover_months=(2, 3, 2)[variant], customers=customers,
                      positions=[dict(id="warehouse_team", start_month="2024-01", end_month="2025-12", monthly_salary_fen=3_000_000,
                                      monthly_capacity_units=220, pay_day=8)],
                      causal_event="先按已签订单采购安全库存；后续月份允许只耗用库存，达到补货阈值才采购。")
    elif mechanism_id == "maintenance_restart":
        outage_start = (6, 8, 10)[variant]
        customers = []
        for i, qty in enumerate((28 + variant, 25, 22), 1):
            plan = _milestones(0, 23, qty, 5 + i * 5)
            customers.append(dict(contract_id=f"outage-{i}", customer_id=f"maintenance_customer_{i}", start_month="2024-01", end_month="2025-12",
                                  monthly_units=qty, delivery_day=5 + i * 5, credit_days=(14, 30, 45)[i - 1], delivery_plan=plan,
                                  suspend_during_maintenance=True))
        common.update(stock_cover_months=2, customers=customers,
                      maintenance=[dict(contract_id="planned_overhaul", start_month=MONTHS[outage_start], end_month=MONTHS[outage_start + 1],
                                        monthly_service_fen=1_400_000 + variant * 200_000)],
                      positions=[dict(id="maintenance_core", start_month="2024-01", end_month="2025-12", monthly_salary_fen=3_800_000,
                                      monthly_capacity_units=180, pay_day=7)],
                      causal_event="设备检修两个月停工；合同明确允许暂停，工资、租约和检修服务费持续，复工后恢复交付。")
    else:  # tender_delivery
        customers = []
        for i, (start, end, qty) in enumerate(((0, 4, 55 + variant), (5, 11, 70), (12, 17, 62 + variant), (18, 23, 80)), 1):
            customers.append(dict(contract_id=f"tender-{i}", customer_id=f"project_owner_{i}", start_month=MONTHS[start], end_month=MONTHS[end],
                                  monthly_units=qty, delivery_day=3 + i * 5, credit_days=(30, 45, 21, 60)[i - 1],
                                  delivery_plan=_milestones(start, end, qty, 3 + i * 5)))
        common.update(stock_cover_months=1, customers=customers,
                      positions=[dict(id="delivery_core", start_month="2024-01", end_month="2025-12", monthly_salary_fen=3_700_000,
                                      monthly_capacity_units=170, pay_day=7),
                                 dict(id="bid_project_team", start_month="2024-01", end_month="2024-12", monthly_salary_fen=2_100_000,
                                      monthly_capacity_units=90, pay_day=14)],
                      causal_event="不同项目业主的中标、交付期限和验收账期独立冻结；项目岗位在期限届满后停止工资。")
    # A finite 3*3*3*2*3 grid creates real alternative enterprises.  These
    # dimensions are deliberately contract/lifecycle/capacity facts, never a
    # seed, name, date translation or whole-budget multiplier.
    lifecycle = (candidate_index // 3) % 3
    credit_variant = (candidate_index // 9) % 3
    workforce_variant = (candidate_index // 27) % 2
    stock_variant = (candidate_index // 54) % 3
    for ordinal, customer in enumerate(common["customers"]):
        customer["quantity_variation_clause_percent"] = 15
        plan = customer.get("delivery_plan", [])
        if lifecycle == 1 and ordinal % 2 == 0 and len(plan) > 4:
            customer["delivery_plan"] = plan[:-2]
            customer["end_month"] = customer["delivery_plan"][-1]["month"]
        elif lifecycle == 2 and ordinal % 2 == 1 and len(plan) > 4:
            customer["delivery_plan"] = plan[2:]
            customer["start_month"] = customer["delivery_plan"][0]["month"]
        customer["credit_days"] = min(60, customer["credit_days"] + (0, 7, 14)[credit_variant])
    if workforce_variant:
        # Rebalance actual roles while retaining a capacity-safe operating
        # plan.  A project role remains a distinct finite-term employment fact.
        common["positions"][0]["monthly_capacity_units"] -= 10
        common["positions"][0]["monthly_salary_fen"] -= 200_000
        if len(common["positions"]) > 1:
            common["positions"][1]["monthly_capacity_units"] += 10
            common["positions"][1]["monthly_salary_fen"] += 200_000
    # Run-off contracts carry a retained field-and-warranty team through the
    # end of the lease.  This is a signed employment commitment, not a cash
    # adjustment: it makes expiring customers produce a genuine downtrend.
    if target_id in {"T05", "T06"}:
        common["positions"][0]["monthly_salary_fen"] += 8_000_000
        common["runoff_commitment"] = "到期客户退出后，质保与场地班组在完整租约期内继续履约"
    if target_id in {"T05", "T06"}:
        # A declining business with a fixed warranty crew is contractually
        # funded through its longer runoff window.  This is selected before
        # any cash forecast and is not a post-failure balance top-up.
        common["capital_coverage_months"] = 18
        common["capital_commitment_reason"] = "客户退出后的18个月质保、场地和技术支持合同覆盖"
    if target_id in {"T05", "T06"}:
        # Each runoff customer has a non-equal project-task calendar.  Tasks
        # correspond to installation/maintenance lots and their quantities;
        # they are not a jitter of an otherwise monthly payment template.
        # The tail customer in particular has uneven task intervals, so a
        # customer exit does not leave a repeated monthly sawtooth behind.
        task_calendars = (
            ((0, 2, 5), (0, 1, 4, 6, 9, 11), (0, 2, 3, 6, 9, 11, 14), (0, 1, 4, 7, 8, 11, 13, 15)),
            ((0, 3, 5), (0, 3, 4, 8, 10, 11), (0, 2, 5, 8, 10, 13, 14), (0, 2, 5, 6, 9, 12, 14, 15)),
            ((0, 1, 4, 5), (0, 2, 5, 7, 8, 11), (0, 1, 4, 7, 8, 12, 14), (0, 3, 4, 7, 10, 11, 14, 15)),
        )
        demand_calendars = (
            (87, 115, 96, 121, 90, 107, 117, 84, 111, 93, 124, 89, 102, 118, 86, 113, 95, 120, 88, 109, 123, 92, 105, 116),
            (112, 91, 123, 86, 108, 94, 119, 88, 114, 97, 121, 85, 110, 93, 125, 89, 116, 96, 122, 87, 107, 98, 118, 90),
            (95, 122, 88, 110, 84, 117, 93, 124, 90, 106, 119, 86, 113, 97, 121, 89, 115, 92, 123, 85, 109, 99, 118, 91),
        )
        task_calendar = task_calendars[(candidate_index // 3) % len(task_calendars)]
        calendar = demand_calendars[candidate_index % len(demand_calendars)]
        package_counts = (1, 3, 2, 1, 2, 3, 1, 2, 3, 2, 1, 3, 2, 1, 3, 2, 3, 1, 2, 1, 3, 2, 1, 3)
        acceptance_days = {1: (12,), 2: (5, 22), 3: (3, 13, 26)}
        for ordinal, customer in enumerate(common["customers"]):
            start, end = MONTHS.index(customer["start_month"]), MONTHS.index(customer["end_month"])
            # Low-D runoff work retains month-by-month service coverage, but
            # every month has a predeclared, non-periodic count of separately
            # accepted task lots.  Medium-D uses the sparser project packages.
            task_months = (list(range(start, end + 1)) if target_id == "T05"
                           else [start + offset for offset in task_calendar[ordinal] if start + offset <= end])
            if not task_months:
                task_months = [start]
            total_units = sum(item["units"] for item in customer["delivery_plan"])
            weights = [calendar[(month + ordinal * 5) % len(calendar)] for month in task_months]
            denominator = sum(weights)
            allocated = [max(1, total_units * weight // denominator) for weight in weights]
            allocated[-1] += total_units - sum(allocated)
            task_plan = []
            for month, units in zip(task_months, allocated, strict=True):
                milestone = {"month": MONTHS[month], "delivery_day": customer["delivery_day"], "units": units}
                if target_id == "T05":
                    count = package_counts[(month + ordinal * 7) % len(package_counts)]
                    days = acceptance_days[count]
                    parts = [max(1, units // count) for _ in days]
                    parts[-1] += units - sum(parts)
                    task_plan.extend({**milestone, "delivery_day": day, "units": part}
                                     for day, part in zip(days, parts, strict=True))
                elif target_id == "T06":
                    first = units // 2
                    task_plan.extend((
                        {**milestone, "delivery_day": max(1, milestone["delivery_day"] - 2), "units": first},
                        {**milestone, "delivery_day": min(28, milestone["delivery_day"] + 10), "units": units - first},
                    ))
                else:
                    task_plan.append(milestone)
            customer["delivery_plan"] = task_plan
            customer["project_task_clause"] = "非等长装机/维保任务包；每项交付后按该客户账期收款"
            customer["demand_schedule_clause"] = "客户装机与消耗计划按任务分别约定，非统一比例缩放"
            if target_id == "T05":
                customer["partial_acceptance_clause"] = "每月按不同数量的装机/维保任务包分批独立验收、各自按合同账期收款"
            elif target_id == "T06":
                customer["partial_acceptance_clause"] = "每个任务包分两批独立验收、各自按合同账期收款"
    common["stock_cover_months"] = (1, 2, 3)[stock_variant]
    common["grid_coordinates"] = dict(price_cost_variant=variant, customer_lifecycle_variant=lifecycle,
                                       credit_terms_variant=credit_variant, workforce_variant=workforce_variant,
                                       stock_cover_variant=stock_variant)
    return common


def _capital_for(spec: dict) -> int:
    payroll = sum(row["monthly_salary_fen"] for row in spec["positions"])
    # Pre-generation coverage is conservative and deterministic.  It is based
    # on the contractual expenditure ceiling, not any future cash curve.
    planned_service = sum(row.get("monthly_service_fen", 0) for row in spec.get("maintenance", []))
    # 3,000,000 fen is the fixed 1,800,000-fen lease plus 1,200,000-fen
    # support service already in the template; do not count rent twice.
    monthly_commitment = payroll + 3_000_000 + planned_service
    coverage_months = spec.get("capital_coverage_months", 12)
    if type(coverage_months) is not int or not 6 <= coverage_months <= 18:
        raise ValueError("期初资金覆盖期必须为合同支出的6至18个月")
    return max(72_000_000, 6_000_000 + monthly_commitment * coverage_months)


def _structural_features(spec: dict, facts: dict) -> dict:
    book = facts["contract_book"]
    inventory = facts["inventory"]
    return dict(mechanism_id=spec["mechanism_id"], customer_terms=sorted((c["start_month"], c["end_month"], c["credit_days"], len(c.get("delivery_plan", []))) for c in spec["customers"]),
                position_terms=sorted((p["start_month"], p["end_month"], p["monthly_capacity_units"]) for p in spec["positions"]),
                maintenance=tuple((m["start_month"], m["end_month"]) for m in spec.get("maintenance", [])),
                stock_cover_months=spec["stock_cover_months"], purchase_months=tuple(row["month"] for row in inventory if row["purchase_fen"]),
                consumed_without_purchase_months=tuple(row["month"] for row in inventory if row["consumed_fen"] and not row["purchase_fen"]),
                customer_count=len({row["customer_id"] for row in book["orders"]}))


def _economic_features(spec: dict) -> dict:
    """Comparable facts intentionally excluding IDs, seeds and calendar shifts."""
    return dict(mechanism_id=spec["mechanism_id"], unit_price_fen=spec["unit_price_fen"], unit_cost_fen=spec["unit_cost_fen"],
                stock_cover_months=spec["stock_cover_months"], supplier_credit_days=spec["supplier_credit_days"],
                customer_contracts=sorted((len(c.get("delivery_plan", [])), c["monthly_units"], c["credit_days"],
                                           tuple(m["units"] for m in c.get("delivery_plan", []))) for c in spec["customers"]),
                position_capacity_and_term=sorted((p["monthly_capacity_units"], p["monthly_salary_fen"],
                                                   _month_position(p["end_month"]) - _month_position(p["start_month"]) + 1) for p in spec["positions"]),
                maintenance_duration_and_cost=sorted((_month_position(m["end_month"]) - _month_position(m["start_month"]) + 1,
                                                       m["monthly_service_fen"]) for m in spec.get("maintenance", [])))


def _core_metrics(target: dict | str) -> dict:
    """Frozen acceptance metadata; it never feeds contract quantities."""
    if isinstance(target, str):
        return dict(stages=[], turns=[], note="调用方未提供目标数值；不得从此预算或未来余额补填")
    slopes = list(target.get("monthly_net_cash_slope_cny", []))
    breaks = list(target.get("breakpoint_months", []))
    volatility = target.get("volatility", "中")
    # The actual hinge assessor observes month-end coordinates 0..23.  The
    # final amplitude therefore ends at 23, rather than inventing a Jan-2026
    # point at 24.  Main stages have a fixed minimum of three full months;
    # allowed turn movement is assessed independently in the gate.
    observed_bounds = [0, *breaks, 23]
    stages = [dict(slope_cny_per_month=slope,
                   peak_or_valley_delta_cny=abs(slope) * (observed_bounds[i + 1] - observed_bounds[i]),
                   min_stage_months=3,
                   # Declare the original category.  The accepting gate, not
                   # this metadata, applies its one 20% tolerance exactly once.
                   D_reference=[.01, .04] if volatility == "低" else [.04, .10])
              for i, slope in enumerate(slopes)]
    target_id = target.get("target_id", "")
    turns = []
    for index, month in enumerate(breaks):
        before, after = slopes[index], slopes[index + 1]
        # Every turn is inferred from its adjacent frozen stage signs, rather
        # than applying one target-wide label to a multi-turn profile.
        upward = before < 0 < after
        slow = target_id in {"T09", "T10", "T13", "T14"}
        kind = "U" if slow and upward else "inverted_U" if slow else "V" if upward else "inverted_V"
        turns.append(dict(month=month, width_months=3 if kind in {"U", "inverted_U"} else 1, kind=kind))
    # This nested object is deliberately the strict gate schema: only stages
    # and turns.  Context belongs in the outer sidecar, otherwise a formal
    # profile consumer must reject it as an unknown metric.
    return dict(stages=stages, turns=turns)


def _percentile(values: list[float], q: float) -> float:
    """Linear percentile compatible with numpy's default quantile method."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("无法为零条日余额计算离散度")
    position = (len(ordered) - 1) * q
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


# 按显式起点计算日内OLS坐标，默认保持旧2024起点。
def _daily_coordinate(day: date, start: date = date(2024, 1, 1)) -> float:
    """The gate's x coordinate: every calendar month end is exactly 0..23."""
    # 从显式起点计算当前日期的合同月偏移。
    month_index = (day.year - start.year) * 12 + day.month - start.month
    next_month = (day.replace(day=28) + timedelta(days=4)).replace(day=1)
    month_length = (next_month - day.replace(day=1)).days
    return month_index - 1 + (day.day - 1) / max(1, month_length - 1)


def _ols(values: list[float]) -> tuple[float, float]:
    """Intercept and slope of the frozen 24 month-end x=0..23 regression."""
    xbar, ybar = (len(values) - 1) / 2, mean(values)
    denominator = sum((index - xbar) ** 2 for index in range(len(values)))
    slope = sum((index - xbar) * (value - ybar) for index, value in enumerate(values)) / denominator
    return ybar - slope * xbar, slope


def _planned_rows(daily: list[int], capital_fen: int, start: date) -> list[dict]:
    cash = capital_fen
    rows = []
    for offset, amount in enumerate(daily):
        cash += amount
        rows.append({"calendar_date": (start + timedelta(days=offset)).isoformat(), "ending_balance_cny": f"{cash / 100:.2f}"})
    return rows


# 使用显式合同起点完成单段预算预检。
def _single_stage_preflight(rows: list[dict], capital_cny: float, core_metrics: dict, target: dict, start: date) -> dict:
    """Same 24-month OLS/D definition as the no-turn formal acceptance path."""
    month_ends = [index for index, row in enumerate(rows)
                  if index + 1 == len(rows) or rows[index + 1]["calendar_date"][5:7] != row["calendar_date"][5:7]]
    balances = [float(rows[index]["ending_balance_cny"]) for index in month_ends]
    intercept, slope = _ols(balances)
    # 残差坐标使用调用方日期合同，避免2026预算落入旧坐标。
    residuals = [float(row["ending_balance_cny"]) - (intercept + slope * _daily_coordinate(date.fromisoformat(row["calendar_date"]), start))
                 for row in rows]
    median = _percentile(residuals, .50)
    d_value = _percentile([abs(value - median) for value in residuals], .90) / capital_cny
    expected = core_metrics["stages"][0]["slope_cny_per_month"]
    platform = target.get("target_id") in {"T03", "T04"}
    direction = abs(slope) <= capital_cny * .006 if platform else (slope >= 0) == (expected >= 0)
    stage = dict(expected_slope_cny_per_month=expected, planned_slope_cny_per_month=round(slope, 2),
                 direction_match=direction, OLS_method="24_month_end_x0_to23")
    return dict(acceptance_version="single_stage_ols_d_v1", stages=[stage], planned_D=round(d_value, 8),
                planned_stage_D=[round(d_value, 8)], target_screen_only=dict(stages=[stage], all_directions_match=direction))


def _multistage_preflight(rows: list[dict], capital_cny: float, core_metrics: dict, target: dict) -> dict:
    """Delegate the planned measurement to the public formal hinge assessor."""
    try:
        from controlled_sample_acceptance import assess_multistage_daily_rows
    except ImportError as exc:
        raise RuntimeError("多段预算前筛需要受控形状验收器") from exc
    profile = {"initial_capital_cny": capital_cny, "core_metrics": core_metrics}
    result = assess_multistage_daily_rows(rows, target, profile)
    slopes = result.get("stage_ols_slopes_cny_per_month", [])
    stages = [dict(expected_slope_cny_per_month=spec["slope_cny_per_month"], planned_slope_cny_per_month=actual,
                   direction_match=(actual >= 0) == (spec["slope_cny_per_month"] >= 0), OLS_method="continuous_smooth_hinge_month_end_v2")
              for actual, spec in zip(slopes, core_metrics["stages"], strict=True)]
    return dict(acceptance_version=result.get("acceptance_version", "continuous_smooth_hinge_month_end_v2"),
                stages=stages, planned_D=round(float(result.get("D", float("inf"))), 8),
                planned_stage_D=[round(float(value), 8) for value in result.get("stage_D", [])],
                fitted_turn_months=result.get("fitted_turn_months", []),
                fitted_transition_width_months=result.get("fitted_transition_width_months", []),
                target_screen_only=dict(stages=stages, all_directions_match=all(row["direction_match"] for row in stages),
                                        hinge_passed=bool(result.get("passed"))))


def _planned_repetition(raw: dict) -> dict:
    """Run the frozen waveform rule on compiled contract events before central.

    These are budget-event rows only, so a pass is explicitly not an actual
    flow admission.  It does, however, catches a mechanically recurring order
    cadence before a costly central run.
    """
    try:
        from curve_repetition_check import check
    except ImportError as exc:
        raise RuntimeError("预算前筛需要曲线内重复检查器") from exc
    # 从已编译画像读取显式合同起点，旧画像未给值时保留原2024默认。
    start = date.fromisoformat(raw.get("start_date", "2024-01-01"))
    # 从已编译画像读取显式合同终点，旧画像未给值时保留原2025默认。
    end = date.fromisoformat(raw.get("end_date", "2025-12-31"))
    # 用闭区间自然日数适配闰年，避免2025--2026合同被错误扩成731天。
    day_count = (end - start).days + 1
    by_day = {}
    for event in raw["operating"]["business_nodes"]["settlements"]:
        when = event["due_date"]
        signed = event["amount_fen"] if event["category"] == "collection" else -event["amount_fen"]
        by_day[when] = by_day.get(when, 0) + signed
    daily = []
    # 逐一覆盖画像中的真实合同日，不再写死旧2024--2025的731天。
    for offset in range(day_count):
        when = (start + timedelta(days=offset)).isoformat()
        signed = by_day.get(when, 0)
        daily.append({"calendar_date": when, "inflow_cny": _money(max(0, signed)), "outflow_cny": _money(max(0, -signed))})
    result = check(daily)
    return dict(passed=bool(result["passed"]), status=result["status"], actual_days=result["actual_days"],
                monthly_evidence=result["monthly_evidence"], rolling_evidence=result["rolling_evidence"],
                whole_curve_density=result["whole_curve_density"], scope="planned_contract_event_preflight_only")


# 从显式日期合同的已签事件计算预算预检，不产生中央流水。
def _budget_response(raw: dict, target: dict | str, start: date = date(2024, 1, 1), end: date = date(2025, 12, 31)) -> dict:
    """Cheap, event-only forecast for selecting a budget before central runs."""
    # 重建本次合同期的月名，避免使用固定MONTHS常量。
    months = tuple(f"{start.year + (start.month - 1 + index) // 12}-{(start.month - 1 + index) % 12 + 1:02d}"
                   for index in range((end.year - start.year) * 12 + end.month - start.month + 1))
    # 初始化每个合同月的净现金累计。
    month_net = {month: 0 for month in months}
    # 初始化显式合同期内的逐日净现金累计。
    daily = [0] * ((end - start).days + 1)
    # 逐个合同结算事件归集月度和日度净额。
    for event in raw["operating"]["business_nodes"]["settlements"]:
        month = event["due_date"][:7]
        if month in month_net:
            signed = event["amount_fen"] if event["category"] == "collection" else -event["amount_fen"]
            month_net[month] += signed
            index = (date.fromisoformat(event["due_date"]) - start).days
            if 0 <= index < len(daily):
                daily[index] += signed
    capital = int(round(float(raw["initial_capital"]["amount_cny"]) * 100))
    balances, running = [], capital
    for month in months:
        running += month_net[month]
        balances.append(running / 100)
    daily_balances, cash = [], capital
    for value in daily:
        cash += value
        daily_balances.append(cash / 100)
    _, ols = _ols(balances)
    response = dict(month_net_cny=[round(month_net[m] / 100, 2) for m in months], month_end_operating_cash_cny=[round(v, 2) for v in balances],
                    overall_ols_cny_per_month=round(ols, 2), minimum_planned_cash_cny=round(min(daily_balances), 2),
                    planned_capital_adequate=min(daily_balances) >= 0, generated_from="冻结合同到期日与金额；非中央流水")
    if isinstance(target, dict) and target.get("monthly_net_cash_slope_cny"):
        core_metrics = _core_metrics(target)
        rows = _planned_rows(daily, capital, start)
        measured = (_multistage_preflight(rows, capital / 100, core_metrics, target)
                    if target.get("breakpoint_months") else _single_stage_preflight(rows, capital / 100, core_metrics, target, start))
        response.update(measured)
    response["preflight"] = dict(passed=False, reasons=["planned_repetition_not_evaluated"],
                                 planned_slope_cny_per_month=[row["planned_slope_cny_per_month"] for row in response.get("stages", [])],
                                 planned_D=response.get("planned_D"), repetition=None)
    return response


def _revision(spec: dict, revision: int) -> tuple[dict, dict]:
    if type(revision) is not int or revision not in {0, 1, 2}:
        raise ValueError("修订仅允许0至2；每版必须有单一业务变量组依据")
    revised = deepcopy(spec)
    if revision == 0:
        return revised, dict(revision=0, variable_group="none", changed_fields=[], economic_reason="初始事前预算")
    # One variable group only: delivery volume under a frozen, explicitly
    # bounded contract adjustment clause.
    # Dates, terms, identities, prices, cost, capital and the frozen seed do
    # not change.  This is the documented response to a magnitude mismatch.
    multiplier = (90, 110)[revision - 1]
    for customer in revised["customers"]:
        clause = customer.get("quantity_variation_clause_percent", 15)
        if not 0 < clause <= 20 or abs(multiplier - 100) > clause:
            raise ValueError("交付数量修订超出合同事前约定范围")
        customer["monthly_units"] = max(1, customer["monthly_units"] * multiplier // 100)
        for milestone in customer.get("delivery_plan", []):
            milestone["units"] = max(1, milestone["units"] * multiplier // 100)
    return revised, dict(revision=revision, variable_group="delivery_volume", changed_fields=["customers[].delivery_plan[].units"],
                         multiplier_percent=multiplier, economic_reason="仅按已签合同的数量变更条款修订交付规模；其他合同事实冻结")


# 构建带显式日期合同的受控预算，默认参数保留历史行为。
def build_candidate(target: dict | str, candidate_index: int, revision: int, base_spec: dict | None = None,
                    *, start_date: str = "2024-01-01", end_date: str = "2025-12-31",
                    prediction_cutoff: str = "2025-12-31"):
    """Return ``raw_profile, entity_facts, mechanism_spec, change_record``.

    ``base_spec`` is an archived revision-zero mechanism specification.  It
    may be supplied by a resume controller; identity fields are verified so a
    revision cannot silently become a different enterprise.
    """
    # 先校验日期合同，后续迁移和预算计算共用该边界。
    start, end = _schedule(start_date, end_date, prediction_cutoff)
    target_id = target if isinstance(target, str) else target.get("target_id")
    candidate_id, seed = _candidate_identity(target_id, candidate_index)
    spec = deepcopy(base_spec) if base_spec is not None else _mechanism_spec(target_id, candidate_index, seed)
    if spec.get("candidate_seed", seed) != seed or spec.get("mechanism_id") != TARGET_MECHANISMS[target_id]:
        raise ValueError("恢复的预算规格与冻结候选身份或目标机制不一致")
    spec["candidate_seed"] = seed
    spec["candidate_id"] = candidate_id
    # A mechanism family must remain comparable across targets; target labels
    # may not be used to split a shared factory and leak it across partitions.
    spec["family_id"] = f"controlled_family_{spec['mechanism_id']}"
    revised, change_record = _revision(spec, revision)
    raw, facts = plan_mechanism(_template(candidate_id, seed, _capital_for(revised)), revised)
    # 非默认合同期才迁移冻结的2024—2025合同事实。
    if (start, end) != (date(2024, 1, 1), date(2025, 12, 31)):
        # 同步迁移画像、合同事实和输出规格中的日期字段。
        raw, facts, revised = (_rebase_schedule(value, start) for value in (raw, facts, revised))
        # 将顶层经营期间和日历截止日写为调用方输入。
        raw["start_date"], raw["end_date"], raw["calendar"]["prediction_cutoff"] = start_date, end_date, prediction_cutoff
        # 使用中央画像编译器确认所有重建的合同、资产与截止日仍自洽。
        from syn_b1.formal_profile import resolve_profile
        resolve_profile(raw)
    # 基于已迁移的合同日期计算预算预检结果。
    budget_response = _budget_response(raw, target, start, end)
    # 默认期间继续复用历史签名器，保证旧调用输出不变。
    if (start, end) == (date(2024, 1, 1), date(2025, 12, 31)):
        try:
            from shape_contract_search import planned_signature
            signature = planned_signature(raw)
            planned = {"cash_total_cny": round(float(signature["cash"].sum()), 2), "nonzero_days": int((signature["cash"] != 0).sum())}
        except ImportError:  # direct library consumers need no tools path.
            planned = {"cash_total_cny": sum(e["amount_fen"] if e["category"] == "collection" else -e["amount_fen"] for e in raw["operating"]["business_nodes"]["settlements"]) / 100}
    # 新日期合同不调用固定2024窗口的旧签名器。
    else:
        # 旧签名器只接受2024起点；此处保留同一合同事件汇总，避免静默空序列。
        # 按真实到期日聚合同一日的合同净现金。
        event_net = {}
        # 遍历全部已签结算事件，保留收入与支出的方向。
        for event in raw["operating"]["business_nodes"]["settlements"]:
            # 客户回款为正，其它结算为负。
            signed = event["amount_fen"] if event["category"] == "collection" else -event["amount_fen"]
            # 合并同一到期日的多笔合同现金。
            event_net[event["due_date"]] = event_net.get(event["due_date"], 0) + signed
        # 保存合同事件总额和真实非零日数，供编排器后续复用重复检查器。
        planned = {"cash_total_cny": sum(event_net.values()) / 100,
                   "nonzero_days": sum(value != 0 for value in event_net.values()),
                   "method": "contract-event-total; 2026日期签名由S5编排器复用既有重复检查器核验"}
    features = _structural_features(revised, facts)
    feature_hash = sha256(dumps(features, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    economic_features = _economic_features(revised)
    economic_fingerprint = sha256(dumps(economic_features, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    mechanism_spec = dict(revised)
    mechanism_spec.update(candidate_id=candidate_id, family_id=spec["family_id"], target_id=target_id, seed=seed,
                          target_used_only_for_mechanism_selection=True, structural_features=features, structural_hash=feature_hash,
                          economic_features=economic_features, economic_fingerprint=economic_fingerprint,
                          planned_signature=planned, budget_response=budget_response, core_metrics=_core_metrics(target),
                          # 记录日期合同，供S5在生成前核对参数指纹。
                          date_contract={"start_date": start_date, "end_date": end_date, "prediction_cutoff": prediction_cutoff},
                          capital_basis="12个月合同支出覆盖，初始生成前冻结；不允许中途补资")
    facts.update(candidate_id=candidate_id, family_id=spec["family_id"], mechanism_id=revised["mechanism_id"],
                 inventory_identity_exact=all(row["ending_fen"] >= 0 for row in facts["inventory"]),
                 structural_hash=feature_hash, economic_fingerprint=economic_fingerprint)
    return raw, facts, mechanism_spec, change_record


def search_budget_candidates(target: dict, *, limit: int = 162) -> list[dict]:
    """Enumerate the finite real-business grid and return cheap preflight hits.

    The loss is applied only after compiling contracts and their due dates.  It
    never writes a ledger or derives an order amount from a desired balance.
    """
    if type(limit) is not int or not 1 <= limit <= 162:
        raise ValueError("预算搜索上限必须为1至162个冻结经营组合")
    rows = []
    for index in range(limit):
        raw, _, spec, _ = build_candidate(target, index, 0)
        screen = spec["budget_response"].get("target_screen_only")
        if not screen:
            continue
        stages = screen["stages"]
        capital = float(raw["initial_capital"]["amount_cny"])
        platform = target.get("target_id") in {"T03", "T04"}
        within = [(abs(row["planned_slope_cny_per_month"]) <= capital * .006 if platform
                   else abs(row["planned_slope_cny_per_month"] - row["expected_slope_cny_per_month"]) <= abs(row["expected_slope_cny_per_month"]) * .20)
                  for row in stages]
        declared_d = [stage["D_reference"] for stage in _core_metrics(target)["stages"]]
        planned_d = spec["budget_response"].get("planned_stage_D", [spec["budget_response"]["planned_D"]])
        d_pass = len(planned_d) == len(declared_d) and all(low * .8 <= actual <= high * 1.2
                                                          for actual, (low, high) in zip(planned_d, declared_d, strict=True))
        capital_pass = spec["budget_response"]["planned_capital_adequate"]
        if screen["all_directions_match"] and all(within) and d_pass and capital_pass:
            repetition = _planned_repetition(raw)
            if not repetition["passed"]:
                continue
            preflight = dict(passed=True, reasons=[],
                             planned_slope_cny_per_month=[row["planned_slope_cny_per_month"] for row in stages],
                             planned_D=spec["budget_response"]["planned_D"], repetition=repetition)
            spec["budget_response"]["preflight"] = preflight
            rows.append(dict(candidate_index=index, candidate_id=spec["candidate_id"], family_id=spec["family_id"],
                             mechanism_id=spec["mechanism_id"], economic_fingerprint=spec["economic_fingerprint"],
                             budget_response=spec["budget_response"], planned_repetition=repetition,
                             planned_cumulative_similarity_warning="由控制器以冻结实际基线作预警；预算签名不构成正式排重",
                             preflight_status="planned_budget_direction_amplitude_D_repetition_pass_not_formal"))
    return rows
