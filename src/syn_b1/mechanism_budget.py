"""将事前经营原因编译为物理预算和合同，不计算或修改实际银行余额。"""
# 金额与随机参数均在看实际流水之前确定。
from copy import deepcopy
from random import Random

# 复用中央参数外壳，不另建流水引擎。
from syn_b1.formal_profile import default_profile


def mechanism_budget(spec, variant, config, *, counterfactual=False):
    # 同一机制和变体的反事实共用抽样参数，不能更换种子。
    rng = Random(spec["seed"] + variant)
    months = [f"{2024 + i // 12}-{i % 12 + 1:02d}" for i in range(24)]
    mid = spec["id"]
    if mid not in {"cohort_expiry", "margin_compression", "committed_inventory", "team_consolidation"}:
        raise ValueError("未实现的经营机制，不允许仅用说明充当支持")
    # 所有变体同一机制共用保守家族，样本编号不同不等于经营家族不同。
    sid = f"causal_{mid}_{variant:02d}"
    p = default_profile(sid, config["version"])
    p.update(schema_version="syn_b1_formal_profile_2_0", start_date="2024-01-01", end_date="2025-12-31", random_seed=spec["seed"] + variant)
    p["calendar"]["prediction_cutoff"] = "2025-12-31"
    p["initial_capital"]["amount_cny"] = config["initial_capital_cny"]
    op = p["operating"]
    # 删除旧比例预算；新预算只有一组显式经营节点。
    for key in ("base_monthly_sales_cny", "sales_monthly_growth_percent", "sales_month_weights_percent", "gross_margin_percent", "monthly_payroll_cny", "monthly_rent_and_utilities_cny", "monthly_other_operating_expense_cny"):
        op.pop(key)
    for key in ("collection_schedule", "supplier_payment_schedule", "payroll_payment_schedule", "rent_and_utilities_payment_schedule", "other_operating_expense_payment_schedule"):
        op[key] = [{"delay_days": 0, "share_percent": "100"}]
    op["fixed_assets"] = [{"event_id": "EQ01", "asset_type": "equipment", "purchase_month": months[0], "purchase_amount_cny": "90000.00", "payment_schedule": [{"delay_days": 0, "share_percent": "100"}], "ready_for_use_date": "2024-01-01", "useful_life_months": 60, "residual_value_percent": "5"}]
    # 独立抽取业务变量；没有按机制序号计算曲线或循环取余模板。
    initial_units = rng.randint(*spec["units"])
    initial_payroll = rng.randint(*spec["payroll_cny"]) * 100
    lost_per_month = rng.randint(7, 10)
    expiry_months = sorted(rng.sample(range(5, 21), 4))
    expired_units = [rng.randint(18, 35) for _ in expiry_months]
    price_step, cost_step = rng.randint(200, 350), rng.randint(100, 220)
    headcount = 20
    expiry_jobs = [6, 12, 18]
    # 本次固定共同日历，不能靠日期扰动宣称机制不同。
    dates = config["calendar"]
    nodes, events, facts = [], [], []
    inventory_before = 0
    money = lambda value: f"{value // 100}.{value % 100:02d}"

    def event(category, month, amount, suffix="base", asset=""):
        # 每笔金额对应真实预算来源；本探针无回款宽限或期外付款。
        if amount:
            day = 3 if asset else dates[category]
            events.append(dict(event_id=f"{month}-{category}-{suffix}", category=category, recognition_month=month,
                               asset_id=asset, due_date=f"{month}-{day:02d}", amount_fen=amount,
                               note_cn="设备购买付款" if asset else f"{spec['name']}：当月已确认合同结算"))

    for m, month in enumerate(months):
        # 四种原因分别作用于数量、单位毛利、采购承诺和固定人工。
        units, price, cost, staff, payroll = initial_units, 45000, 22000, headcount, initial_payroll
        rent, other = 2600000, 4800000
        if mid == "cohort_expiry":
            units -= 0 if counterfactual else sum(q for when, q in zip(expiry_months, expired_units) if m >= when)
        elif mid == "margin_compression":
            price -= 0 if counterfactual else m * price_step
            cost += 0 if counterfactual else m * cost_step
        elif mid == "committed_inventory":
            units -= m * lost_per_month
            price, cost, other = 41000, 20000, 4400000
        elif mid == "team_consolidation":
            units -= m * lost_per_month
            staff -= 0 if counterfactual else 2 * sum(m >= when for when in expiry_jobs)
            payroll = initial_payroll * staff // headcount
            other = 5000000
        # 显式人员与设备能力双约束，不因缩编而默认为无限生产能力。
        if units <= 0 or price <= cost or units > min(2000, staff * 55):
            raise ValueError("经营参数违反正毛利、销量或人员产能约束")
        material = units * cost
        if mid == "committed_inventory":
            # 锁量合同驱动采购，不是先决定余额再反推库存。
            purchase = material if counterfactual else initial_units * cost
            inventory = inventory_before + purchase - material
            if inventory > 8 * material:
                raise ValueError("承诺采购积压超过八个月耗用，需另审库存用途")
        else:
            # 非锁量机制维持少量安全库存，不引入额外采购承诺。
            inventory = material // 10
            purchase = material + inventory - inventory_before
        nodes.append(dict(month=month, stage_id=f"{mid}-{m+1}", business_reason=spec["counterfactual"] if counterfactual else spec["cause"],
                          sales_units=units, unit_price_cny=money(price), unit_variable_cost_cny=money(cost),
                          target_inventory_cny=money(inventory), payroll_cny=money(payroll), rent_cny=money(rent), other_expense_cny=money(other)))
        amounts = dict(collection=units * price, supplier_payment=purchase, payroll_payment=payroll, rent_payment=rent, other_expense_payment=other)
        for category, amount in amounts.items():
            event(category, month, amount)
        facts.append(dict(month=month, units=units, unit_price_fen=price, unit_cost_fen=cost, purchase_fen=purchase,
                          ending_inventory_fen=inventory, staff=staff, payroll_fen=payroll))
        inventory_before = inventory
    event("fixed_asset_payment", months[0], 9000000, asset="EQ01")
    op["business_nodes"] = dict(version="physical_contract_business_v1", nodes=nodes, settlements=events,
                                asset_capacities=[dict(event_id="EQ01", monthly_units=2000)],
                                capacity_basis="自有设备月产能2000标准件；每名在岗人员月能力55件；人员能力在预算编译时逐月验证。")
    basis = dict(mechanism_id=mid, family_id=f"mechanism_family_{mid}", sample_id=sid, counterfactual=counterfactual,
                 cause=spec["cause"], counterfactual_rule=spec["counterfactual"], source_parameters=dict(initial_units=initial_units,
                 initial_payroll_fen=initial_payroll, lost_per_month=lost_per_month, expiry_months=expiry_months,
                 expired_units=expired_units, price_step_fen=price_step, cost_step_fen=cost_step, job_expiry_months=expiry_jobs), monthly_facts=facts,
                 contract_status="事前声明的合成合同，不声称存在真实外部合同", split_rule="同机制变体保守归一业务家族，不跨训练开发划分")
    return p, basis


def causal_evidence(raw, baseline):
    # 反事实只检查真正进入中央输入的数值字段，不比较说明和身份。
    left = raw["operating"]["business_nodes"]
    right = baseline["operating"]["business_nodes"]
    fields = ("sales_units", "unit_price_cny", "unit_variable_cost_cny", "target_inventory_cny", "payroll_cny")
    changes = {key: [a["month"] for a, b in zip(left["nodes"], right["nodes"], strict=True) if a[key] != b[key]] for key in fields}
    # 合同身份和日期不变时，金额必须受到机制影响。
    amount_changes = [a["event_id"] for a, b in zip(left["settlements"], right["settlements"], strict=True) if a["amount_fen"] != b["amount_fen"]]
    if not any(changes.values()) or not amount_changes:
        raise ValueError("机制没有进入经营数值及合同金额，仅有故事标签")
    return dict(changed_fields={k:v for k,v in changes.items() if v}, changed_contract_amount_count=len(amount_changes),
                same_contract_dates=[e["due_date"] for e in left["settlements"]] == [e["due_date"] for e in right["settlements"]])


def replenishment_entity_budget(variant, seed_base):
    # 客户库存触发订单；没有读取目标斜率、历史或实际余额来选择日期。
    from datetime import date, timedelta
    rng = Random(seed_base + variant)
    spec = dict(id="cohort_expiry", seed=seed_base, units=[640, 690], payroll_cny=[88000, 94000],
                name="装机存量收缩下按耗用补货", cause="客户设备逐季退出，备件按实际耗用补货，固定场地与服务团队持续", counterfactual="不适用于本实体库存模型")
    config = dict(version="entity_replenishment_v1", initial_capital_cny="1800000.00", calendar=dict(collection=21, supplier_payment=18, payroll_payment=7, rent_payment=3, other_expense_payment=25))
    raw, _ = mechanism_budget(spec, variant, config)
    raw["sample_id"] = f"entity_t05_replenishment_{variant:02d}"
    # 人员按明确岗位连续在岗，固定薪酬不随库存触发事件随机改变。
    book = dict(schema="entity_contracts_v1", orders=[], purchases=[], positions=[dict(id=f"J{i}", start_month="2024-01", end_month="2025-12", monthly_salary_fen=1050000,
                monthly_capacity_units=250, pay_day=7) for i in range(6)])
    customers = [dict(id=f"C{i}", installed_units=rng.randint(70, 100), stock=rng.randint(12, 25), reorder_point=rng.randint(5, 10),
                     target_stock=rng.randint(30, 50), credit_days=rng.choice([3, 7, 14]), quarterly_retirements=rng.randint(3, 6)) for i in range(8)]
    source = deepcopy(customers)
    day, end = date(2024, 1, 4), date(2025, 12, 30)
    sequence = 0
    while day <= end:
        quarter = ((day.year - 2024) * 12 + day.month - 1) // 3
        for customer in customers:
            # 客户耗用是设备存量与独立需求冲击的结果，不给银行余额加噪声。
            active = max(20, customer["installed_units"] - quarter * customer["quarterly_retirements"])
            usage = rng.randint(0, 4) * active // customer["installed_units"]
            customer["stock"] -= usage
            if customer["stock"] <= customer["reorder_point"]:
                quantity = customer["target_stock"] - customer["stock"]
                sequence += 1
                customer["stock"] += quantity
                oid = f"O{sequence:05d}"
                book["orders"].append(dict(id=oid, customer_id=customer["id"], delivery_date=day.isoformat(), units=quantity,
                                           unit_price_fen=45000, unit_cost_fen=21000, credit_days=customer["credit_days"]))
                # 本试点按单取得通用件并当日完成适配交付；供应合同5天付款。
                book["purchases"].append(dict(id=f"P{sequence:05d}", supplier_id="标准件供应商", arrival_date=day.isoformat(), amount_fen=quantity * 21000, credit_days=5))
        day += timedelta(days=1)
    return raw, book, dict(family_id="installed_base_consumption_replenishment", customers=source,
                          mechanism="装机存量退出→实际耗用减少→库存触发订单减少；同日到货适配交付，账期逐客户事前声明",
                          customer_inventory_assumption="客户库存只用于触发其订单，不混入本企业会计库存", seed=seed_base + variant)
