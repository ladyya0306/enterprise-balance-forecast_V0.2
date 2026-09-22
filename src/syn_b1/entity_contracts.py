"""目标无关的实体合同编译器；只编译事前订单、到货及岗位，不接收余额目标。"""
# 原参数保留为独立副本，旧样本不可原地更改。
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal

from syn_b1.formal_profile import resolve_profile


def compile_entity_profile(template, book):
    # 限定显式协议，未知字段不得静默忽略。
    if set(book) != {"schema", "orders", "purchases", "positions"} or book["schema"] != "entity_contracts_v1":
        raise ValueError("实体合同协议或字段无效")
    raw = deepcopy(template)
    plan = raw["operating"]["business_nodes"]
    if plan["version"] != "physical_contract_business_v1":
        raise ValueError("当前实体编译器只支持既有物理精确合同模式，其他模式不能自动降级")
    by_month = {n["month"]: n for n in plan["nodes"]}
    grouped = {m: dict(orders=[], purchases=[], positions=[]) for m in by_month}
    used, links = set(), []
    # 固定资产、租赁、支持费用沿用已声明合同；三类被编译来源全部重建。
    events = [e for e in plan["settlements"] if e["category"] not in {"collection", "supplier_payment", "payroll_payment"}]

    def integer(value, minimum=0):
        if type(value) is not int or value < minimum:
            raise ValueError("实体数量、金额、账期须为合法整数")
        return value

    def add(kind, row, anchor, amount, entity):
        # 到期日期只能由真实合成交付/到货日和冻结账期推导。
        due = anchor + timedelta(days=integer(row["credit_days"]))
        eid = f"entity-{kind}-{row['id']}"
        events.append(dict(event_id=eid, category=kind, recognition_month=anchor.isoformat()[:7], asset_id="",
                           due_date=due.isoformat(), amount_fen=amount, note_cn="已确认业务合同结算"))
        links.append(dict(event_id=eid, entity_id=entity, source_id=row["id"], source_date=anchor.isoformat(), credit_days=row["credit_days"]))

    for kind, required, date_field in (
        ("orders", {"id", "customer_id", "delivery_date", "units", "unit_price_fen", "unit_cost_fen", "credit_days"}, "delivery_date"),
        ("purchases", {"id", "supplier_id", "arrival_date", "amount_fen", "credit_days"}, "arrival_date"),
        ("positions", {"id", "start_month", "end_month", "monthly_salary_fen", "monthly_capacity_units", "pay_day"}, None),
    ):
        for row in book[kind]:
            if set(row) != required or not isinstance(row["id"], str) or not row["id"] or row["id"] in used:
                raise ValueError("实体字段无效或来源编号重复")
            used.add(row["id"])
            if kind == "positions":
                start, end = row["start_month"], row["end_month"]
                if start not in by_month or end not in by_month or start > end or not 1 <= integer(row["pay_day"]) <= 28:
                    raise ValueError("岗位必须声明观察期内完整月份及有效发薪日；暂不支持日内入离职折算")
                integer(row["monthly_salary_fen"], 1); integer(row["monthly_capacity_units"])
                for month in by_month:
                    if start <= month <= end: grouped[month][kind].append(row)
                continue
            anchor = date.fromisoformat(row[date_field])
            if anchor.isoformat()[:7] not in by_month:
                raise ValueError("订单交付或采购到货超出声明月份")
            grouped[anchor.isoformat()[:7]][kind].append(row)
            if kind == "orders":
                amount = integer(row["units"], 1) * integer(row["unit_price_fen"], 1)
                if integer(row["unit_cost_fen"]) >= row["unit_price_fen"]:
                    raise ValueError("本模式不支持无独立依据的单位成本不低于售价")
                entity, category = row["customer_id"], "collection"
            else:
                amount = integer(row["amount_fen"], 1)
                entity, category = row["supplier_id"], "supplier_payment"
            if not isinstance(entity, str) or not entity.strip():
                raise ValueError("客户或供应商身份缺失")
            add(category, row, anchor, amount, entity)
    inventory = 0
    money = lambda fen: f"{Decimal(fen) / 100:.2f}"
    for month, node in by_month.items():
        orders, purchases, positions = (grouped[month][k] for k in ("orders", "purchases", "positions"))
        prices = {(r["unit_price_fen"], r["unit_cost_fen"]) for r in orders}
        if len(prices) > 1:
            raise ValueError("当前物理节点要求同月同标准品价格成本一致；多SKU不得伪装加权标准件")
        price, cost = next(iter(prices), (0, 0))
        units = sum(r["units"] for r in orders)
        if units > sum(r["monthly_capacity_units"] for r in positions):
            raise ValueError("交付超过在岗人员声明能力")
        inventory += sum(r["amount_fen"] for r in purchases) - units * cost
        if inventory < 0:
            raise ValueError("材料到货不足以支持月度耗用，不能虚构库存")
        node.update(sales_units=units, unit_price_cny=money(price), unit_variable_cost_cny=money(cost),
                    target_inventory_cny=money(inventory), payroll_cny=money(sum(r["monthly_salary_fen"] for r in positions)))
        for row in positions:
            eid = f"entity-payroll-{month}-{row['id']}"
            events.append(dict(event_id=eid, category="payroll_payment", recognition_month=month, asset_id="",
                               due_date=f"{month}-{row['pay_day']:02d}", amount_fen=row["monthly_salary_fen"], note_cn="在岗人员当月工资"))
            links.append(dict(event_id=eid, entity_id=row["id"], source_id=row["id"], start_month=row["start_month"], end_month=row["end_month"]))
    plan["settlements"] = events
    # 财务恒等式、合同来源、设备能力继续由中央唯一规则检查。
    resolve_profile(raw)
    return raw, dict(schema="entity_event_lineage_v1", links=links, inventory_granularity="monthly_only", target_used_to_construct_budget=False)
