"""Compile finite customer lifetimes, stock replenishment and planned outages.

No target slope, target balance, noise or random seed enters this planner.
"""
from copy import deepcopy
from decimal import Decimal

from syn_b1.entity_contracts import compile_entity_profile


def plan_mechanism(template, specification):
    raw = deepcopy(template)
    nodes = raw['operating']['business_nodes']['nodes']
    months = [n['month'] for n in nodes]
    price, cost = specification['unit_price_fen'], specification['unit_cost_fen']
    if not (type(price) is int and type(cost) is int and 0 < cost < price):
        raise ValueError('单位售价和材料成本必须为正整数分，且材料成本低于售价')
    cover = specification['stock_cover_months']
    if type(cover) is not int or not 1 <= cover <= 3:
        raise ValueError('采购覆盖范围为1至3个月已签订单')
    outages = specification.get('maintenance', [])
    for item in outages:
        if item['start_month'] not in months or item['end_month'] not in months or item['start_month'] > item['end_month']:
            raise ValueError('检修期限无效')
        if type(item['monthly_service_fen']) is not int or item['monthly_service_fen'] < 0:
            raise ValueError('检修服务合同金额无效')
    paused = {m for m in months if any(o['start_month'] <= m <= o['end_month'] for o in outages)}
    book = dict(schema='entity_contracts_v1', orders=[], purchases=[], positions=deepcopy(specification['positions']))
    quantity = dict.fromkeys(months, 0)
    contracts = specification['customers']
    ids = [c['contract_id'] for c in contracts]
    if len(set(ids)) != len(ids): raise ValueError('合同编号重复')
    for c in contracts:
        if c['start_month'] not in months or c['end_month'] not in months or c['start_month'] > c['end_month']:
            raise ValueError('客户合同期限无效')
        if type(c['monthly_units']) is not int or c['monthly_units'] <= 0:
            raise ValueError('客户合同用量无效')
        if not 0 <= c['credit_days'] <= 60 or not 1 <= c['delivery_day'] <= 28:
            raise ValueError('客户交付日或账期超出范围')
        # A contract may define a finite delivery-milestone schedule.  It is
        # business evidence (quantities and dates frozen before generation),
        # not a post-hoc calendar jitter applied to an otherwise monthly
        # template.  The legacy monthly_units form remains supported.
        plan = c.get('delivery_plan')
        if plan is not None:
            if not isinstance(plan, list) or not plan:
                raise ValueError('合同交付计划必须是非空列表')
            scheduled_months = set()
            for milestone in plan:
                month, day, units = milestone.get('month'), milestone.get('delivery_day'), milestone.get('units')
                if month not in months or not c['start_month'] <= month <= c['end_month']:
                    raise ValueError('合同交付计划超出合同期限')
                if not isinstance(day, int) or not 1 <= day <= 28 or not isinstance(units, int) or units <= 0:
                    raise ValueError('合同交付计划日期或数量无效')
                key=(month, day)
                if key in scheduled_months:
                    raise ValueError('同一合同交付计划日期重复')
                scheduled_months.add(key)
                if month in paused:
                    if not c.get('suspend_during_maintenance', False):
                        raise ValueError('停工月份仍有不可暂停订单，须先解决合同履约冲突')
                    continue
                quantity[month] += units
                book['orders'].append(dict(id=f"{c['contract_id']}-{month}-{day:02d}", customer_id=c['customer_id'],
                    delivery_date=f"{month}-{day:02d}", units=units, unit_price_fen=price,
                    unit_cost_fen=cost, credit_days=c['credit_days']))
            continue
        for month in months:
            if not c['start_month'] <= month <= c['end_month']: continue
            if month in paused:
                if not c.get('suspend_during_maintenance', False):
                    raise ValueError('停工月份仍有不可暂停订单，须先解决合同履约冲突')
                continue
            quantity[month] += c['monthly_units']
            book['orders'].append(dict(id=f"{c['contract_id']}-{month}", customer_id=c['customer_id'],
                delivery_date=f"{month}-{c['delivery_day']:02d}", units=c['monthly_units'],
                unit_price_fen=price, unit_cost_fen=cost, credit_days=c['credit_days']))
    stock = 0; inventory = []; service_facts = []
    for i, node in enumerate(nodes):
        month = node['month']; consumption = quantity[month]*cost
        purchase = 0
        if stock < consumption:
            # Purchase ahead against already signed orders; stock may be
            # consumed over later months without new monthly purchases.
            purchase = max(0, sum(quantity[m]*cost for m in months[i:i+cover])-stock)
            book['purchases'].append(dict(id=f"stock-{month}", supplier_id='signed_supplier_01',
                arrival_date=f'{month}-01', amount_fen=purchase, credit_days=specification['supplier_credit_days']))
        stock += purchase-consumption
        if stock < 0: raise ValueError('库存不足')
        inventory.append(dict(month=month, purchase_fen=purchase, consumed_fen=consumption, ending_fen=stock,
                              production_open=month not in paused, units=quantity[month]))
        node['stage_id'] = f"{specification['mechanism_id']}-{month}"
        node['business_reason'] = specification['reason']
        for outage in outages:
            if outage['start_month'] <= month <= outage['end_month'] and outage['monthly_service_fen']:
                amount = outage['monthly_service_fen']
                node['other_expense_cny'] = f"{Decimal(node['other_expense_cny']) + Decimal(amount)/100:.2f}"
                event = dict(event_id=f"maintenance-{outage['contract_id']}-{month}", category='other_expense_payment',
                    recognition_month=month, asset_id='', due_date=f'{month}-20', amount_fen=amount,
                    note_cn='已签检修停工期间的外部维修服务合同')
                raw['operating']['business_nodes']['settlements'].append(event)
                service_facts.append(event)
    compiled, lineage = compile_entity_profile(raw, book)
    assert not any(o['delivery_date'][:7] in paused for o in book['orders'])
    return compiled, dict(specification=deepcopy(specification), contract_book=book, inventory=inventory,
                          maintenance_contracts=service_facts, lineage=lineage,
                          target_balance_used=False, maturity='engineering_only_not_formal_sample')
