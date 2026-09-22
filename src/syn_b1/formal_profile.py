"""SYN-B1 profile validation only; readable YAML documents are handled separately."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from syn_b1.calendar_adapter import FORMAL_CALENDAR_VERSION, SynB1BankCalendar
from syn_b1.profile_document import ProfileDocumentError, read_profile_document, write_user_profile
from syn_b1.funding_contracts import FundingKind, LoanContract, RepaymentMethod
from syn_b1.funding_timeline import ShareholderCapitalPlan
from syn_b1.initial_capital import InitialCapitalDistribution, InitialCapitalInput, InitialCapitalMode
from syn_b1.operating_cycle import DelayShare, FixedAssetEvent
from syn_b1.operating_regime import (
    OperatingRegimeError, OperatingRegimePlan, OperatingRegimeSegment,
    adjusted_collection_schedule, build_monthly_regime_drivers,
)
from syn_b1.scenario_runner import ScenarioRunRequest, ScenarioRuntimeSettings
from syn_b1.statement_rollforward import AssetDepreciationPolicy
from syn_b1.tax_routing import (
    AccountRole, CashCategory, NonBankDayPolicy, RoutingRule, TaxFrequency,
    TaxPaymentRule, TaxScheduleMode, TaxType,
)
from syn_b1.transaction_planner import CategoryTransactionPolicy, MonthConcentration


FORMAL_TRANSACTION_PRESET = "standard_transactions_v1"


class FormalProfileError(ValueError):
    """A user editable SYN-B1 parameter profile is incomplete or inconsistent."""


@dataclass(frozen=True)
class ResolvedFormalProfile:
    """Validated input handed unchanged to the central scenario runner."""

    sample_id: str
    run_id: str
    request: ScenarioRunRequest
    resolved_profile: Mapping[str, Any]


def default_profile(sample_id: str = "syn_b1_demo_001", run_id: str = "run_001") -> dict[str, Any]:
    """Return the fully visible F7F base template; it is not a hidden default."""
    return {
        "schema_version": "syn_b1_formal_profile_1_0",
        "sample_id": sample_id,
        "run_id": run_id,
        "start_date": "2026-01-01",
        "end_date": "2026-12-31",
        "random_seed": 94301,
        "calendar": {"prediction_cutoff": "2026-12-31", "version": FORMAL_CALENDAR_VERSION},
        "account": {"role": "primary", "routing_preset": "primary_all_v1"},
        "initial_capital": {
            "capital_item_id": "initial_capital_001", "plan_version": "v1", "mode": "manual", "amount_cny": "1000000.00",
        },
        "additional_capital_plans": [],
        "loan_contracts": [],
        "operating": {
            "base_monthly_sales_cny": "300000.00", "sales_monthly_growth_percent": "0",
            "sales_month_weights_percent": None, "gross_margin_percent": "35", "production_cycle_days": 30,
            "inventory_target_days": None,
            "collection_schedule": [{"delay_days": 30, "share_percent": "100"}],
            "supplier_payment_schedule": [{"delay_days": 30, "share_percent": "100"}],
            "monthly_payroll_cny": "30000.00", "monthly_rent_and_utilities_cny": "10000.00",
            "monthly_other_operating_expense_cny": "0.00",
            "payroll_payment_schedule": [{"delay_days": 0, "share_percent": "100"}],
            "rent_and_utilities_payment_schedule": [{"delay_days": 0, "share_percent": "100"}],
            "other_operating_expense_payment_schedule": [{"delay_days": 0, "share_percent": "100"}],
            "fixed_assets": [],
        },
        "tax": {
            "income_tax_rate_percent": "20", "vat_burden_rate_percent": "1", "surcharge_rate_percent": "12",
            "rules": {
                "corporate_income_tax": _default_tax_rule("primary"),
                "simulated_vat": _default_tax_rule("primary"),
                "surcharge": _default_tax_rule("primary"),
            },
        },
        "transactions": {
            "preset": FORMAL_TRANSACTION_PRESET,
            "default_policy": {"transaction_count": 3, "month_concentration": "uniform", "amount_variation_percent": "20"},
            "category_overrides": {},
        },
    }


def default_profile_v2(sample_id: str = "syn_b1_v2_demo_001", run_id: str = "run_001") -> dict[str, Any]:
    """Return a visible profile-2.0 template without changing the v1 template."""
    profile = default_profile(sample_id, run_id)
    profile["schema_version"] = "syn_b1_formal_profile_2_0"
    operating = profile["operating"]
    base_sales = operating.pop("base_monthly_sales_cny")
    operating.pop("sales_monthly_growth_percent")
    operating.pop("sales_month_weights_percent")
    margin = operating.pop("gross_margin_percent")
    base_collection = operating.pop("collection_schedule")
    start_month = str(profile["start_date"])[:7]
    cutoff_month = str(profile["calendar"]["prediction_cutoff"])[:7]
    operating["regime_plan"] = {
        "shape": "stable",
        "declared_through_month": cutoff_month,
        "base_latent_monthly_sales_cny": base_sales,
        "segments": [{
            "segment_id": "stable_phase", "start_month": start_month,
            "monthly_linear_change_percent": "0", "gross_margin_percent": margin,
        }],
    }
    operating["sales_seasonality"] = {
        "enabled": False,
        "month_weights_percent": {str(month): "100" for month in range(1, 13)},
    }
    operating["collection_policy"] = {
        "base_schedule": base_collection,
        "seasonal_timing": {
            "enabled": False,
            "delay_adjustment_days_by_sales_month": {str(month): 0 for month in range(1, 13)},
        },
    }
    operating["procurement_inertia"] = {
        "opening_expectation": {
            "mode": "ratio_to_first_month_cogs",
            "opening_expected_cogs_ratio_percent": "100",
        },
        "adjustment_speed_percent": "50",
    }
    return profile


def write_template(path: str | Path, *, sample_id: str, run_id: str, profile_schema_version: str = "1.0") -> Path:
    if not str(path).strip():
        raise FormalProfileError("参数文件位置不能为空。双击BAT时请选择“新建企业参数方案”，无需手填路径。")
    try:
        factory = default_profile_v2 if profile_schema_version == "2.0" else default_profile
        return write_user_profile(path, factory(sample_id, run_id))
    except ProfileDocumentError as error:
        raise FormalProfileError(str(error)) from error


def load_profile(path: str | Path) -> ResolvedFormalProfile:
    try:
        raw = read_profile_document(path)
    except ProfileDocumentError as error:
        raise FormalProfileError(str(error)) from error
    return resolve_profile(raw)


def resolve_profile(raw: Mapping[str, Any]) -> ResolvedFormalProfile:
    if not isinstance(raw, Mapping):
        raise FormalProfileError("参数文件最外层必须是JSON对象。")
    schema_version = _text(raw, "schema_version")
    if schema_version not in {"syn_b1_formal_profile_1_0", "syn_b1_formal_profile_2_0"}:
        raise FormalProfileError("schema_version必须是受支持的画像1.0或画像2.0。")
    sample_id, run_id = _text(raw, "sample_id"), _text(raw, "run_id")
    start, end = _date(raw, "start_date"), _date(raw, "end_date")
    seed = _integer(raw, "random_seed")
    calendar_raw = _mapping(raw, "calendar")
    calendar_version = _text(calendar_raw, "version")
    # 旧版日历保持原样，新小样显式选择补齐资料的新版本。
    from syn_b1.calendar_adapter import ORDER_CALENDAR_VERSION, CYCLE_CALENDAR_VERSION
    # 不接受未实现的任意日历名称。
    if calendar_version not in {FORMAL_CALENDAR_VERSION, ORDER_CALENDAR_VERSION, CYCLE_CALENDAR_VERSION}:
        raise FormalProfileError(f"银行日历版本必须为已冻结的{FORMAL_CALENDAR_VERSION}。")
    calendar = SynB1BankCalendar(_date(calendar_raw, "prediction_cutoff"), calendar_version)

    account_raw = _mapping(raw, "account")
    role = _enum(AccountRole, _text(account_raw, "role"), "账户角色")
    routes = _routes(account_raw, role)

    initial = _initial_capital(_mapping(raw, "initial_capital"))
    additional_capital = tuple(_capital_plan(value) for value in _list(raw, "additional_capital_plans"))
    loans = tuple(_loan(value) for value in _list(raw, "loan_contracts"))

    operating = _mapping(raw, "operating")
    # 先解析唯一设备合同，新节点产能必须对应这些设备。
    fixed_events, policies = _fixed_assets(_list(operating, "fixed_assets"))
    # 新模式显式启用；缺少该字段的历史画像完全走原分支。
    node_plan = _business_nodes(operating, start, end, fixed_events) if "business_nodes" in operating else None
    # 节点模式与历史趋势、季节、固定费用及其他订单模式不能混用。
    if node_plan is not None and (schema_version != "syn_b1_formal_profile_2_0" or any(key in operating for key in ("regime_plan", "sales_seasonality", "collection_policy", "base_monthly_sales_cny", "gross_margin_percent", "sales_monthly_growth_percent", "sales_month_weights_percent", "monthly_payroll_cny", "monthly_rent_and_utilities_cny", "monthly_other_operating_expense_cny")) or any(key in raw for key in ("cycle_orders", "supplementary_orders"))):
        # 防止可见参数被忽略或同时贡献两次销售。
        raise FormalProfileError("经营节点仅用于画像2.0，且不能混入旧趋势、固定费用或其他订单模式。")
    # 节点模式没有旧的潜在销售增长驱动。
    regime_plan = _regime_plan(operating, start) if schema_version == "syn_b1_formal_profile_2_0" and node_plan is None else None
    # 首户使用逐月明确节点，收付仍调用已有延迟份额排期。
    if node_plan is not None:
        # 回款份额必须显式填写，不从预算脚本的默认值读取。
        collection_schedule = _schedule(operating, "collection_schedule")
        # 内部兼容字段保留首月真实零销售，不制造虚假营业额。
        base_monthly_sales_fen = node_plan.nodes[0].sales_fen
        # 毛利驱动不参与节点成本核算，成本在中央模块按明确金额确认。
        gross_margin_rate_bp, sales_monthly_growth_bp, sales_month_weights = 0, 0, None
    # 旧1.0画像继续使用原字段。
    elif regime_plan is None:
        collection_schedule = _schedule(operating, "collection_schedule")
        base_monthly_sales_fen = _fen(operating, "base_monthly_sales_cny")
        gross_margin_rate_bp = _bp(operating, "gross_margin_percent")
        sales_monthly_growth_bp = _signed_growth_bp(operating, "sales_monthly_growth_percent")
        sales_month_weights = _month_weights(operating.get("sales_month_weights_percent"))
    else:
        collection_schedule = _schedule(_mapping(operating, "collection_policy"), "base_schedule")
        try:
            build_monthly_regime_drivers(regime_plan, start_date=start, end_date=end)
            for adjustment in set(regime_plan.collection_delay_adjustment_days_by_sales_month.values()):
                adjusted_collection_schedule(collection_schedule, adjustment)
        except OperatingRegimeError as error:
            raise FormalProfileError(
                f"画像2.0字段 operating.regime_plan 或 operating.collection_policy.seasonal_timing 无效：{error} "
                "请核对声明截止月、阶段开始月和回款延迟调整后重新核验。"
            ) from error
        base_monthly_sales_fen = regime_plan.base_latent_monthly_sales_fen
        gross_margin_rate_bp = regime_plan.segments[0].gross_margin_bp
        sales_monthly_growth_bp = 0
        sales_month_weights = None
    tax_raw = _mapping(raw, "tax")
    income_tax_rate = _bp(tax_raw, "income_tax_rate_percent")
    tax_rules = _tax_rules(tax_raw, role)
    transaction_raw = _mapping(raw, "transactions")
    if _text(transaction_raw, "preset") != FORMAL_TRANSACTION_PRESET:
        raise FormalProfileError(f"普通流水预设必须为已冻结的{FORMAL_TRANSACTION_PRESET}；具体笔数、集中和波动仍由下方参数明确填写。")
    default_policy = _policy(_mapping(transaction_raw, "default_policy"))
    overrides = {
        _enum(CashCategory, name, "类别覆盖") : _policy(value)
        for name, value in _mapping(transaction_raw, "category_overrides").items()
    }

    # 完整订单必须与生成前保存的规则一致。
    from syn_b1.supplementary_orders import load_orders
    # 参数解析阶段就拒绝被改动的金额、日期和重复编号。
    supplementary_orders = load_orders(raw.get("supplementary_orders"), start, end, calendar)
    # 周期主要订单采用独立验证，不放宽九组规则。
    from syn_b1.cycle_orders import load_cycle_orders
    # 本轮不同时混入补充订单，避免超出八组范围。
    if "cycle_orders" in raw and "supplementary_orders" in raw:
        # 后续综合组合需要另定完整规则。
        raise FormalProfileError("本批周期主要订单与九组补充订单不能同时启用。")
    # 显式空值也拒绝，防止关闭旧主业却没有新订单。
    if "cycle_orders" in raw and raw["cycle_orders"] is None:
        # 缺少计划必须报错。
        raise FormalProfileError("周期订单必须提供完整计划。")
    # 无新字段时原订单完全不变。
    supplementary_orders = load_cycle_orders(raw["cycle_orders"], start, end, calendar) if "cycle_orders" in raw else supplementary_orders
    # 原运行配置仅增加一个默认关闭的字段。
    settings = ScenarioRuntimeSettings(
        collection_schedule=collection_schedule,
        supplier_payment_schedule=_schedule(operating, "supplier_payment_schedule"),
        payroll_payment_schedule=_schedule(operating, "payroll_payment_schedule"),
        rent_and_utilities_payment_schedule=_schedule(operating, "rent_and_utilities_payment_schedule"),
        other_operating_expense_payment_schedule=_schedule(operating, "other_operating_expense_payment_schedule"),
        tax_payment_schedule=(DelayShare(0, 10_000),),
        sales_month_weights_bp=sales_month_weights,
        inventory_target_days=_optional_nonnegative_int(operating.get("inventory_target_days"), "目标存货天数"),
        # 新模式首月费用仅作为内部兼容字段，每月真实费用由节点提供。
        monthly_other_operating_expense_fen=node_plan.nodes[0].other_expense_fen if node_plan is not None else _fen(operating, "monthly_other_operating_expense_cny"),
        fixed_asset_events=fixed_events,
        asset_depreciation_policies=policies,
        income_tax_rate_bp=income_tax_rate,
        account_role=role,
        routing_rules=routes,
        tax_rules=tax_rules,
        bank_calendar=calendar,
        additional_capital_plans=additional_capital,
        category_policies=overrides or None,
        regime_plan=regime_plan,
        # 完整节点传到中央运行器，参数和业务理由随原画像保存。
        business_node_plan=node_plan,
        # 后续只传这份已核验订单，不重新随机抽取。
        supplementary_orders=supplementary_orders,
        # 新字段存在才明确替代原主业。
        cycle_orders_only="cycle_orders" in raw,
    )
    request = ScenarioRunRequest(
        synthetic_account_id=sample_id, run_id=run_id, start_date=start, end_date=end, random_seed=seed,
        base_monthly_sales_fen=base_monthly_sales_fen,
        gross_margin_rate_bp=gross_margin_rate_bp,
        collection_delay_days=collection_schedule[0].delay_days,
        supplier_delay_days=_schedule(operating, "supplier_payment_schedule")[0].delay_days,
        production_cycle_days=_integer(operating, "production_cycle_days"),
        # 新模式首月工资与节点一致，运行时每月读取对应节点。
        monthly_payroll_fen=node_plan.nodes[0].payroll_fen if node_plan is not None else _fen(operating, "monthly_payroll_cny"),
        # 房租水电同样不保留一个可见但无效的旧参数。
        monthly_rent_and_utilities_fen=node_plan.nodes[0].rent_fen if node_plan is not None else _fen(operating, "monthly_rent_and_utilities_cny"),
        initial_capital=initial,
        sales_monthly_growth_bp=sales_monthly_growth_bp,
        loan_contracts=loans, transaction_policy=default_policy, runtime_settings=settings,
    )
    resolved = _jsonable(raw)
    resolved["resolved_presets"] = {
        "routing_preset": account_raw.get("routing_preset"), "transaction_preset": transaction_raw.get("preset"),
        "calendar_version": calendar.version,
    }
    return ResolvedFormalProfile(sample_id, run_id, request, resolved)


# 将可读月度输入转换为带约束的业务节点，金额复用既有解析器。
def _business_nodes(operating, start, end, assets):
    # 延迟导入，保持旧画像加载路径不变。
    from syn_b1.business_nodes import BusinessMonthlyNode, BusinessNodePlan
    # 显式空值或非对象均属于输入错误。
    raw = _mapping(operating, "business_nodes")
    if raw.get("version") == "physical_contract_business_v1":
        # 新显式版本将物理节点与精确结算连接，旧版本保持原日期语义。
        required = {"version", "nodes", "asset_capacities", "settlements", "capacity_basis"}
        if set(raw) != required:
            raise FormalProfileError("数量合同需完整节点、设备产能、结算清单与产能说明。")
        physical_raw = {key: raw[key] for key in ("nodes", "asset_capacities")}
        physical_raw["version"] = "monthly_business_nodes_v1"
        physical = _business_nodes({"business_nodes": physical_raw}, start, end, assets)
        from syn_b1.contract_business import load_contract_business
        contract_raw = {key: raw[key] for key in ("nodes", "settlements", "capacity_basis")}
        contract_raw.update(version="physical_contract_business_v1", capacity_mode="owned_assets_with_physical_nodes")
        return load_contract_business(contract_raw, start, end, assets, physical_plan=physical)
    # 新合同金额模式不借用旧设备产能字段，也不改旧版本行为。
    if raw.get("version") == "monthly_contract_business_v1":
        # 完整合同在专用输入模块核验，现金仍由原中央计算。
        from syn_b1.contract_business import load_contract_business
        # 返回与原月度节点兼容的明确金额驱动。
        return load_contract_business(raw, start, end, assets)
    # 限定首户已实现的输入版本。
    _require_equal(raw, "version", "monthly_business_nodes_v1")
    # 拒绝拼错字段而被静默忽略。
    if set(raw) != {"version", "nodes", "asset_capacities"}:
        # 新模式必须完整且只有这些明确字段。
        raise FormalProfileError("business_nodes字段应为version、nodes、asset_capacities。")
    # 初始化不可变节点的构造列表。
    nodes = []
    # 每月输入必须同时声明经营原因及全部费用。
    for value in _list(raw, "nodes"):
        # 验证行对象的结构。
        row = _as_mapping(value, "经营节点")
        # 完整列出本版本接受的月度字段。
        fields = {"month", "stage_id", "business_reason", "sales_units", "unit_price_cny", "unit_variable_cost_cny", "target_inventory_cny", "payroll_cny", "rent_cny", "other_expense_cny"}
        # 金额缺失或字段拼错均应在运行前报错。
        if set(row) != fields:
            # 明确拒绝未接通的临时高级字段。
            raise FormalProfileError("经营节点缺字段或含未实现字段。")
        # 使用原金额和整数解析器构造本月经营输入。
        nodes.append(BusinessMonthlyNode(_text(row, "month"), _text(row, "stage_id"), _text(row, "business_reason"), _integer(row, "sales_units"), _fen(row, "unit_price_cny"), _fen(row, "unit_variable_cost_cny"), _fen(row, "target_inventory_cny"), _fen(row, "payroll_cny"), _fen(row, "rent_cny"), _fen(row, "other_expense_cny")))
    # 设备能力仅从明确填写的事项读取。
    capacities = []
    # 同时检查设备产能输入结构。
    for value in _list(raw, "asset_capacities"):
        # 拒绝非对象的产能项目。
        row = _as_mapping(value, "设备产能")
        # 当前版本不接受隐含扩产触发器。
        if set(row) != {"event_id", "monthly_units"}:
            # 无法接通的字段不静默保留。
            raise FormalProfileError("设备产能仅接受event_id与monthly_units。")
        # 记录与设备合同对应的月度有效产能。
        capacities.append((_text(row, "event_id"), _integer(row, "monthly_units")))
    # 冻结已解析节点。
    plan = BusinessNodePlan(tuple(nodes), tuple(capacities))
    # 在运行前就检查覆盖、零销售和已投用产能约束。
    plan.validate_period(start, end, assets)
    # 返回业务驱动，所有现金仍由中央模块生成。
    return plan


def _regime_plan(operating: Mapping[str, Any], start: date) -> OperatingRegimePlan:
    try:
        return _regime_plan_checked(operating, start)
    except FormalProfileError as error:
        if str(error).startswith("画像2.0字段"):
            raise
        raise FormalProfileError(
            f"画像2.0字段 operating.regime_plan / operating.sales_seasonality / "
            f"operating.collection_policy / operating.procurement_inertia 无效：{error} "
            "请按YAML中对应#中文注释调整冒号右侧的值后重新核验。"
        ) from error
    except OperatingRegimeError as error:
        raise FormalProfileError(
            f"画像2.0字段 operating.regime_plan 无效：{error} "
            "请核对shape、segments的开始月、变化率方向和毛利率，再重新核验。"
        ) from error


def _regime_plan_checked(operating: Mapping[str, Any], start: date) -> OperatingRegimePlan:
    raw = _mapping(operating, "regime_plan")
    shape = _text(raw, "shape")
    allowed_shapes = {
        "stable", "sustained_growth", "sustained_contraction",
        "growth_to_contraction", "contraction_to_growth",
    }
    if shape not in allowed_shapes:
        raise FormalProfileError(
            "画像2.0字段 operating.regime_plan.shape 无效；请填写 stable、sustained_growth、"
            "sustained_contraction、growth_to_contraction 或 contraction_to_growth。"
        )
    segments_raw = _list(raw, "segments")
    segments = tuple(OperatingRegimeSegment(
        segment_id=_text(item, "segment_id"),
        start_month=_text(item, "start_month"),
        monthly_linear_change_bp=_signed_percent_bp(item, "monthly_linear_change_percent", -20, 20),
        gross_margin_bp=_signed_percent_bp(item, "gross_margin_percent", -100, 100),
    ) for item in (_as_mapping(value, "经营阶段") for value in segments_raw))
    sales = _mapping(operating, "sales_seasonality")
    sales_enabled = _bool(sales, "enabled")
    weights = _month_map_bp(_mapping(sales, "month_weights_percent"), 0, 300, "销售季节权重")
    if not sales_enabled and any(value != 10_000 for value in weights.values()):
        raise FormalProfileError("销售季节性关闭时，1至12月权重必须全部为100。")
    collection = _mapping(operating, "collection_policy")
    seasonal = _mapping(collection, "seasonal_timing")
    seasonal_enabled = _bool(seasonal, "enabled")
    adjustments = _month_map_int(_mapping(seasonal, "delay_adjustment_days_by_sales_month"), -60, 60, "回款季节调整")
    if not seasonal_enabled and any(adjustments.values()):
        raise FormalProfileError("回款季节性关闭时，1至12月调整必须全部为0。")
    procurement = _mapping(operating, "procurement_inertia")
    opening = _mapping(procurement, "opening_expectation")
    _require_equal(opening, "mode", "ratio_to_first_month_cogs")
    plan = OperatingRegimePlan(
        shape=shape,
        declared_through_month=_text(raw, "declared_through_month"),
        base_latent_monthly_sales_fen=_fen(raw, "base_latent_monthly_sales_cny"),
        segments=segments,
        sales_month_weights_bp=weights,
        collection_delay_adjustment_days_by_sales_month=adjustments,
        opening_expected_cogs_ratio_bp=_signed_percent_bp(opening, "opening_expected_cogs_ratio_percent", 0, 300),
        procurement_adjustment_speed_bp=_signed_percent_bp(procurement, "adjustment_speed_percent", 0, 100),
    )
    if plan.segments and plan.segments[0].start_month != start.strftime("%Y-%m"):
        raise FormalProfileError("第一经营阶段必须从start_date所在月份开始。")
    return plan


def _routes(raw: Mapping[str, Any], role: AccountRole) -> Mapping[CashCategory, RoutingRule]:
    preset = raw.get("routing_preset")
    supplied = raw.get("routing_rules")
    if preset == "primary_all_v1":
        if role is not AccountRole.PRIMARY:
            raise FormalProfileError("‘主账户全部经过’预设只能用于主账户。")
        return {category: RoutingRule(True, 10_000) for category in CashCategory}
    if not isinstance(supplied, Mapping):
        raise FormalProfileError("次要账户或非预设主账户必须逐类填写routing_rules。")
    result: dict[CashCategory, RoutingRule] = {}
    for category in CashCategory:
        item = supplied.get(category.value)
        if not isinstance(item, Mapping):
            raise FormalProfileError(f"缺少现金路由：{category.value}。")
        enabled = item.get("enabled")
        if not isinstance(enabled, bool):
            raise FormalProfileError(f"现金路由开关无效：{category.value}。")
        result[category] = RoutingRule(enabled, _bp(item, "routed_share_percent"))
    return result


def _initial_capital(raw: Mapping[str, Any]) -> InitialCapitalInput:
    mode = _enum(InitialCapitalMode, _text(raw, "mode"), "首次注资方式")
    common = {"capital_item_id": _text(raw, "capital_item_id"), "plan_version": _text(raw, "plan_version"), "mode": mode}
    if mode is InitialCapitalMode.MANUAL:
        return InitialCapitalInput(**common, amount_fen=_fen(raw, "amount_cny"))
    return InitialCapitalInput(
        **common,
        minimum_amount_fen=_fen(raw, "minimum_amount_cny"), maximum_amount_fen=_fen(raw, "maximum_amount_cny"),
        distribution=_enum(InitialCapitalDistribution, _text(raw, "distribution"), "随机分布"),
        profile_version=_text(raw, "profile_version"),
    )


def _capital_plan(raw: Mapping[str, Any]) -> ShareholderCapitalPlan:
    return ShareholderCapitalPlan(_text(raw, "capital_item_id"), _text(raw, "plan_version"), _date(raw, "due_date"), _fen(raw, "amount_cny"))


def _loan(raw: Mapping[str, Any]) -> LoanContract:
    maturity = raw.get("maturity_date")
    term = raw.get("term_days")
    return LoanContract(
        loan_contract_id=_text(raw, "loan_contract_id"), contract_version=_text(raw, "contract_version"),
        funding_kind=_enum(FundingKind, _text(raw, "funding_kind"), "借款类型"), draw_date=_date(raw, "draw_date"),
        principal_fen=_fen(raw, "principal_cny"), annual_interest_rate_bp=_bp(raw, "annual_interest_rate_percent"),
        repayment_method=_enum(RepaymentMethod, _text(raw, "repayment_method"), "还款方式"),
        maturity_date=_parse_date(maturity, "到期日") if maturity is not None else None,
        term_days=_optional_nonnegative_int(term, "期限天数") if term is not None else None,
    )


def _fixed_assets(values: list[Any]) -> tuple[tuple[FixedAssetEvent, ...], tuple[AssetDepreciationPolicy, ...]]:
    events: list[FixedAssetEvent] = []
    policies: list[AssetDepreciationPolicy] = []
    for value in values:
        raw = _as_mapping(value, "固定资产事项")
        event_id = _text(raw, "event_id")
        ready = raw.get("ready_for_use_date")
        events.append(FixedAssetEvent(
            event_id, _text(raw, "asset_type"), _text(raw, "purchase_month"), _fen(raw, "purchase_amount_cny"),
            _schedule(raw, "payment_schedule"), _parse_date(ready, "达到可使用日期") if ready else None,
        ))
        if ready:
            policies.append(AssetDepreciationPolicy(event_id, _integer(raw, "useful_life_months"), _bp(raw, "residual_value_percent")))
    return tuple(events), tuple(policies)


def _tax_rules(raw: Mapping[str, Any], role: AccountRole) -> Mapping[TaxType, TaxPaymentRule]:
    rules = _mapping(raw, "rules")
    vat_rate, surcharge_rate = _bp(raw, "vat_burden_rate_percent"), _bp(raw, "surcharge_rate_percent")
    result: dict[TaxType, TaxPaymentRule] = {}
    for tax_type in TaxType:
        item = _mapping(rules, tax_type.value)
        mode = _enum(TaxScheduleMode, _text(item, "schedule_mode"), "税费日期方式")
        explicit = item.get("explicit_period_dates")
        explicit_dates = ({key: _parse_date(value, "明确税费日期") for key, value in _as_mapping(explicit, "明确税费日期").items()} if explicit else None)
        result[tax_type] = TaxPaymentRule(
            enabled=_bool(item, "enabled"), frequency=_enum(TaxFrequency, _text(item, "frequency"), "税费周期"),
            schedule_mode=mode, payer_account_role=_enum(AccountRole, _text(item, "payer_account_role"), "税费扣款账户"),
            non_bank_day_policy=_enum(NonBankDayPolicy, _text(item, "non_bank_day_policy"), "非工作日处理"),
            recurring_debit_day=_integer(item, "recurring_debit_day") if mode is TaxScheduleMode.RECURRING_DAY else None,
            explicit_period_dates=explicit_dates if mode is TaxScheduleMode.EXPLICIT_PERIOD_DATES else None,
            effective_vat_burden_rate_bp=vat_rate if tax_type is TaxType.SIMULATED_VAT else None,
            surcharge_rate_on_simulated_vat_bp=surcharge_rate if tax_type is TaxType.SURCHARGE else None,
        )
    if result[TaxType.CORPORATE_INCOME_TAX].payer_account_role not in {AccountRole.PRIMARY, AccountRole.SECONDARY}:
        raise FormalProfileError("所得税扣款账户无效。")
    return result


def _policy(raw: Mapping[str, Any]) -> CategoryTransactionPolicy:
    return CategoryTransactionPolicy(
        transaction_count=_integer(raw, "transaction_count"),
        month_concentration=_enum(MonthConcentration, _text(raw, "month_concentration"), "月内集中方式"),
        amount_variation_bp=_bp(raw, "amount_variation_percent"),
    )


def _schedule(raw: Mapping[str, Any], key: str) -> tuple[DelayShare, ...]:
    values = _list(raw, key)
    return tuple(DelayShare(_integer(_as_mapping(item, key), "delay_days"), _bp(_as_mapping(item, key), "share_percent")) for item in values)


def _month_weights(value: Any) -> Mapping[int, int] | None:
    if value is None:
        return None
    raw = _as_mapping(value, "月度季节系数")
    if set(raw) != {str(number) for number in range(1, 13)}:
        raise FormalProfileError("月度季节系数必须完整填写1至12月。")
    return {int(month): _bp({"value": amount}, "value") for month, amount in raw.items()}


def _month_map_bp(raw: Mapping[str, Any], minimum: int, maximum: int, label: str) -> Mapping[int, int]:
    required = {str(number) for number in range(1, 13)}
    if set(raw) != required:
        raise FormalProfileError(f"{label}必须完整填写1至12月。")
    return {
        int(month): _signed_percent_bp({"value": value}, "value", minimum, maximum)
        for month, value in raw.items()
    }


def _month_map_int(raw: Mapping[str, Any], minimum: int, maximum: int, label: str) -> Mapping[int, int]:
    required = {str(number) for number in range(1, 13)}
    if set(raw) != required:
        raise FormalProfileError(f"{label}必须完整填写1至12月。")
    result = {int(month): _integer({"value": value}, "value") for month, value in raw.items()}
    if any(not minimum <= value <= maximum for value in result.values()):
        raise FormalProfileError(f"{label}必须在{minimum}至{maximum}之间。")
    return result


def _default_tax_rule(role: str) -> dict[str, Any]:
    return {"enabled": True, "frequency": "monthly", "schedule_mode": "recurring_day", "payer_account_role": role, "non_bank_day_policy": "next_bank_workday", "recurring_debit_day": 28}


def _fen(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise FormalProfileError(f"{key}必须是金额。") from error
    result = int((decimal * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if result < 0:
        raise FormalProfileError(f"{key}不能为负数。")
    return result


def _bp(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise FormalProfileError(f"{key}必须是百分比数字。") from error
    result = int((decimal * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if not 0 <= result <= 10_000:
        raise FormalProfileError(f"{key}必须在0至100之间。")
    return result


def _signed_growth_bp(raw: Mapping[str, Any], key: str) -> int:
    """Parse the one signed percentage in the formal profile: linear sales trend."""

    value = raw.get(key)
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise FormalProfileError(f"{key}必须是百分比数字。") from error
    result = int((decimal * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if not -10_000 < result <= 10_000:
        raise FormalProfileError(f"{key}必须大于-100且不超过100。")
    return result


def _signed_percent_bp(raw: Mapping[str, Any], key: str, minimum: int, maximum: int) -> int:
    value = raw.get(key)
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise FormalProfileError(f"{key}必须是百分比数字。") from error
    result = int((decimal * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if not minimum * 100 <= result <= maximum * 100:
        raise FormalProfileError(f"{key}必须在{minimum}至{maximum}之间。")
    return result


def _integer(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FormalProfileError(f"{key}必须为整数。")
    return value


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FormalProfileError(f"{label}必须为非负整数或留空。")
    return value


def _date(raw: Mapping[str, Any], key: str) -> date:
    return _parse_date(raw.get(key), key)


def _parse_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise FormalProfileError(f"{label}必须使用YYYY-MM-DD。")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise FormalProfileError(f"{label}必须使用YYYY-MM-DD。") from error


def _text(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FormalProfileError(f"{key}不能为空。")
    return value.strip()


def _bool(raw: Mapping[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise FormalProfileError(f"{key}必须为true或false。")
    return value


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _as_mapping(raw.get(key), key)


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalProfileError(f"{label}必须是对象。")
    return value


def _list(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise FormalProfileError(f"{key}必须是列表。")
    return value


def _enum(kind, value: str, label: str):
    try:
        return kind(value)
    except ValueError as error:
        raise FormalProfileError(f"{label}取值无效：{value}。") from error


def _require_equal(raw: Mapping[str, Any], key: str, expected: str) -> None:
    if raw.get(key) != expected:
        raise FormalProfileError(f"{key}必须为{expected}。")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    return value
