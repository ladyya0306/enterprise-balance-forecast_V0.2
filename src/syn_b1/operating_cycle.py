"""SYN-B1-I1A: fictional monthly operating-cycle planning only.

This module deliberately separates business confirmation from bank cash due.
It creates a monthly plan for sales, collections, purchases, supplier
payments, inventory, recurring expenses, simplified taxes, and fixed-asset
payment plans.  It does *not* produce final financial statements, financing,
dates, transactions, files, databases, or model inputs.

Amounts are integer fen.  All inputs are fictional scenario parameters.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Mapping, Sequence


class OperatingCycleError(ValueError):
    """The stated operating assumptions cannot form a valid monthly plan."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DelayShare:
    """One share of an amount that becomes cash due after calendar days.

    I1A is monthly, so an internal mid-month planning anchor converts days to
    the month containing the due date.  It never creates a bank transaction.
    """

    delay_days: int
    share_bp: int

    def __post_init__(self) -> None:
        if not isinstance(self.delay_days, int) or isinstance(self.delay_days, bool) or self.delay_days < 0:
            raise OperatingCycleError("INVALID_DELAY_DAYS", "现金到期延迟天数必须为非负整数。")
        if (
            not isinstance(self.share_bp, int)
            or isinstance(self.share_bp, bool)
            or not 0 <= self.share_bp <= 10_000
        ):
            raise OperatingCycleError("INVALID_BASIS_POINTS", "share_bp必须是0至10000之间的整数。")


@dataclass(frozen=True)
class OperatingOpeningState:
    """Beginning operating balances; not a complete balance sheet."""

    accounts_receivable_fen: int = 0
    inventory_fen: int = 0
    accounts_payable_fen: int = 0
    accrued_payroll_and_utilities_fen: int = 0
    tax_payable_fen: int = 0
    capital_expenditure_payable_fen: int = 0

    def __post_init__(self) -> None:
        for field_name, value in self.__dict__.items():
            _require_nonnegative_fen(field_name, value)


@dataclass(frozen=True)
class FixedAssetEvent:
    """A fictional equipment or factory investment planned before financing.

    Acquisition is recognized in ``purchase_month``.  The payment schedule
    determines only monthly investing cash due; depreciation and final fixed
    asset balances belong to I1B.
    """

    event_id: str
    asset_type: str
    purchase_month: str
    purchase_amount_fen: int
    payment_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    ready_for_use_date: date | None = None

    def __post_init__(self) -> None:
        if not self.event_id:
            raise OperatingCycleError("EMPTY_FIXED_ASSET_EVENT_ID", "固定资产事件编号不能为空。")
        if self.asset_type not in {"equipment", "factory_or_construction"}:
            raise OperatingCycleError("INVALID_ASSET_TYPE", "固定资产类型只能是设备或厂房建设。")
        _validate_month_key(self.purchase_month)
        _require_positive_fen("purchase_amount_fen", self.purchase_amount_fen)
        _validate_schedule("固定资产付款安排", self.payment_schedule)
        if self.ready_for_use_date is not None and not isinstance(self.ready_for_use_date, date):
            raise OperatingCycleError("INVALID_READY_FOR_USE_DATE", "达到可使用日期必须为date或留空。")


@dataclass(frozen=True)
class OperatingCycleRequest:
    """Inputs for I1A's monthly operating plan.

    ``random_seed`` is saved as scenario identity only.  I1A contains no
    random draw: later authorised down-drilling may use it for dates and
    individual amounts without changing this monthly business plan.
    """

    synthetic_account_id: str
    start_date: date
    end_date: date
    random_seed: int
    base_monthly_sales_fen: int
    gross_margin_rate_bp: int
    collection_schedule: tuple[DelayShare, ...]
    production_cycle_days: int
    supplier_payment_schedule: tuple[DelayShare, ...]
    monthly_payroll_fen: int
    monthly_rent_and_utilities_fen: int
    tax_and_surcharge_rate_bp: int
    sales_monthly_growth_bp: int = 0
    sales_month_weights_bp: Mapping[int, int] | None = None
    inventory_target_days: int | None = None
    monthly_other_operating_expense_fen: int = 0
    payroll_payment_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    rent_and_utilities_payment_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    other_operating_expense_payment_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    tax_payment_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    fixed_asset_events: tuple[FixedAssetEvent, ...] = ()
    opening_state: OperatingOpeningState | None = None
    opening_receivable_collection_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    opening_payable_payment_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    opening_accrual_payment_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    opening_tax_payment_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    opening_capex_payment_schedule: tuple[DelayShare, ...] = (DelayShare(0, 10_000),)
    regime_plan: Any | None = None
    # 新增订单显式传入，旧请求默认没有补充业务。
    supplementary_orders: tuple = ()
    # 周期贸易显式替代旧主业，不重复计入销售。
    cycle_orders_only: bool = False
    # 新节点模式只提供预先声明的业务输入，默认不启用。
    business_node_plan: Any | None = None

    def __post_init__(self) -> None:
        if not self.synthetic_account_id:
            raise OperatingCycleError("EMPTY_ACCOUNT_ID", "虚构账户编号不能为空。")
        if not isinstance(self.start_date, date) or not isinstance(self.end_date, date):
            raise OperatingCycleError("INVALID_PERIOD", "开始和结束日期必须是date。")
        if self.start_date > self.end_date:
            raise OperatingCycleError("INVALID_PERIOD", "开始日期不能晚于结束日期。")
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise OperatingCycleError("INVALID_RANDOM_SEED", "随机种子必须为整数。")
        # 新节点模式允许首月筹建销售为零，旧模式仍要求正销售锚点。
        if self.business_node_plan is not None:
            # 不使用极小正数伪装筹建期。
            _require_nonnegative_fen("base_monthly_sales_fen", self.base_monthly_sales_fen)
        # 旧画像校验要求保持不变。
        else:
            # 原锚点仍不能为零。
            _require_positive_fen("base_monthly_sales_fen", self.base_monthly_sales_fen)
        _require_rate("gross_margin_rate_bp", self.gross_margin_rate_bp)
        _require_rate("tax_and_surcharge_rate_bp", self.tax_and_surcharge_rate_bp)
        if not isinstance(self.sales_monthly_growth_bp, int) or self.sales_monthly_growth_bp <= -10_000:
            raise OperatingCycleError("INVALID_SALES_GROWTH", "月销售增长率必须大于-100%。")
        _require_positive_days("production_cycle_days", self.production_cycle_days)
        if self.inventory_target_days is not None:
            _require_nonnegative_days("inventory_target_days", self.inventory_target_days)
        for field_name in (
            "monthly_payroll_fen",
            "monthly_rent_and_utilities_fen",
            "monthly_other_operating_expense_fen",
        ):
            _require_nonnegative_fen(field_name, getattr(self, field_name))
        for label, schedule in (
            ("回款安排", self.collection_schedule),
            ("供应商付款安排", self.supplier_payment_schedule),
            ("工资付款安排", self.payroll_payment_schedule),
            ("房租水电付款安排", self.rent_and_utilities_payment_schedule),
            ("其他费用付款安排", self.other_operating_expense_payment_schedule),
            ("税费付款安排", self.tax_payment_schedule),
            ("期初应收回款安排", self.opening_receivable_collection_schedule),
            ("期初应付付款安排", self.opening_payable_payment_schedule),
            ("期初费用付款安排", self.opening_accrual_payment_schedule),
            ("期初税费付款安排", self.opening_tax_payment_schedule),
            ("期初设备工程付款安排", self.opening_capex_payment_schedule),
        ):
            _validate_schedule(label, schedule)
        if self.sales_month_weights_bp is not None:
            if set(self.sales_month_weights_bp) != set(range(1, 13)):
                raise OperatingCycleError("INVALID_SALES_WEIGHTS", "销售季节权重必须完整包含1至12月。")
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.sales_month_weights_bp.values()):
                raise OperatingCycleError("INVALID_SALES_WEIGHTS", "销售季节权重必须为非负整数。")
            if sum(self.sales_month_weights_bp.values()) == 0:
                raise OperatingCycleError("INVALID_SALES_WEIGHTS", "销售季节权重之和必须大于0。")
        event_ids = [event.event_id for event in self.fixed_asset_events]
        if len(event_ids) != len(set(event_ids)):
            raise OperatingCycleError("DUPLICATE_FIXED_ASSET_EVENT_ID", "固定资产事件编号不能重复。")


@dataclass(frozen=True)
class MonthlyOperatingPlanRow:
    """A monthly business plan and its scheduled bank cash amounts."""

    month: str
    opening_accounts_receivable_fen: int
    sales_confirmed_fen: int
    cash_collections_fen: int
    ending_accounts_receivable_fen: int
    opening_inventory_fen: int
    cost_of_goods_sold_fen: int
    material_purchases_fen: int
    ending_inventory_fen: int
    opening_accounts_payable_fen: int
    supplier_cash_payments_fen: int
    ending_accounts_payable_fen: int
    payroll_expense_fen: int
    payroll_cash_payment_fen: int
    rent_and_utilities_expense_fen: int
    rent_and_utilities_cash_payment_fen: int
    other_operating_expense_fen: int
    other_operating_cash_payment_fen: int
    opening_accrued_payroll_and_utilities_fen: int
    ending_accrued_payroll_and_utilities_fen: int
    tax_and_surcharge_accrual_fen: int
    tax_cash_payment_fen: int
    opening_tax_payable_fen: int
    ending_tax_payable_fen: int
    capitalized_asset_acquisition_fen: int
    fixed_asset_cash_payment_fen: int
    opening_capital_expenditure_payable_fen: int
    ending_capital_expenditure_payable_fen: int
    regime_segment_id: str | None = None
    latent_sales_fen: int = 0
    gross_margin_rate_bp: int = 0
    procurement_expected_cogs_fen: int = 0
    collection_delay_adjustment_days: int = 0

    @property
    def operating_cash_inflow_fen(self) -> int:
        return self.cash_collections_fen

    @property
    def operating_cash_outflow_fen(self) -> int:
        return (
            self.supplier_cash_payments_fen
            + self.payroll_cash_payment_fen
            + self.rent_and_utilities_cash_payment_fen
            + self.other_operating_cash_payment_fen
            + self.tax_cash_payment_fen
        )


@dataclass(frozen=True)
class DeferredCashDue:
    """A cash amount due after the requested planning horizon."""

    due_month: str
    cash_category: str
    amount_fen: int


@dataclass(frozen=True)
class OperatingCycleDiagnostic:
    severity: str
    code: str
    message_cn: str


@dataclass(frozen=True)
class OperatingCycleResult:
    request: OperatingCycleRequest
    opening_state: OperatingOpeningState
    monthly_rows: tuple[MonthlyOperatingPlanRow, ...]
    deferred_cash_due: tuple[DeferredCashDue, ...]
    diagnostics: tuple[OperatingCycleDiagnostic, ...]


def build_monthly_operating_cycle(request: OperatingCycleRequest) -> OperatingCycleResult:
    """Build an entirely fictional, deterministic monthly operating plan.

    The output keeps business recognition and cash due separate.  It has no
    bank balance, financing event, journal entry, transaction date, file
    output, or public model field.
    """

    months = _period_months(request.start_date, request.end_date)
    # 节点模式不得与其他业务驱动混合或继承隐含期初库存。
    if request.business_node_plan is not None and (request.regime_plan is not None or request.supplementary_orders or request.cycle_orders_only or request.opening_state != OperatingOpeningState()):
        # 首户从空白经营状态开始，禁止双重计入销售。
        raise OperatingCycleError("INCOMPATIBLE_BUSINESS_NODES", "月度节点需显式空白期初，且不能混用趋势、补充订单或周期订单。")
    # 在中央运行入口重新检查期间与产能，直接调用不能绕过校验。
    nodes = request.business_node_plan.validate_period(request.start_date, request.end_date, request.fixed_asset_events) if request.business_node_plan is not None else {}
    # 合同定日仅替代现金排期，不替代本模块库存和应收应付递推。
    from syn_b1.contract_business import schedule_node_cash
    month_set = set(months)
    for event in request.fixed_asset_events:
        if event.purchase_month not in month_set:
            raise OperatingCycleError("FIXED_ASSET_EVENT_OUT_OF_PERIOD", "固定资产事件发生在规划期间外。")

    diagnostics: list[OperatingCycleDiagnostic] = []
    opening = request.opening_state or _derive_opening_state(request)
    if request.opening_state is None:
        diagnostics.append(
            OperatingCycleDiagnostic(
                severity="info",
                code="DERIVED_OPENING_OPERATING_STATE",
                message_cn="未填写期初经营状态，系统已按基础销售、账期和生产周期反算，并在结果中保留该事实。",
            )
        )

    schedules: dict[str, dict[str, int]] = {
        category: {} for category in _CASH_CATEGORIES
    }
    first_month = months[0]
    _schedule_amount(schedules["collection"], first_month, opening.accounts_receivable_fen, request.opening_receivable_collection_schedule)
    _schedule_amount(schedules["supplier_payment"], first_month, opening.accounts_payable_fen, request.opening_payable_payment_schedule)
    _schedule_amount(schedules["payroll_payment"], first_month, opening.accrued_payroll_and_utilities_fen, request.opening_accrual_payment_schedule)
    _schedule_amount(schedules["tax_payment"], first_month, opening.tax_payable_fen, request.opening_tax_payment_schedule)
    _schedule_amount(schedules["fixed_asset_payment"], first_month, opening.capital_expenditure_payable_fen, request.opening_capex_payment_schedule)

    events_by_month: dict[str, list[FixedAssetEvent]] = {month: [] for month in months}
    for event in request.fixed_asset_events:
        events_by_month[event.purchase_month].append(event)

    current_ar = opening.accounts_receivable_fen
    current_inventory = opening.inventory_fen
    current_ap = opening.accounts_payable_fen
    current_accrual = opening.accrued_payroll_and_utilities_fen
    current_tax = opening.tax_payable_fen
    current_capex_payable = opening.capital_expenditure_payable_fen
    rows: list[MonthlyOperatingPlanRow] = []
    # 新订单只提供业务驱动，库存和应收应付仍由下方唯一递推计算。
    from syn_b1.supplementary_orders import month_drivers
    # 直接程序调用也不能重复添加同一订单。
    if len({o.order_id for o in request.supplementary_orders}) != len(request.supplementary_orders) or any(o.stock_date < request.start_date for o in request.supplementary_orders):
        # 期初之前的订单需要专门期初账规则，本轮不默认为零。
        raise OperatingCycleError("INVALID_SUPPLEMENTARY_ORDER", "补充订单重复或早于模拟开始日。")
    # 收付款按逐单真实到期月排入原有现金日程，尾款继续由原日程保存。
    for order in request.supplementary_orders:
        # 客户款和供应商款不能再次套用旧月度账期。
        for category, due, amount in (("collection", order.client_date, order.sale_fen), ("supplier_payment", order.supplier_date, order.cost_fen)):
            # 月度日程仅汇总已确定的现金日期。
            due_month = due.strftime("%Y-%m")
            # 同一月份可以包含多张订单，金额只加一次。
            schedules[category][due_month] = schedules[category].get(due_month, 0) + amount
    normalized_weights = _normalized_month_weights(request.sales_month_weights_bp)
    regime_by_month = None
    if request.regime_plan is not None:
        from syn_b1.operating_regime import build_monthly_regime_drivers
        regime_by_month = {
            item.month: item
            for item in build_monthly_regime_drivers(
                request.regime_plan, start_date=request.start_date, end_date=request.end_date,
            )
        }
    previous_actual_cogs: int | None = None
    expected_cogs: int | None = None

    for index, month in enumerate(months):
        coverage_days = _covered_days_in_month(request.start_date, request.end_date, month)
        month_days = _month_days(month)
        regime = regime_by_month[month] if regime_by_month is not None else None
        # 取本月明确输入，旧模式没有该节点。
        node = nodes.get(month)
        if regime is None:
            sales = _monthly_sales(request, index, month, coverage_days, month_days, normalized_weights)
            margin_bp = request.gross_margin_rate_bp
            latent_sales = sales
            collection_schedule = request.collection_schedule
            collection_adjustment = 0
        else:
            from syn_b1.operating_regime import adjusted_collection_schedule
            sales = _prorate(regime.full_month_sales_fen, coverage_days, month_days)
            latent_sales = _prorate(regime.latent_sales_fen, coverage_days, month_days)
            margin_bp = regime.gross_margin_bp
            collection_adjustment = regime.collection_delay_adjustment_days
            collection_schedule = adjusted_collection_schedule(request.collection_schedule, collection_adjustment)
        # 新周期模式关闭旧主业销售驱动，费用及唯一账务递推仍保留。
        if request.cycle_orders_only:
            # 主业全部由已冻结的周期订单提供。
            sales = latent_sales = 0
        cogs = sales * (10_000 - margin_bp) // 10_000
        # 节点模式直接使用数量乘单价及明确单位成本，允许筹建期销售为零。
        if node is not None:
            # 销售与耗用来自业务计划，绝不读取账户余额。
            sales, latent_sales, cogs = node.sales_fen, node.sales_fen, node.variable_cost_fen
            # 比例只保留作解释字段，后续库存和利润使用上面的精确成本。
            margin_bp = (sales - cogs) * 10_000 // sales if sales else 0
        if regime is None:
            expected_cogs = cogs
        elif expected_cogs is None:
            expected_cogs = cogs * request.regime_plan.opening_expected_cogs_ratio_bp // 10_000
        else:
            expected_cogs += (
                request.regime_plan.procurement_adjustment_speed_bp
                * (previous_actual_cogs - expected_cogs)
                // 10_000
            )
        target_days = max(request.production_cycle_days, request.inventory_target_days or 0)
        target_inventory = _round_up_div(expected_cogs * target_days, month_days)
        # 只有新模式使用明确库存金额，旧采购惯性与库存天数保持原样。
        if node is not None:
            # 覆盖目标而非另建库存递推。
            target_inventory = node.target_inventory_fen
        # 新订单的库存不能被原主业采购预算视为可挪用存货。
        extra = month_drivers(request.supplementary_orders, month)
        # 从合计库存扣除有明确订单归属的期初库存。
        baseline_inventory = current_inventory - extra["inventory_before"]
        # 原采购公式保持不变，只读取原业务自己的库存。
        purchases = max(0, expected_cogs + target_inventory - baseline_inventory, cogs - baseline_inventory)
        # 节点采购仍在唯一中央经营模块内形成，并使用同一库存桥接。
        if node is not None:
            # 当月耗用加库存变化构成预先预算的采购。
            purchases = cogs + target_inventory - baseline_inventory
            # 不用负采购或虚构库存出售来筹资。
            if purchases < 0:
                # 明确要求重新声明过剩库存处理方式。
                raise OperatingCycleError("NEGATIVE_NODE_PURCHASE", f"{month}目标库存会产生负采购，须明确库存消耗计划。")
        # 超额库存的判断同样只针对原业务。
        if purchases == 0 and baseline_inventory - cogs > target_inventory:
            diagnostics.append(
                OperatingCycleDiagnostic(
                    severity="info",
                    code="EXCESS_OPENING_INVENTORY_CONSUMED",
                    message_cn=f"{month}期初存货超过目标，材料采购按0处理，未用虚构采购抵消存货。",
                )
            )
        acquisition = sum(event.purchase_amount_fen for event in events_by_month[month])

        # 收款总额来自本月业务确认，日期可由明确合同给定。
        schedule_node_cash(schedules, "collection", month, sales, collection_schedule, request.business_node_plan)
        # 采购承诺仍等于中央计算的采购，禁止另加付款。
        schedule_node_cash(schedules, "supplier_payment", month, purchases, request.supplier_payment_schedule, request.business_node_plan)
        # 保存原主业成本，下一月采购惯性不混入补充订单。
        baseline_cogs = cogs
        # 收入在交货时合并，然后统一计算税费和三表。
        sales += extra["sales"]
        # 交货成本引用逐单冻结值，不在这里第二次舍入。
        cogs += extra["cogs"]
        # 新订单备货在实际发生月份计入统一库存递推。
        purchases += extra["purchases"]
        # 费用确认与现金排期共用同一金额，防止只改流水而未改利润。
        payroll_expense = node.payroll_fen if node is not None else _prorate(request.monthly_payroll_fen, coverage_days, month_days)
        # 房租水电同样读取本月显式节点或原固定配置。
        rent_expense = node.rent_fen if node is not None else _prorate(request.monthly_rent_and_utilities_fen, coverage_days, month_days)
        # 其他费用包含筹建或正常维护费用，但不重复包括设备支出。
        other_expense = node.other_expense_fen if node is not None else _prorate(request.monthly_other_operating_expense_fen, coverage_days, month_days)
        # 复用原工资应付排期。
        schedule_node_cash(schedules, "payroll_payment", month, payroll_expense, request.payroll_payment_schedule, request.business_node_plan)
        # 复用原租金应付排期。
        schedule_node_cash(schedules, "rent_payment", month, rent_expense, request.rent_and_utilities_payment_schedule, request.business_node_plan)
        # 复用原其他费用应付排期。
        schedule_node_cash(schedules, "other_expense_payment", month, other_expense, request.other_operating_expense_payment_schedule, request.business_node_plan)
        tax_accrual = sales * request.tax_and_surcharge_rate_bp // 10_000
        _schedule_amount(schedules["tax_payment"], month, tax_accrual, request.tax_payment_schedule)
        for event in events_by_month[month]:
            # 设备分期必须绑定具体资产，不能按月总额重复入账。
            schedule_node_cash(schedules, "fixed_asset_payment", month, event.purchase_amount_fen, event.payment_schedule, request.business_node_plan, event.event_id)

        cash_collections = _take_due(schedules["collection"], month)
        supplier_cash_payments = _take_due(schedules["supplier_payment"], month)
        payroll_cash_payment = _take_due(schedules["payroll_payment"], month)
        rent_cash_payment = _take_due(schedules["rent_payment"], month)
        other_cash_payment = _take_due(schedules["other_expense_payment"], month)
        tax_cash_payment = _take_due(schedules["tax_payment"], month)
        fixed_asset_cash_payment = _take_due(schedules["fixed_asset_payment"], month)

        ending_ar = current_ar + sales - cash_collections
        ending_inventory = current_inventory + purchases - cogs
        ending_ap = current_ap + purchases - supplier_cash_payments
        ending_accrual = current_accrual + payroll_expense + rent_expense + other_expense - payroll_cash_payment - rent_cash_payment - other_cash_payment
        ending_tax = current_tax + tax_accrual - tax_cash_payment
        ending_capex_payable = current_capex_payable + acquisition - fixed_asset_cash_payment
        _assert_nonnegative_states(month, ending_ar, ending_inventory, ending_ap, ending_accrual, ending_tax, ending_capex_payable)

        rows.append(
            MonthlyOperatingPlanRow(
                month=month,
                opening_accounts_receivable_fen=current_ar,
                sales_confirmed_fen=sales,
                cash_collections_fen=cash_collections,
                ending_accounts_receivable_fen=ending_ar,
                opening_inventory_fen=current_inventory,
                cost_of_goods_sold_fen=cogs,
                material_purchases_fen=purchases,
                ending_inventory_fen=ending_inventory,
                opening_accounts_payable_fen=current_ap,
                supplier_cash_payments_fen=supplier_cash_payments,
                ending_accounts_payable_fen=ending_ap,
                payroll_expense_fen=payroll_expense,
                payroll_cash_payment_fen=payroll_cash_payment,
                rent_and_utilities_expense_fen=rent_expense,
                rent_and_utilities_cash_payment_fen=rent_cash_payment,
                other_operating_expense_fen=other_expense,
                other_operating_cash_payment_fen=other_cash_payment,
                opening_accrued_payroll_and_utilities_fen=current_accrual,
                ending_accrued_payroll_and_utilities_fen=ending_accrual,
                tax_and_surcharge_accrual_fen=tax_accrual,
                tax_cash_payment_fen=tax_cash_payment,
                opening_tax_payable_fen=current_tax,
                ending_tax_payable_fen=ending_tax,
                capitalized_asset_acquisition_fen=acquisition,
                fixed_asset_cash_payment_fen=fixed_asset_cash_payment,
                opening_capital_expenditure_payable_fen=current_capex_payable,
                ending_capital_expenditure_payable_fen=ending_capex_payable,
                # 阶段编号仅用于受限经营核查，不进入公共余额特征。
                regime_segment_id=node.stage_id if node is not None else regime.segment_id if regime is not None else None,
                latent_sales_fen=latent_sales,
                gross_margin_rate_bp=margin_bp,
                procurement_expected_cogs_fen=expected_cogs,
                collection_delay_adjustment_days=collection_adjustment,
            )
        )
        # 下一期主业采购仅跟随主业成本，补充订单有自己的采购日。
        previous_actual_cogs = baseline_cogs
        current_ar, current_inventory, current_ap = ending_ar, ending_inventory, ending_ap
        current_accrual, current_tax, current_capex_payable = ending_accrual, ending_tax, ending_capex_payable

    return OperatingCycleResult(
        request=request,
        opening_state=opening,
        monthly_rows=tuple(rows),
        deferred_cash_due=_deferred_due(schedules, months[-1]),
        diagnostics=tuple(diagnostics),
    )


_CASH_CATEGORIES = (
    "collection",
    "supplier_payment",
    "payroll_payment",
    "rent_payment",
    "other_expense_payment",
    "tax_payment",
    "fixed_asset_payment",
)


def _derive_opening_state(request: OperatingCycleRequest) -> OperatingOpeningState:
    """Visible simple-mode estimate; detailed users may supply all balances."""

    cogs = request.base_monthly_sales_fen * (10_000 - request.gross_margin_rate_bp) // 10_000
    average_collection_delay = _weighted_delay_days(request.collection_schedule)
    average_supplier_delay = _weighted_delay_days(request.supplier_payment_schedule)
    inventory_days = max(request.production_cycle_days, request.inventory_target_days or 0)
    return OperatingOpeningState(
        accounts_receivable_fen=_round_up_div(request.base_monthly_sales_fen * average_collection_delay, 30),
        inventory_fen=_round_up_div(cogs * inventory_days, 30),
        accounts_payable_fen=_round_up_div(cogs * average_supplier_delay, 30),
    )


def _monthly_sales(
    request: OperatingCycleRequest,
    index: int,
    month: str,
    coverage_days: int,
    month_days: int,
    weights: Mapping[int, int],
) -> int:
    _, month_number = _parse_month_key(month)
    growth_factor_bp = 10_000 + request.sales_monthly_growth_bp * index
    if growth_factor_bp < 0:
        raise OperatingCycleError("NEGATIVE_SALES_PLAN", "销售趋势使某个月销售确认额变为负数。")
    full_month_sales = request.base_monthly_sales_fen * weights[month_number] * growth_factor_bp // 100_000_000
    return _prorate(full_month_sales, coverage_days, month_days)


def _normalized_month_weights(supplied: Mapping[int, int] | None) -> dict[int, int]:
    if supplied is None:
        return {month: 10_000 for month in range(1, 13)}
    total = sum(supplied.values())
    return {month: supplied[month] * 120_000 // total for month in range(1, 13)}


def _schedule_amount(destination: dict[str, int], source_month: str, amount_fen: int, schedule: Sequence[DelayShare]) -> None:
    if amount_fen == 0:
        return
    remaining = amount_fen
    for share in schedule[:-1]:
        allocation = amount_fen * share.share_bp // 10_000
        due_month = _due_month(source_month, share.delay_days)
        destination[due_month] = destination.get(due_month, 0) + allocation
        remaining -= allocation
    final_share = schedule[-1]
    due_month = _due_month(source_month, final_share.delay_days)
    destination[due_month] = destination.get(due_month, 0) + remaining


def _take_due(schedule: dict[str, int], month: str) -> int:
    return schedule.pop(month, 0)


def _deferred_due(schedules: Mapping[str, Mapping[str, int]], last_month: str) -> tuple[DeferredCashDue, ...]:
    rows: list[DeferredCashDue] = []
    for category in _CASH_CATEGORIES:
        for due_month, amount in schedules[category].items():
            if due_month > last_month and amount:
                rows.append(DeferredCashDue(due_month, category, amount))
    return tuple(sorted(rows, key=lambda item: (item.due_month, item.cash_category)))


def _due_month(source_month: str, delay_days: int) -> str:
    year, month = _parse_month_key(source_month)
    anchor_day = min(15, monthrange(year, month)[1])
    due = date(year, month, anchor_day) + timedelta(days=delay_days)
    return _month_key(due)


def _covered_days_in_month(start_date: date, end_date: date, month: str) -> int:
    year, month_number = _parse_month_key(month)
    first = date(year, month_number, 1)
    last = date(year, month_number, monthrange(year, month_number)[1])
    return (min(end_date, last) - max(start_date, first)).days + 1


def _month_days(month: str) -> int:
    year, month_number = _parse_month_key(month)
    return monthrange(year, month_number)[1]


def _prorate(monthly_amount_fen: int, covered_days: int, month_days: int) -> int:
    return monthly_amount_fen * covered_days // month_days


def _weighted_delay_days(schedule: Sequence[DelayShare]) -> int:
    return sum(item.delay_days * item.share_bp for item in schedule) // 10_000


def _assert_nonnegative_states(month: str, *values: int) -> None:
    if any(value < 0 for value in values):
        raise OperatingCycleError("NEGATIVE_OPERATING_STATE", f"{month}出现负的应收、存货、应付或应计状态，已停止。")


def _period_months(start_date: date, end_date: date) -> tuple[str, ...]:
    cursor = date(start_date.year, start_date.month, 1)
    last = date(end_date.year, end_date.month, 1)
    months: list[str] = []
    while cursor <= last:
        months.append(_month_key(cursor))
        cursor = _next_month(cursor)
    return tuple(months)


def _validate_schedule(label: str, schedule: Sequence[DelayShare]) -> None:
    if not schedule:
        raise OperatingCycleError("EMPTY_PAYMENT_SCHEDULE", f"{label}不能为空。")
    if sum(item.share_bp for item in schedule) != 10_000:
        raise OperatingCycleError("INVALID_PAYMENT_SCHEDULE", f"{label}的比例合计必须为100%。")


def _require_positive_fen(name: str, value: object) -> None:
    _require_nonnegative_fen(name, value)
    if value == 0:
        raise OperatingCycleError("ZERO_MONEY", f"{name}必须大于0分。")


def _require_nonnegative_fen(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OperatingCycleError("INVALID_MONEY", f"{name}必须为非负整数分。")


def _require_basis_points(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10_000:
        raise OperatingCycleError("INVALID_BASIS_POINTS", f"{name}必须是0至10000之间的整数。")


def _require_rate(name: str, value: object) -> None:
    _require_basis_points(name, value)


def _require_positive_days(name: str, value: object) -> None:
    _require_nonnegative_days(name, value)
    if value == 0:
        raise OperatingCycleError("ZERO_DAYS", f"{name}必须大于0天。")


def _require_nonnegative_days(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise OperatingCycleError("INVALID_DAYS", f"{name}必须为非负整数天。")


def _round_up_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _validate_month_key(value: str) -> None:
    try:
        year, month = _parse_month_key(value)
        date(year, month, 1)
    except (TypeError, ValueError) as exc:
        raise OperatingCycleError("INVALID_MONTH_KEY", f"月份必须为YYYY-MM：{value!r}") from exc


def _parse_month_key(value: str) -> tuple[int, int]:
    if not isinstance(value, str) or len(value) != 7 or value[4] != "-":
        raise ValueError(value)
    return int(value[:4]), int(value[5:])


def _month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)
