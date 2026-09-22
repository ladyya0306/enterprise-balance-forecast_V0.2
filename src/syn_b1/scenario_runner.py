"""One narrow SYN-B1 scenario orchestrator for acceptance and continuation.

It contains no cash, tax, financing, candidate, or ledger formula.  Its only
job is to connect the already approved owners in their frozen order so later
stages have one repeatable in-memory run boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

from syn_b1.cash_plan import (
    CashComponent, CashFlowClass, CashPlanRequest, Direction, PlanningMode,
    build_monthly_cash_plan,
)
from syn_b1.contract_planning_bridge import ContractPlanningBridgeRequest, ContractPlanningBridgeResult, build_contract_planning_bridge
from syn_b1.funding_contracts import LoanContract, build_loan_schedule
from syn_b1.funding_timeline import (
    FundingLedgerPosting, FundingTimelineState, ShareholderCapitalPlan,
    build_timeline_state, settle_funding_timeline_from_ledger,
)
from syn_b1.final_cash_reconciliation import FinalCashReconciliation, build_final_cash_reconciliation
from syn_b1.initial_capital import InitialCapitalInput, ResolvedInitialCapital, resolve_initial_capital
from syn_b1.ledger_rollforward import LedgerRollforwardResult, LedgerRunManifest, build_ledger_rollforward
from syn_b1.operating_cycle import DelayShare, FixedAssetEvent, OperatingCycleRequest, OperatingCycleResult, OperatingOpeningState, build_monthly_operating_cycle
from syn_b1.statement_rollforward import AssetDepreciationPolicy, StatementOpeningState, StatementRollforwardResult, StatementRollforwardRequest, build_statement_rollforward
from syn_b1.tax_routing import (
    AccountRole, CashCategory, NonBankDayPolicy, ObservedAccount, RoutingRule,
    TaxFrequency, TaxPaymentRule, TaxRoutingRequest, TaxRoutingResult,
    TaxScheduleMode, TaxType, build_tax_and_routing_plan,
)
from syn_b1.calendar_adapter import SynB1BankCalendar
from syn_b1.transaction_planner import (
    CategoryTransactionPolicy, TransactionPlanningRequest, TransactionPlanningResult,
    build_funding_dated_source_events, build_transaction_candidates,
)


class ScenarioRunError(ValueError):
    """A complete fictional scenario cannot safely reach the ledger."""


@dataclass(frozen=True)
class ScenarioRuntimeSettings:
    """All explicit formal controls passed through the one central runner.

    ``None`` on ``ScenarioRunRequest`` retains only the old internal acceptance
    fixture.  Formal I7 profiles always provide this object, so no business
    choice is silently taken from that legacy fixture.
    """

    collection_schedule: tuple[DelayShare, ...]
    supplier_payment_schedule: tuple[DelayShare, ...]
    payroll_payment_schedule: tuple[DelayShare, ...]
    rent_and_utilities_payment_schedule: tuple[DelayShare, ...]
    other_operating_expense_payment_schedule: tuple[DelayShare, ...]
    tax_payment_schedule: tuple[DelayShare, ...]
    sales_month_weights_bp: Mapping[int, int] | None
    inventory_target_days: int | None
    monthly_other_operating_expense_fen: int
    fixed_asset_events: tuple[FixedAssetEvent, ...]
    asset_depreciation_policies: tuple[AssetDepreciationPolicy, ...]
    income_tax_rate_bp: int
    account_role: AccountRole
    routing_rules: Mapping[CashCategory, RoutingRule]
    tax_rules: Mapping[TaxType, TaxPaymentRule]
    bank_calendar: SynB1BankCalendar
    additional_capital_plans: tuple[ShareholderCapitalPlan, ...] = ()
    category_policies: Mapping[CashCategory, CategoryTransactionPolicy] | None = None
    regime_plan: object | None = None
    # 补充业务只由显式画像开启，不改变旧请求。
    supplementary_orders: tuple = ()
    # 默认不开启替代旧主业，保护历史运行结果。
    cycle_orders_only: bool = False
    # 显式月度经营节点默认关闭，旧画像的驱动保持不变。
    business_node_plan: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.account_role, AccountRole):
            raise ScenarioRunError("观察账户角色必须为主账户或次要账户。")
        if not isinstance(self.bank_calendar, SynB1BankCalendar):
            raise ScenarioRunError("正式运行必须提供版本化银行日历。")
        if not 0 <= self.income_tax_rate_bp <= 10_000:
            raise ScenarioRunError("所得税率必须在0%至100%之间。")
        if set(self.tax_rules) != set(TaxType):
            raise ScenarioRunError("正式运行必须分别提供三种税费规则。")
        if not all(isinstance(rule, RoutingRule) for rule in self.routing_rules.values()):
            raise ScenarioRunError("账户路由规则无效。")
        if not all(isinstance(item, ShareholderCapitalPlan) for item in self.additional_capital_plans):
            raise ScenarioRunError("后续股东注资计划无效。")


@dataclass(frozen=True)
class ScenarioRunRequest:
    """A deliberately small, all-fictional public-control candidate.

    Advanced controls remain outside this request until they have their own
    frozen evidence.  The request is not a BAT input and does not write files.
    """

    synthetic_account_id: str
    run_id: str
    start_date: date
    end_date: date
    random_seed: int
    base_monthly_sales_fen: int
    gross_margin_rate_bp: int
    collection_delay_days: int
    supplier_delay_days: int
    production_cycle_days: int
    monthly_payroll_fen: int
    monthly_rent_and_utilities_fen: int
    initial_capital: InitialCapitalInput
    sales_monthly_growth_bp: int = 0
    loan_contracts: tuple[LoanContract, ...] = ()
    transaction_policy: CategoryTransactionPolicy = CategoryTransactionPolicy()
    runtime_settings: ScenarioRuntimeSettings | None = None

    def __post_init__(self) -> None:
        if not self.synthetic_account_id or not self.run_id:
            raise ScenarioRunError("虚构账户编号和运行编号不能为空。")
        if self.start_date > self.end_date:
            raise ScenarioRunError("结束日期不能早于开始日期。")
        if not isinstance(self.initial_capital, InitialCapitalInput):
            raise ScenarioRunError("新建虚构企业必须提供首次股东注资。")

    def with_end_date(self, end_date: date) -> "ScenarioRunRequest":
        """Return a new period request without mutating the frozen history request."""
        return ScenarioRunRequest(
            synthetic_account_id=self.synthetic_account_id,
            run_id=self.run_id,
            start_date=self.start_date,
            end_date=end_date,
            random_seed=self.random_seed,
            base_monthly_sales_fen=self.base_monthly_sales_fen,
            gross_margin_rate_bp=self.gross_margin_rate_bp,
            collection_delay_days=self.collection_delay_days,
            supplier_delay_days=self.supplier_delay_days,
            production_cycle_days=self.production_cycle_days,
            monthly_payroll_fen=self.monthly_payroll_fen,
            monthly_rent_and_utilities_fen=self.monthly_rent_and_utilities_fen,
            initial_capital=self.initial_capital,
            sales_monthly_growth_bp=self.sales_monthly_growth_bp,
            loan_contracts=self.loan_contracts,
            transaction_policy=self.transaction_policy,
            runtime_settings=self.runtime_settings,
        )


@dataclass(frozen=True)
class ScenarioRunResult:
    request: ScenarioRunRequest
    initial_capital: ResolvedInitialCapital
    operating_cycle: OperatingCycleResult
    statements: StatementRollforwardResult
    funding_timeline: FundingTimelineState
    bridge: ContractPlanningBridgeResult
    tax_routing: TaxRoutingResult
    transaction_plan: TransactionPlanningResult
    ledger: LedgerRollforwardResult
    final_cash_reconciliation: FinalCashReconciliation


@dataclass(frozen=True)
class ScenarioPreLedgerPlan:
    """共享的合同、报表和税费计划边界；止于税费候选且不建立逐笔账。"""
    request: ScenarioRunRequest
    initial_capital: ResolvedInitialCapital
    settings: ScenarioRuntimeSettings
    operating_cycle: OperatingCycleResult
    statements: StatementRollforwardResult
    funding_timeline: FundingTimelineState
    bridge: ContractPlanningBridgeResult
    tax_routing: TaxRoutingResult


# 中文逐行注释：抽出中央运行器在逐笔候选前的成熟计划链，供正式运行和静态预算共同复用。
def build_scenario_preledger_plan(request: ScenarioRunRequest) -> ScenarioPreLedgerPlan:
    """运行唯一经营、报表、合同桥和税费规划链；不生成transaction candidates或ledger。"""
    initial_capital = resolve_initial_capital(
        request.initial_capital, start_date=request.start_date, root_seed=request.random_seed,
    )
    settings = request.runtime_settings or _legacy_runtime_settings(request)
    operating = build_monthly_operating_cycle(OperatingCycleRequest(
        synthetic_account_id=request.synthetic_account_id,
        start_date=request.start_date,
        end_date=request.end_date,
        random_seed=request.random_seed,
        base_monthly_sales_fen=request.base_monthly_sales_fen,
        gross_margin_rate_bp=request.gross_margin_rate_bp,
        collection_schedule=settings.collection_schedule,
        production_cycle_days=request.production_cycle_days,
        supplier_payment_schedule=settings.supplier_payment_schedule,
        monthly_payroll_fen=request.monthly_payroll_fen,
        monthly_rent_and_utilities_fen=request.monthly_rent_and_utilities_fen,
        tax_and_surcharge_rate_bp=0,
        sales_monthly_growth_bp=request.sales_monthly_growth_bp,
        sales_month_weights_bp=settings.sales_month_weights_bp,
        inventory_target_days=settings.inventory_target_days,
        monthly_other_operating_expense_fen=settings.monthly_other_operating_expense_fen,
        payroll_payment_schedule=settings.payroll_payment_schedule,
        rent_and_utilities_payment_schedule=settings.rent_and_utilities_payment_schedule,
        other_operating_expense_payment_schedule=settings.other_operating_expense_payment_schedule,
        tax_payment_schedule=settings.tax_payment_schedule,
        fixed_asset_events=settings.fixed_asset_events,
        opening_state=OperatingOpeningState(),
        regime_plan=settings.regime_plan,
        # 同一份冻结订单交给唯一经营账务所有者。
        supplementary_orders=settings.supplementary_orders,
        # 只传递已确认的业务模式，不另算金额。
        cycle_orders_only=settings.cycle_orders_only,
        # 仅把已解析节点传给唯一的经营和记账链。
        business_node_plan=settings.business_node_plan,
    ))
    cash_plan = build_monthly_cash_plan(CashPlanRequest(
        synthetic_account_id=request.synthetic_account_id,
        start_date=request.start_date,
        end_date=request.end_date,
        opening_balance_fen=0,
        planning_mode=PlanningMode.FIXED_BANK_TOTALS,
        fixed_total_inflow_fen=sum(row.cash_collections_fen for row in operating.monthly_rows),
        fixed_total_outflow_fen=sum(
            row.operating_cash_outflow_fen + row.fixed_asset_cash_payment_fen
            for row in operating.monthly_rows
        ),
        components=(
            CashComponent(
                "collections", CashFlowClass.OPERATING, Direction.INFLOW,
                {row.month: row.cash_collections_fen for row in operating.monthly_rows},
            ),
            CashComponent(
                "payments", CashFlowClass.OPERATING, Direction.OUTFLOW,
                {row.month: row.operating_cash_outflow_fen for row in operating.monthly_rows},
            ),
            CashComponent(
                "fixed_asset_payments", CashFlowClass.INVESTING, Direction.OUTFLOW,
                {row.month: row.fixed_asset_cash_payment_fen for row in operating.monthly_rows},
            ),
        ),
    ))
    statements = build_statement_rollforward(StatementRollforwardRequest(
        operating_cycle=operating,
        cash_plan=cash_plan,
        opening_state=StatementOpeningState(bank_cash_fen=0),
        income_tax_rate_bp=settings.income_tax_rate_bp,
        asset_depreciation_policies=settings.asset_depreciation_policies,
    ))
    loan_schedules = tuple(build_loan_schedule(item) for item in request.loan_contracts)
    timeline = build_timeline_state(loan_schedules, (initial_capital.plan, *settings.additional_capital_plans))
    financing_inflow = sum(
        item.event.amount_fen
        for item in timeline.pending_tail
        if item.event.cash_direction.value == "inflow" and request.start_date <= item.event.due_date <= request.end_date
    )
    financing_outflow = sum(
        item.event.amount_fen
        for item in timeline.pending_tail
        if item.event.cash_direction.value == "outflow" and request.start_date <= item.event.due_date <= request.end_date
    )
    bridge = build_contract_planning_bridge(ContractPlanningBridgeRequest(
        pre_financing_statements=statements,
        timeline_state=timeline,
        planning_mode=PlanningMode.FIXED_BANK_TOTALS,
        minimum_liquidity_fen=0,
        fixed_total_inflow_fen=cash_plan.total_inflow_fen + financing_inflow,
        fixed_total_outflow_fen=cash_plan.total_outflow_fen + financing_outflow,
        allow_monthly_cash_shortage_for_daily_stop=request.runtime_settings is not None,
    ))
    if not bridge.is_feasible or bridge.post_contract_statements is None:
        raise ScenarioRunError(f"用户已设合同与固定总收付不能对平：{bridge.failure}")
    tax_routing = build_tax_and_routing_plan(TaxRoutingRequest(
        operating_cycle=operating,
        post_contract_statements=bridge.post_contract_statements,
        observed_account=ObservedAccount(request.synthetic_account_id, settings.account_role, 0),
        routing_rules=settings.routing_rules,
        tax_rules=settings.tax_rules,
        is_bank_workday=settings.bank_calendar.is_bank_workday,
    ))
    # 中文逐行注释：返回止于现有税费候选的共享计划，不进入逐笔交易或余额账本。
    return ScenarioPreLedgerPlan(request, initial_capital, settings, operating, statements, timeline, bridge, tax_routing)


# 中文逐行注释：完整运行器复用同一前置计划，再接入原有逐笔候选、账本和最终核对。
def build_scenario_run(request: ScenarioRunRequest) -> ScenarioRunResult:
    """在共享税费前置计划后运行原有逐笔账本链。"""
    preledger = build_scenario_preledger_plan(request)
    # 中文逐行注释：逐笔入口读取同一订单的合同日期，不复制现金计算。
    from syn_b1.supplementary_orders import order_cash_events
    # 合同节点也只向原逐笔模块提供已对平的日期。
    from syn_b1.contract_business import contract_cash_events
    # 原税费、融资和普通业务仍走既有逐笔入口。
    plan = build_transaction_candidates(TransactionPlanningRequest(
        tax_routing_result=preledger.tax_routing,
        random_seed=request.random_seed,
        default_policy=preledger.request.transaction_policy,
        category_policies=preledger.settings.category_policies,
        dated_source_events=build_funding_dated_source_events(preledger.tax_routing, preledger.funding_timeline),
        # 新业务现金已包含在月合计中，逐笔模块会扣除后再拆普通业务。
        business_order_events=order_cash_events(preledger.settings.supplementary_orders, request.start_date, request.end_date),
        # 销售、采购、工资、租赁和设备合同在原账本统一记账。
        contract_source_events=contract_cash_events(preledger.settings.business_node_plan, request.start_date, request.end_date),
    ))
    ledger = build_ledger_rollforward(plan, LedgerRunManifest(request.run_id, request.synthetic_account_id))
    settled_timeline = _settle_funding_after_ledger(preledger.funding_timeline, plan, ledger, request.end_date)
    final_cash_reconciliation = build_final_cash_reconciliation(preledger.tax_routing, ledger)
    return ScenarioRunResult(
        request, preledger.initial_capital, preledger.operating_cycle, preledger.statements, settled_timeline,
        preledger.bridge, preledger.tax_routing, plan, ledger, final_cash_reconciliation,
    )


def _settle_funding_after_ledger(
    timeline: FundingTimelineState,
    plan: TransactionPlanningResult,
    ledger: LedgerRollforwardResult,
    planned_end_date: date,
) -> FundingTimelineState:
    """Connect I4B's posted prefix back to the one frozen funding timeline."""

    candidate_by_id = {item.candidate_id: item for item in plan.candidates}
    postings: list[FundingLedgerPosting] = []
    balance_before = ledger.opening_balance_fen
    for movement, transaction in zip(plan.cash_preview.completed_movements, ledger.transactions, strict=True):
        candidate = candidate_by_id[movement.movement_id]
        if candidate.funding_tail_item_id is not None:
            postings.append(FundingLedgerPosting(
                tail_item_id=candidate.funding_tail_item_id,
                booking_date=transaction.booking_datetime.date(),
                posted_amount_fen=transaction.credit_fen or transaction.debit_fen,
                cash_before_fen=balance_before,
                cash_after_fen=transaction.post_transaction_balance_fen,
            ))
        balance_before = transaction.post_transaction_balance_fen
    blocked_tail_id = None
    blocked_cash = None
    fact = ledger.fact_sheet
    if fact is not None:
        blocked_candidate = candidate_by_id.get(fact.blocked_movement_id)
        if blocked_candidate is not None:
            blocked_tail_id = blocked_candidate.funding_tail_item_id
            blocked_cash = fact.available_balance_before_block_fen if blocked_tail_id is not None else None
    return settle_funding_timeline_from_ledger(
        timeline,
        tuple(postings),
        generated_through_date=planned_end_date if ledger.is_complete else fact.blocked_booking_datetime.date(),
        blocked_tail_item_id=blocked_tail_id,
        blocked_cash_fen=blocked_cash,
    )


def _legacy_runtime_settings(request: ScenarioRunRequest) -> ScenarioRuntimeSettings:
    """Keep pre-I7 acceptance tests stable; this fixture is never a formal profile."""
    return ScenarioRuntimeSettings(
        collection_schedule=(DelayShare(request.collection_delay_days, 10_000),),
        supplier_payment_schedule=(DelayShare(request.supplier_delay_days, 10_000),),
        payroll_payment_schedule=(DelayShare(0, 10_000),),
        rent_and_utilities_payment_schedule=(DelayShare(0, 10_000),),
        other_operating_expense_payment_schedule=(DelayShare(0, 10_000),),
        tax_payment_schedule=(DelayShare(0, 10_000),),
        sales_month_weights_bp=None,
        inventory_target_days=None,
        monthly_other_operating_expense_fen=0,
        fixed_asset_events=(),
        asset_depreciation_policies=(),
        income_tax_rate_bp=2_000,
        account_role=AccountRole.PRIMARY,
        routing_rules={category: RoutingRule(True, 10_000) for category in CashCategory},
        tax_rules={tax_type: _monthly_tax_rule(tax_type) for tax_type in TaxType},
        bank_calendar=SynB1BankCalendar(request.end_date, "legacy_acceptance_fixture_1_0"),
    )


def _monthly_tax_rule(tax_type: TaxType) -> TaxPaymentRule:
    return TaxPaymentRule(
        enabled=True,
        frequency=TaxFrequency.MONTHLY,
        schedule_mode=TaxScheduleMode.RECURRING_DAY,
        payer_account_role=AccountRole.PRIMARY,
        non_bank_day_policy=NonBankDayPolicy.KEEP_CALENDAR_DATE,
        recurring_debit_day=28,
        effective_vat_burden_rate_bp=100 if tax_type is TaxType.SIMULATED_VAT else None,
        surcharge_rate_on_simulated_vat_bp=1_200 if tax_type is TaxType.SURCHARGE else None,
    )
