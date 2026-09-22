"""SYN-B1-I2A: orchestration for the three monthly planning modes.

The module intentionally contains no independent cash, interest, tax,
statement, or random-generation formula.  It only validates the joint request
and passes named monthly results through I1A, I1, I1B and I2 in the frozen
order.  It does not create transactions, files, databases or model inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum
import hashlib
import json
from typing import Sequence

from syn_b1.cash_plan import (
    BalanceRange,
    CashComponent,
    CashFlowClass,
    CashPlanRequest,
    Direction,
    FinalNamedCashPlanResult,
    NamedMonthlyCashComponent,
    PlanningMode,
    build_final_named_cash_plan,
    build_monthly_cash_plan,
)
from syn_b1.financing_lifecycle import (
    DeclaredFinancingEvent,
    FinancingClosingState,
    FinancingInstrument,
    FinancingLifecycleRequest,
    FinancingLifecycleResult,
    FinancingOpeningState,
    FinancingReservation,
    FixedFinancingPolicy,
    build_financing_lifecycle,
)
from syn_b1.operating_cycle import OperatingCycleRequest, OperatingCycleResult, build_monthly_operating_cycle
from syn_b1.statement_rollforward import (
    AssetDepreciationPolicy,
    StatementOpeningState,
    StatementRollforwardRequest,
    StatementRollforwardResult,
    build_statement_rollforward,
)


class JointMonthlyPlanningError(ValueError):
    """A frozen joint-planning requirement has not been met."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def resolved_profile_id_for(operating_request: OperatingCycleRequest) -> str:
    """Return the immutable identity of one resolved fictional enterprise.

    I2A has no profile file or database by design.  The identity therefore
    derives from the complete I1A request (including its seed), rather than
    trusting a user-entered nickname.  A later UI may display a friendly name,
    but it must preserve this identity alongside it.
    """

    payload = json.dumps(
        asdict(operating_request),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_profile_json_value,
    )
    return f"profile_sha256_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _profile_json_value(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    raise TypeError(f"无法计算虚构企业画像指纹：{type(value).__name__}")


@dataclass(frozen=True)
class JointMonthlyPlanningRequest:
    """One fictional enterprise request for all three planning modes."""

    synthetic_account_id: str
    start_date: date
    end_date: date
    planning_mode: PlanningMode
    opening_bank_cash_fen: int
    minimum_liquidity_fen: int
    operating_cycle_request: OperatingCycleRequest
    statement_opening_state: StatementOpeningState
    income_tax_rate_bp: int
    financing_instruments: tuple[FinancingInstrument, ...] = ()
    financing_opening_state: FinancingOpeningState = FinancingOpeningState()
    target_ending_balance_range: BalanceRange | None = None
    fixed_total_inflow_fen: int | None = None
    fixed_total_outflow_fen: int | None = None
    fixed_financing_policy: FixedFinancingPolicy | None = None
    declared_financing_events: tuple[DeclaredFinancingEvent, ...] = ()
    asset_depreciation_policies: tuple[AssetDepreciationPolicy, ...] = ()
    resolved_profile_id: str | None = None
    auto_profile_confirmed: bool = False

    def __post_init__(self) -> None:
        if not self.synthetic_account_id:
            raise JointMonthlyPlanningError("EMPTY_ACCOUNT_ID", "联合规划请求必须填写虚构账户编号。")
        if self.start_date > self.end_date:
            raise JointMonthlyPlanningError("INVALID_PERIOD", "联合规划开始日期不能晚于结束日期。")
        if self.opening_bank_cash_fen < 0 or self.minimum_liquidity_fen < 0:
            raise JointMonthlyPlanningError("INVALID_MONEY", "期初余额和最低保留余额必须为非负整数分。")
        operating = self.operating_cycle_request
        if (
            operating.synthetic_account_id != self.synthetic_account_id
            or operating.start_date != self.start_date
            or operating.end_date != self.end_date
        ):
            raise JointMonthlyPlanningError("OPERATING_REQUEST_MISMATCH", "I1A经营请求必须与联合请求使用同一账户和期间。")
        if self.statement_opening_state.bank_cash_fen != self.opening_bank_cash_fen:
            raise JointMonthlyPlanningError("OPENING_CASH_MISMATCH", "期初资产负债表银行现金必须等于联合请求期初余额。")
        if self.planning_mode is PlanningMode.FIXED_BANK_TOTALS:
            if self.fixed_total_inflow_fen is None or self.fixed_total_outflow_fen is None:
                raise JointMonthlyPlanningError("MISSING_FIXED_TOTAL", "固定总额方式必须填写总流入和总流出。")
            if self.fixed_total_inflow_fen < 0 or self.fixed_total_outflow_fen < 0:
                raise JointMonthlyPlanningError("INVALID_MONEY", "固定总收付必须为非负整数分。")
            if self.fixed_financing_policy is None:
                raise JointMonthlyPlanningError("MISSING_FIXED_FINANCING_POLICY", "固定总额方式必须选择一种融资政策。")
            if self.target_ending_balance_range is not None:
                raise JointMonthlyPlanningError("UNUSED_TARGET_RANGE", "固定总额方式不能同时独立填写期末余额目标。")
        else:
            if self.fixed_total_inflow_fen is not None or self.fixed_total_outflow_fen is not None:
                raise JointMonthlyPlanningError("UNUSED_FIXED_TOTAL", "非固定总额方式不能填写固定总收付。")
            if self.target_ending_balance_range is None:
                raise JointMonthlyPlanningError("MISSING_TARGET_RANGE", "目标余额和自动方式必须填写期末余额目标范围。")
        if self.planning_mode is PlanningMode.CONSTRAINED_AUTO:
            if not self.resolved_profile_id:
                raise JointMonthlyPlanningError("AUTO_PROFILE_NOT_CONFIRMED", "自动方式必须保存一份完整虚构企业画像编号。")
            if not self.auto_profile_confirmed:
                raise JointMonthlyPlanningError("AUTO_PROFILE_NOT_CONFIRMED", "自动方式必须先确认该企业画像和月度计划。")
            if self.resolved_profile_id != resolved_profile_id_for(operating):
                raise JointMonthlyPlanningError(
                    "AUTO_PROFILE_BINDING_MISMATCH",
                    "自动方式的企业画像编号必须由完整经营参数和随机种子生成，不能用同一编号指向不同设定。",
                )
        elif self.resolved_profile_id is not None or self.auto_profile_confirmed:
            raise JointMonthlyPlanningError("UNUSED_AUTO_PROFILE", "只有自动方式可以填写企业画像确认信息。")
        if self.planning_mode is not PlanningMode.FIXED_BANK_TOTALS and self.declared_financing_events:
            raise JointMonthlyPlanningError("DECLARED_EVENTS_ONLY_FOR_FIXED_MODE", "用户声明融资计划只允许用于固定总额方式。")
        if (
            self.planning_mode is PlanningMode.FIXED_BANK_TOTALS
            and self.fixed_financing_policy is FixedFinancingPolicy.MINIMUM_REQUIRED_WITHIN_RESERVE
            and self.declared_financing_events
        ):
            raise JointMonthlyPlanningError(
                "MIXED_FIXED_FINANCING_POLICY",
                "固定总额选择系统最小必要融资时，不能同时填写用户声明融资计划。",
            )


@dataclass(frozen=True)
class JointPlanningFailure:
    planning_mode: PlanningMode
    first_conflict_month_or_date: str
    conflict_code: str
    required_amount_fen: int
    available_or_reserved_amount_fen: int
    affected_named_components: tuple[str, ...]
    allowed_user_adjustments_cn: tuple[str, ...]


@dataclass(frozen=True)
class JointMonthlyPlanningResult:
    request: JointMonthlyPlanningRequest
    is_feasible: bool
    operating_cycle: OperatingCycleResult | None
    pre_financing_cash_plan: object | None
    pre_financing_statements: StatementRollforwardResult | None
    financing_result: FinancingLifecycleResult | None
    final_named_plan: FinalNamedCashPlanResult | None
    closing_financing_state: FinancingClosingState | None
    failure: JointPlanningFailure | None = None


def build_joint_monthly_plan(request: JointMonthlyPlanningRequest) -> JointMonthlyPlanningResult:
    """Build the frozen I1A → I1 → I1B → I2 joint monthly plan."""

    operating = build_monthly_operating_cycle(request.operating_cycle_request)
    operating_components = _operating_components(operating)
    # The pre-financing plan has only named operating/investing cash.  It uses
    # I1's existing target-mode carrier so fixed-mode reserves never enter the
    # pre-financing balance.
    pre_cash = build_monthly_cash_plan(
        CashPlanRequest(
            synthetic_account_id=request.synthetic_account_id,
            start_date=request.start_date,
            end_date=request.end_date,
            opening_balance_fen=request.opening_bank_cash_fen,
            planning_mode=PlanningMode.TARGET_ENDING_BALANCE,
            components=operating_components,
            minimum_liquidity_fen=request.minimum_liquidity_fen,
            target_ending_balance_fen=request.target_ending_balance_range or BalanceRange(0, 10**18),
        )
    )
    pre_statements = build_statement_rollforward(
        StatementRollforwardRequest(
            operating_cycle=operating,
            cash_plan=pre_cash,
            opening_state=request.statement_opening_state,
            income_tax_rate_bp=request.income_tax_rate_bp,
            asset_depreciation_policies=request.asset_depreciation_policies,
        )
    )
    reservation = _fixed_reservation(request, pre_cash)
    financing = build_financing_lifecycle(
        FinancingLifecycleRequest(
            cash_plan=pre_cash,
            statements=pre_statements,
            instruments=request.financing_instruments,
            planning_mode=request.planning_mode,
            ending_balance_target_range=request.target_ending_balance_range,
            fixed_financing_reservation=reservation,
            fixed_financing_policy=request.fixed_financing_policy,
            declared_events=request.declared_financing_events,
            opening_state=request.financing_opening_state,
        )
    )
    if not financing.is_feasible:
        assert financing.failure is not None
        failure = financing.failure
        return JointMonthlyPlanningResult(
            request=request,
            is_feasible=False,
            operating_cycle=operating,
            pre_financing_cash_plan=pre_cash,
            pre_financing_statements=pre_statements,
            financing_result=financing,
            final_named_plan=None,
            closing_financing_state=None,
            failure=JointPlanningFailure(
                planning_mode=request.planning_mode,
                first_conflict_month_or_date=failure.month,
                conflict_code=failure.conflict_code,
                required_amount_fen=failure.shortfall_fen,
                available_or_reserved_amount_fen=_available_amount(request, reservation),
                affected_named_components=tuple(item.instrument_id for item in request.financing_instruments),
                allowed_user_adjustments_cn=failure.adjustable_items_cn,
            ),
        )

    final_plan = build_final_named_cash_plan(
        start_date=request.start_date,
        end_date=request.end_date,
        opening_balance_fen=request.opening_bank_cash_fen,
        components=_final_components(operating, financing),
    )
    _assert_joint_outputs_align(final_plan, financing)
    return JointMonthlyPlanningResult(
        request=request,
        is_feasible=True,
        operating_cycle=operating,
        pre_financing_cash_plan=pre_cash,
        pre_financing_statements=pre_statements,
        financing_result=financing,
        final_named_plan=final_plan,
        closing_financing_state=financing.closing_state,
    )


def _operating_components(operating: OperatingCycleResult) -> tuple[CashComponent, ...]:
    inflows = {row.month: row.cash_collections_fen for row in operating.monthly_rows if row.cash_collections_fen}
    outflows = {row.month: row.operating_cash_outflow_fen for row in operating.monthly_rows if row.operating_cash_outflow_fen}
    components: list[CashComponent] = [
        CashComponent("operating_collections", CashFlowClass.OPERATING, Direction.INFLOW, inflows),
    ]
    if outflows:
        components.append(CashComponent("operating_payments", CashFlowClass.OPERATING, Direction.OUTFLOW, outflows))
    capex = {row.month: row.fixed_asset_cash_payment_fen for row in operating.monthly_rows if row.fixed_asset_cash_payment_fen}
    if capex:
        components.append(CashComponent("investing_asset_payments", CashFlowClass.INVESTING, Direction.OUTFLOW, capex))
    return tuple(components)


def _fixed_reservation(request: JointMonthlyPlanningRequest, pre_cash) -> FinancingReservation | None:
    if request.planning_mode is not PlanningMode.FIXED_BANK_TOTALS:
        return None
    assert request.fixed_total_inflow_fen is not None
    assert request.fixed_total_outflow_fen is not None
    inflow = request.fixed_total_inflow_fen - pre_cash.total_inflow_fen
    outflow = request.fixed_total_outflow_fen - pre_cash.total_outflow_fen
    if inflow < 0:
        raise JointMonthlyPlanningError("FIXED_NAMED_INFLOW_EXCEEDS_TOTAL", "已命名经营和投资流入超过固定总流入。")
    if outflow < 0:
        raise JointMonthlyPlanningError("FIXED_NAMED_OUTFLOW_EXCEEDS_TOTAL", "已命名经营和投资流出超过固定总流出。")
    return FinancingReservation(inflow, outflow)


def _final_components(
    operating: OperatingCycleResult, financing: FinancingLifecycleResult
) -> tuple[NamedMonthlyCashComponent, ...]:
    components: list[NamedMonthlyCashComponent] = []
    for row in operating.monthly_rows:
        if row.cash_collections_fen:
            components.append(NamedMonthlyCashComponent(f"operating_collection_{row.month}", row.month, "operating", Direction.INFLOW, row.cash_collections_fen, "I1A"))
        if row.operating_cash_outflow_fen:
            components.append(NamedMonthlyCashComponent(f"operating_payment_{row.month}", row.month, "operating", Direction.OUTFLOW, row.operating_cash_outflow_fen, "I1A"))
        if row.fixed_asset_cash_payment_fen:
            components.append(NamedMonthlyCashComponent(f"investing_asset_payment_{row.month}", row.month, "investing", Direction.OUTFLOW, row.fixed_asset_cash_payment_fen, "I1A"))
    for index, event in enumerate(financing.schedule_events):
        direction = Direction.INFLOW if event.event_type in {"drawdown", "capital_injection"} else Direction.OUTFLOW
        components.append(
            NamedMonthlyCashComponent(
                component_id=f"financing_{event.instrument_id}_{event.event_type}_{index}",
                month=event.month,
                cash_flow_class="financing",
                direction=direction,
                amount_fen=event.amount_fen,
                source_stage="I2",
                truth_visibility="restricted_reason_and_schedule",
            )
        )
    return tuple(components)


def _assert_joint_outputs_align(final_plan: FinalNamedCashPlanResult, financing: FinancingLifecycleResult) -> None:
    if not financing.is_fully_reconciled:
        raise JointMonthlyPlanningError("FINANCED_STATEMENTS_NOT_RECONCILED", "融资后的现金、债务或三张表未完全勾稽。")
    by_month = {row.month: row for row in final_plan.monthly_rows}
    for row in financing.financing_cash_plan_rows:
        final = by_month[row.month]
        if final.ending_balance_fen != row.ending_bank_cash_fen:
            raise JointMonthlyPlanningError("FINAL_CASH_PLAN_MISMATCH", "最终命名计划与I2融资后现金余额不一致。")
    if any(item.amount_fen < 0 for item in final_plan.components):
        raise AssertionError("SYN-B1最终命名计划内部出现负金额")


def _available_amount(request: JointMonthlyPlanningRequest, reservation: FinancingReservation | None) -> int:
    if reservation is not None:
        return reservation.inflow_fen + reservation.outflow_fen
    return sum(item.limit_fen for item in request.financing_instruments)
