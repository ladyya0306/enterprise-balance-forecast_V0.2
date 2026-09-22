"""SYN-B1-I1: monthly cash planning and pre-financing feasibility only.

This module deliberately stops before transaction scheduling, financing
posting, ledger creation, CSV/SQLite output, or model training.  It turns a
fully fictional enterprise plan into monthly operating/investing budgets and
a conservative pre-financing cash-gap preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Mapping, Sequence


class CashPlanError(ValueError):
    """A plan cannot be formed without changing a user-declared constraint."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PlanningMode(str, Enum):
    FIXED_BANK_TOTALS = "fixed_bank_totals"
    TARGET_ENDING_BALANCE = "target_ending_balance"
    CONSTRAINED_AUTO = "constrained_auto"


class CashFlowClass(str, Enum):
    OPERATING = "operating"
    INVESTING = "investing"


class Direction(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"


@dataclass(frozen=True)
class NamedMonthlyCashComponent:
    """One fully named monthly item in the final joint plan.

    This is deliberately a monthly planning item, rather than a transaction.
    It lets the I1 cash module check the final operating, investing and
    financing cash identity without teaching the joint orchestrator a second
    cash formula.
    """

    component_id: str
    month: str
    cash_flow_class: str
    direction: Direction
    amount_fen: int
    source_stage: str
    truth_visibility: str = "public_amount_only"

    def __post_init__(self) -> None:
        if not self.component_id:
            raise CashPlanError("EMPTY_NAMED_COMPONENT_ID", "最终命名计划的项目编号不能为空。")
        _validate_month_key(self.month)
        if self.cash_flow_class not in {"operating", "investing", "financing"}:
            raise CashPlanError("INVALID_NAMED_CASH_CLASS", "最终命名计划的现金类别必须为经营、投资或筹资。")
        if self.source_stage not in {"I1A", "I2", "owner_declared_other"}:
            raise CashPlanError("INVALID_NAMED_COMPONENT_SOURCE", "最终命名计划项目必须说明来源阶段。")
        if self.truth_visibility not in {"public_amount_only", "restricted_reason_and_schedule"}:
            raise CashPlanError("INVALID_TRUTH_VISIBILITY", "最终命名计划项目的可见性标记无效。")
        _require_nonnegative_fen("amount_fen", self.amount_fen)


@dataclass(frozen=True)
class FinalMonthlyCashPlanRow:
    """A reconciled final row containing only named monthly cash amounts."""

    month: str
    opening_balance_fen: int
    operating_inflow_fen: int
    operating_outflow_fen: int
    investing_inflow_fen: int
    investing_outflow_fen: int
    financing_inflow_fen: int
    financing_outflow_fen: int
    total_inflow_fen: int
    total_outflow_fen: int
    ending_balance_fen: int


@dataclass(frozen=True)
class FinalNamedCashPlanResult:
    """The only cash plan eligible for a future daily/transaction drilldown."""

    monthly_rows: tuple[FinalMonthlyCashPlanRow, ...]
    components: tuple[NamedMonthlyCashComponent, ...]
    total_inflow_fen: int
    total_outflow_fen: int
    ending_balance_fen: int


def build_final_named_cash_plan(
    *,
    start_date: date,
    end_date: date,
    opening_balance_fen: int,
    components: Sequence[NamedMonthlyCashComponent],
) -> FinalNamedCashPlanResult:
    """Use the central I1 cash identity to reconcile a fully named final plan.

    Callers must provide every operating, investing and financing monthly
    amount by name.  There is intentionally no residual or balancing item.
    """

    _require_nonnegative_fen("opening_balance_fen", opening_balance_fen)
    months = _period_months(start_date, end_date)
    month_set = set(months)
    component_keys = [(item.component_id, item.month) for item in components]
    if len(component_keys) != len(set(component_keys)):
        raise CashPlanError("DUPLICATE_NAMED_COMPONENT", "最终命名计划中同一项目编号和月份不能重复。")
    if any(item.month not in month_set for item in components):
        raise CashPlanError("NAMED_COMPONENT_MONTH_OUT_OF_PERIOD", "最终命名计划包含期间外月份。")

    buckets = {
        month: {
            "operating_inflow": 0,
            "operating_outflow": 0,
            "investing_inflow": 0,
            "investing_outflow": 0,
            "financing_inflow": 0,
            "financing_outflow": 0,
        }
        for month in months
    }
    for item in components:
        buckets[item.month][f"{item.cash_flow_class}_{item.direction.value}"] += item.amount_fen

    rows: list[FinalMonthlyCashPlanRow] = []
    current = opening_balance_fen
    for month in months:
        values = buckets[month]
        inflow = values["operating_inflow"] + values["investing_inflow"] + values["financing_inflow"]
        outflow = values["operating_outflow"] + values["investing_outflow"] + values["financing_outflow"]
        ending = current + inflow - outflow
        _assert_cash_identity(current, inflow, outflow, ending)
        rows.append(
            FinalMonthlyCashPlanRow(
                month=month,
                opening_balance_fen=current,
                operating_inflow_fen=values["operating_inflow"],
                operating_outflow_fen=values["operating_outflow"],
                investing_inflow_fen=values["investing_inflow"],
                investing_outflow_fen=values["investing_outflow"],
                financing_inflow_fen=values["financing_inflow"],
                financing_outflow_fen=values["financing_outflow"],
                total_inflow_fen=inflow,
                total_outflow_fen=outflow,
                ending_balance_fen=ending,
            )
        )
        current = ending
    return FinalNamedCashPlanResult(
        monthly_rows=tuple(rows),
        components=tuple(components),
        total_inflow_fen=sum(row.total_inflow_fen for row in rows),
        total_outflow_fen=sum(row.total_outflow_fen for row in rows),
        ending_balance_fen=current,
    )


@dataclass(frozen=True)
class BalanceRange:
    """Inclusive target range in integer fen."""

    minimum_fen: int
    maximum_fen: int

    def __post_init__(self) -> None:
        _require_nonnegative_fen("minimum_fen", self.minimum_fen)
        _require_nonnegative_fen("maximum_fen", self.maximum_fen)
        if self.minimum_fen > self.maximum_fen:
            raise CashPlanError("INVALID_TARGET_RANGE", "期末余额目标下限不能大于上限。")


@dataclass(frozen=True)
class CashComponent:
    """One normal cash component, already allocated to calendar months.

    ``monthly_amounts_fen`` uses keys in ``YYYY-MM`` form.  Values are
    positive amounts; direction determines whether they are an inflow or an
    outflow.  This is a plan entry, not a bank transaction.
    """

    component_id: str
    cash_flow_class: CashFlowClass
    direction: Direction
    monthly_amounts_fen: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.component_id:
            raise CashPlanError("EMPTY_COMPONENT_ID", "普通现金项目编号不能为空。")
        if not self.monthly_amounts_fen:
            raise CashPlanError("EMPTY_COMPONENT", f"现金项目{self.component_id}没有月度金额。")
        for month, amount in self.monthly_amounts_fen.items():
            _validate_month_key(month)
            _require_nonnegative_fen(f"{self.component_id}:{month}", amount)


@dataclass(frozen=True)
class SpecialEvent:
    """A selected special event that must occupy the plan before scheduling.

    I1 stores no transaction time, counterparty, or reason in model-visible
    data.  Later I4 will isolate cause information in restricted scenario
    truth.  I1 only uses a fictional event identifier for diagnostics.
    """

    event_id: str
    cash_flow_class: CashFlowClass
    direction: Direction
    month: str
    amount_fen: int

    def __post_init__(self) -> None:
        if not self.event_id:
            raise CashPlanError("EMPTY_EVENT_ID", "特殊事件编号不能为空。")
        _validate_month_key(self.month)
        _require_nonnegative_fen(f"事件{self.event_id}", self.amount_fen)
        if self.amount_fen == 0:
            raise CashPlanError("ZERO_EVENT_AMOUNT", f"事件{self.event_id}金额必须大于0。")


@dataclass(frozen=True)
class AutoOperatingSpec:
    """Small, deterministic input set for constrained automatic planning.

    ``base_monthly_collection_fen`` is the first-month operating inflow.
    ``operating_outflow_ratio_bp`` is paid operating cash per 10,000 fen of
    collection.  ``monthly_growth_bp`` may be negative but may not make a
    month negative.  Month weights express concentration; omitted means an
    even allocation before the growth adjustment.
    """

    base_monthly_collection_fen: int
    operating_outflow_ratio_bp: int
    monthly_growth_bp: int = 0
    month_weights_bp: Mapping[int, int] | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_fen("base_monthly_collection_fen", self.base_monthly_collection_fen)
        if self.base_monthly_collection_fen == 0:
            raise CashPlanError("ZERO_AUTO_SCALE", "自动规划的首月回款规模必须大于0。")
        if not isinstance(self.operating_outflow_ratio_bp, int) or self.operating_outflow_ratio_bp < 0:
            raise CashPlanError("INVALID_OUTFLOW_RATIO", "自动规划的经营流出比例必须是非负整数。")
        if not isinstance(self.monthly_growth_bp, int) or self.monthly_growth_bp <= -10_000:
            raise CashPlanError("INVALID_GROWTH_RATE", "月度增长率必须大于-100%。")
        if self.month_weights_bp is not None:
            if set(self.month_weights_bp) != set(range(1, 13)):
                raise CashPlanError("INVALID_MONTH_WEIGHTS", "月度集中权重必须完整包含1至12月。")
            if any(not isinstance(value, int) or value < 0 for value in self.month_weights_bp.values()):
                raise CashPlanError("INVALID_MONTH_WEIGHTS", "月度集中权重必须为非负整数。")
            if sum(self.month_weights_bp.values()) == 0:
                raise CashPlanError("INVALID_MONTH_WEIGHTS", "月度集中权重之和必须大于0。")


@dataclass(frozen=True)
class CashPlanRequest:
    """All inputs accepted by I1.

    All monetary values are integer fen.  Finance sources are intentionally
    absent: I1 reports the pre-financing gap for I2 to solve later.
    """

    synthetic_account_id: str
    start_date: date
    end_date: date
    opening_balance_fen: int
    planning_mode: PlanningMode
    components: Sequence[CashComponent] = ()
    special_events: Sequence[SpecialEvent] = ()
    fixed_total_inflow_fen: int | None = None
    fixed_total_outflow_fen: int | None = None
    target_ending_balance_fen: BalanceRange | None = None
    minimum_liquidity_fen: int = 0
    residual_month_weights_bp: Mapping[str, int] | None = None
    auto_operating_spec: AutoOperatingSpec | None = None

    def __post_init__(self) -> None:
        if not self.synthetic_account_id:
            raise CashPlanError("EMPTY_ACCOUNT_ID", "虚构账户编号不能为空。")
        if not isinstance(self.start_date, date) or not isinstance(self.end_date, date):
            raise CashPlanError("INVALID_PERIOD", "开始和结束日期必须是date。")
        if self.start_date > self.end_date:
            raise CashPlanError("INVALID_PERIOD", "开始日期不能晚于结束日期。")
        _require_nonnegative_fen("opening_balance_fen", self.opening_balance_fen)
        _require_nonnegative_fen("minimum_liquidity_fen", self.minimum_liquidity_fen)
        if len({component.component_id for component in self.components}) != len(self.components):
            raise CashPlanError("DUPLICATE_COMPONENT_ID", "普通现金项目编号不能重复。")
        if len({event.event_id for event in self.special_events}) != len(self.special_events):
            raise CashPlanError("DUPLICATE_EVENT_ID", "特殊事件编号不能重复。")
        if self.residual_month_weights_bp is not None:
            for month, weight in self.residual_month_weights_bp.items():
                _validate_month_key(month)
                if not isinstance(weight, int) or weight < 0:
                    raise CashPlanError("INVALID_RESIDUAL_WEIGHTS", "未分配预算月度权重必须为非负整数。")


@dataclass(frozen=True)
class MonthlyCashPlanRow:
    month: str
    opening_balance_fen: int
    operating_inflow_fen: int
    operating_outflow_fen: int
    investing_inflow_fen: int
    investing_outflow_fen: int
    unallocated_inflow_fen: int
    unallocated_outflow_fen: int
    total_inflow_fen: int
    total_outflow_fen: int
    ending_balance_fen: int


@dataclass(frozen=True)
class PreFinancingGap:
    date: date
    month: str
    balance_before_financing_fen: int
    required_financing_fen: int
    reason_code: str


@dataclass(frozen=True)
class PlanDiagnostic:
    severity: str
    code: str
    message_cn: str


@dataclass(frozen=True)
class CashPlanResult:
    request: CashPlanRequest
    monthly_rows: tuple[MonthlyCashPlanRow, ...]
    total_inflow_fen: int
    total_outflow_fen: int
    ending_balance_fen: int
    pre_financing_gaps: tuple[PreFinancingGap, ...]
    diagnostics: tuple[PlanDiagnostic, ...]
    target_financing_to_ending_range_fen: int

    @property
    def has_pre_financing_gap(self) -> bool:
        return bool(self.pre_financing_gaps)


def build_monthly_cash_plan(request: CashPlanRequest) -> CashPlanResult:
    """Build a deterministic, pre-financing monthly plan from fictional inputs.

    This is the only public I1 entry point.  It never creates a transaction,
    writes a file, selects a financing instrument, or mutates the request.
    """

    months = _period_months(request.start_date, request.end_date)
    month_set = set(months)
    _validate_plan_months(request.components, request.special_events, month_set)

    ordinary = _empty_budget(months)
    for component in request.components:
        _add_component(ordinary, component)
    for event in request.special_events:
        _add_event(ordinary, event)

    diagnostics: list[PlanDiagnostic] = []
    if request.planning_mode is PlanningMode.CONSTRAINED_AUTO:
        if request.auto_operating_spec is None:
            raise CashPlanError("MISSING_AUTO_SPEC", "受约束自动规划必须提供自动经营参数。")
        _add_auto_operating_plan(ordinary, months, request.auto_operating_spec)
    elif request.auto_operating_spec is not None:
        raise CashPlanError("UNUSED_AUTO_SPEC", "只有受约束自动规划可以提供自动经营参数。")

    if request.planning_mode is PlanningMode.FIXED_BANK_TOTALS:
        _require_nonnegative_fen("fixed_total_inflow_fen", request.fixed_total_inflow_fen)
        _require_nonnegative_fen("fixed_total_outflow_fen", request.fixed_total_outflow_fen)
        _allocate_fixed_total_residuals(
            ordinary,
            months,
            request.fixed_total_inflow_fen,
            request.fixed_total_outflow_fen,
            request.residual_month_weights_bp,
        )
    elif request.fixed_total_inflow_fen is not None or request.fixed_total_outflow_fen is not None:
        raise CashPlanError("UNUSED_FIXED_TOTAL", "只有固定银行总收付方式可以填写期间总收付。")

    rows = _build_rows(months, request.opening_balance_fen, ordinary)
    total_inflow = sum(row.total_inflow_fen for row in rows)
    total_outflow = sum(row.total_outflow_fen for row in rows)
    ending_balance = rows[-1].ending_balance_fen
    _assert_cash_identity(request.opening_balance_fen, total_inflow, total_outflow, ending_balance)

    if request.planning_mode is PlanningMode.FIXED_BANK_TOTALS:
        assert request.fixed_total_inflow_fen is not None
        assert request.fixed_total_outflow_fen is not None
        if (total_inflow, total_outflow) != (
            request.fixed_total_inflow_fen,
            request.fixed_total_outflow_fen,
        ):
            raise CashPlanError("FIXED_TOTAL_NOT_PRESERVED", "固定总收付在计划后发生变化，已拒绝继续。")

    gaps = _pre_financing_gaps(rows, request.start_date, request.end_date, request.minimum_liquidity_fen)
    if gaps:
        diagnostics.append(
            PlanDiagnostic(
                severity="warning",
                code="PRE_FINANCING_LIQUIDITY_GAP",
                message_cn="融资前现金路径低于最低可用余额；I1只报告缺口，不能自动借款或注资。",
            )
        )

    target_financing = 0
    if request.target_ending_balance_fen is not None:
        target_financing = max(0, request.target_ending_balance_fen.minimum_fen - ending_balance)
        if ending_balance < request.target_ending_balance_fen.minimum_fen:
            diagnostics.append(
                PlanDiagnostic(
                    severity="warning",
                    code="ENDING_BALANCE_BELOW_TARGET",
                    message_cn="融资前期末余额低于目标下限；I2将判断已声明融资是否足以填补。",
                )
            )
        elif ending_balance > request.target_ending_balance_fen.maximum_fen:
            diagnostics.append(
                PlanDiagnostic(
                    severity="warning",
                    code="ENDING_BALANCE_ABOVE_TARGET",
                    message_cn="融资前期末余额高于目标上限；I1不通过虚构支出强行压低余额。",
                )
            )
    elif request.planning_mode is PlanningMode.TARGET_ENDING_BALANCE:
        raise CashPlanError("MISSING_TARGET_RANGE", "目标期末余额方式必须填写期末余额目标范围。")

    return CashPlanResult(
        request=request,
        monthly_rows=tuple(rows),
        total_inflow_fen=total_inflow,
        total_outflow_fen=total_outflow,
        ending_balance_fen=ending_balance,
        pre_financing_gaps=tuple(gaps),
        diagnostics=tuple(diagnostics),
        target_financing_to_ending_range_fen=target_financing,
    )


def _period_months(start_date: date, end_date: date) -> tuple[str, ...]:
    month_cursor = date(start_date.year, start_date.month, 1)
    final_month = date(end_date.year, end_date.month, 1)
    months: list[str] = []
    while month_cursor <= final_month:
        months.append(_month_key(month_cursor))
        month_cursor = _next_month(month_cursor)
    return tuple(months)


def _validate_plan_months(
    components: Sequence[CashComponent], events: Sequence[SpecialEvent], months: set[str]
) -> None:
    for component in components:
        invalid = set(component.monthly_amounts_fen) - months
        if invalid:
            raise CashPlanError("COMPONENT_MONTH_OUT_OF_PERIOD", f"项目{component.component_id}包含期间外月份：{sorted(invalid)}")
    for event in events:
        if event.month not in months:
            raise CashPlanError("EVENT_MONTH_OUT_OF_PERIOD", f"事件{event.event_id}发生在期间外月份：{event.month}")


def _empty_budget(months: Sequence[str]) -> dict[str, dict[str, int]]:
    return {
        month: {
            "operating_inflow": 0,
            "operating_outflow": 0,
            "investing_inflow": 0,
            "investing_outflow": 0,
            "unallocated_inflow": 0,
            "unallocated_outflow": 0,
        }
        for month in months
    }


def _add_component(budget: dict[str, dict[str, int]], component: CashComponent) -> None:
    key = f"{component.cash_flow_class.value}_{component.direction.value}"
    for month, amount in component.monthly_amounts_fen.items():
        budget[month][key] += amount


def _add_event(budget: dict[str, dict[str, int]], event: SpecialEvent) -> None:
    key = f"{event.cash_flow_class.value}_{event.direction.value}"
    budget[event.month][key] += event.amount_fen


def _add_auto_operating_plan(
    budget: dict[str, dict[str, int]], months: Sequence[str], spec: AutoOperatingSpec
) -> None:
    weights = _auto_month_weights(spec)
    for index, month in enumerate(months):
        year, month_number = _parse_month_key(month)
        del year  # The weight is seasonal; calendar year does not change its meaning.
        growth_factor_bp = 10_000 + spec.monthly_growth_bp * index
        if growth_factor_bp < 0:
            raise CashPlanError("AUTO_PLAN_NEGATIVE_MONTH", "自动规划趋势使某个月经营规模变成负数。")
        collection = (
            spec.base_monthly_collection_fen
            * weights[month_number]
            * growth_factor_bp
            // (10_000 * 10_000)
        )
        payment = collection * spec.operating_outflow_ratio_bp // 10_000
        budget[month]["operating_inflow"] += collection
        budget[month]["operating_outflow"] += payment


def _auto_month_weights(spec: AutoOperatingSpec) -> dict[int, int]:
    if spec.month_weights_bp is None:
        return {month: 10_000 for month in range(1, 13)}
    total = sum(spec.month_weights_bp.values())
    return {month: spec.month_weights_bp[month] * 120_000 // total for month in range(1, 13)}


def _allocate_fixed_total_residuals(
    budget: dict[str, dict[str, int]],
    months: Sequence[str],
    total_inflow_fen: int | None,
    total_outflow_fen: int | None,
    supplied_weights: Mapping[str, int] | None,
) -> None:
    assert total_inflow_fen is not None
    assert total_outflow_fen is not None
    known_inflow = sum(values["operating_inflow"] + values["investing_inflow"] for values in budget.values())
    known_outflow = sum(values["operating_outflow"] + values["investing_outflow"] for values in budget.values())
    if known_inflow > total_inflow_fen:
        raise CashPlanError(
            "FIXED_INFLOW_BUDGET_EXCEEDED",
            "普通项目和特殊事件的流入合计超过用户固定总流入；请调整总额或事件。",
        )
    if known_outflow > total_outflow_fen:
        raise CashPlanError(
            "FIXED_OUTFLOW_BUDGET_EXCEEDED",
            "普通项目和特殊事件的流出合计超过用户固定总流出；请调整总额或事件。",
        )
    weights = _resolve_month_weights(months, supplied_weights)
    _allocate_by_weights(budget, "unallocated_inflow", total_inflow_fen - known_inflow, weights)
    _allocate_by_weights(budget, "unallocated_outflow", total_outflow_fen - known_outflow, weights)


def _resolve_month_weights(months: Sequence[str], supplied: Mapping[str, int] | None) -> dict[str, int]:
    if supplied is None:
        return {month: 1 for month in months}
    unknown = set(supplied) - set(months)
    if unknown:
        raise CashPlanError("RESIDUAL_WEIGHT_OUT_OF_PERIOD", f"未分配预算权重包含期间外月份：{sorted(unknown)}")
    weights = {month: supplied.get(month, 0) for month in months}
    if sum(weights.values()) == 0:
        raise CashPlanError("ZERO_RESIDUAL_WEIGHT", "未分配预算月度权重之和必须大于0。")
    return weights


def _allocate_by_weights(
    budget: dict[str, dict[str, int]], key: str, amount_fen: int, weights: Mapping[str, int]
) -> None:
    total_weight = sum(weights.values())
    remaining = amount_fen
    ordered_months = tuple(weights)
    for month in ordered_months[:-1]:
        allocated = amount_fen * weights[month] // total_weight
        budget[month][key] += allocated
        remaining -= allocated
    budget[ordered_months[-1]][key] += remaining


def _build_rows(
    months: Sequence[str], opening_balance_fen: int, budget: Mapping[str, Mapping[str, int]]
) -> list[MonthlyCashPlanRow]:
    rows: list[MonthlyCashPlanRow] = []
    current = opening_balance_fen
    for month in months:
        values = budget[month]
        inflow = values["operating_inflow"] + values["investing_inflow"] + values["unallocated_inflow"]
        outflow = values["operating_outflow"] + values["investing_outflow"] + values["unallocated_outflow"]
        ending = current + inflow - outflow
        rows.append(
            MonthlyCashPlanRow(
                month=month,
                opening_balance_fen=current,
                operating_inflow_fen=values["operating_inflow"],
                operating_outflow_fen=values["operating_outflow"],
                investing_inflow_fen=values["investing_inflow"],
                investing_outflow_fen=values["investing_outflow"],
                unallocated_inflow_fen=values["unallocated_inflow"],
                unallocated_outflow_fen=values["unallocated_outflow"],
                total_inflow_fen=inflow,
                total_outflow_fen=outflow,
                ending_balance_fen=ending,
            )
        )
        current = ending
    return rows


def _pre_financing_gaps(
    rows: Sequence[MonthlyCashPlanRow], start_date: date, end_date: date, minimum_liquidity_fen: int
) -> list[PreFinancingGap]:
    """Conservative preview: each month's payments occur before its receipts.

    This is deliberately not a transaction schedule.  I3 will later choose
    dates and individual transactions.  The conservative order prevents I1
    from hiding a temporary monthly shortage behind a month-end net surplus.
    """

    gaps: list[PreFinancingGap] = []
    for row in rows:
        month_start = _month_start(row.month)
        preview_date = max(start_date, month_start)
        if preview_date > end_date:
            continue
        balance_after_monthly_outflow = row.opening_balance_fen - row.total_outflow_fen
        if balance_after_monthly_outflow < minimum_liquidity_fen:
            gaps.append(
                PreFinancingGap(
                    date=preview_date,
                    month=row.month,
                    balance_before_financing_fen=balance_after_monthly_outflow,
                    required_financing_fen=minimum_liquidity_fen - balance_after_monthly_outflow,
                    reason_code="MINIMUM_LIQUIDITY_SHORTFALL",
                )
            )
    return gaps


def _assert_cash_identity(opening: int, inflow: int, outflow: int, ending: int) -> None:
    if opening + inflow - outflow != ending:
        raise AssertionError("SYN-B1 I1现金恒等式内部错误")


def _month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def _validate_month_key(value: str) -> None:
    try:
        year, month = _parse_month_key(value)
        date(year, month, 1)
    except (TypeError, ValueError) as exc:
        raise CashPlanError("INVALID_MONTH_KEY", f"月份必须为YYYY-MM：{value!r}") from exc


def _parse_month_key(value: str) -> tuple[int, int]:
    if not isinstance(value, str) or len(value) != 7 or value[4] != "-":
        raise ValueError(value)
    return int(value[:4]), int(value[5:])


def _month_start(value: str) -> date:
    year, month = _parse_month_key(value)
    return date(year, month, 1)


def _next_month(month_start: date) -> date:
    if month_start.month == 12:
        return date(month_start.year + 1, 1, 1)
    return date(month_start.year, month_start.month + 1, 1)


def _require_nonnegative_fen(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CashPlanError("INVALID_MONEY", f"{name}必须为非负整数分。")
