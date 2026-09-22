"""SYN-B1-I3A: user-declared loan contracts and deterministic repayment schedules.

This module has one responsibility: validate one user-declared loan contract
and turn it into immutable future funding events.  It does not read a cash
balance, execute an event, decide whether to borrow, roll financial
statements, or create product files.  Those responsibilities belong to later
I3B and I3C stages.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP, localcontext
from enum import Enum


class FundingContractError(ValueError):
    """A user-declared funding contract cannot produce a valid schedule."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class FundingKind(str, Enum):
    """The two loan types in the frozen I3A contract."""

    BANK_LOAN = "bank_loan"
    SHAREHOLDER_LOAN = "shareholder_loan"


class RepaymentMethod(str, Enum):
    """The only repayment methods admitted by F4F."""

    BULLET_PRINCIPAL_AND_INTEREST = "bullet_principal_and_interest"
    MONTHLY_INTEREST_BULLET_PRINCIPAL = "monthly_interest_bullet_principal"
    EQUAL_PRINCIPAL_MONTHLY = "equal_principal_monthly"
    EQUAL_PAYMENT_MONTHLY = "equal_payment_monthly"


class FundingEventType(str, Enum):
    """Future-tail event kinds produced from one loan contract."""

    LOAN_DRAWDOWN = "loan_drawdown"
    INTEREST_PAYMENT = "interest_payment"
    PRINCIPAL_REPAYMENT = "principal_repayment"
    SHAREHOLDER_CAPITAL_INJECTION = "shareholder_capital_injection"


class CashDirection(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"


@dataclass(frozen=True)
class LoanContract:
    """A complete, user-declared contract for one finite loan.

    ``maturity_date`` and ``term_days`` are mutually exclusive.  The contract
    intentionally has no field for early repayment, facilities, limits,
    renewal, extension, automatic financing, or shareholder capability.
    """

    loan_contract_id: str
    contract_version: str
    funding_kind: FundingKind
    draw_date: date
    principal_fen: int
    annual_interest_rate_bp: int
    repayment_method: RepaymentMethod
    maturity_date: date | None = None
    term_days: int | None = None

    def __post_init__(self) -> None:
        _require_nonempty("loan_contract_id", self.loan_contract_id)
        _require_nonempty("contract_version", self.contract_version)
        if not isinstance(self.funding_kind, FundingKind):
            raise FundingContractError("INVALID_FUNDING_KIND", "资金类型必须是银行借款或股东借款。")
        if not isinstance(self.draw_date, date):
            raise FundingContractError("INVALID_DRAW_DATE", "借款日必须是date。")
        _require_positive_int("principal_fen", self.principal_fen, "本金必须是正整数分。")
        _require_nonnegative_int(
            "annual_interest_rate_bp", self.annual_interest_rate_bp, "年利率必须是非负整数基点。"
        )
        if not isinstance(self.repayment_method, RepaymentMethod):
            raise FundingContractError("INVALID_REPAYMENT_METHOD", "还款方式不在F4F冻结的四种方案内。")
        if (self.maturity_date is None) == (self.term_days is None):
            raise FundingContractError(
                "INVALID_MATURITY", "到期日与期限天数必须且只能填写一个。"
            )
        if self.maturity_date is not None:
            if not isinstance(self.maturity_date, date):
                raise FundingContractError("INVALID_MATURITY_DATE", "到期日必须是date。")
            if self.maturity_date <= self.draw_date:
                raise FundingContractError("INVALID_MATURITY_DATE", "到期日必须晚于借款日。")
        if self.term_days is not None:
            _require_positive_int("term_days", self.term_days, "期限天数必须是正整数。")

    @property
    def resolved_maturity_date(self) -> date:
        """Return the one deterministic maturity date for this contract."""

        if self.maturity_date is not None:
            return self.maturity_date
        assert self.term_days is not None
        return self.draw_date + timedelta(days=self.term_days)


@dataclass(frozen=True)
class ScheduledFundingEvent:
    """One immutable future-tail item; I3A never executes it."""

    tail_item_id: str
    loan_contract_id: str
    contract_version: str
    due_date: date
    frozen_sequence: int
    event_type: FundingEventType
    cash_direction: CashDirection
    amount_fen: int
    interest_period_start: date | None = None
    interest_period_end: date | None = None

    def __post_init__(self) -> None:
        _require_nonempty("tail_item_id", self.tail_item_id)
        _require_positive_int("frozen_sequence", self.frozen_sequence, "冻结顺序必须是正整数。")
        _require_positive_int("amount_fen", self.amount_fen, "事项金额必须是正整数分。")
        if self.event_type is FundingEventType.INTEREST_PAYMENT:
            if self.interest_period_start is None or self.interest_period_end is None:
                raise FundingContractError("MISSING_INTEREST_PERIOD", "付息事项必须保留计息区间。")
            if self.interest_period_end <= self.interest_period_start:
                raise FundingContractError("INVALID_INTEREST_PERIOD", "计息区间结束日必须晚于开始日。")
        elif self.interest_period_start is not None or self.interest_period_end is not None:
            raise FundingContractError("UNEXPECTED_INTEREST_PERIOD", "非付息事项不能填写计息区间。")


@dataclass(frozen=True)
class LoanSchedule:
    """The full immutable schedule generated from exactly one contract."""

    contract: LoanContract
    maturity_date: date
    events: tuple[ScheduledFundingEvent, ...]

    @property
    def total_interest_fen(self) -> int:
        return sum(
            event.amount_fen
            for event in self.events
            if event.event_type is FundingEventType.INTEREST_PAYMENT
        )

    @property
    def total_principal_repayment_fen(self) -> int:
        return sum(
            event.amount_fen
            for event in self.events
            if event.event_type is FundingEventType.PRINCIPAL_REPAYMENT
        )

    @property
    def drawdown_fen(self) -> int:
        return sum(
            event.amount_fen
            for event in self.events
            if event.event_type is FundingEventType.LOAN_DRAWDOWN
        )


def build_loan_schedule(contract: LoanContract) -> LoanSchedule:
    """Create the complete, reproducible future schedule for one loan.

    Interest is calculated as ``outstanding_principal * annual_rate * actual
    days / 365`` and rounded half-up to the nearest fen for each due period.
    The module creates no zero-value tail items.
    """

    maturity_date = contract.resolved_maturity_date
    due_dates = _repayment_due_dates(contract.draw_date, maturity_date, contract.repayment_method)
    events: list[ScheduledFundingEvent] = []
    sequence = 1
    events.append(
        _event(
            contract,
            sequence,
            contract.draw_date,
            FundingEventType.LOAN_DRAWDOWN,
            CashDirection.INFLOW,
            contract.principal_fen,
        )
    )
    sequence += 1

    if contract.repayment_method is RepaymentMethod.BULLET_PRINCIPAL_AND_INTEREST:
        sequence = _append_interest(
            events, contract, sequence, maturity_date, contract.draw_date, contract.principal_fen
        )
        _append_principal(events, contract, sequence, maturity_date, contract.principal_fen)
    elif contract.repayment_method is RepaymentMethod.MONTHLY_INTEREST_BULLET_PRINCIPAL:
        sequence = _append_interest_only_bullet(events, contract, sequence, due_dates)
    elif contract.repayment_method is RepaymentMethod.EQUAL_PRINCIPAL_MONTHLY:
        sequence = _append_equal_principal(events, contract, sequence, due_dates)
    elif contract.repayment_method is RepaymentMethod.EQUAL_PAYMENT_MONTHLY:
        sequence = _append_equal_payment(events, contract, sequence, due_dates)
    else:  # defensive: LoanContract has already checked this enum.
        raise FundingContractError("INVALID_REPAYMENT_METHOD", "还款方式不在F4F冻结的四种方案内。")

    schedule = LoanSchedule(contract=contract, maturity_date=maturity_date, events=tuple(events))
    _validate_complete_schedule(schedule)
    return schedule


def _append_interest_only_bullet(
    events: list[ScheduledFundingEvent],
    contract: LoanContract,
    sequence: int,
    due_dates: tuple[date, ...],
) -> int:
    period_start = contract.draw_date
    for index, due_date in enumerate(due_dates):
        sequence = _append_interest(
            events, contract, sequence, due_date, period_start, contract.principal_fen
        )
        if index == len(due_dates) - 1:
            sequence = _append_principal(events, contract, sequence, due_date, contract.principal_fen)
        period_start = due_date
    return sequence


def _append_equal_principal(
    events: list[ScheduledFundingEvent],
    contract: LoanContract,
    sequence: int,
    due_dates: tuple[date, ...],
) -> int:
    period_start = contract.draw_date
    outstanding = contract.principal_fen
    periods = len(due_dates)
    base_principal = contract.principal_fen // periods
    for index, due_date in enumerate(due_dates):
        sequence = _append_interest(events, contract, sequence, due_date, period_start, outstanding)
        principal_payment = (
            outstanding if index == periods - 1 else base_principal
        )
        if principal_payment:
            sequence = _append_principal(events, contract, sequence, due_date, principal_payment)
            outstanding -= principal_payment
        period_start = due_date
    return sequence


def _append_equal_payment(
    events: list[ScheduledFundingEvent],
    contract: LoanContract,
    sequence: int,
    due_dates: tuple[date, ...],
) -> int:
    period_days = _period_days(contract.draw_date, due_dates)
    periodic_payment = _equal_payment_fen(
        contract.principal_fen, contract.annual_interest_rate_bp, period_days
    )
    period_start = contract.draw_date
    outstanding = contract.principal_fen

    for index, due_date in enumerate(due_dates):
        is_final = index == len(due_dates) - 1
        interest_fen = _interest_fen(
            outstanding, contract.annual_interest_rate_bp, (due_date - period_start).days
        )
        if is_final:
            principal_payment = outstanding
        else:
            proposed_principal = max(0, periodic_payment - interest_fen)
            # Preserve at least one fen for the final deterministic adjustment
            # when possible.  This only affects tiny rounding edge cases.
            principal_payment = min(proposed_principal, max(0, outstanding - 1))
        if interest_fen:
            events.append(
                _event(
                    contract,
                    sequence,
                    due_date,
                    FundingEventType.INTEREST_PAYMENT,
                    CashDirection.OUTFLOW,
                    interest_fen,
                    interest_period_start=period_start,
                    interest_period_end=due_date,
                )
            )
            sequence += 1
        if principal_payment:
            sequence = _append_principal(events, contract, sequence, due_date, principal_payment)
            outstanding -= principal_payment
        period_start = due_date
    return sequence


def _append_interest(
    events: list[ScheduledFundingEvent],
    contract: LoanContract,
    sequence: int,
    due_date: date,
    period_start: date,
    outstanding_fen: int,
) -> int:
    interest_fen = _interest_fen(
        outstanding_fen, contract.annual_interest_rate_bp, (due_date - period_start).days
    )
    if not interest_fen:
        return sequence
    events.append(
        _event(
            contract,
            sequence,
            due_date,
            FundingEventType.INTEREST_PAYMENT,
            CashDirection.OUTFLOW,
            interest_fen,
            interest_period_start=period_start,
            interest_period_end=due_date,
        )
    )
    return sequence + 1


def _append_principal(
    events: list[ScheduledFundingEvent],
    contract: LoanContract,
    sequence: int,
    due_date: date,
    amount_fen: int,
) -> int:
    if amount_fen:
        events.append(
            _event(
                contract,
                sequence,
                due_date,
                FundingEventType.PRINCIPAL_REPAYMENT,
                CashDirection.OUTFLOW,
                amount_fen,
            )
        )
        return sequence + 1
    return sequence


def _event(
    contract: LoanContract,
    sequence: int,
    due_date: date,
    event_type: FundingEventType,
    cash_direction: CashDirection,
    amount_fen: int,
    *,
    interest_period_start: date | None = None,
    interest_period_end: date | None = None,
) -> ScheduledFundingEvent:
    tail_item_id = (
        f"{contract.loan_contract_id}:{contract.contract_version}:"
        f"{sequence:04d}:{event_type.value}"
    )
    return ScheduledFundingEvent(
        tail_item_id=tail_item_id,
        loan_contract_id=contract.loan_contract_id,
        contract_version=contract.contract_version,
        due_date=due_date,
        frozen_sequence=sequence,
        event_type=event_type,
        cash_direction=cash_direction,
        amount_fen=amount_fen,
        interest_period_start=interest_period_start,
        interest_period_end=interest_period_end,
    )


def _repayment_due_dates(
    draw_date: date, maturity_date: date, repayment_method: RepaymentMethod
) -> tuple[date, ...]:
    if repayment_method is RepaymentMethod.BULLET_PRINCIPAL_AND_INTEREST:
        return (maturity_date,)
    due_dates: list[date] = []
    year, month = _next_month(draw_date.year, draw_date.month)
    while True:
        regular_due_date = date(year, month, min(draw_date.day, monthrange(year, month)[1]))
        if regular_due_date >= maturity_date:
            break
        due_dates.append(regular_due_date)
        year, month = _next_month(year, month)
    due_dates.append(maturity_date)
    return tuple(due_dates)


def _period_days(start_date: date, due_dates: tuple[date, ...]) -> tuple[int, ...]:
    days: list[int] = []
    previous_date = start_date
    for due_date in due_dates:
        interval_days = (due_date - previous_date).days
        if interval_days <= 0:
            raise FundingContractError("INVALID_REPAYMENT_DATES", "还款日必须严格递增。")
        days.append(interval_days)
        previous_date = due_date
    return tuple(days)


def _equal_payment_fen(principal_fen: int, annual_rate_bp: int, period_days: tuple[int, ...]) -> int:
    """Solve the equal-payment amount for variable actual-day periods.

    The continuous calculation establishes one common payment.  Individual
    interest amounts are then rounded to fen when the event list is produced;
    only the final period absorbs the resulting one-to-few-fen difference.
    """

    if annual_rate_bp == 0:
        return _round_fen(Decimal(principal_fen) / Decimal(len(period_days)))
    with localcontext() as context:
        context.prec = 50
        growth = Decimal(1)
        payment_factor = Decimal(0)
        annual_rate = Decimal(annual_rate_bp) / Decimal(10_000)
        for days in period_days:
            period_growth = Decimal(1) + annual_rate * Decimal(days) / Decimal(365)
            growth *= period_growth
            payment_factor = payment_factor * period_growth + Decimal(1)
        payment = Decimal(principal_fen) * growth / payment_factor
    return max(1, _round_fen(payment))


def _interest_fen(outstanding_fen: int, annual_rate_bp: int, days: int) -> int:
    if days <= 0:
        raise FundingContractError("INVALID_INTEREST_PERIOD", "计息天数必须为正数。")
    with localcontext() as context:
        context.prec = 50
        interest = (
            Decimal(outstanding_fen)
            * Decimal(annual_rate_bp)
            * Decimal(days)
            / Decimal(10_000)
            / Decimal(365)
        )
    return _round_fen(interest)


def _round_fen(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _validate_complete_schedule(schedule: LoanSchedule) -> None:
    events = schedule.events
    if not events or events[0].event_type is not FundingEventType.LOAN_DRAWDOWN:
        raise FundingContractError("MISSING_DRAWDOWN", "日程必须以提款事项开始。")
    if schedule.drawdown_fen != schedule.contract.principal_fen:
        raise FundingContractError("DRAWDOWN_MISMATCH", "提款金额必须等于合同本金。")
    if schedule.total_principal_repayment_fen != schedule.contract.principal_fen:
        raise FundingContractError("PRINCIPAL_MISMATCH", "全部还本金额必须等于合同本金。")
    expected_sequences = tuple(range(1, len(events) + 1))
    actual_sequences = tuple(event.frozen_sequence for event in events)
    if actual_sequences != expected_sequences:
        raise FundingContractError("INVALID_FROZEN_SEQUENCE", "尾巴冻结顺序必须连续且从1开始。")
    if len({event.tail_item_id for event in events}) != len(events):
        raise FundingContractError("DUPLICATE_TAIL_ITEM_ID", "同一合同版本不能生成重复尾巴编号。")
    dates_and_sequences = tuple((event.due_date, event.frozen_sequence) for event in events)
    if dates_and_sequences != tuple(sorted(dates_and_sequences)):
        raise FundingContractError("INVALID_EVENT_ORDER", "未来事项必须按日期和冻结顺序排列。")


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FundingContractError("EMPTY_FIELD", f"{name}不能为空。")


def _require_positive_int(name: str, value: int, message: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FundingContractError(f"INVALID_{name.upper()}", message)


def _require_nonnegative_int(name: str, value: int, message: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FundingContractError(f"INVALID_{name.upper()}", message)
