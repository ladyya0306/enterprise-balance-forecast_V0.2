"""SYN-B1-I3B: execute user-declared funding tails and preserve history.

This module receives only the immutable schedules created by I3A plus a
caller-supplied opening cash amount.  It executes due funding events in date
and frozen-order order, records history, and stops at the first unfunded
outflow.  It never creates new funding, changes a contract, rolls financial
statements, or creates transaction files.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from syn_b1.funding_contracts import (
    CashDirection,
    FundingEventType,
    LoanContract,
    LoanSchedule,
    ScheduledFundingEvent,
)


class FundingTimelineError(ValueError):
    """A timeline state or execution request violates the frozen I3B contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ContractStatus(str, Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SUPERSEDED_BEFORE_FIRST_EVENT = "superseded_before_first_event"


class TailStatus(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    SUPERSEDED_BEFORE_FIRST_EVENT = "superseded_before_first_event"


class HistoryStatus(str, Enum):
    EXECUTED = "executed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ShareholderCapitalPlan:
    """One user-declared shareholder capital injection, not an inferred capacity."""

    capital_item_id: str
    plan_version: str
    due_date: date
    amount_fen: int

    def __post_init__(self) -> None:
        if not isinstance(self.capital_item_id, str) or not self.capital_item_id.strip():
            raise FundingTimelineError("EMPTY_CAPITAL_ITEM_ID", "股东投入事项编号不能为空。")
        if not isinstance(self.plan_version, str) or not self.plan_version.strip():
            raise FundingTimelineError("EMPTY_CAPITAL_PLAN_VERSION", "股东投入计划版本不能为空。")
        if not isinstance(self.due_date, date):
            raise FundingTimelineError("INVALID_CAPITAL_DUE_DATE", "股东投入日期必须是date。")
        _require_positive_int("amount_fen", self.amount_fen)


@dataclass(frozen=True)
class TimelineContractRecord:
    """A retained contract-book entry with an immutable cross-contract order."""

    contract: LoanContract | ShareholderCapitalPlan
    contract_frozen_order: int
    status: ContractStatus

    def __post_init__(self) -> None:
        _require_positive_int("contract_frozen_order", self.contract_frozen_order)
        if not isinstance(self.status, ContractStatus):
            raise FundingTimelineError("INVALID_CONTRACT_STATUS", "合同状态不合法。")

    @property
    def source_key(self) -> tuple[str, str]:
        """Stable source identity owned by the contract-book record."""

        return _source_id(self.contract), _source_version(self.contract)


@dataclass(frozen=True)
class TimelineTailItem:
    """A not-yet-executed funding event, including a retained blocked item."""

    event: ScheduledFundingEvent
    contract_frozen_order: int
    status: TailStatus = TailStatus.PENDING
    blocked_on: date | None = None
    blocked_reason_code: str | None = None
    superseded_by_contract_version: str | None = None

    def __post_init__(self) -> None:
        _require_positive_int("contract_frozen_order", self.contract_frozen_order)
        if not isinstance(self.status, TailStatus):
            raise FundingTimelineError("INVALID_TAIL_STATUS", "尾巴状态不合法。")
        if self.status is TailStatus.PENDING:
            if (
                self.blocked_on is not None
                or self.blocked_reason_code is not None
                or self.superseded_by_contract_version is not None
            ):
                raise FundingTimelineError("INVALID_PENDING_TAIL", "待执行尾巴不能有额外状态信息。")
        elif self.status is TailStatus.BLOCKED:
            if self.blocked_on is None or not self.blocked_reason_code:
                raise FundingTimelineError("INVALID_BLOCKED_TAIL", "被阻断尾巴必须保留日期和原因。")
            if self.superseded_by_contract_version is not None:
                raise FundingTimelineError("INVALID_BLOCKED_TAIL", "被阻断尾巴不能同时标为被替代。")
        elif not self.superseded_by_contract_version:
            raise FundingTimelineError("INVALID_SUPERSEDED_TAIL", "被替代尾巴必须保留替代版本。")


@dataclass(frozen=True)
class FundingHistoryItem:
    """One append-only record of an executed or blocked funding event."""

    event: ScheduledFundingEvent
    status: HistoryStatus
    recorded_on: date
    cash_before_fen: int
    cash_after_fen: int
    reason_code: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_int("cash_before_fen", self.cash_before_fen)
        _require_nonnegative_int("cash_after_fen", self.cash_after_fen)
        if not isinstance(self.status, HistoryStatus):
            raise FundingTimelineError("INVALID_HISTORY_STATUS", "历史状态不合法。")
        if self.status is HistoryStatus.EXECUTED:
            if self.reason_code is not None:
                raise FundingTimelineError("INVALID_EXECUTED_HISTORY", "已执行事项不能填写阻断原因。")
        elif not self.reason_code:
            raise FundingTimelineError("INVALID_BLOCKED_HISTORY", "被阻断事项必须保留原因。")


@dataclass(frozen=True)
class FundingTimelineState:
    """The three persisted books: contracts, future tails, and append-only history."""

    contract_book: tuple[TimelineContractRecord, ...]
    pending_tail: tuple[TimelineTailItem, ...]
    event_history: tuple[FundingHistoryItem, ...] = ()
    processed_through_date: date | None = None

    def __post_init__(self) -> None:
        _validate_state(self)


@dataclass(frozen=True)
class FundingTimelineRequest:
    """A bounded execution request; cash is supplied by the later planning bridge."""

    state: FundingTimelineState
    start_date: date
    end_date: date
    opening_cash_fen: int

    def __post_init__(self) -> None:
        if not isinstance(self.start_date, date) or not isinstance(self.end_date, date):
            raise FundingTimelineError("INVALID_DATE_RANGE", "执行起止日期必须是date。")
        if self.end_date < self.start_date:
            raise FundingTimelineError("INVALID_DATE_RANGE", "结束日期不能早于开始日期。")
        _require_nonnegative_int("opening_cash_fen", self.opening_cash_fen)
        previous = self.state.processed_through_date
        if previous is not None and self.start_date != previous + timedelta(days=1):
            raise FundingTimelineError(
                "NON_CONTIGUOUS_EXECUTION", "跨期执行必须从上次处理日期的下一自然日开始。"
            )


@dataclass(frozen=True)
class FundingTimelineFailure:
    """The first item that could not be fully paid under the supplied cash."""

    code: str
    due_date: date
    tail_item_id: str
    required_fen: int
    available_cash_fen: int
    shortfall_fen: int


@dataclass(frozen=True)
class FundingTimelineResult:
    """Execution outcome, including the new immutable state and first failure."""

    state: FundingTimelineState
    closing_cash_fen: int
    new_history: tuple[FundingHistoryItem, ...]
    failure: FundingTimelineFailure | None = None

    @property
    def is_complete(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class FundingLedgerPosting:
    """One funding event that the central ledger actually posted.

    This is deliberately not a second repayment calculator.  It carries only
    the identity and observed-account before/after balances already produced
    by I4B, so the frozen contract book can move an executed item out of its
    future tail without guessing a new event or changing an amount.
    """

    tail_item_id: str
    booking_date: date
    posted_amount_fen: int
    cash_before_fen: int
    cash_after_fen: int

    def __post_init__(self) -> None:
        if not self.tail_item_id or not isinstance(self.booking_date, date):
            raise FundingTimelineError("INVALID_LEDGER_FUNDING_POSTING", "已记账融资事项必须提供尾巴编号和日期。")
        _require_positive_int("posted_amount_fen", self.posted_amount_fen)
        _require_nonnegative_int("cash_before_fen", self.cash_before_fen)
        _require_nonnegative_int("cash_after_fen", self.cash_after_fen)


def settle_funding_timeline_from_ledger(
    state: FundingTimelineState,
    postings: tuple[FundingLedgerPosting, ...],
    *,
    generated_through_date: date,
    blocked_tail_item_id: str | None = None,
    blocked_cash_fen: int | None = None,
) -> FundingTimelineState:
    """Move only actually posted funding events from tail into history.

    The ledger remains the sole owner of balance sufficiency and transaction
    order.  This function only settles matching contract-tail identities after
    that result is known, preserving all later events as the future tail.
    """

    if not isinstance(generated_through_date, date):
        raise FundingTimelineError("INVALID_GENERATED_THROUGH_DATE", "流水处理截止日必须是日期。")
    posting_by_id = {item.tail_item_id: item for item in postings}
    if len(posting_by_id) != len(postings):
        raise FundingTimelineError("DUPLICATE_LEDGER_FUNDING_POSTING", "同一融资尾巴不能重复记账。")
    tail_by_id = {item.event.tail_item_id: item for item in state.pending_tail}
    unknown_ids = set(posting_by_id) - set(tail_by_id)
    if unknown_ids:
        raise FundingTimelineError("UNKNOWN_LEDGER_FUNDING_POSTING", "账本包含未声明的融资尾巴事项。")
    if blocked_tail_item_id is not None and blocked_tail_item_id not in tail_by_id:
        raise FundingTimelineError("UNKNOWN_BLOCKED_FUNDING_TAIL", "资金不足事项没有对应融资尾巴。")
    if blocked_tail_item_id is not None and blocked_cash_fen is None:
        raise FundingTimelineError("MISSING_BLOCKED_FUNDING_CASH", "被阻断融资事项必须保留可用余额。")

    remaining: list[TimelineTailItem] = []
    new_history: list[FundingHistoryItem] = []
    for item in _ordered_tails(state.pending_tail):
        posting = posting_by_id.get(item.event.tail_item_id)
        if posting is not None:
            _validate_ledger_posting_matches_tail(posting, item)
            new_history.append(FundingHistoryItem(
                event=item.event,
                status=HistoryStatus.EXECUTED,
                recorded_on=posting.booking_date,
                cash_before_fen=posting.cash_before_fen,
                cash_after_fen=posting.cash_after_fen,
            ))
            continue
        if item.event.tail_item_id == blocked_tail_item_id:
            assert blocked_cash_fen is not None
            remaining.append(TimelineTailItem(
                event=item.event,
                contract_frozen_order=item.contract_frozen_order,
                status=TailStatus.BLOCKED,
                blocked_on=item.event.due_date,
                blocked_reason_code="INSUFFICIENT_CASH_FOR_DUE_EVENT",
            ))
            new_history.append(FundingHistoryItem(
                event=item.event,
                status=HistoryStatus.BLOCKED,
                recorded_on=item.event.due_date,
                cash_before_fen=blocked_cash_fen,
                cash_after_fen=blocked_cash_fen,
                reason_code="INSUFFICIENT_CASH_FOR_DUE_EVENT",
            ))
            continue
        remaining.append(item)

    processed_through = _settled_processed_through_date(
        state, tuple(remaining), generated_through_date,
    )
    return _with_refreshed_contract_statuses(FundingTimelineState(
        contract_book=state.contract_book,
        pending_tail=tuple(_ordered_tails(remaining)),
        event_history=(*state.event_history, *new_history),
        processed_through_date=processed_through,
    ))


def _validate_ledger_posting_matches_tail(posting: FundingLedgerPosting, tail: TimelineTailItem) -> None:
    if tail.status is not TailStatus.PENDING:
        raise FundingTimelineError("NON_PENDING_LEDGER_FUNDING_POSTING", "只有待执行融资尾巴可以进入已记账历史。")
    if posting.booking_date != tail.event.due_date:
        raise FundingTimelineError("LEDGER_FUNDING_DATE_MISMATCH", "账本融资日期必须等于已声明合同日期。")
    if posting.posted_amount_fen != tail.event.amount_fen:
        raise FundingTimelineError("LEDGER_FUNDING_AMOUNT_MISMATCH", "实际已记账融资金额必须精确等于冻结合同金额。")
    balance_delta = posting.cash_after_fen - posting.cash_before_fen
    expected_delta = posting.posted_amount_fen if tail.event.cash_direction is CashDirection.INFLOW else -posting.posted_amount_fen
    if balance_delta != expected_delta:
        raise FundingTimelineError("LEDGER_FUNDING_BALANCE_DELTA_MISMATCH", "融资事项的账前账后余额变化必须精确等于实际记账金额。")


def _settled_processed_through_date(
    prior_state: FundingTimelineState,
    remaining: tuple[TimelineTailItem, ...],
    generated_through_date: date,
) -> date:
    unsettled_in_period = [
        item.event.due_date
        for item in remaining
        if item.status in {TailStatus.PENDING, TailStatus.BLOCKED}
        and item.event.due_date <= generated_through_date
    ]
    candidate = min(unsettled_in_period) - timedelta(days=1) if unsettled_in_period else generated_through_date
    if prior_state.processed_through_date is not None and candidate < prior_state.processed_through_date:
        raise FundingTimelineError("FUNDING_PROCESSING_REGRESSION", "资金尾巴处理日期不能倒退。")
    return candidate


def build_timeline_state(
    schedules: tuple[LoanSchedule, ...],
    capital_plans: tuple[ShareholderCapitalPlan, ...] = (),
) -> FundingTimelineState:
    """Create the initial three-book state in the caller's frozen contract order.

    The schedule tuple is the user-confirmed contract creation order.  That
    order is stored once and is later used to break same-day ties; sorting by
    loan amount, identifier, or any inferred priority is forbidden.
    """

    if not schedules and not capital_plans:
        raise FundingTimelineError("EMPTY_CONTRACT_BOOK", "至少需要一份用户已设资金合同或股东投入计划。")
    records: list[TimelineContractRecord] = []
    tails: list[TimelineTailItem] = []
    for contract_order, schedule in enumerate(schedules, start=1):
        _validate_schedule_for_timeline(schedule)
        records.append(
            TimelineContractRecord(
                contract=schedule.contract,
                contract_frozen_order=contract_order,
                status=ContractStatus.PLANNED,
            )
        )
        tails.extend(
            TimelineTailItem(event=event, contract_frozen_order=contract_order)
            for event in schedule.events
        )
    offset = len(records)
    for capital_order, capital_plan in enumerate(capital_plans, start=1):
        contract_order = offset + capital_order
        records.append(
            TimelineContractRecord(
                contract=capital_plan,
                contract_frozen_order=contract_order,
                status=ContractStatus.PLANNED,
            )
        )
        tails.append(
            TimelineTailItem(
                event=_capital_event(capital_plan),
                contract_frozen_order=contract_order,
            )
        )
    return _with_refreshed_contract_statuses(
        FundingTimelineState(contract_book=tuple(records), pending_tail=tuple(_ordered_tails(tails)))
    )


def append_declared_schedule(state: FundingTimelineState, schedule: LoanSchedule) -> FundingTimelineState:
    """Append one newly declared future contract without changing prior records.

    A completed contract is never reused.  Before any event executes, a user
    may replace a planned contract by saving a new version with the same
    contract ID; the old version and its tails remain as superseded history.
    Any new contract draw date must be after the processed date, preventing
    retroactive financing from repairing an already evaluated run.
    """

    _validate_schedule_for_timeline(schedule)
    return _append_declared_source(state, schedule.contract, schedule.events)


def append_declared_capital_plan(
    state: FundingTimelineState, capital_plan: ShareholderCapitalPlan
) -> FundingTimelineState:
    """Append a user-declared capital injection without inferring shareholder capacity."""

    return _append_declared_source(state, capital_plan, (_capital_event(capital_plan),))


def _append_declared_source(
    state: FundingTimelineState,
    source: LoanContract | ShareholderCapitalPlan,
    events: tuple[ScheduledFundingEvent, ...],
) -> FundingTimelineState:
    if state.processed_through_date is not None and _source_start_date(source) <= state.processed_through_date:
        raise FundingTimelineError("RETROACTIVE_CONTRACT", "新增资金事项日期必须晚于已处理日期。")
    source_id = _source_id(source)
    source_version = _source_version(source)
    same_id_records = [record for record in state.contract_book if _source_id(record.contract) == source_id]
    if any(_source_version(record.contract) == source_version for record in same_id_records):
        raise FundingTimelineError("DUPLICATE_CONTRACT_VERSION", "资金事项编号和版本组合不能重复。")
    if same_id_records:
        has_executed_history = any(item.event.loan_contract_id == source_id for item in state.event_history)
        if has_executed_history:
            raise FundingTimelineError("REUSED_LOAN_CONTRACT_ID", "已发生事项的合同不能改版本或重新使用。")
        state = _supersede_unstarted_contract_versions(state, source_id, source_version)
    order = len(state.contract_book) + 1
    record = TimelineContractRecord(
        contract=source,
        contract_frozen_order=order,
        status=ContractStatus.PLANNED,
    )
    tails = (*state.pending_tail, *(TimelineTailItem(event=event, contract_frozen_order=order) for event in events))
    return _with_refreshed_contract_statuses(
        FundingTimelineState(
            contract_book=(*state.contract_book, record),
            pending_tail=tuple(_ordered_tails(tails)),
            event_history=state.event_history,
            processed_through_date=state.processed_through_date,
        )
    )


def _supersede_unstarted_contract_versions(
    state: FundingTimelineState, loan_contract_id: str, replacement_version: str
) -> FundingTimelineState:
    """Retain unstarted versions and tails when the user saves a replacement."""

    records: list[TimelineContractRecord] = []
    for record in state.contract_book:
        status = (
            ContractStatus.SUPERSEDED_BEFORE_FIRST_EVENT
            if _source_id(record.contract) == loan_contract_id
            else record.status
        )
        records.append(
            TimelineContractRecord(
                contract=record.contract,
                contract_frozen_order=record.contract_frozen_order,
                status=status,
            )
        )
    tails: list[TimelineTailItem] = []
    for item in state.pending_tail:
        if item.event.loan_contract_id == loan_contract_id:
            tails.append(
                TimelineTailItem(
                    event=item.event,
                    contract_frozen_order=item.contract_frozen_order,
                    status=TailStatus.SUPERSEDED_BEFORE_FIRST_EVENT,
                    superseded_by_contract_version=replacement_version,
                )
            )
        else:
            tails.append(item)
    return FundingTimelineState(
        contract_book=tuple(records),
        pending_tail=tuple(_ordered_tails(tails)),
        event_history=state.event_history,
        processed_through_date=state.processed_through_date,
    )


def execute_funding_timeline(request: FundingTimelineRequest) -> FundingTimelineResult:
    """Execute only due, user-declared funding events within the requested range.

    Inflows increase the supplied cash.  An outflow must be payable in full.
    On the first shortfall, prior events remain recorded, the failed event is
    retained as blocked, and no later item is executed or silently rescheduled.
    """

    state = request.state
    _raise_if_state_already_blocked(state)
    ordered_tails = _ordered_tails(state.pending_tail)
    _raise_if_overdue_pending_tail(ordered_tails, request.start_date)

    cash_fen = request.opening_cash_fen
    remaining_tails = list(ordered_tails)
    new_history: list[FundingHistoryItem] = []

    for item in ordered_tails:
        if item.event.due_date > request.end_date:
            break
        if item.event.due_date < request.start_date:
            continue  # guarded by _raise_if_overdue_pending_tail above
        if item.status is TailStatus.SUPERSEDED_BEFORE_FIRST_EVENT:
            continue
        if item.status is not TailStatus.PENDING:
            raise FundingTimelineError("BLOCKED_TAIL_REQUIRES_RESOLUTION", "被阻断尾巴不能自动重试。")

        event = item.event
        cash_before_fen = cash_fen
        if event.cash_direction is CashDirection.OUTFLOW and cash_fen < event.amount_fen:
            failure = FundingTimelineFailure(
                code="INSUFFICIENT_CASH_FOR_DUE_EVENT",
                due_date=event.due_date,
                tail_item_id=event.tail_item_id,
                required_fen=event.amount_fen,
                available_cash_fen=cash_fen,
                shortfall_fen=event.amount_fen - cash_fen,
            )
            blocked_item = TimelineTailItem(
                event=event,
                contract_frozen_order=item.contract_frozen_order,
                status=TailStatus.BLOCKED,
                blocked_on=event.due_date,
                blocked_reason_code=failure.code,
            )
            remaining_tails[remaining_tails.index(item)] = blocked_item
            blocked_history = FundingHistoryItem(
                event=event,
                status=HistoryStatus.BLOCKED,
                recorded_on=event.due_date,
                cash_before_fen=cash_before_fen,
                cash_after_fen=cash_before_fen,
                reason_code=failure.code,
            )
            new_state = _with_refreshed_contract_statuses(
                FundingTimelineState(
                    contract_book=state.contract_book,
                    pending_tail=tuple(_ordered_tails(remaining_tails)),
                    event_history=(*state.event_history, *new_history, blocked_history),
                    processed_through_date=event.due_date - timedelta(days=1),
                )
            )
            return FundingTimelineResult(
                state=new_state,
                closing_cash_fen=cash_fen,
                new_history=tuple((*new_history, blocked_history)),
                failure=failure,
            )

        cash_fen = cash_fen + event.amount_fen if event.cash_direction is CashDirection.INFLOW else cash_fen - event.amount_fen
        executed_history = FundingHistoryItem(
            event=event,
            status=HistoryStatus.EXECUTED,
            recorded_on=event.due_date,
            cash_before_fen=cash_before_fen,
            cash_after_fen=cash_fen,
        )
        new_history.append(executed_history)
        remaining_tails.remove(item)

    new_state = _with_refreshed_contract_statuses(
        FundingTimelineState(
            contract_book=state.contract_book,
            pending_tail=tuple(_ordered_tails(remaining_tails)),
            event_history=(*state.event_history, *new_history),
            processed_through_date=request.end_date,
        )
    )
    return FundingTimelineResult(
        state=new_state,
        closing_cash_fen=cash_fen,
        new_history=tuple(new_history),
    )


def _with_refreshed_contract_statuses(state: FundingTimelineState) -> FundingTimelineState:
    """Return a state whose retained contract-book statuses match tails/history."""

    blocked_ids = {
        item.event.tail_item_id for item in state.pending_tail if item.status is TailStatus.BLOCKED
    }
    executed_ids = {
        item.event.tail_item_id for item in state.event_history if item.status is HistoryStatus.EXECUTED
    }
    blocked_history_ids = {
        item.event.tail_item_id for item in state.event_history if item.status is HistoryStatus.BLOCKED
    }
    refreshed: list[TimelineContractRecord] = []
    for record in state.contract_book:
        if record.status is ContractStatus.SUPERSEDED_BEFORE_FIRST_EVENT:
            refreshed.append(record)
            continue
        contract_events = {
            item.event.tail_item_id
            for item in state.pending_tail
            if _matches_source(item.event, record.contract)
        } | {
            item.event.tail_item_id
            for item in state.event_history
            if _matches_source(item.event, record.contract)
        }
        contract_blocked = contract_events & (blocked_ids | blocked_history_ids)
        contract_executed = contract_events & executed_ids
        contract_pending = any(
            _matches_source(item.event, record.contract) and item.status is TailStatus.PENDING
            for item in state.pending_tail
        )
        if contract_blocked:
            status = ContractStatus.BLOCKED
        elif not contract_pending and contract_executed:
            status = ContractStatus.COMPLETED
        elif contract_executed:
            status = ContractStatus.ACTIVE
        else:
            status = ContractStatus.PLANNED
        refreshed.append(
            TimelineContractRecord(
                contract=record.contract,
                contract_frozen_order=record.contract_frozen_order,
                status=status,
            )
        )
    return FundingTimelineState(
        contract_book=tuple(refreshed),
        pending_tail=state.pending_tail,
        event_history=state.event_history,
        processed_through_date=state.processed_through_date,
    )


def _validate_state(state: FundingTimelineState) -> None:
    records = state.contract_book
    if not records:
        raise FundingTimelineError("EMPTY_CONTRACT_BOOK", "合同本不能为空。")
    expected_orders = tuple(range(1, len(records) + 1))
    actual_orders = tuple(record.contract_frozen_order for record in records)
    if actual_orders != expected_orders:
        raise FundingTimelineError("INVALID_CONTRACT_ORDER", "合同冻结顺序必须连续且从1开始。")
    contract_keys = {(_source_id(record.contract), _source_version(record.contract)) for record in records}
    if len(contract_keys) != len(records):
        raise FundingTimelineError("DUPLICATE_CONTRACT_VERSION", "合同编号和版本组合不能重复。")

    tail_ids = [item.event.tail_item_id for item in state.pending_tail]
    history_ids = [item.event.tail_item_id for item in state.event_history]
    if len(tail_ids) != len(set(tail_ids)) or len(history_ids) != len(set(history_ids)):
        raise FundingTimelineError("DUPLICATE_TAIL_ITEM", "尾巴或历史中出现重复事项编号。")
    for item in (*state.pending_tail, *state.event_history):
        key = (item.event.loan_contract_id, item.event.contract_version)
        if key not in contract_keys:
            raise FundingTimelineError("UNKNOWN_CONTRACT_EVENT", "事项没有对应的合同本记录。")
    blocked_tail_ids = {
        item.event.tail_item_id for item in state.pending_tail if item.status is TailStatus.BLOCKED
    }
    blocked_history_ids = {
        item.event.tail_item_id for item in state.event_history if item.status is HistoryStatus.BLOCKED
    }
    if blocked_tail_ids != blocked_history_ids:
        raise FundingTimelineError("BLOCKED_HISTORY_MISMATCH", "被阻断尾巴必须有且只能有一条阻断历史。")
    superseded_tail_ids = {
        item.event.tail_item_id
        for item in state.pending_tail
        if item.status is TailStatus.SUPERSEDED_BEFORE_FIRST_EVENT
    }
    if superseded_tail_ids & set(history_ids):
        raise FundingTimelineError("SUPERSEDED_HISTORY_MISMATCH", "被替代尾巴不能进入执行历史。")
    executed_history_ids = {
        item.event.tail_item_id for item in state.event_history if item.status is HistoryStatus.EXECUTED
    }
    if set(tail_ids) & executed_history_ids:
        raise FundingTimelineError("EXECUTED_TAIL_RETAINED", "已执行事项必须从未来尾巴移入历史。")
    if state.processed_through_date is not None and not isinstance(state.processed_through_date, date):
        raise FundingTimelineError("INVALID_PROCESSED_DATE", "已处理日期必须是date。")


def _validate_schedule_for_timeline(schedule: LoanSchedule) -> None:
    if not isinstance(schedule, LoanSchedule):
        raise FundingTimelineError("INVALID_SCHEDULE", "资金日程必须由I3A生成。")
    if not schedule.events:
        raise FundingTimelineError("EMPTY_SCHEDULE", "资金日程不能为空。")
    if any(not _matches_source(event, schedule.contract) for event in schedule.events):
        raise FundingTimelineError("SCHEDULE_CONTRACT_MISMATCH", "日程事项与合同不一致。")


def _raise_if_state_already_blocked(state: FundingTimelineState) -> None:
    blocked = [item for item in state.pending_tail if item.status is TailStatus.BLOCKED]
    if blocked:
        raise FundingTimelineError("BLOCKED_TAIL_REQUIRES_RESOLUTION", "存在被阻断尾巴，不能自动重试或跳过。")


def _raise_if_overdue_pending_tail(tails: tuple[TimelineTailItem, ...], start_date: date) -> None:
    overdue = [
        item for item in tails if item.status is TailStatus.PENDING and item.event.due_date < start_date
    ]
    if overdue:
        raise FundingTimelineError("OVERDUE_PENDING_EVENT", "存在早于本次开始日的未执行事项，不能静默延期。")


def _ordered_tails(items: tuple[TimelineTailItem, ...] | list[TimelineTailItem]) -> tuple[TimelineTailItem, ...]:
    return tuple(
        sorted(
            items,
            key=lambda item: (
                item.event.due_date,
                item.contract_frozen_order,
                item.event.frozen_sequence,
            ),
        )
    )


def _capital_event(plan: ShareholderCapitalPlan) -> ScheduledFundingEvent:
    return ScheduledFundingEvent(
        tail_item_id=f"{plan.capital_item_id}:{plan.plan_version}:0001:shareholder_capital_injection",
        loan_contract_id=plan.capital_item_id,
        contract_version=plan.plan_version,
        due_date=plan.due_date,
        frozen_sequence=1,
        event_type=FundingEventType.SHAREHOLDER_CAPITAL_INJECTION,
        cash_direction=CashDirection.INFLOW,
        amount_fen=plan.amount_fen,
    )


def _matches_source(event: ScheduledFundingEvent, contract: LoanContract | ShareholderCapitalPlan) -> bool:
    return (
        event.loan_contract_id == _source_id(contract)
        and event.contract_version == _source_version(contract)
    )


def _source_id(source: LoanContract | ShareholderCapitalPlan) -> str:
    return source.loan_contract_id if isinstance(source, LoanContract) else source.capital_item_id


def _source_version(source: LoanContract | ShareholderCapitalPlan) -> str:
    return source.contract_version if isinstance(source, LoanContract) else source.plan_version


def _source_start_date(source: LoanContract | ShareholderCapitalPlan) -> date:
    return source.draw_date if isinstance(source, LoanContract) else source.due_date


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FundingTimelineError(f"INVALID_{name.upper()}", f"{name}必须是正整数。")


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FundingTimelineError(f"INVALID_{name.upper()}", f"{name}必须是非负整数。")
