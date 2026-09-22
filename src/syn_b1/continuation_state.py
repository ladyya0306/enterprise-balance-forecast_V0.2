"""Append-only continuation guard for complete SYN-B1 scenario runs.

This module owns run identity and historic-prefix protection only.  It does
not calculate cash, choose financing, create transactions, or write files.
The current implementation deliberately uses deterministic replay to verify a
candidate extension before accepting its new suffix.  That keeps the first
product version small while making any attempted rewrite visible and blocked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
import json

from syn_b1.scenario_runner import ScenarioRunRequest, ScenarioRunResult, build_scenario_run


class ContinuationStateError(ValueError):
    """A requested extension would alter frozen history or its identity."""


@dataclass(frozen=True)
class ContinuationState:
    synthetic_account_id: str
    run_id: str
    root_seed: int
    request_fingerprint: str
    generated_through_date: date
    transaction_count: int
    historic_prefix_digest: str
    closing_balance_fen: int
    ledger_version: str
    historical_prediction_cutoff: date | None = None
    calendar_version: str | None = None
    random_derivation_version: str = "syn_b1_i4a2_derived_seed_1_0"


@dataclass(frozen=True)
class ContinuationResult:
    state: ContinuationState
    appended_transaction_count: int
    appended_first_date: date | None
    appended_last_date: date | None
    full_run: ScenarioRunResult


def begin_continuation_state(run: ScenarioRunResult) -> ContinuationState:
    """Freeze the completed history identity after a complete ledger run."""
    if not run.ledger.is_complete:
        raise ContinuationStateError("中断流水没有完整期末，不能作为可续生成历史。")
    return ContinuationState(
        synthetic_account_id=run.request.synthetic_account_id,
        run_id=run.request.run_id,
        root_seed=run.request.random_seed,
        request_fingerprint=_request_fingerprint(run.request),
        generated_through_date=run.request.end_date,
        transaction_count=len(run.ledger.transactions),
        historic_prefix_digest=_transaction_digest(run.ledger),
        closing_balance_fen=run.ledger.closing_balance_fen,
        ledger_version=run.ledger.manifest.ledger_version,
        historical_prediction_cutoff=_calendar_cutoff(run.request),
        calendar_version=_calendar_version(run.request),
    )


def verify_persisted_formal_prefix(
    state: ContinuationState,
    *,
    sample_id: str,
    logical_run_id: str,
    transaction_count: int,
    transaction_digest: str,
) -> None:
    """Refuse continuation unless replay also matches the original CSV prefix."""

    if state.synthetic_account_id != sample_id or state.run_id != logical_run_id:
        raise ContinuationStateError("原正式目录与保存画像的账户或逻辑运行编号不一致。")
    if state.transaction_count != transaction_count or state.historic_prefix_digest != transaction_digest:
        raise ContinuationStateError("重新计算结果与原正式逐笔流水不一致，已拒绝续生成。")


def continue_verified_run(state: ContinuationState, request: ScenarioRunRequest) -> ContinuationResult:
    """Verify the saved prefix, then accept only newly generated transactions.

    The request must retain all historic scenario controls and root seed.  A
    different end date may extend the run; it may never shorten it.  The full
    deterministic replay is a verification implementation, not a permission
    to overwrite stored history: callers receive an append count only, and a
    mismatching prefix raises before a new state is returned.
    """
    _validate_identity(state, request)
    full_run = build_scenario_run(request)
    if not full_run.ledger.is_complete:
        raise ContinuationStateError("追加期间在记账前已中断，不能把不完整流水并入已冻结历史。")
    prefix = full_run.ledger.transactions[:state.transaction_count]
    if len(prefix) != state.transaction_count or _transaction_digest_items(prefix) != state.historic_prefix_digest:
        raise ContinuationStateError("续生成结果改写了既有交易、余额或随机路径，已拒绝追加。")
    appended = full_run.ledger.transactions[state.transaction_count:]
    next_state = begin_continuation_state(full_run)
    return ContinuationResult(
        state=next_state,
        appended_transaction_count=len(appended),
        appended_first_date=appended[0].booking_datetime.date() if appended else None,
        appended_last_date=appended[-1].booking_datetime.date() if appended else None,
        full_run=full_run,
    )


def _validate_identity(state: ContinuationState, request: ScenarioRunRequest) -> None:
    if request.synthetic_account_id != state.synthetic_account_id or request.run_id != state.run_id:
        raise ContinuationStateError("续生成必须使用同一虚构账户和运行编号。")
    if request.random_seed != state.root_seed:
        raise ContinuationStateError("续生成必须恢复同一主随机种子，不能重抽历史。")
    if request.end_date <= state.generated_through_date:
        raise ContinuationStateError("续生成结束日期必须晚于已生成截止日。")
    _validate_calendar_extension(state, request)
    if _request_fingerprint(request) != state.request_fingerprint:
        raise ContinuationStateError("续生成只能延长日期；历史企业设置、合同和拆分规则不能静默修改。")


def _request_fingerprint(request: ScenarioRunRequest) -> str:
    payload = asdict(request)
    payload.pop("end_date")
    # A later public-calendar publication date may extend a run into newly
    # generated months.  It does not alter cash, contracts, seed or the stored
    # transaction prefix, so it is deliberately stored and validated apart.
    runtime = payload.get("runtime_settings")
    if isinstance(runtime, dict):
        calendar = runtime.get("bank_calendar")
        if isinstance(calendar, dict):
            calendar.pop("prediction_cutoff", None)
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _calendar_cutoff(request: ScenarioRunRequest) -> date | None:
    settings = request.runtime_settings
    return settings.bank_calendar.prediction_cutoff if settings is not None else None


def _calendar_version(request: ScenarioRunRequest) -> str | None:
    settings = request.runtime_settings
    return settings.bank_calendar.version if settings is not None else None


def _validate_calendar_extension(state: ContinuationState, request: ScenarioRunRequest) -> None:
    """Allow only a forward public-calendar cutoff while retaining its version."""

    cutoff = _calendar_cutoff(request)
    version = _calendar_version(request)
    if state.calendar_version != version:
        raise ContinuationStateError("续生成必须使用同一银行日历版本，不能替换历史日历规则。")
    if state.historical_prediction_cutoff is None:
        if cutoff is not None:
            raise ContinuationStateError("原运行未保存日历截止日，不能静默改为新的日历范围。")
        return
    if cutoff is None or cutoff < state.historical_prediction_cutoff:
        raise ContinuationStateError("银行日历截止日只能保持或向后延长，不能缩短或移除。")
    if cutoff < request.end_date:
        raise ContinuationStateError("银行日历截止日不得早于新的生成结束日。")


def _transaction_digest(ledger) -> str:
    return _transaction_digest_items(ledger.transactions)


def _transaction_digest_items(items) -> str:
    payload = [
        (item.transaction_id, item.sequence_no, item.booking_datetime.isoformat(), item.debit_fen, item.credit_fen, item.post_transaction_balance_fen)
        for item in items
    ]
    return sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()
