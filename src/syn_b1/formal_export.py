"""SYN-B1-I7 formal by-ID files, with public and restricted outputs separated.

The scenario runner remains the sole owner of business calculations.  This
module only translates one already completed in-memory run into the F7F file
contract.  It never plans transactions, repairs a cash shortage, or reads a
real account.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from runtime_bank_calendar import infer_bank_calendar
from runtime_daily_aggregation import DailyRecord
from syn_b1.formal_profile import ResolvedFormalProfile
from syn_b1.scenario_runner import ScenarioRunResult
from syn_b1.tax_routing import CashCategory, CashDirection


FORMAL_GENERATOR_VERSION = "syn_b1_pretrain_i0_closure_1_0"
FORMAL_GENERATOR_VERSION_2_0_CANDIDATE = "syn_b1_pretrain_i1_regime_transition_2_0_candidate"


class FormalExportError(ValueError):
    """A formal output would overwrite history or violate the output contract."""


@dataclass(frozen=True)
class FormalExportBundle:
    """Paths to one non-overwriting SYN-B1 formal generation result."""

    output_dir: Path
    transactions_csv: Path
    daily_csv: Path
    cash_flow_review_notes_csv: Path
    profile_json: Path
    manifest_json: Path
    quality_report_json: Path
    fact_sheet_json: Path | None


def write_formal_bundle(
    profile: ResolvedFormalProfile,
    run: ScenarioRunResult,
    *,
    output_root: str | Path,
    output_run_id: str | None = None,
    continuation_of_run_id: str | None = None,
) -> FormalExportBundle:
    """Persist a completed run exactly once under ``sample_id/run_id``.

    The standard transaction/day CSVs deliberately contain only model-visible
    values.  A separate human-review file may carry Chinese cash-flow notes;
    it is explicitly excluded from model and database input.  Scenario truth,
    tax detail, finance tails, and three-statement detail go into the
    physically separated ``_restricted`` directory.
    """

    if profile.sample_id != run.request.synthetic_account_id:
        raise FormalExportError("参数画像与运行结果的虚构账户编号不一致。")
    effective_run_id = output_run_id or profile.run_id
    if not effective_run_id:
        raise FormalExportError("正式输出运行编号不能为空。")
    target = Path(output_root) / profile.sample_id / effective_run_id
    if target.exists():
        raise FormalExportError("正式输出目录已存在，不能覆盖历史运行。")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.writing-{uuid4().hex}"
    temporary.mkdir()
    try:
        transactions = temporary / "transactions_total.csv"
        daily = temporary / "account_daily_total.csv"
        review_notes = temporary / "cash_flow_review_notes.csv"
        profile_path = temporary / "scenario_profile.json"
        quality_path = temporary / "quality_report.json"
        manifest_path = temporary / "generation_manifest.json"
        _write_transactions(transactions, run)
        _write_cash_flow_review_notes(review_notes, run)
        daily_rows = _formal_daily_rows(profile, run)
        _write_daily(daily, daily_rows)
        _write_json(profile_path, _profile_payload(profile, run, effective_run_id))
        _write_json(quality_path, _quality_payload(run, daily_rows))
        fact_path = _write_fact_if_stopped(temporary, run)
        _write_restricted(temporary / "_restricted", run)
        _write_json(manifest_path, _manifest_payload(
            profile, run, effective_run_id, temporary, continuation_of_run_id,
        ))
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return FormalExportBundle(
        output_dir=target,
        transactions_csv=target / "transactions_total.csv",
        daily_csv=target / "account_daily_total.csv",
        cash_flow_review_notes_csv=target / "cash_flow_review_notes.csv",
        profile_json=target / "scenario_profile.json",
        manifest_json=target / "generation_manifest.json",
        quality_report_json=target / "quality_report.json",
        fact_sheet_json=(target / "generation_fact_sheet.json") if not run.ledger.is_complete else None,
    )


def _write_transactions(path: Path, run: ScenarioRunResult) -> None:
    fields = (
        "transaction_id", "booking_datetime", "debit_cny", "credit_cny",
        "post_transaction_balance_cny",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in run.ledger.transactions:
            writer.writerow({
                "transaction_id": item.transaction_id,
                "booking_datetime": item.booking_datetime.isoformat(timespec="seconds"),
                "debit_cny": _cny(item.debit_fen),
                "credit_cny": _cny(item.credit_fen),
                "post_transaction_balance_cny": _cny(item.post_transaction_balance_fen),
            })


def _write_cash_flow_review_notes(path: Path, run: ScenarioRunResult) -> None:
    """Write human-only cash-flow explanations without changing model CSVs."""

    fields = (
        "transaction_id", "booking_datetime", "direction_cn", "amount_cny",
        "cash_flow_type_cn", "usage_scope_cn",
    )
    candidate_by_id = {item.candidate_id: item for item in run.transaction_plan.candidates}
    movements = run.transaction_plan.cash_preview.completed_movements
    if len(movements) != len(run.ledger.transactions):
        raise FormalExportError("可支付候选与已记账流水笔数不一致，不能写资金备注表。")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for movement, transaction in zip(movements, run.ledger.transactions, strict=True):
            candidate = candidate_by_id.get(movement.movement_id)
            if candidate is None:
                raise FormalExportError("已记账流水没有对应候选，不能写资金备注表。")
            amount_fen = transaction.credit_fen or transaction.debit_fen
            expected_direction = CashDirection.INFLOW if transaction.credit_fen else CashDirection.OUTFLOW
            if candidate.amount_fen != amount_fen or candidate.direction is not expected_direction:
                raise FormalExportError("候选的金额或方向与已记账流水不一致，不能写资金备注表。")
            writer.writerow({
                "transaction_id": transaction.transaction_id,
                "booking_datetime": transaction.booking_datetime.isoformat(timespec="seconds"),
                "direction_cn": "流入" if candidate.direction is CashDirection.INFLOW else "流出",
                "amount_cny": _cny(amount_fen),
                # 新订单只写已发生业务备注，旧业务仍用原类别说明。
                "cash_flow_type_cn": candidate.historical_note_cn or _cash_flow_note(candidate.category),
                "usage_scope_cn": "仅供人工查看；不得作为模型输入或导入数据库",
            })


def _cash_flow_note(category: CashCategory) -> str:
    """Translate an already-set category; never infer a new business reason."""

    notes = {
        CashCategory.SALES_COLLECTION: "销售回款",
        CashCategory.SUPPLIER_PAYMENT: "原材料或采购付款",
        CashCategory.PAYROLL: "工资发放",
        CashCategory.RENT_AND_UTILITIES: "房租或水电费",
        CashCategory.BANK_LOAN_DRAWDOWN: "银行借款到账",
        CashCategory.BANK_LOAN_REPAYMENT: "偿还银行借款本金",
        CashCategory.BANK_INTEREST_PAYMENT: "支付银行借款利息",
        CashCategory.SHAREHOLDER_FUNDING: "股东注资或股东借款",
        CashCategory.FIXED_ASSET_PAYMENT: "设备或厂房等固定资产投入",
        CashCategory.CORPORATE_INCOME_TAX_PAYMENT: "缴纳企业所得税",
        CashCategory.SIMULATED_VAT_PAYMENT: "缴纳模拟增值税",
        CashCategory.SURCHARGE_PAYMENT: "缴纳附加税",
        CashCategory.OTHER_DECLARED_EVENT: "其他已声明事项",
    }
    return notes[category]


def _formal_daily_rows(profile: ResolvedFormalProfile, run: ScenarioRunResult) -> tuple[dict[str, Any], ...]:
    """Extend ledger roll-up to natural dates, then reuse frozen R6 dates."""

    final_date = run.request.end_date
    if not run.ledger.is_complete:
        fact = run.ledger.fact_sheet
        if fact is None:
            raise FormalExportError("中断运行缺少资金不足事实单。")
        final_date = fact.blocked_booking_datetime.date()
    daily_by_date = {item.calendar_date: item for item in run.ledger.daily_rows}
    last_transaction_by_date = {
        item.booking_datetime.date(): item.transaction_id for item in run.ledger.transactions
    }
    daily_records: list[DailyRecord] = []
    balance_fen = run.ledger.opening_balance_fen
    current = run.request.start_date
    while current <= final_date:
        source = daily_by_date.get(current)
        if source is not None:
            balance_fen = source.ending_balance_fen
            inflow_fen, outflow_fen, count = source.inflow_fen, source.outflow_fen, source.transaction_count
        else:
            inflow_fen = outflow_fen = count = 0
        daily_records.append(DailyRecord(
            account_token=profile.sample_id,
            calendar_date=current.isoformat(),
            debit_total_cny=Decimal(outflow_fen) / 100,
            credit_total_cny=Decimal(inflow_fen) / 100,
            transaction_count=count,
            closing_balance_cny=Decimal(balance_fen) / 100,
            source_last_transaction_id=last_transaction_by_date.get(current),
            reconciliation_status="trusted",
        ))
        current = date.fromordinal(current.toordinal() + 1)
    calendar = infer_bank_calendar(
        daily_records,
        prediction_cutoff=profile.request.runtime_settings.bank_calendar.prediction_cutoff,  # type: ignore[union-attr]
    )
    calendar_by_date = {item.calendar_date: item for item in calendar.records}
    # 新日历日表与订单所用工作日一致；旧版推断结果保持原样。
    from syn_b1.calendar_adapter import ORDER_CALENDAR_VERSION, CYCLE_CALENDAR_VERSION
    # 仅对显式新版本替换日历标记，账务金额不变。
    if profile.request.runtime_settings.bank_calendar.version in {ORDER_CALENDAR_VERSION, CYCLE_CALENDAR_VERSION}:
        # 原业务仍可在自然日交易，但不因此将假日改称工作日。
        calendar_by_date = {item.calendar_date: item for item in profile.request.runtime_settings.bank_calendar.public_records(daily_records)}
    status = "complete" if run.ledger.is_complete else "stopped_before_blocked_outflow"
    rows: list[dict[str, Any]] = []
    for record in daily_records:
        item = calendar_by_date[record.calendar_date]
        rows.append({
            "calendar_date": record.calendar_date,
            "inflow_cny": _decimal_cny(record.credit_total_cny),
            "outflow_cny": _decimal_cny(record.debit_total_cny),
            "transaction_count": record.transaction_count,
            "ending_balance_cny": _decimal_cny(record.closing_balance_cny),
            "day_cut_basis": "last_posted_transaction_or_prior_closing_balance",
            "source_last_transaction_id": record.source_last_transaction_id,
            "reconciliation_status": record.reconciliation_status,
            "calendar_source": item.calendar_source,
            "calendar_inference_status": item.calendar_inference_status,
            "is_bank_workday": item.is_bank_workday,
            "business_step": item.business_step,
            "target_date": item.target_date,
            "execution_status": status,
        })
    return tuple(rows)


def _write_daily(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    fields = (
        "calendar_date", "inflow_cny", "outflow_cny", "transaction_count",
        "ending_balance_cny", "day_cut_basis", "source_last_transaction_id",
        "reconciliation_status", "calendar_source", "calendar_inference_status",
        "is_bank_workday", "business_step", "target_date", "execution_status",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _profile_payload(profile: ResolvedFormalProfile, run: ScenarioRunResult, effective_run_id: str) -> dict[str, Any]:
    return {
        "schema_version": "syn_b1_formal_profile_output_1_0",
        "sample_id": profile.sample_id,
        "run_id": effective_run_id,
        "input_profile": _jsonable(profile.resolved_profile),
        "resolved_runtime_settings": _jsonable(run.request.runtime_settings),
        "initial_capital_resolution": _jsonable(run.initial_capital),
    }


def _quality_payload(run: ScenarioRunResult, daily_rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    transactions = run.ledger.transactions
    debit_total = sum(item.debit_fen for item in transactions)
    credit_total = sum(item.credit_fen for item in transactions)
    unsettled_funding_in_period = [
        item for item in run.funding_timeline.pending_tail
        if item.status.value != "superseded_before_first_event"
        and item.event.due_date <= run.request.end_date
    ]
    future_training_data_quality_ready = (
        run.ledger.is_complete
        and run.transaction_plan.is_amount_reconciled
        and not unsettled_funding_in_period
        and run.final_cash_reconciliation.is_reconciled
    )
    return {
        "schema_version": "syn_b1_quality_report_1_0",
        "execution_status": "complete" if run.ledger.is_complete else "stopped_before_blocked_outflow",
        # F7F explicitly defers model training.  A complete CSV can be marked
        # ready for a later data-quality gate, but cannot claim current model
        # eligibility merely because it reached its planned end date.
        "training_eligible": False,
        "training_eligibility_reason_cn": "模型训练仍属F7F明确延期范围；本报告不授予训练资格。",
        "future_training_data_quality_ready": future_training_data_quality_ready,
        "human_review_notes": {
            "file": "cash_flow_review_notes.csv",
            "purpose_cn": "仅供人工理解合成现金流来源和用途。",
            "model_input_allowed": False,
            "database_import_allowed": False,
        },
        "checks": {
            "candidate_amount_reconciled": run.transaction_plan.is_amount_reconciled,
            "transaction_count": len(transactions),
            "daily_transaction_count": sum(int(row["transaction_count"]) for row in daily_rows),
            "debit_total_cny": _cny(debit_total),
            "daily_outflow_total_cny": _decimal_cny(sum((Decimal(str(row["outflow_cny"])) for row in daily_rows), Decimal("0"))),
            "credit_total_cny": _cny(credit_total),
            "daily_inflow_total_cny": _decimal_cny(sum((Decimal(str(row["inflow_cny"])) for row in daily_rows), Decimal("0"))),
            "closing_balance_cny": _cny(run.ledger.closing_balance_fen),
            "daily_closing_balance_cny": daily_rows[-1]["ending_balance_cny"] if daily_rows else _cny(run.ledger.opening_balance_fen),
            "natural_day_count": len(daily_rows),
            "funding_tail_settled_through_planned_end": not unsettled_funding_in_period,
            "final_cash_reconciled": run.final_cash_reconciliation.is_reconciled,
        },
        "failure_diagnostic_available": not run.ledger.is_complete,
    }


def _write_fact_if_stopped(directory: Path, run: ScenarioRunResult) -> Path | None:
    if run.ledger.is_complete:
        return None
    fact = run.ledger.fact_sheet
    if fact is None:
        raise FormalExportError("中断运行缺少资金不足事实单。")
    path = directory / "generation_fact_sheet.json"
    _write_json(path, {
        "schema_version": "syn_b1_generation_fact_sheet_1_0",
        "execution_status": "stopped_before_blocked_outflow",
        "opening_balance_cny": _cny(fact.opening_balance_fen),
        "cumulative_inflow_cny": _cny(fact.cumulative_inflow_fen),
        "cumulative_outflow_cny": _cny(fact.cumulative_outflow_fen),
        "available_balance_before_block_cny": _cny(fact.available_balance_before_block_fen),
        "blocked_booking_datetime": fact.blocked_booking_datetime.isoformat(timespec="seconds"),
        "blocked_outflow_cny": _cny(fact.blocked_outflow_fen),
        "shortage_cny": _cny(fact.shortage_fen),
        "completed_movement_count": fact.completed_movement_count,
        "planned_movement_count": fact.planned_movement_count,
        "next_action_zh": "请调整可见的首次注资、后续注资、借款合同、经营收付或事项参数后，以新的运行编号重新生成。",
    })
    return path


def _write_restricted(directory: Path, run: ScenarioRunResult) -> None:
    directory.mkdir()
    # 新订单完整未来计划只写入受限目录，不混入对外逐笔备注。
    if run.operating_cycle.request.supplementary_orders:
        # 保存同一份冻结数据类供逐单追溯。
        _write_json(directory / "supplementary_orders.json", _jsonable(run.operating_cycle.request.supplementary_orders))
    # The final tax-adjusted statements are the only statement set whose cash
    # is allowed to be called the final scenario cash.  Earlier stages stay
    # available separately for audit; no statement calculation is duplicated.
    _write_json(directory / "three_statements.json", _jsonable(run.tax_routing.tax_adjusted_statements))
    _write_json(directory / "statement_stages.json", {
        "pre_financing": _jsonable(run.statements),
        "post_financing_before_tax": _jsonable(run.bridge.post_contract_statements),
        "final_after_tax": _jsonable(run.tax_routing.tax_adjusted_statements),
    })
    _write_json(directory / "final_cash_reconciliation.json", {
        **_jsonable(run.final_cash_reconciliation),
        "is_reconciled": run.final_cash_reconciliation.is_reconciled,
    })
    _write_json(directory / "tax_ledger.json", _jsonable(run.tax_routing))
    _write_json(directory / "funding_tail.json", _jsonable(run.funding_timeline))
    _write_json(directory / "scenario_truth.json", {
        "operating_cycle": _jsonable(run.operating_cycle),
        "cash_plan_bridge": _jsonable(run.bridge),
        "transaction_plan": _jsonable(run.transaction_plan),
    })


def _manifest_payload(
    profile: ResolvedFormalProfile,
    run: ScenarioRunResult,
    effective_run_id: str,
    directory: Path,
    continuation_of_run_id: str | None,
) -> dict[str, Any]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    # 新日历及订单升级显式登记新生成器版本，不冒充历史冻结位元。
    from syn_b1.calendar_adapter import ORDER_CALENDAR_VERSION, CYCLE_CALENDAR_VERSION
    # 只对明确开启新版本的运行改变版本标识。
    new_orders_version = run.request.runtime_settings is not None and run.request.runtime_settings.bank_calendar.version == ORDER_CALENDAR_VERSION
    return {
        "schema_version": "syn_b1_generation_manifest_1_0",
        "sample_id": profile.sample_id,
        "run_id": effective_run_id,
        "continuation_of_run_id": continuation_of_run_id,
        "execution_status": "complete" if run.ledger.is_complete else "stopped_before_blocked_outflow",
        "generator_version": (
            # 数量节点和精确合同联合通道独立登记，旧画像版本不变。
            "syn_b1_physical_contract_business_20260913_v1" if getattr(getattr(run.request.runtime_settings, "business_node_plan", None), "capacity_mode", None) == "owned_assets_with_physical_nodes" else
            # 新月度节点有独立版本标识，不能冒充历史季节或订单版本。
            "syn_b1_monthly_business_nodes_20260909_v1" if run.request.runtime_settings is not None and run.request.runtime_settings.business_node_plan is not None else
            # 周期版本独立标识，不冒充九组或历史生成器。
            "syn_b1_cycle_trade_20260908_1_0" if run.request.runtime_settings is not None and run.request.runtime_settings.bank_calendar.version == CYCLE_CALENDAR_VERSION else
            # 新样板及同日历对照绑定本轮获准的代码范围。
            "syn_b1_supplementary_orders_20260908_1_0" if new_orders_version else
            FORMAL_GENERATOR_VERSION_2_0_CANDIDATE
            if profile.resolved_profile.get("schema_version") == "syn_b1_formal_profile_2_0"
            else FORMAL_GENERATOR_VERSION
        ),
        "ledger_version": run.ledger.manifest.ledger_version,
        "calendar_version": run.request.runtime_settings.bank_calendar.version if run.request.runtime_settings else None,
        "random_seed": run.request.random_seed,
        "period": {
            "start_date": run.request.start_date.isoformat(),
            "planned_end_date": run.request.end_date.isoformat(),
            "written_through_date": (run.request.end_date if run.ledger.is_complete else run.ledger.fact_sheet.blocked_booking_datetime.date()).isoformat(),
        },
        "files": {str(path.relative_to(directory)).replace("\\", "/"): _sha256(path) for path in files},
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key.value if isinstance(key, Enum) else key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, Decimal)):
        return str(value) if isinstance(value, Decimal) else value.isoformat()
    if callable(value):
        # A callable is an implementation link (for example the frozen R6
        # calendar adapter), not hidden scenario truth.  Persist its name only
        # so the restricted audit file remains readable and serializable.
        return getattr(value, "__qualname__", type(value).__name__)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cny(value_fen: int) -> str:
    return f"{value_fen // 100}.{value_fen % 100:02d}"


def _decimal_cny(value: Decimal | None) -> str:
    if value is None:
        return ""
    return f"{value.quantize(Decimal('0.01'))}"
