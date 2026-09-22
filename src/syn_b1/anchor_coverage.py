"""Restricted SYN-B1 anchor coverage report.

The report reads only the pre-existing anonymous feature cards.  It never
opens workbooks, never imports the sealed SG3 generator, and never writes a
formal dataset.  Exact per-anchor values stay in a caller-chosen restricted
directory; the returned public summary contains counts only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from syn_b1.initial_capital import InitialCapitalInput, InitialCapitalMode
from syn_b1.scenario_runner import ScenarioRunRequest, build_scenario_run
from syn_b1.transaction_planner import CategoryTransactionPolicy, MonthConcentration


class AnchorCoverageError(ValueError):
    """The restricted anonymous anchor input or output boundary is invalid."""


@dataclass(frozen=True)
class AnchorCoverageSummary:
    anonymous_anchor_count: int
    evaluated_anchor_count: int
    qualified_anchor_count: int
    unmatched_anchor_count: int
    seeds_per_anchor: int
    real_workbook_read: bool = False
    formal_csv_written: bool = False
    database_written: bool = False


def run_anchor_coverage_review(
    card_directory: str | Path,
    *,
    seeds: tuple[int, ...] = (94111, 94112, 94113),
) -> tuple[AnchorCoverageSummary, tuple[dict[str, Any], ...]]:
    """Evaluate all twenty cards against the current, deliberately narrow B1 controls.

    An anchor is marked *unmatched* when a current B1 control cannot express a
    reference dimension.  This is a reported limitation, not an excuse to drop
    the account or to invent a parameter from its raw transaction history.
    """
    directory = Path(card_directory)
    cards = _load_cards(directory)
    rows = []
    for anonymous_id, card in cards:
        reference_balance = _reference_balance_fen(card)
        seed_results = []
        for seed in seeds:
            run = build_scenario_run(_anchor_request(anonymous_id, reference_balance, seed))
            generated_balance = round(mean(row.ending_balance_fen for row in run.ledger.daily_rows))
            generated_monthly_gross = round(
                (sum(item.debit_fen + item.credit_fen for item in run.ledger.transactions) * 30)
                / max(1, len(run.ledger.daily_rows))
            )
            seed_results.append({
                "seed": seed,
                "ledger_complete": run.ledger.is_complete,
                "generated_average_balance_fen": generated_balance,
                "generated_monthly_gross_fen": generated_monthly_gross,
            })
        reasons = [
            "当前B1尚未开放日活跃度、跨月集中、金额长尾和微观结构参数，不能声称完整风格覆盖。",
            "本报告只比较可由当前B1公开控制表达的余额尺度和收付强度；不推断经营原因。",
        ]
        rows.append({
            "anonymous_id": anonymous_id,
            "evaluated": True,
            "qualified": False,
            "coverage_status": "evaluated_but_unmatched_due_to_deferred_dimensions",
            "reference_average_balance_fen": reference_balance,
            "seed_results": seed_results,
            "gap_reasons_cn": reasons,
        })
    summary = AnchorCoverageSummary(
        anonymous_anchor_count=len(cards),
        evaluated_anchor_count=len(rows),
        qualified_anchor_count=0,
        unmatched_anchor_count=len(rows),
        seeds_per_anchor=len(seeds),
    )
    return summary, tuple(rows)


def write_restricted_anchor_report(
    summary: AnchorCoverageSummary,
    rows: tuple[dict[str, Any], ...],
    output_directory: str | Path,
) -> Path:
    """Write a non-overwriting private report; no values enter ordinary docs."""
    target = Path(output_directory)
    if target.exists():
        raise AnchorCoverageError("受限锚点报告目录已存在，禁止覆盖历史报告。")
    target.mkdir(parents=True)
    payload = {
        "stage": "SYN-B1-I5-RC1",
        "summary": summary.__dict__,
        "rows": rows,
        "real_workbook_read": False,
        "formal_csv_written": False,
        "database_written": False,
    }
    path = target / "syn_b1_anchor_coverage_private.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def safe_anchor_summary(summary: AnchorCoverageSummary) -> dict[str, int | bool]:
    """Return only aggregate counts safe for a normal acceptance record."""
    return summary.__dict__.copy()


def _load_cards(directory: Path) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    paths = sorted(directory.glob("cohort_*.json"))
    if len(paths) != 20:
        raise AnchorCoverageError("新版SYN-B1锚点报告必须恰好读取20张匿名特征卡。")
    cards = []
    for path in paths:
        card = json.loads(path.read_text(encoding="utf-8"))
        anonymous_id = str(card.get("anonymous_id", ""))
        if not anonymous_id or "full" not in card:
            raise AnchorCoverageError("匿名特征卡缺少编号或完整特征块。")
        cards.append((anonymous_id, card))
    if len({item[0] for item in cards}) != 20:
        raise AnchorCoverageError("20张匿名特征卡编号必须一一对应。")
    return tuple(cards)


def _reference_balance_fen(card: Mapping[str, Any]) -> int:
    values = card["full"].get("values", {})
    for key in ("average_deposit_balance_30cd", "average_deposit_balance_90cd", "deposit_balance_scale"):
        value = values.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return max(1, round(value * 100))
    return 1_000_000


def _anchor_request(anonymous_id: str, reference_balance_fen: int, seed: int) -> ScenarioRunRequest:
    # A fixed period after the reviewed data avoids copying dates or trajectories.
    return ScenarioRunRequest(
        synthetic_account_id=f"syn_b1_anchor_{anonymous_id}",
        run_id=f"syn-b1-anchor-{anonymous_id}",
        start_date=date(2027, 1, 1),
        end_date=date(2027, 6, 30),
        random_seed=seed,
        base_monthly_sales_fen=max(1_000_000, reference_balance_fen * 4),
        gross_margin_rate_bp=5_000,
        collection_delay_days=0,
        supplier_delay_days=0,
        production_cycle_days=30,
        monthly_payroll_fen=max(100_000, reference_balance_fen // 5),
        monthly_rent_and_utilities_fen=max(50_000, reference_balance_fen // 10),
        initial_capital=InitialCapitalInput(
            capital_item_id=f"anchor_initial_capital_{anonymous_id}",
            plan_version="v1",
            mode=InitialCapitalMode.MANUAL,
            amount_fen=reference_balance_fen,
        ),
        transaction_policy=CategoryTransactionPolicy(3, MonthConcentration.UNIFORM, 2_000),
    )
