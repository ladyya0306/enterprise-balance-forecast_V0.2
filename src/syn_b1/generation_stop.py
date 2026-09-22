"""Deterministic stop-and-explain control for future transaction generation.

This module does not invent financing, reschedule cash, or change amounts.  It
checks an already planned, date-ordered stream and stops before the first
outflow that the available balance cannot pay in full.  I4B can reuse the
result to show a fact sheet and a parameter-adjustment panel before the user
starts a new generation run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence


class GenerationStopError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class MovementDirection(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"


@dataclass(frozen=True)
class PlannedCashMovement:
    movement_id: str
    booking_datetime: datetime
    sequence_no: int
    direction: MovementDirection
    amount_fen: int

    def __post_init__(self) -> None:
        if not self.movement_id:
            raise GenerationStopError("EMPTY_MOVEMENT_ID", "计划现金事项编号不能为空。")
        if not isinstance(self.booking_datetime, datetime):
            raise GenerationStopError("INVALID_BOOKING_DATETIME", "计划现金事项必须提供日期时间。")
        if not isinstance(self.sequence_no, int) or isinstance(self.sequence_no, bool) or self.sequence_no < 0:
            raise GenerationStopError("INVALID_SEQUENCE_NO", "同一时点顺序必须为非负整数。")
        if not isinstance(self.direction, MovementDirection):
            raise GenerationStopError("INVALID_MOVEMENT_DIRECTION", "现金方向必须为流入或流出。")
        if not isinstance(self.amount_fen, int) or isinstance(self.amount_fen, bool) or self.amount_fen <= 0:
            raise GenerationStopError("INVALID_MOVEMENT_AMOUNT", "计划现金金额必须为正整数分。")


@dataclass(frozen=True)
class ParameterAdjustmentField:
    parameter_code: str
    display_name_cn: str
    current_fact_cn: str
    allowed_change_cn: str


@dataclass(frozen=True)
class GenerationAdjustmentPanel:
    reason_cn: str
    fields: tuple[ParameterAdjustmentField, ...]
    confirmation_cn: str


@dataclass(frozen=True)
class CashExhaustionFactSheet:
    opening_balance_fen: int
    cumulative_inflow_fen: int
    cumulative_outflow_fen: int
    available_balance_before_block_fen: int
    blocked_movement_id: str
    blocked_booking_datetime: datetime
    blocked_outflow_fen: int
    shortage_fen: int
    completed_movement_count: int
    planned_movement_count: int


@dataclass(frozen=True)
class CashExecutionPreviewResult:
    is_complete: bool
    completed_movements: tuple[PlannedCashMovement, ...]
    closing_balance_fen: int
    fact_sheet: CashExhaustionFactSheet | None = None
    adjustment_panel: GenerationAdjustmentPanel | None = None


def preview_cash_sufficiency(
    opening_balance_fen: int,
    movements: Sequence[PlannedCashMovement],
) -> CashExecutionPreviewResult:
    """Return the payable prefix; never alter a movement to force completion."""
    if not isinstance(opening_balance_fen, int) or isinstance(opening_balance_fen, bool) or opening_balance_fen < 0:
        raise GenerationStopError("INVALID_OPENING_BALANCE", "期初余额必须为非负整数分。")
    ids = [item.movement_id for item in movements]
    if len(ids) != len(set(ids)):
        raise GenerationStopError("DUPLICATE_MOVEMENT_ID", "计划现金事项编号不能重复。")
    order_keys = [(item.booking_datetime, item.sequence_no) for item in movements]
    if len(order_keys) != len(set(order_keys)):
        raise GenerationStopError("DUPLICATE_MOVEMENT_ORDER", "同一日期时间和顺序号只能对应一笔计划事项。")

    ordered = tuple(sorted(movements, key=lambda item: (item.booking_datetime, item.sequence_no, item.movement_id)))
    balance = opening_balance_fen
    cumulative_inflow = 0
    cumulative_outflow = 0
    completed: list[PlannedCashMovement] = []
    for movement in ordered:
        if movement.direction is MovementDirection.INFLOW:
            balance += movement.amount_fen
            cumulative_inflow += movement.amount_fen
            completed.append(movement)
            continue
        if movement.amount_fen > balance:
            fact = CashExhaustionFactSheet(
                opening_balance_fen=opening_balance_fen,
                cumulative_inflow_fen=cumulative_inflow,
                cumulative_outflow_fen=cumulative_outflow,
                available_balance_before_block_fen=balance,
                blocked_movement_id=movement.movement_id,
                blocked_booking_datetime=movement.booking_datetime,
                blocked_outflow_fen=movement.amount_fen,
                shortage_fen=movement.amount_fen - balance,
                completed_movement_count=len(completed),
                planned_movement_count=len(ordered),
            )
            return CashExecutionPreviewResult(
                is_complete=False,
                completed_movements=tuple(completed),
                closing_balance_fen=balance,
                fact_sheet=fact,
                adjustment_panel=_adjustment_panel(fact),
            )
        balance -= movement.amount_fen
        cumulative_outflow += movement.amount_fen
        completed.append(movement)
    return CashExecutionPreviewResult(
        is_complete=True,
        completed_movements=tuple(completed),
        closing_balance_fen=balance,
    )


def _adjustment_panel(fact: CashExhaustionFactSheet) -> GenerationAdjustmentPanel:
    fields = (
        ParameterAdjustmentField(
            "opening_balance_fen", "期初余额", f"当前为{fact.opening_balance_fen}分",
            "用户可以提高或降低设定值；系统不会自行改动。",
        ),
        ParameterAdjustmentField(
            "operating_inflow", "经营流入金额、频率和到账时间",
            f"中断前累计流入{fact.cumulative_inflow_fen}分",
            "用户可以调整回款总额、回款频率或到账日期。",
        ),
        ParameterAdjustmentField(
            "operating_outflow", "经营流出金额、频率和付款时间",
            f"中断前累计流出{fact.cumulative_outflow_fen}分",
            "用户可以调整采购、工资、房租水电、税费等金额、频率或日期。",
        ),
        ParameterAdjustmentField(
            "funding_contracts", "股东资金和银行借款合同",
            f"被阻断支出尚缺{fact.shortage_fen}分",
            "只有用户新增或修改明确合同，系统才重新计算融资流入和后续还本付息。",
        ),
        ParameterAdjustmentField(
            "special_outflow_events", "设备、厂房等特殊支出",
            f"本次被阻断支出为{fact.blocked_outflow_fen}分",
            "用户可以修改事件金额、日期或取消事件，并以新参数版本重新生成。",
        ),
    )
    return GenerationAdjustmentPanel(
        reason_cn="现有余额不足以全额支付下一笔计划流出，流水已在记账前停止。",
        fields=fields,
        confirmation_cn="修改参数后必须创建新的计划版本并重新预览；不得覆盖已完成历史或由系统自动补资。",
    )
