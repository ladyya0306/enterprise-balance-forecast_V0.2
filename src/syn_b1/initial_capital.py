"""Resolve one frozen starting-capital input before a SYN-B1 run begins.

This module owns only the choice and identity of the first shareholder
capital injection.  It does not plan loans, change cash, write transactions,
or persist files.  The resolved plan is passed unchanged to I3B.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from hashlib import sha256
from random import Random

from syn_b1.funding_timeline import ShareholderCapitalPlan


class InitialCapitalError(ValueError):
    """The frozen first-capital input cannot be resolved safely."""


class InitialCapitalMode(str, Enum):
    MANUAL = "manual"
    PROFILE_RANDOM = "profile_random"


class InitialCapitalDistribution(str, Enum):
    UNIFORM = "uniform"


@dataclass(frozen=True)
class InitialCapitalInput:
    """The one required starting-capital instruction for a new enterprise."""

    capital_item_id: str
    plan_version: str
    mode: InitialCapitalMode
    amount_fen: int | None = None
    minimum_amount_fen: int | None = None
    maximum_amount_fen: int | None = None
    distribution: InitialCapitalDistribution | None = None
    profile_version: str | None = None

    def __post_init__(self) -> None:
        if not self.capital_item_id or not self.plan_version:
            raise InitialCapitalError("首次股东注资必须提供编号和版本。")
        if not isinstance(self.mode, InitialCapitalMode):
            raise InitialCapitalError("首次股东注资方式必须为手工或随机画像。")
        if self.mode is InitialCapitalMode.MANUAL:
            _require_positive("amount_fen", self.amount_fen)
            if any(value is not None for value in (self.minimum_amount_fen, self.maximum_amount_fen, self.distribution, self.profile_version)):
                raise InitialCapitalError("手工首次注资不能同时填写随机画像范围或版本。")
        else:
            if self.amount_fen is not None:
                raise InitialCapitalError("随机首次注资不能同时填写固定金额。")
            _require_positive("minimum_amount_fen", self.minimum_amount_fen)
            _require_positive("maximum_amount_fen", self.maximum_amount_fen)
            if self.minimum_amount_fen > self.maximum_amount_fen:
                raise InitialCapitalError("随机首次注资下限不能高于上限。")
            if self.distribution is not InitialCapitalDistribution.UNIFORM:
                raise InitialCapitalError("首期随机首次注资只支持均匀分布。")
            if not self.profile_version:
                raise InitialCapitalError("随机首次注资必须保存画像版本。")


@dataclass(frozen=True)
class ResolvedInitialCapital:
    """The immutable pre-run result passed to the funding timeline."""

    input: InitialCapitalInput
    plan: ShareholderCapitalPlan
    root_seed: int
    derived_seed: int | None


def resolve_initial_capital(
    value: InitialCapitalInput,
    *,
    start_date: date,
    root_seed: int,
) -> ResolvedInitialCapital:
    """Resolve the one first-inflow plan before any cash calculation starts."""
    if not isinstance(start_date, date):
        raise InitialCapitalError("首次生成开始日期必须是date。")
    if not isinstance(root_seed, int) or isinstance(root_seed, bool):
        raise InitialCapitalError("主随机种子必须是整数。")
    if value.mode is InitialCapitalMode.MANUAL:
        amount = value.amount_fen
        derived = None
    else:
        derived = _derived_seed(root_seed, value.capital_item_id, value.plan_version, value.profile_version or "")
        amount = Random(derived).randint(value.minimum_amount_fen, value.maximum_amount_fen)
    assert amount is not None
    return ResolvedInitialCapital(
        input=value,
        plan=ShareholderCapitalPlan(
            capital_item_id=value.capital_item_id,
            plan_version=value.plan_version,
            due_date=start_date,
            amount_fen=amount,
        ),
        root_seed=root_seed,
        derived_seed=derived,
    )


def _derived_seed(root_seed: int, capital_item_id: str, plan_version: str, profile_version: str) -> int:
    digest = sha256(
        f"syn_b1_initial_capital_1_0:{root_seed}:{capital_item_id}:{plan_version}:{profile_version}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _require_positive(name: str, value: int | None) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InitialCapitalError(f"{name}必须为正整数分。")
