"""Managed local workspace for the SYN-B1 user-facing generator only.

This module owns file locations and versioned parameter files.  It does not
parse business parameters, generate transactions, or write formal CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from syn_b1.formal_profile import FormalProfileError, default_profile, default_profile_v2, resolve_profile
from syn_b1.profile_document import ProfileDocumentError, read_profile_document, write_user_profile


@dataclass(frozen=True)
class GeneratorWorkspace:
    """One physically isolated local workspace for the SYN-B1 generator."""

    root: Path
    generated_output_root: Path | None = None
    profile_schema_version: str = "1.0"

    @property
    def profiles_root(self) -> Path:
        return self.root / "profiles"

    @property
    def generated_root(self) -> Path:
        return self.generated_output_root or self.root / "generated"

    def profile_path(self, enterprise_id: str, profile_version: str) -> Path:
        return self.profiles_root / _folder_name(enterprise_id, "虚构企业编号") / f"{_folder_name(profile_version, '参数版本')}.yaml"

    def create_profile(self, enterprise_id: str, profile_version: str) -> Path:
        target = self.profile_path(enterprise_id, profile_version)
        if self._version_exists(target):
            raise FormalProfileError("该企业的同名参数版本已存在。请填写新的参数版本；历史版本不会被覆盖。")
        factory = default_profile_v2 if self.profile_schema_version == "2.0" else default_profile
        payload = factory(enterprise_id, f"{enterprise_id}_{profile_version}")
        try:
            return write_user_profile(target, payload)
        except ProfileDocumentError as error:
            raise FormalProfileError(str(error)) from error

    def copy_profile_as_new_version(
        self,
        source: Path,
        enterprise_id: str,
        profile_version: str,
    ) -> Path:
        target = self.profile_path(enterprise_id, profile_version)
        if self._version_exists(target):
            raise FormalProfileError("该企业的同名参数版本已存在。请使用新的参数版本；历史版本不会被覆盖。")
        try:
            payload = read_profile_document(source)
        except ProfileDocumentError as error:
            raise FormalProfileError("原参数文件无法读取，不能创建新版本。") from error
        if not isinstance(payload, dict):
            raise FormalProfileError("原参数文件格式无效，不能创建新版本。")
        payload["sample_id"] = enterprise_id
        payload["run_id"] = f"{enterprise_id}_{profile_version}"
        if self.profile_schema_version == "2.0" and payload.get("schema_version") != "syn_b1_formal_profile_2_0":
            upgraded = default_profile_v2(enterprise_id, f"{enterprise_id}_{profile_version}")
            for key in ("start_date", "end_date", "random_seed", "calendar", "account", "initial_capital", "additional_capital_plans", "loan_contracts", "tax", "transactions"):
                if key in payload:
                    upgraded[key] = payload[key]
            old_operating = payload.get("operating") if isinstance(payload.get("operating"), dict) else {}
            new_operating = upgraded["operating"]
            start_month = str(upgraded.get("start_date", ""))[:7]
            end_month = str(upgraded.get("end_date", ""))[:7]
            calendar = upgraded.get("calendar") if isinstance(upgraded.get("calendar"), dict) else {}
            declared_month = max(end_month, str(calendar.get("prediction_cutoff", ""))[:7])
            legacy_growth = old_operating.get("sales_monthly_growth_percent", "0")
            new_operating["regime_plan"]["shape"] = _legacy_shape(legacy_growth)
            new_operating["regime_plan"]["declared_through_month"] = declared_month
            new_operating["regime_plan"]["base_latent_monthly_sales_cny"] = old_operating.get("base_monthly_sales_cny", "300000.00")
            new_operating["regime_plan"]["segments"][0]["start_month"] = start_month
            new_operating["regime_plan"]["segments"][0]["monthly_linear_change_percent"] = legacy_growth
            new_operating["regime_plan"]["segments"][0]["gross_margin_percent"] = old_operating.get("gross_margin_percent", "35")
            old_weights = old_operating.get("sales_month_weights_percent")
            if isinstance(old_weights, dict):
                new_operating["sales_seasonality"]["month_weights_percent"] = old_weights
                new_operating["sales_seasonality"]["enabled"] = any(
                    Decimal(str(value)) != Decimal("100") for value in old_weights.values()
                )
            new_operating["collection_policy"]["base_schedule"] = old_operating.get("collection_schedule", [{"delay_days": 30, "share_percent": "100"}])
            for key in (
                "production_cycle_days", "inventory_target_days", "supplier_payment_schedule",
                "monthly_payroll_cny", "monthly_rent_and_utilities_cny", "monthly_other_operating_expense_cny",
                "payroll_payment_schedule", "rent_and_utilities_payment_schedule",
                "other_operating_expense_payment_schedule", "fixed_assets",
            ):
                if key in old_operating:
                    new_operating[key] = old_operating[key]
            payload = upgraded
        # Never save a migrated or copied profile that the selected generator
        # cannot immediately validate.  The source remains untouched on error.
        resolve_profile(payload)
        try:
            return write_user_profile(target, payload)
        except ProfileDocumentError as error:
            raise FormalProfileError(str(error)) from error

    def list_profiles(self) -> tuple[Path, ...]:
        if not self.profiles_root.is_dir():
            return ()
        allowed_suffixes = {".yaml", ".yml", ".json"}
        return tuple(sorted(path for path in self.profiles_root.rglob("*") if path.is_file() and path.suffix.lower() in allowed_suffixes))

    def list_generated_runs(self) -> tuple[Path, ...]:
        if not self.generated_root.is_dir():
            return ()
        return tuple(sorted(
            path.parent for path in self.generated_root.rglob("generation_manifest.json") if path.is_file()
        ))

    @staticmethod
    def _version_exists(target: Path) -> bool:
        return any(target.with_suffix(suffix).exists() for suffix in (".yaml", ".yml", ".json"))


def _folder_name(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or normalized in {".", ".."} or any(marker in normalized for marker in ("/", "\\", "\x00")):
        raise FormalProfileError(f"{label}不能为空，也不能包含路径符号。")
    return normalized


def _legacy_shape(value: object) -> str:
    try:
        growth = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise FormalProfileError("旧画像字段 operating.sales_monthly_growth_percent 无效；请填写-20至20之间的数字后再复制。") from error
    if growth > 0:
        return "sustained_growth"
    if growth < 0:
        return "sustained_contraction"
    return "stable"
