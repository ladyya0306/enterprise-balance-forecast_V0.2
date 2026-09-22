"""SYN-B1-I7 formal generation boundary: profile -> central run -> formal files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from syn_b1.formal_export import FormalExportBundle, write_formal_bundle
from syn_b1.formal_profile import ResolvedFormalProfile, load_profile
from syn_b1.scenario_runner import ScenarioRunResult, build_scenario_run


@dataclass(frozen=True)
class FormalGenerationResult:
    """One inspectable, reproducible formal generation result."""

    profile: ResolvedFormalProfile
    run: ScenarioRunResult
    bundle: FormalExportBundle


def generate_from_profile(
    profile_path: str | Path,
    *,
    output_root: str | Path,
    output_run_id: str | None = None,
) -> FormalGenerationResult:
    """Generate exactly one new formal run; no SQLite or model action occurs."""

    profile = load_profile(profile_path)
    run = build_scenario_run(profile.request)
    bundle = write_formal_bundle(profile, run, output_root=output_root, output_run_id=output_run_id)
    return FormalGenerationResult(profile, run, bundle)


def preview_from_profile(profile_path: str | Path) -> tuple[ResolvedFormalProfile, ScenarioRunResult]:
    """Validate and calculate in memory without writing any output file."""

    profile = load_profile(profile_path)
    return profile, build_scenario_run(profile.request)
