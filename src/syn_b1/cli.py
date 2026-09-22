"""Chinese, thin command-line entry for the SYN-B1 formal generator."""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence

from syn_b1.continuation_state import (
    begin_continuation_state, continue_verified_run, verify_persisted_formal_prefix,
)
from syn_b1.formal_export import write_formal_bundle
from syn_b1.formal_history import load_verified_formal_history
from syn_b1.formal_generator import generate_from_profile, preview_from_profile
from syn_b1.formal_profile import FormalProfileError, load_profile, write_template
from syn_b1.profile_workspace import GeneratorWorkspace
from syn_b1.scenario_runner import build_scenario_run


DEFAULT_WORKSPACE_ROOT = Path(os.environ.get("SYN_B1_WORKSPACE", str(Path("apps") / "syn_b1_generator" / "workspace")))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("SYN_B1_OUTPUT_ROOT", str(Path("data") / "synthetic_generated" / "syn_b1")))
DEFAULT_PROFILE_SCHEMA_VERSION = os.environ.get("SYN_B1_PROFILE_SCHEMA_VERSION", "1.0")
APP_DISPLAY_VERSION = os.environ.get("SYN_B1_APP_DISPLAY_VERSION", "1.0")
ENTRY_ISOLATION_ENFORCED = os.environ.get("SYN_B1_ENFORCE_ISOLATION", "0") == "1"


def main(argv: Sequence[str] | None = None) -> int:
    """Run an explicit generator action; this entry never touches SQLite or models."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        return _interactive()
    try:
        if args.command == "new-profile":
            _assert_entry_path(args.path, DEFAULT_WORKSPACE_ROOT / "profiles", "新参数方案")
            path = write_template(
                args.path, sample_id=args.sample_id, run_id=args.run_id,
                profile_schema_version=DEFAULT_PROFILE_SCHEMA_VERSION,
            )
            print(f"已新建参数文件：{path}")
            print("此命令供高级用户使用；双击BAT时请使用菜单，让程序自动管理保存位置。")
            return 0
        if args.command == "validate":
            profile = load_profile(args.profile)
            _assert_entry_profile(profile)
            print(f"核验通过：虚构企业 {profile.sample_id}，参数版本运行标识 {profile.run_id}。未生成任何文件。")
            return 0
        if args.command == "preview":
            profile = load_profile(args.profile)
            _assert_entry_profile(profile)
            run = build_scenario_run(profile.request)
            print(_preview_text(profile.sample_id, profile.run_id, run))
            return 0
        if args.command == "generate":
            profile = load_profile(args.profile)
            _assert_entry_profile(profile)
            _assert_entry_output_root(args.output_root)
            result = generate_from_profile(
                args.profile,
                output_root=args.output_root,
                output_run_id=args.output_run_id,
            )
            print(_result_text(
                result.profile.sample_id,
                args.output_run_id or result.profile.run_id,
                result.bundle.output_dir,
                result.run.ledger.is_complete,
            ))
            return 0
        if args.command == "show":
            _assert_entry_path(args.directory, DEFAULT_OUTPUT_ROOT, "生成结果")
            return _show(args.directory)
        if args.command == "continue":
            return _continue(args)
    except (FormalProfileError, ValueError) as error:
        print(f"未执行写入：{error}")
        return 2
    parser.error("未知命令")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic_generator_2_0.bat" if APP_DISPLAY_VERSION == "2.0" else "synthetic_generator.bat",
        description="SYN-B1 虚构制造企业现金流水生成器（只生成合成数据，不读取真实流水）。",
    )
    commands = parser.add_subparsers(dest="command")
    new = commands.add_parser("new-profile", help="新建一份可见参数模板（高级用法）")
    new.add_argument("--path", required=True, help="新YAML参数文件路径")
    new.add_argument("--sample-id", required=True, help="虚构企业ID")
    new.add_argument("--run-id", required=True, help="参数版本运行标识")
    check = commands.add_parser("validate", help="核验参数，不生成文件")
    check.add_argument("--profile", required=True, help="参数YAML路径（历史JSON仍可读取）")
    preview = commands.add_parser("preview", help="内存预演，不写入文件")
    preview.add_argument("--profile", required=True, help="参数YAML路径（历史JSON仍可读取）")
    generate = commands.add_parser("generate", help="正式生成一套按ID保存的CSV")
    generate.add_argument("--profile", required=True, help="已核验的参数YAML路径（历史JSON仍可读取）")
    generate.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="正式输出根目录")
    generate.add_argument("--output-run-id", help="本次输出运行编号，不覆盖历史结果")
    show = commands.add_parser("show", help="查看已生成结果的简要状态")
    show.add_argument("--directory", required=True, help="某次正式输出目录")
    extend = commands.add_parser("continue", help="按同一参数和种子向后续生成")
    extend.add_argument("--base-run-directory", required=True, help="原正式运行目录（必须含清单、保存参数和逐笔CSV）")
    extend.add_argument("--extension-profile", required=True, help="只延长日期后的同参数YAML")
    extend.add_argument("--output-run-id", required=True, help="新续生成输出编号，不覆盖原运行")
    extend.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="正式输出根目录")
    return parser


def _continue(args: argparse.Namespace) -> int:
    _assert_entry_output_root(args.output_root)
    _assert_entry_path(args.base_run_directory, DEFAULT_OUTPUT_ROOT, "续生成原运行")
    history = load_verified_formal_history(args.base_run_directory)
    _assert_entry_profile(history.profile)
    base_run = build_scenario_run(history.profile.request)
    state = begin_continuation_state(base_run)
    verify_persisted_formal_prefix(
        state,
        sample_id=history.profile.sample_id,
        logical_run_id=history.profile.run_id,
        transaction_count=history.transaction_count,
        transaction_digest=history.transaction_digest,
    )
    extension_profile = load_profile(args.extension_profile)
    _assert_entry_profile(extension_profile)
    continuation = continue_verified_run(state, extension_profile.request)
    bundle = write_formal_bundle(
        extension_profile,
        continuation.full_run,
        output_root=args.output_root,
        output_run_id=args.output_run_id,
        continuation_of_run_id=history.formal_run_id,
    )
    print(_result_text(
        extension_profile.sample_id,
        args.output_run_id,
        bundle.output_dir,
        continuation.full_run.ledger.is_complete,
    ))
    print(f"已验证历史前缀未被改写，本次新增 {continuation.appended_transaction_count} 笔交易。")
    return 0


def _assert_entry_profile(profile) -> None:
    if not ENTRY_ISOLATION_ENFORCED:
        return
    expected = f"syn_b1_formal_profile_{DEFAULT_PROFILE_SCHEMA_VERSION.replace('.', '_')}"
    actual = profile.resolved_profile.get("schema_version")
    if actual != expected:
        raise FormalProfileError(
            f"当前是生成器{APP_DISPLAY_VERSION}入口，只能使用{DEFAULT_PROFILE_SCHEMA_VERSION}画像；"
            f"当前文件为{actual or '未知版本'}。请回到对应版本的BAT选择该画像，不要跨版本生成。"
        )


def _assert_entry_output_root(value: str | Path) -> None:
    if not ENTRY_ISOLATION_ENFORCED:
        return
    if Path(value).resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise FormalProfileError(
            f"生成器{APP_DISPLAY_VERSION}的正式输出根目录已固定为{DEFAULT_OUTPUT_ROOT}；"
            "不能通过--output-root改写到其他版本或任意目录。"
        )


def _assert_entry_path(value: str | Path, root: Path, label: str) -> None:
    if not ENTRY_ISOLATION_ENFORCED:
        return
    candidate, allowed_root = Path(value).resolve(), root.resolve()
    try:
        candidate.relative_to(allowed_root)
    except ValueError as error:
        raise FormalProfileError(
            f"{label}必须位于当前生成器{APP_DISPLAY_VERSION}的隔离目录{root}内；请使用BAT菜单选择，不要手填跨版本路径。"
        ) from error


def _interactive() -> int:
    workspace = GeneratorWorkspace(DEFAULT_WORKSPACE_ROOT, DEFAULT_OUTPUT_ROOT, DEFAULT_PROFILE_SCHEMA_VERSION)
    while True:
        print(f"\nSYN-B1 合成流水生成器 {APP_DISPLAY_VERSION}")
        print(f"参数工作区：{workspace.root}")
        print("1. 新建企业方案  2. 管理参数方案  3. 管理生成结果  0. 退出")
        choice = input("请选择：").strip()
        if choice == "0":
            return 0
        try:
            if choice == "1":
                _new_profile_interactively(workspace)
            elif choice == "2":
                _manage_profiles_interactively(workspace)
            elif choice == "3":
                _manage_runs_interactively(workspace)
            else:
                print("未选择可用操作；没有写入任何文件。")
        except (FormalProfileError, ValueError) as error:
            print(f"未执行写入：{error}")
        input("按回车键返回主菜单：")


def _new_profile_interactively(workspace: GeneratorWorkspace) -> None:
    enterprise_id = input("虚构企业编号（直接回车自动生成）：").strip() or _automatic_enterprise_id()
    profile_version = input("参数版本（直接回车自动生成）：").strip() or _automatic_profile_version()
    path = workspace.create_profile(enterprise_id, profile_version)
    print(f"已新建参数方案：{path}")
    print(f"已采用企业编号：{enterprise_id}；参数版本：{profile_version}。")
    print("现在请用记事本直接填写或调整参数；保存后的YAML就是程序下次核验、预演和生成时读取的实际参数。")
    _offer_open(path)


def _manage_profiles_interactively(workspace: GeneratorWorkspace) -> None:
    path = _choose_profile(workspace, "请选择参数方案")
    if path is None:
        return
    print("1. 打开查看或修改  2. 复制为新版本  3. 核验  4. 预演  5. 正式生成  0. 返回")
    choice = input("请选择：").strip()
    if choice == "0":
        return
    if choice == "1":
        if path.suffix.lower() == ".json":
            print("这是历史JSON方案，不带中文说明。请选择第2项复制为新版本，程序会生成可直接编辑的中文YAML。")
            return
        print("为保留历史，请不要直接修改已用于正式生成的旧方案；建议选择第2项复制为新版本。")
        _open_editor(path)
        return
    if choice == "2":
        enterprise_id = input("新版本所属企业编号（直接回车沿用）：").strip() or path.parent.name
        profile_version = input("新参数版本（直接回车自动生成）：").strip() or _automatic_profile_version()
        target = workspace.copy_profile_as_new_version(path, enterprise_id, profile_version)
        print(f"已创建新版本：{target}")
        _offer_open(target)
        return
    if choice == "3":
        main(("validate", "--profile", str(path)))
        return
    if choice == "4":
        main(("preview", "--profile", str(path)))
        return
    if choice == "5":
        output_run_id = input("本次生成运行编号（直接回车自动生成）：").strip() or _automatic_run_id()
        main((
            "generate", "--profile", str(path),
            "--output-root", str(workspace.generated_root),
            "--output-run-id", output_run_id,
        ))
        return
    print("未选择可用操作；没有写入任何文件。")


def _manage_runs_interactively(workspace: GeneratorWorkspace) -> None:
    directory = _choose_generated_run(workspace, "请选择生成结果")
    if directory is None:
        return
    print("1. 查看结果  2. 续生成  0. 返回")
    choice = input("请选择：").strip()
    if choice == "0":
        return
    if choice == "1":
        _show(str(directory))
        return
    if choice == "2":
        extension_profile = _choose_profile(workspace, "请选择延长日期后的参数方案")
        if extension_profile is None:
            return
        output_run_id = input("新的续生成运行编号（直接回车自动生成）：").strip() or _automatic_run_id()
        main((
            "continue", "--base-run-directory", str(directory),
            "--extension-profile", str(extension_profile),
            "--output-run-id", output_run_id,
            "--output-root", str(workspace.generated_root),
        ))
        return
    print("未选择可用操作；没有写入任何文件。")


def _automatic_enterprise_id() -> str:
    return f"factory_{_automatic_suffix()}"


def _automatic_profile_version() -> str:
    return f"v1_{_automatic_suffix()}"


def _automatic_run_id() -> str:
    return f"run_{_automatic_suffix()}"


def _automatic_suffix() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(2)}"


def _print_profiles(workspace: GeneratorWorkspace) -> tuple[Path, ...]:
    profiles = workspace.list_profiles()
    if not profiles:
        print("还没有参数方案。请先选择“1. 新建企业参数方案”。")
        return ()
    print("现有参数方案：")
    for index, path in enumerate(profiles, start=1):
        legacy = "（历史JSON；复制新版本后可获得中文YAML说明）" if path.suffix.lower() == ".json" else ""
        print(f"{index}. {path.relative_to(workspace.profiles_root)}{legacy}")
    return profiles


def _choose_profile(workspace: GeneratorWorkspace, prompt: str) -> Path | None:
    profiles = _print_profiles(workspace)
    return _choose_from(profiles, prompt) if profiles else None


def _choose_generated_run(workspace: GeneratorWorkspace, prompt: str) -> Path | None:
    runs = workspace.list_generated_runs()
    if not runs:
        print("还没有正式生成结果。请先进入“2. 管理参数方案”，再选择“正式生成”。")
        return None
    print("现有生成结果：")
    for index, path in enumerate(runs, start=1):
        print(f"{index}. {path.relative_to(workspace.generated_root)}")
    return _choose_from(runs, prompt)


def _choose_from(items: tuple[Path, ...], prompt: str) -> Path | None:
    raw = input(f"{prompt}（输入序号，0取消）：").strip()
    if raw == "0":
        return None
    try:
        index = int(raw)
    except ValueError as error:
        raise FormalProfileError("请输入列表中的序号。") from error
    if not 1 <= index <= len(items):
        raise FormalProfileError("选择的序号不在列表中。")
    return items[index - 1]


def _offer_open(path: Path) -> None:
    if input("现在用记事本打开参数方案吗？（Y/N）：").strip().lower() in {"y", "yes"}:
        _open_editor(path)


def _open_editor(path: Path) -> None:
    subprocess.Popen(["notepad.exe", str(path)])
    print("已打开记事本。这是普通YAML文本文件，可直接修改、保存；若记事本无法打开，可用VS Code等文本编辑器。保存后回到本菜单选择“核验”。")


def _preview_text(sample_id: str, run_id: str, run) -> str:
    status = "可完整生成" if run.ledger.is_complete else "会在资金不足处停止"
    lines = [
        f"预演完成：虚构企业 {sample_id}，参数版本运行标识 {run_id}。",
        f"预计逐笔数：{len(run.ledger.transactions)}；预计日度数：{len(run.ledger.daily_rows)}；结果：{status}。",
        "预演只在内存计算，不写CSV、不写数据库、不训练模型。",
    ]
    if run.request.runtime_settings is not None and run.request.runtime_settings.regime_plan is not None:
        monthly_ending_balances: dict[str, int] = {}
        for daily_row in run.ledger.daily_rows:
            monthly_ending_balances[daily_row.calendar_date.strftime("%Y-%m")] = daily_row.ending_balance_fen
        lines.append("逐月经营桥（阶段和采购预期仅供本地人工预演，禁止入模）：")
        lines.append("月份 | 阶段 | 潜在销售 | 确认销售 | 毛利率 | 销售成本 | 采购预期 | 采购 | 存货 | 回款 | 供应商付款 | 固定费用 | 月末账户余额")
        for row in run.operating_cycle.monthly_rows:
            fixed = row.payroll_expense_fen + row.rent_and_utilities_expense_fen + row.other_operating_expense_fen
            values = (
                row.month, row.regime_segment_id or "-", _cny_text(row.latent_sales_fen),
                _cny_text(row.sales_confirmed_fen), f"{row.gross_margin_rate_bp / 100:.2f}%",
                _cny_text(row.cost_of_goods_sold_fen), _cny_text(row.procurement_expected_cogs_fen),
                _cny_text(row.material_purchases_fen), _cny_text(row.ending_inventory_fen),
                _cny_text(row.cash_collections_fen), _cny_text(row.supplier_cash_payments_fen), _cny_text(fixed),
                _cny_text(monthly_ending_balances[row.month]),
            )
            lines.append(" | ".join(values))
    return "\n".join(lines)


def _cny_text(value_fen: int) -> str:
    return f"{value_fen / 100:.2f}"


def _result_text(sample_id: str, run_id: str, directory: Path, complete: bool) -> str:
    if complete:
        ending = "已完成：可查看逐笔CSV、日度CSV、参数、质量报告和受限审计文件。"
    else:
        ending = "已按资金不足规则停止：公共目录保留可支付前缀和事实单，不能用于训练。"
    return f"已生成虚构企业 {sample_id} 的运行 {run_id}。\n目录：{directory}\n{ending}\n未写入SQLite，未启动训练或预测。"


def _show(directory_value: str) -> int:
    directory = Path(directory_value)
    manifest = directory / "generation_manifest.json"
    quality = directory / "quality_report.json"
    if not manifest.is_file() or not quality.is_file():
        raise ValueError("该目录不是完整的正式生成结果：缺少运行清单或质量报告。")
    import json
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    quality_payload = json.loads(quality.read_text(encoding="utf-8"))
    print(f"虚构企业：{manifest_payload['sample_id']}；运行：{manifest_payload['run_id']}。")
    print(f"期间：{manifest_payload['period']['start_date']} 至 {manifest_payload['period']['written_through_date']}。")
    print(f"状态：{manifest_payload['execution_status']}；可训练：{quality_payload['training_eligible']}。")
    print(f"逐笔文件：{directory / 'transactions_total.csv'}")
    print(f"日度文件：{directory / 'account_daily_total.csv'}")
    print(f"资金备注表（仅人工查看）：{directory / 'cash_flow_review_notes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
