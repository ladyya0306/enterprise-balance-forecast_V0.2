"""第四轮两项隔离的循环网络试验。

只在 525 户学习题及其较晚预测日检查题上选版本。113 户和 142 户的
答案只会在版本固定以后读取一次，用来给选用版本定范围和作开发汇报。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from datetime import date
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import v3_model_comparison as base


ROOT = Path(__file__).resolve().parents[1]
OLD_RUN = ROOT / "runs" / "第三轮_开发比较_20260926_完整修正"
SEEDS = base.SEEDS
KIND_NAMES = ("LSTM", "GRU")


def transaction_with_last_gap(question: base.Question) -> np.ndarray:
    """在每个有效逐笔行重复末笔到预测日的自然日数。

    无交易题没有可见末笔；此时最后一列是 28 个银行日覆盖窗口的自然日长度，
    它只是“本窗口无交易的可见天数下界”，不是不存在交易前的真实末笔距离。
    """
    values = base.transaction_array(question)
    start = date.fromisoformat(question.basic[0]["calendar_date"])
    origin = date.fromisoformat(question.origin)
    if question.transactions:
        last = date.fromisoformat(question.transactions[-1]["booking_date"])
        gap = float((origin - last).days)
    else:
        gap = float((origin - start).days)
    return np.column_stack((values, np.full((len(values), 1), gap, dtype=np.float64)))


def feature(question: base.Question, variant: str) -> np.ndarray:
    if variant == "日表":
        return base.daily_array(question, False)
    if variant == "逐笔12字段":
        return base.transaction_array(question)
    if variant == "逐笔13字段_末笔距预测日天数":
        return transaction_with_last_gap(question)
    raise ValueError(variant)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)


def initial_state(kind: str, seed: int, fields: int) -> dict[str, torch.Tensor]:
    set_seed(seed)
    return {name: value.detach().cpu().clone() for name, value in base.SequenceRegressor(fields, kind).state_dict().items()}


def expand_initial_state(kind: str, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """给 B 返回相同未训练 12 字段状态和新增列为零的 13 字段状态。"""
    old = initial_state(kind, seed, 12)
    set_seed(seed)
    new_model = base.SequenceRegressor(13, kind)
    new = {name: value.detach().cpu().clone() for name, value in new_model.state_dict().items()}
    for name, value in old.items():
        if name == "layer.weight_ih_l0":
            new[name].zero_()
            new[name][:, :12] = value
        else:
            new[name] = value.clone()
    return old, new


def prepare(questions: list[base.Question], variant: str, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = [feature(question, variant) for question in questions]
    padded, lengths = base.pad_sequences(raw)
    if mean is None:
        flat = np.concatenate(raw, axis=0)
        mean, std = flat.mean(axis=0), flat.std(axis=0)
        std = np.where(std == 0.0, 1.0, std)
    assert std is not None
    return ((padded - mean) / std).astype(np.float32), lengths, mean, std


def train(kind: str, learning: list[base.Question], early: list[base.Question], variant: str, seed: int, maximum_epochs: int, state: dict[str, torch.Tensor], output: Path, learning_rate: float = 0.001) -> tuple[base.SequenceRegressor, dict]:
    set_seed(seed)
    padded, lengths, mean, std = prepare(learning, variant)
    y_raw = base.normalized_targets(learning)
    y_mean, y_std = y_raw.mean(axis=0), y_raw.std(axis=0)
    y_std = np.where(y_std == 0.0, 1.0, y_std)
    y = ((y_raw - y_mean) / y_std).astype(np.float32)
    early_padded, early_lengths, _, _ = prepare(early, variant, mean, std)
    early_y = ((base.normalized_targets(early) - y_mean) / y_std).astype(np.float32)
    model = base.SequenceRegressor(padded.shape[2], kind)
    model.load_state_dict(copy.deepcopy(state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    x, lengths_t, target = torch.from_numpy(padded), torch.from_numpy(lengths), torch.from_numpy(y)
    early_x, early_lengths_t, early_target = torch.from_numpy(early_padded), torch.from_numpy(early_lengths), torch.from_numpy(early_y)
    losses: list[dict[str, float]] = []
    saved_error = float("inf"); saved_epoch = 0; saved_state = None; stale = 0
    absolute_error = float("inf"); absolute_epoch = 0
    for _ in range(maximum_epochs):
        model.train()
        optimizer.zero_grad(); predicted = model(x, lengths_t); loss = torch.abs(predicted - target).mean(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad():
            early_loss = float(torch.abs(model(early_x, early_lengths_t) - early_target).mean())
        losses.append({"学习误差": float(loss.detach()), "日期检查误差": early_loss})
        if early_loss < absolute_error:
            absolute_error, absolute_epoch = early_loss, len(losses)
        # 保留第三轮的阈值语义，确保此轮与旧 30 次控制能比较。
        if early_loss < saved_error - 0.0001:
            saved_error, saved_epoch, stale = early_loss, len(losses), 0
            saved_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 5:
            break
    if saved_state is None:
        raise ValueError("没有按旧阈值保存的检查点")
    model.load_state_dict(saved_state)
    package = {"kind": kind, "fields": int(padded.shape[2]), "feature_variant": variant, "learning_rate": learning_rate, "mean": mean.tolist(), "std": std.tolist(), "target_mean": y_mean.tolist(), "target_std": y_std.tolist(), "state_dict": model.state_dict(), "seed": seed, "maximum_epochs": maximum_epochs, "actual_epochs": len(losses), "best_epoch_by_old_threshold": saved_epoch, "best_date_check_error_by_old_threshold": saved_error, "absolute_lowest_date_check_error": absolute_error, "absolute_lowest_epoch": absolute_epoch, "losses": losses}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, output)
    return model, package


def predict(model: base.SequenceRegressor, package: dict, questions: list[base.Question]) -> np.ndarray:
    padded, lengths, _, _ = prepare(questions, package["feature_variant"], np.asarray(package["mean"]), np.asarray(package["std"]))
    model.eval()
    with torch.no_grad():
        normalized = model(torch.from_numpy(padded), torch.from_numpy(lengths)).numpy()
    normalized = normalized * np.asarray(package["target_std"]) + np.asarray(package["target_mean"])
    return np.asarray([q.base for q in questions])[:, None] + normalized * np.asarray([q.scale for q in questions])[:, None]


def ensemble_internal_error(models: list[base.SequenceRegressor], packages: list[dict], questions: list[base.Question]) -> float:
    """三起点逐题取中位数后，按共同目标标准化计算平均绝对差。"""
    prediction = np.median(np.stack([predict(model, package, questions) for model, package in zip(models, packages)]), axis=0)
    normalized = (prediction - np.asarray([q.base for q in questions])[:, None]) / np.asarray([q.scale for q in questions])[:, None]
    target_mean, target_std = np.asarray(packages[0]["target_mean"]), np.asarray(packages[0]["target_std"])
    return float(np.abs((normalized - target_mean) / target_std - (base.normalized_targets(questions) - target_mean) / target_std).mean())


def historic(kind: str, material: str, seed: int, variant: str) -> tuple[base.SequenceRegressor, dict]:
    path = OLD_RUN / "模型文件" / material / kind / f"seed_{seed}.pt"
    package = torch.load(path, map_location="cpu", weights_only=False)
    package = dict(package); package["feature_variant"] = variant
    model = base.SequenceRegressor(int(package["fields"]), kind); model.load_state_dict(package["state_dict"])
    package["best_date_check_error_by_old_threshold"] = package["best_date_check_error"]
    package["best_epoch_by_old_threshold"] = package["best_epoch"]
    package["absolute_lowest_date_check_error"] = min(item["日期检查误差"] for item in package["losses"])
    package["absolute_lowest_epoch"] = int(np.argmin([item["日期检查误差"] for item in package["losses"]])) + 1
    return model, package


def score_selected(label: str, kind: str, variant: str, models: list[base.SequenceRegressor], packages: list[dict], calibration: list[base.Question], development: list[base.Question], output: Path) -> list[dict[str, object]]:
    p_cal = np.median(np.stack([predict(m, p, calibration) for m, p in zip(models, packages)]), axis=0)
    radii, _ = base.radius(p_cal, calibration)
    p_dev = np.median(np.stack([predict(m, p, development) for m, p in zip(models, packages)]), axis=0)
    lower, upper = base.make_intervals(p_dev, development, radii)
    rows = base.scored_rows(label, variant, development, p_dev, lower, upper)
    base.write_csv(output / f"{label}_{kind}_开发逐题预测与评分.csv", rows)
    return rows


def curve_plot(results: dict[str, list[dict]], path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for axis, (key, packages) in zip(axes.flat, results.items()):
        for package in packages:
            values = [row["日期检查误差"] for row in package["losses"]]
            axis.plot(range(1, len(values) + 1), values, label=f"seed {package['seed']}")
        axis.set_title(key); axis.set_xlabel("整批更新次数"); axis.set_ylabel("较晚日期检查标准化绝对误差"); axis.legend(fontsize=7)
    for axis in axes.flat[len(results):]:
        axis.remove()
    fig.savefig(path, dpi=160); plt.close(fig)


def package_from_path(kind: str, path: Path, variant: str) -> tuple[base.SequenceRegressor, dict]:
    package = dict(torch.load(path, map_location="cpu", weights_only=False))
    package["feature_variant"] = variant
    model = base.SequenceRegressor(int(package["fields"]), kind)
    model.load_state_dict(package["state_dict"])
    return model, package


def selected_paths(kind: str, trial: str, choice: str, output: Path) -> tuple[str, list[str]]:
    if trial == "A" and choice == "原版本_30次已保存权重":
        return "日表", [str(OLD_RUN / "模型文件" / "每天余额收入支出" / kind / f"seed_{seed}.pt") for seed in SEEDS]
    if trial == "A":
        return "日表", [str(output / "模型文件" / "A_270次整批更新" / kind / f"seed_{seed}.pt") for seed in SEEDS]
    folder = "B_13字段末笔距预测日" if choice == "13字段末笔距预测日" else "B_12字段配对控制"
    variant = "逐笔13字段_末笔距预测日天数" if choice == "13字段末笔距预测日" else "逐笔12字段"
    return variant, [str(output / "模型文件" / folder / kind / f"seed_{seed}.pt") for seed in SEEDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "第三轮_两项改进试验_20260926")
    parser.add_argument("--stage", choices=("train-select", "recompute-ensemble", "learning-rate-search", "score-frozen"), required=True)
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if args.stage == "score-frozen":
        frozen = json.loads((output / "最终冻结选择.json").read_text(encoding="utf-8"))
        # 最终冻结后的独立评分；先前初始四候选已有一次评分留痕，二者都不参与后续选参。
        calibration = base.build_questions("区间校准", 0.60, None); development = base.build_questions("开发测试", 0.70, None)
        rows: list[dict[str, object]] = []
        for entry in frozen["固定选用版本"]:
            models, packages = zip(*[package_from_path(entry["模型"], Path(path), entry["输入版本"]) for path in entry["权重文件"]])
            label = f"{entry['试验']}_{entry['模型']}_选用版本"
            rows += score_selected(label, entry["模型"], entry["资料"], list(models), list(packages), calibration, development, output)
        base.write_csv(output / "选用版本开发测试逐题预测与评分.csv", rows)
        base.write_csv(output / "选用版本开发测试成绩汇总.csv", base.summarize(rows))
        (output / "最终冻结后评分记录.json").write_text(json.dumps({"冻结文件": str(output / "最终冻结选择.json"), "范围确定企业数": len(calibration), "开发测试企业数": len(development), "开发逐题行数": len(rows)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": args.stage, "rows": len(rows)}, ensure_ascii=False)); return
    if args.stage == "learning-rate-search":
        learning = base.build_questions("学习", 0.45, None); early = base.build_questions("学习", 0.70, None)
        initial = json.loads((output / "冻结选择.json").read_text(encoding="utf-8"))
        records = json.loads((output / "运行记录.json").read_text(encoding="utf-8"))
        lr_records: dict[str, object] = {}
        final_entries = []
        for item in initial["固定选用版本"]:
            if item["试验"] != "B":
                continue
            item = dict(item)
            folder = "B_13字段末笔距预测日" if item["选用版本"] == "13字段末笔距预测日" else "B_12字段配对控制"
            item["权重文件"] = [str(output / "模型文件" / folder / item["模型"] / f"seed_{seed}.pt") for seed in SEEDS]
            final_entries.append(item)
        for kind in KIND_NAMES:
            candidates = []
            # 已有 0.001 的初始 A 候选，与两个新增学习率共同按同一内部规则比较。
            old_a = records["runs"][f"A_{kind}"]["试验版本_270次上限"]
            candidates.append({"learning_rate": 0.001, "three_seed_mean": old_a["three_seed_mean"], "source": "初始A_270次"})
            for lr in (0.0005, 0.002):
                models, packages = [], []
                for seed in SEEDS:
                    model, package = train(kind, learning, early, "日表", seed, 270, initial_state(kind, seed, 4), output / "模型文件" / f"A_学习率_{lr}" / kind / f"seed_{seed}.pt", lr)
                    models.append(model); packages.append(package)
                candidates.append({"learning_rate": lr, "three_seed_mean": float(np.mean([p["best_date_check_error_by_old_threshold"] for p in packages])), "three_seed_ensemble_internal_error": ensemble_internal_error(models, packages, early), "per_seed": [{k: p[k] for k in ("seed", "best_date_check_error_by_old_threshold", "best_epoch_by_old_threshold", "actual_epochs")} for p in packages], "source": f"A_学习率_{lr}"})
            chosen = min(candidates, key=lambda item: item["three_seed_mean"])
            lr_records[kind] = {"全部候选": candidates, "选用学习率": chosen["learning_rate"], "主选择依据": "三个随机起点旧阈值保存的较晚日期检查误差平均值最小"}
            if chosen["learning_rate"] == 0.001:
                entry = next(item for item in initial["固定选用版本"] if item["试验"] == "A" and item["模型"] == kind)
                entry = dict(entry)
                entry["权重文件"] = [str(output / "模型文件" / "A_270次整批更新" / kind / f"seed_{seed}.pt") for seed in SEEDS]
            else:
                entry = {"试验": "A", "模型": kind, "资料": "每天余额收入支出", "输入版本": "日表", "选用版本": f"学习率_{chosen['learning_rate']}", "权重文件": [str(output / "模型文件" / f"A_学习率_{chosen['learning_rate']}" / kind / f"seed_{seed}.pt") for seed in SEEDS]}
            final_entries.append(entry)
        final_entries.sort(key=lambda item: (item["试验"], item["模型"]))
        (output / "学习率增补运行记录.json").write_text(json.dumps(lr_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output / "最终冻结选择.json").write_text(json.dumps({"冻结时刻": "初始A/B与学习率增补候选均完成；评分前冻结", "固定选用版本": final_entries, "学习率增补全部候选记录": "学习率增补运行记录.json"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": args.stage}, ensure_ascii=False)); return
    if args.stage == "recompute-ensemble":
        # 只重新计算已经训练权重在 525 户较晚日期检查题上的三模型中位数误差。
        early = base.build_questions("学习", 0.70, None)
        records_path = output / "运行记录.json"; records = json.loads(records_path.read_text(encoding="utf-8"))
        for kind in KIND_NAMES:
            old_models, old_packages = zip(*[historic(kind, "每天余额收入支出", seed, "日表") for seed in SEEDS])
            new_models, new_packages = zip(*[package_from_path(kind, output / "模型文件" / "A_270次整批更新" / kind / f"seed_{seed}.pt", "日表") for seed in SEEDS])
            records["runs"][f"A_{kind}"]["原版本_30次已保存权重"]["three_seed_ensemble_internal_error"] = ensemble_internal_error(list(old_models), list(old_packages), early)
            records["runs"][f"A_{kind}"]["试验版本_270次上限"]["three_seed_ensemble_internal_error"] = ensemble_internal_error(list(new_models), list(new_packages), early)
            control_models, control_packages = zip(*[package_from_path(kind, output / "模型文件" / "B_12字段配对控制" / kind / f"seed_{seed}.pt", "逐笔12字段") for seed in SEEDS])
            field_models, field_packages = zip(*[package_from_path(kind, output / "模型文件" / "B_13字段末笔距预测日" / kind / f"seed_{seed}.pt", "逐笔13字段_末笔距预测日天数") for seed in SEEDS])
            records["runs"][f"B_{kind}"]["12字段配对控制"]["three_seed_ensemble_internal_error"] = ensemble_internal_error(list(control_models), list(control_packages), early)
            records["runs"][f"B_{kind}"]["13字段末笔距预测日"]["three_seed_ensemble_internal_error"] = ensemble_internal_error(list(field_models), list(field_packages), early)
        records["三起点中位数内部误差重算说明"] = "2026-09-26：修正了预测值未按目标均值和离散程度标准化的记录错误；未重训、未改变任何候选选择。"
        records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": args.stage}, ensure_ascii=False)); return
    learning = base.build_questions("学习", 0.45, None); early = base.build_questions("学习", 0.70, None)
    if {q.sample_id for q in learning} != {q.sample_id for q in early} or any(date.fromisoformat(next(item.origin for item in early if item.sample_id == q.sample_id)) <= date.fromisoformat(q.answer_end) for q in learning):
        raise ValueError("学习题和较晚日期检查题的企业或日期边界不符合第三轮规则")
    started = perf_counter(); records: dict[str, object] = {"purpose": "第四轮两项初始因果比较。本阶段未构造、未读取113户或142户题目。", "learning_count": len(learning), "early_count": len(early), "seeds": list(SEEDS), "runs": {}}
    frozen: list[dict[str, object]] = []
    curve_data: dict[str, list[dict]] = {}
    # A：只改变更新上限。旧版本直接读取已保存 30 次权重；新版本从相同种子重新开始训练。
    for kind in KIND_NAMES:
        old_models, old_packages, new_models, new_packages = [], [], [], []
        for seed in SEEDS:
            old_model, old_package = historic(kind, "每天余额收入支出", seed, "日表"); old_models.append(old_model); old_packages.append(old_package)
            state = initial_state(kind, seed, 4)
            model, package = train(kind, learning, early, "日表", seed, 270, state, output / "模型文件" / "A_270次整批更新" / kind / f"seed_{seed}.pt")
            new_models.append(model); new_packages.append(package)
        old_avg = float(np.mean([p["best_date_check_error_by_old_threshold"] for p in old_packages])); new_avg = float(np.mean([p["best_date_check_error_by_old_threshold"] for p in new_packages]))
        first30_differences = []
        for new_p, old_p in zip(new_packages, old_packages):
            new_curve = np.asarray([x["日期检查误差"] for x in new_p["losses"][:30]])
            old_curve = np.asarray([x["日期检查误差"] for x in old_p["losses"]])
            first30_differences.append(float(np.max(np.abs(new_curve - old_curve))))
        if max(first30_differences) > 1e-7:
            raise ValueError(f"{kind} 的270次候选前30次未复现旧日期检查曲线：{max(first30_differences)}")
        choose_new = new_avg < old_avg
        choice = "试验版本_270次上限" if choose_new else "原版本_30次已保存权重"
        eligible_540 = [p["actual_epochs"] == 270 and p["losses"][-1]["日期检查误差"] < p["losses"][265]["日期检查误差"] - 0.0001 for p in new_packages]
        records["runs"][f"A_{kind}"] = {"270次候选前30次对旧曲线最大绝对差": max(first30_differences), "原版本_30次已保存权重": {"three_seed_mean": old_avg, "three_seed_median": float(np.median([p["best_date_check_error_by_old_threshold"] for p in old_packages])), "three_seed_ensemble_internal_error": ensemble_internal_error(old_models, old_packages, early), "per_seed": [{k: p[k] for k in ("seed", "best_date_check_error_by_old_threshold", "best_epoch_by_old_threshold", "actual_epochs")} for p in old_packages]}, "试验版本_270次上限": {"three_seed_mean": new_avg, "three_seed_median": float(np.median([p["best_date_check_error_by_old_threshold"] for p in new_packages])), "three_seed_ensemble_internal_error": ensemble_internal_error(new_models, new_packages, early), "per_seed": [{k: p[k] for k in ("seed", "best_date_check_error_by_old_threshold", "best_epoch_by_old_threshold", "absolute_lowest_date_check_error", "absolute_lowest_epoch", "actual_epochs")} for p in new_packages]}, "270到540候选资格": {"逐seed符合": eligible_540, "至少两seed符合": sum(eligible_540) >= 2, "本初始阶段尚未执行540": True}, "选用版本": choice}
        variant, paths = selected_paths(kind, "A", choice, output); frozen.append({"试验": "A", "模型": kind, "资料": "每天余额收入支出", "输入版本": variant, "选用版本": choice, "权重文件": paths})
        curve_data[f"A {kind}：270次上限"] = new_packages
    # B：先重新跑配对的 12 字段控制，再跑新字段；训练开始前核验预测相等。
    for kind in KIND_NAMES:
        control_models, control_packages, field_models, field_packages = [], [], [], []
        initial_differences = []
        for seed in SEEDS:
            state12, state13 = expand_initial_state(kind, seed)
            model12 = base.SequenceRegressor(12, kind); model12.load_state_dict(state12)
            model13 = base.SequenceRegressor(13, kind); model13.load_state_dict(state13)
            raw12, length12, _, _ = prepare(learning[:8], "逐笔12字段")
            raw13, length13, _, _ = prepare(learning[:8], "逐笔13字段_末笔距预测日天数")
            with torch.no_grad(): initial_differences.append(float(np.max(np.abs(model12(torch.from_numpy(raw12), torch.from_numpy(length12)).numpy() - model13(torch.from_numpy(raw13), torch.from_numpy(length13)).numpy()))))
            if initial_differences[-1] > 1e-7:
                raise ValueError(f"{kind} seed {seed} 的配对初始化预测不相同：{initial_differences[-1]}")
            c_model, c_package = train(kind, learning, early, "逐笔12字段", seed, 30, state12, output / "模型文件" / "B_12字段配对控制" / kind / f"seed_{seed}.pt")
            f_model, f_package = train(kind, learning, early, "逐笔13字段_末笔距预测日天数", seed, 30, state13, output / "模型文件" / "B_13字段末笔距预测日" / kind / f"seed_{seed}.pt")
            control_models.append(c_model); control_packages.append(c_package); field_models.append(f_model); field_packages.append(f_package)
        control_avg = float(np.mean([p["best_date_check_error_by_old_threshold"] for p in control_packages])); field_avg = float(np.mean([p["best_date_check_error_by_old_threshold"] for p in field_packages]))
        choose_field = field_avg < control_avg
        old_packages = [historic(kind, "逐笔金额余额用途", seed, "逐笔12字段")[1] for seed in SEEDS]
        reproduction_differences = []
        for control_package, old_package in zip(control_packages, old_packages):
            control_curve = np.asarray([x["日期检查误差"] for x in control_package["losses"]])
            old_curve = np.asarray([x["日期检查误差"] for x in old_package["losses"]])
            reproduction_differences.append(float(np.max(np.abs(control_curve - old_curve))))
        if max(reproduction_differences) > 1e-7:
            raise ValueError(f"{kind} 的12字段30次控制未复现旧日期检查曲线：{max(reproduction_differences)}")
        choice = "13字段末笔距预测日" if choose_field else "12字段配对控制"
        records["runs"][f"B_{kind}"] = {"配对初始化预测最大绝对差": max(initial_differences), "12字段控制对旧曲线最大绝对差": max(reproduction_differences), "12字段配对控制": {"three_seed_mean": control_avg, "three_seed_median": float(np.median([p["best_date_check_error_by_old_threshold"] for p in control_packages])), "three_seed_ensemble_internal_error": ensemble_internal_error(control_models, control_packages, early), "per_seed": [{k: p[k] for k in ("seed", "best_date_check_error_by_old_threshold", "best_epoch_by_old_threshold", "actual_epochs")} for p in control_packages]}, "13字段末笔距预测日": {"three_seed_mean": field_avg, "three_seed_median": float(np.median([p["best_date_check_error_by_old_threshold"] for p in field_packages])), "three_seed_ensemble_internal_error": ensemble_internal_error(field_models, field_packages, early), "per_seed": [{k: p[k] for k in ("seed", "best_date_check_error_by_old_threshold", "best_epoch_by_old_threshold", "absolute_lowest_date_check_error", "absolute_lowest_epoch", "actual_epochs")} for p in field_packages]}, "选用版本": choice}
        variant, paths = selected_paths(kind, "B", choice, output); frozen.append({"试验": "B", "模型": kind, "资料": "逐笔金额余额用途", "输入版本": variant, "选用版本": choice, "权重文件": paths})
        curve_data[f"B {kind}：12字段控制"] = control_packages; curve_data[f"B {kind}：13字段"] = field_packages
    records["运行秒数"] = perf_counter() - started
    (output / "运行记录.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    curve_plot(curve_data, output / "较晚日期检查误差曲线.png")
    (output / "冻结选择.json").write_text(json.dumps({"冻结时刻": "初始A/B四项内部选择完成；尚未构造113户或142户题目", "固定选用版本": frozen}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "seconds": records["运行秒数"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
