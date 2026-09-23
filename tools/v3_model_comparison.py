"""第三轮三种资料、八种方法、三项余额预测的运行入口。

本文件只从已经整理好的 CSV 读取资料。每一题只读取预测日及以前的
28 个银行营业日所覆盖的日期；答案从随后 30 个银行营业日的日末余额计算。
它不调用会依据交易有无改变日历的旧运行时程序。

默认 ``--mode smoke`` 只用很少企业检查八种方法、三项预测、上下范围和
成绩文件能完整衔接。正式运行必须明确写 ``--mode development``；本入口
不会因为导入或普通检查而开始 525/113/142 户的正式比较。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Iterable

import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "v3_three_materials_eight_methods_20260923_1"
ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "data" / "training_materials" / "v1"
HORIZON_NAMES = ("下一银行营业日的日末余额", "以后10个银行营业日的日均余额", "以后30个银行营业日的日均余额")
USAGE_CODES = ("CUSTOMER_RECEIPT", "PROCUREMENT", "TAX", "OPERATING_SERVICE", "PREMISES_RENT", "PAYROLL", "UNKNOWN", "FINANCING", "EQUIPMENT")
SEEDS = (2026092301, 2026092302, 2026092303)
LOOKBACK = 28
MAX_TRANSACTION_STEPS = 256


@dataclass
class Question:
    sample_id: str
    family_id: str
    shape: str
    origin: str
    basic: list[dict[str, str]]
    usage: list[dict[str, str]]
    transactions: list[dict[str, str]]
    future: np.ndarray

    @property
    def base(self) -> float:
        return float(self.basic[-1]["ending_balance_cny"])

    @property
    def scale(self) -> float:
        return max(abs(self.base), 1_000_000.0)

    @property
    def target(self) -> np.ndarray:
        return np.asarray((self.future[0], self.future[:10].mean(), self.future.mean()), dtype=np.float64)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def by_sample(path: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        result[row["sample_id"]].append(row)
    return result


def sample_shapes() -> dict[str, str]:
    path = ROOT / "data" / "manifests" / "实测走势与曲线重复.json"
    return {row["sample_id"]: row["measured_shape_v3"] for row in json.loads(path.read_text(encoding="utf-8"))}


def choose_cutoff(rows: list[dict[str, str]], fraction: float) -> str | None:
    """仅依据已保存的银行日标记选预测日；不读取交易表决定日历。"""
    banks = [row for row in rows if row["is_bank_workday"] == "True"]
    # 28 个过去银行日加 30 个答案银行日都必须存在。
    lower, upper = LOOKBACK - 1, len(banks) - 31
    if upper < lower:
        return None
    index = lower + round((upper - lower) * fraction)
    return banks[index]["calendar_date"]


def build_questions(role: str, fraction: float, limit: int | None) -> list[Question]:
    basic = by_sample(MATERIALS / "每天余额收入支出" / f"{role}.csv")
    usage = by_sample(MATERIALS / "每天余额收入支出及用途" / f"{role}.csv")
    transactions = by_sample(MATERIALS / "逐笔金额余额用途" / f"{role}.csv")
    families = {row["sample_id"]: row["family_id"] for row in read_csv(ROOT / "data" / "manifests" / "样本分组名单.csv")}
    shapes = sample_shapes()
    questions: list[Question] = []
    for sample_id in sorted(basic):
        cutoff = choose_cutoff(basic[sample_id], fraction)
        if cutoff is None or sample_id not in usage:
            continue
        history = [row for row in basic[sample_id] if row["calendar_date"] <= cutoff and row["is_bank_workday"] == "True"][-LOOKBACK:]
        usage_history = [row for row in usage[sample_id] if row["calendar_date"] <= cutoff and row["is_bank_workday"] == "True"][-LOOKBACK:]
        answers = [row for row in basic[sample_id] if row["calendar_date"] > cutoff and row["is_bank_workday"] == "True"][:30]
        if len(history) != LOOKBACK or len(usage_history) != LOOKBACK or len(answers) != 30:
            continue
        start = history[0]["calendar_date"]
        visible_transactions = [row for row in transactions.get(sample_id, []) if start <= row["booking_date"] <= cutoff]
        # 这里检查的是输入边界：逐笔资料绝不含预测日后的行。
        if any(row["booking_date"] > cutoff for row in visible_transactions):
            raise ValueError(f"{sample_id}逐笔输入混入预测日后的交易")
        questions.append(Question(sample_id, families[sample_id], shapes.get(sample_id, "未分类"), cutoff, history, usage_history, visible_transactions, np.asarray([float(row["ending_balance_cny"]) for row in answers])))
        if limit is not None and len(questions) >= limit:
            break
    if not questions:
        raise ValueError(f"{role}没有可构造的预测题")
    return questions


def daily_array(question: Question, with_usage: bool) -> np.ndarray:
    rows = question.usage if with_usage else question.basic
    base_columns = ("ending_balance_cny", "inflow_cny", "outflow_cny", "transaction_count")
    usage_columns = tuple(name for name in rows[0] if name.endswith("_cny") or name.endswith("_transaction_count")) if with_usage else ()
    columns = base_columns + tuple(name for name in usage_columns if name not in base_columns)
    values = np.asarray([[float(row[name]) for name in columns] for row in rows], dtype=np.float64)
    # 金额字段都除以预测日余额尺度，避免不同企业的金额绝对大小支配训练。
    for column, name in enumerate(columns):
        if name.endswith("_cny"):
            values[:, column] /= question.scale
    return values


def transaction_array(question: Question) -> np.ndarray:
    """保留逐笔顺序；每行是金额、方向、该笔余额、间隔日数和9个用途标记。"""
    start = date.fromisoformat(question.basic[0]["calendar_date"])
    previous = start
    rows: list[list[float]] = []
    for row in question.transactions[-MAX_TRANSACTION_STEPS:]:
        current = date.fromisoformat(row["booking_date"])
        gap = (current - previous).days
        signed = float(row["amount_cny"]) * (-1.0 if row["direction_cn"] == "流出" else 1.0)
        item = [signed / question.scale, float(row["post_transaction_balance_cny"]) / question.scale, float(gap)]
        item += [1.0 if row["usage_code"] == code else 0.0 for code in USAGE_CODES]
        rows.append(item)
        previous = current
    # 没交易也要提供一行，第三列记录整个可见日期段长度。
    if not rows:
        rows = [[0.0, question.base / question.scale, float((date.fromisoformat(question.origin) - start).days)] + [0.0] * len(USAGE_CODES)]
    return np.asarray(rows, dtype=np.float64)


def transaction_fixed_features(question: Question) -> np.ndarray:
    """供固定数字方法使用；只汇总已可见的逐笔金额、用途和间隔。"""
    values = transaction_array(question)
    amounts = values[:, 0]
    # 金额已经按该企业预测日余额尺度处理。
    result = [question.base / question.scale, float(len(values)), float(amounts.sum()), float(np.mean(amounts)), float(np.median(amounts)), float(np.percentile(amounts, 90)), float(np.min(amounts)), float(np.max(amounts))]
    result += [float((amounts > 0).sum()), float((amounts < 0).sum()), float(values[-1, 2])]
    for position in range(len(USAGE_CODES)):
        mask = values[:, 3 + position] > 0
        result.extend((float(mask.sum()), float(amounts[mask].sum())))
    return np.asarray(result, dtype=np.float64)


def tabular_features(question: Question, material: str) -> np.ndarray:
    if material == "逐笔金额余额用途":
        return transaction_fixed_features(question)
    sequence = daily_array(question, material == "每天余额收入支出及用途")
    # 固定数字方法读完整28天的开始、结束、平均、波动和近5天变化。
    return np.concatenate((sequence[0], sequence[-1], sequence.mean(axis=0), sequence.std(axis=0), sequence[-5:].mean(axis=0), sequence[-1] - sequence[-5:].mean(axis=0)))


def sequence_features(question: Question, material: str) -> np.ndarray:
    if material == "逐笔金额余额用途":
        return transaction_array(question)
    return daily_array(question, material == "每天余额收入支出及用途")


def baseline(question: Question, trend: bool) -> np.ndarray:
    if not trend:
        return np.full(3, question.base, dtype=np.float64)
    balances = np.asarray([float(row["ending_balance_cny"]) for row in question.basic[-5:]])
    drift = float(np.mean(np.diff(balances))) if len(balances) > 1 else 0.0
    path = question.base + drift * np.arange(1, 31)
    return np.asarray((path[0], path[:10].mean(), path.mean()), dtype=np.float64)


class SequenceRegressor(torch.nn.Module):
    def __init__(self, fields: int, kind: str) -> None:
        super().__init__()
        self.layer = torch.nn.LSTM(fields, 16, batch_first=True) if kind == "LSTM" else torch.nn.GRU(fields, 16, batch_first=True)
        self.head = torch.nn.Linear(16, 3)

    def forward(self, padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.layer(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.head(hidden[-1])


def pad_sequences(sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    fields = sequences[0].shape[1]
    max_length = max(len(item) for item in sequences)
    values = np.zeros((len(sequences), max_length, fields), dtype=np.float32)
    lengths = np.zeros(len(sequences), dtype=np.int64)
    for index, item in enumerate(sequences):
        values[index, : len(item)] = item
        lengths[index] = len(item)
    return values, lengths


def targets(questions: list[Question]) -> np.ndarray:
    return np.vstack([question.target for question in questions])


def normalized_targets(questions: list[Question]) -> np.ndarray:
    return np.vstack([(question.target - question.base) / question.scale for question in questions])


def train_tabular(name: str, train: list[Question], material: str):
    x = np.vstack([tabular_features(question, material) for question in train])
    y = normalized_targets(train)
    if name == "岭回归":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="svd")).fit(x, y)
    if name == "XGBoost":
        return [xgb.train({"objective": "reg:squarederror", "tree_method": "hist", "max_depth": 4, "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5.0, "seed": SEEDS[0], "nthread": 4}, xgb.DMatrix(x, label=y[:, index]), num_boost_round=100) for index in range(3)]
    if name == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=96, max_depth=12, min_samples_leaf=2, max_features=0.8, n_jobs=4, random_state=SEEDS[0]).fit(x, y)
    if name == "直方图梯度提升树":
        return MultiOutputRegressor(HistGradientBoostingRegressor(learning_rate=0.1, max_iter=100, max_leaf_nodes=15, min_samples_leaf=2, early_stopping=False, random_state=SEEDS[0])).fit(x, y)
    raise ValueError(name)


def predict_tabular(model, name: str, questions: list[Question], material: str) -> np.ndarray:
    x = np.vstack([tabular_features(question, material) for question in questions])
    normalized = np.column_stack([item.predict(xgb.DMatrix(x)) for item in model]) if name == "XGBoost" else model.predict(x)
    base = np.asarray([question.base for question in questions])
    scale = np.asarray([question.scale for question in questions])
    return base[:, None] + normalized * scale[:, None]


def train_sequence(kind: str, train: list[Question], early: list[Question], material: str, seed: int, epochs: int, output: Path) -> tuple[SequenceRegressor, dict]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    raw = [sequence_features(question, material) for question in train]
    padded, lengths = pad_sequences(raw)
    # 只由学习企业拟合各栏平均和离散程度。
    flat = np.concatenate(raw, axis=0)
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    padded = ((padded - mean) / std).astype(np.float32)
    y = normalized_targets(train)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_std = np.where(y_std == 0.0, 1.0, y_std)
    y = ((y - y_mean) / y_std).astype(np.float32)
    early_raw = [sequence_features(question, material) for question in early]
    early_padded, early_lengths = pad_sequences(early_raw)
    early_padded = ((early_padded - mean) / std).astype(np.float32)
    early_y = ((normalized_targets(early) - y_mean) / y_std).astype(np.float32)
    model = SequenceRegressor(padded.shape[2], kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x_tensor, length_tensor, y_tensor = torch.from_numpy(padded), torch.from_numpy(lengths), torch.from_numpy(y)
    early_x, early_length, early_target = torch.from_numpy(early_padded), torch.from_numpy(early_lengths), torch.from_numpy(early_y)
    losses: list[dict[str, float]] = []; best_loss = float("inf"); best_state = None; best_epoch = 0; without_improvement = 0
    for _ in range(epochs):
        optimizer.zero_grad(); predicted = model(x_tensor, length_tensor); loss = torch.abs(predicted - y_tensor).mean(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        model.eval()
        with torch.no_grad(): early_loss = float(torch.abs(model(early_x, early_length) - early_target).mean())
        losses.append({"学习误差": float(loss.detach()), "日期检查误差": early_loss})
        if early_loss < best_loss - 0.0001:
            best_loss, best_epoch, without_improvement = early_loss, len(losses), 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            without_improvement += 1
        if without_improvement >= 5:
            break
    if best_state is None: raise ValueError("循环网络没有产生可保存的日期检查结果")
    model.load_state_dict(best_state)
    package = {"kind": kind, "fields": int(padded.shape[2]), "mean": mean.tolist(), "std": std.tolist(), "target_mean": y_mean.tolist(), "target_std": y_std.tolist(), "state_dict": model.state_dict(), "seed": seed, "maximum_epochs": epochs, "actual_epochs": len(losses), "best_epoch": best_epoch, "best_date_check_error": best_loss, "losses": losses}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, output)
    return model, package


def predict_sequence(model: SequenceRegressor, package: dict, questions: list[Question], material: str) -> np.ndarray:
    raw = [sequence_features(question, material) for question in questions]
    padded, lengths = pad_sequences(raw)
    padded = ((padded - np.asarray(package["mean"])) / np.asarray(package["std"])).astype(np.float32)
    model.eval()
    with torch.no_grad():
        normalized = model(torch.from_numpy(padded), torch.from_numpy(lengths)).numpy()
    normalized = normalized * np.asarray(package["target_std"]) + np.asarray(package["target_mean"])
    return np.asarray([question.base for question in questions])[:, None] + normalized * np.asarray([question.scale for question in questions])[:, None]


def radius(predictions: np.ndarray, questions: list[Question]) -> tuple[np.ndarray, np.ndarray]:
    actual = targets(questions); scale = np.asarray([question.scale for question in questions])[:, None]
    residual = np.abs(predictions - actual) / scale
    # 目标约80%：取从小到大第 ceil((n+1)*0.80) 项，超过最后一项时取最后一项。
    position = min(len(questions) - 1, math.ceil((len(questions) + 1) * 0.80) - 1)
    return np.sort(residual, axis=0)[position], residual


def make_intervals(predictions: np.ndarray, questions: list[Question], radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray([question.scale for question in questions])[:, None]
    distance = scale * radii[None, :]
    return predictions - distance, predictions + distance


def scored_rows(method: str, material: str, questions: list[Question], prediction: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, question in enumerate(questions):
        for horizon, label in enumerate(HORIZON_NAMES):
            actual = float(question.target[horizon]); point = float(prediction[index, horizon]); lo = float(lower[index, horizon]); hi = float(upper[index, horizon]); scale = question.scale
            miss = max(lo - actual, 0.0) / scale + max(actual - hi, 0.0) / scale
            rows.append({"方法": method, "资料": material, "企业编号": question.sample_id, "来源组": question.family_id, "余额走势": question.shape, "预测日期": question.origin, "预测内容": label, "实际余额_cny": actual, "预测余额_cny": point, "预测下限_cny": lo, "预测上限_cny": hi, "绝对误差_cny": abs(actual - point), "实际余额是否落在范围内": int(lo <= actual <= hi), "范围宽度占预测日余额比例": (hi - lo) / scale, "漏出范围扣分": (hi - lo) / scale + 10.0 * miss})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: raise ValueError(f"{path}没有内容")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows: grouped[(str(row["方法"]), str(row["资料"]), str(row["预测内容"]))].append(row)
    result = []
    for key, items in sorted(grouped.items()):
        result.append({"方法": key[0], "资料": key[1], "预测内容": key[2], "企业数": len({item["企业编号"] for item in items}), "预测题数": len(items), "平均绝对误差_cny": float(np.mean([float(item["绝对误差_cny"]) for item in items])), "实际余额落在范围内比例": float(np.mean([int(item["实际余额是否落在范围内"]) for item in items])), "平均范围宽度占预测日余额比例": float(np.mean([float(item["范围宽度占预测日余额比例"]) for item in items])), "平均漏出范围扣分": float(np.mean([float(item["漏出范围扣分"]) for item in items]))})
    return result


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, list):
        for index, item in enumerate(model): item.save_model(path.with_name(f"{path.stem}_{index + 1}.json"))
    else: joblib.dump(model, path)


def run_one(method: str, material: str, learning: list[Question], early: list[Question], calibration: list[Question], development: list[Question], output: Path, epochs: int, sequence_seeds: Iterable[int]) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = perf_counter()
    model_dir = output / "模型文件" / material / method
    if method in ("余额保持法", "近期趋势延续法"):
        trend = method == "近期趋势延续法"
        p_cal = np.vstack([baseline(question, trend) for question in calibration]); p_dev = np.vstack([baseline(question, trend) for question in development]); seed_count = 0
    elif method in ("岭回归", "XGBoost", "ExtraTrees", "直方图梯度提升树"):
        model = train_tabular(method, learning, material); save_model(model, model_dir / "model.joblib")
        p_cal, p_dev, seed_count = predict_tabular(model, method, calibration, material), predict_tabular(model, method, development, material), 1
    else:
        # 三个固定随机起点都保留，预测取逐项中位数；小规模检查也使用相同规则。
        cal_all, dev_all = [], []
        for seed in sequence_seeds:
            model, package = train_sequence(method, learning, early, material, seed, epochs, model_dir / f"seed_{seed}.pt")
            cal_all.append(predict_sequence(model, package, calibration, material)); dev_all.append(predict_sequence(model, package, development, material))
        p_cal, p_dev, seed_count = np.median(np.stack(cal_all), axis=0), np.median(np.stack(dev_all), axis=0), len(cal_all)
    radii, residuals = radius(p_cal, calibration)
    lower, upper = make_intervals(p_dev, development, radii)
    rows = scored_rows(method, material, development, p_dev, lower, upper)
    detail = {"方法": method, "资料": material, "学习企业数": len(learning), "日期检查企业数": len(early) if method in ("LSTM", "GRU") else 0, "区间确定企业数": len(calibration), "开发测试企业数": len(development), "三个预测上下范围的半径_按预测日余额比例": radii.tolist(), "区间确定时三个预测的每户残差": residuals.tolist(), "循环网络独立训练次数": seed_count if method in ("LSTM", "GRU") else 0, "运行秒数": perf_counter() - started}
    return rows, detail


def transaction_length_audit(groups: Iterable[list[Question]]) -> dict[str, object]:
    """说明逐笔序列是否因固定上限遗漏较早交易，供正式运行前决定上限。"""
    questions = [question for group in groups for question in group]
    counts = [len(question.transactions) for question in questions]
    truncated = [question for question in questions if len(question.transactions) > MAX_TRANSACTION_STEPS]
    return {
        "每题逐笔输入上限": MAX_TRANSACTION_STEPS,
        "题目数": len(questions),
        "原始逐笔数最大值": max(counts),
        "原始逐笔数中位数": float(np.median(counts)),
        "超过上限题目数": len(truncated),
        "超过上限题目比例": len(truncated) / len(questions),
        "超过上限企业和预测日期": [{"企业编号": question.sample_id, "预测日期": question.origin, "原始逐笔数": len(question.transactions)} for question in truncated],
        "处理规则": "仅逐笔LSTM和GRU保留最晚的256笔；树方法仍从预测日以前全部可见逐笔资料算固定数字。若正式运行中超过上限比例较高，必须先重新决定上限或分段读法，再开始正式比较。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="第三轮三种资料八种方法比较入口")
    parser.add_argument("--mode", choices=("smoke", "development"), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "smoke":
        limits, epochs, suffix = (8, 3, "小规模可运行核对")
    else:
        # 这个分支仍只读现有三组资料；调用人必须显式选择，避免平时检查触发全量训练。
        limits, epochs, suffix = (None, 30 if args.epochs is None else args.epochs, "开发比较")
    if args.epochs is not None: epochs = args.epochs
    output = (args.output or ROOT / "runs" / f"第三轮_{suffix}_20260923").resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 学习资料用较早预测日学习，较晚预测日只检查每轮是否继续改善的入口已预留；本次统一训练接口不使用开发测试标签。
    learning = build_questions("学习", 0.45, limits)
    early = build_questions("学习", 0.70, limits)
    calibration = build_questions("区间校准", 0.60, limits)
    development = build_questions("开发测试", 0.70, limits)
    materials = ("每天余额收入支出", "每天余额收入支出及用途", "逐笔金额余额用途")
    methods = ("余额保持法", "近期趋势延续法", "岭回归", "XGBoost", "ExtraTrees", "直方图梯度提升树", "LSTM", "GRU")
    if {item.sample_id for item in learning} != {item.sample_id for item in early} or any(date.fromisoformat(next(item.origin for item in early if item.sample_id == question.sample_id)) <= date.fromisoformat(question.origin) for question in learning):
        raise ValueError("学习资料的日期检查题必须与学习题属于同一企业且预测日更晚")
    all_rows: list[dict[str, object]] = []; details: list[dict[str, object]] = []
    for material in materials:
        for method in methods:
            rows, detail = run_one(method, material, learning, early, calibration, development, output, epochs, SEEDS)
            all_rows.extend(rows); details.append(detail)
    write_csv(output / "开发测试逐题预测与评分.csv", all_rows)
    write_csv(output / "开发测试成绩汇总.csv", summarize(all_rows))
    (output / "运行记录.json").write_text(json.dumps({"version": VERSION, "purpose": "小规模可运行核对" if args.mode == "smoke" else "开发比较", "mode": args.mode, "actual_learning_enterprises": len(learning), "actual_date_check_enterprises": len(early), "actual_calibration_enterprises": len(calibration), "actual_development_enterprises": len(development), "maximum_epochs": epochs, "fixed_sequence_seeds": list(SEEDS), "target_rule": "三种资料共用同一份每日余额表的随后30个银行营业日日末余额；因此同一企业和预测日期的三项正确余额严格相同，包含无交易日延续的余额。", "date_check_rule": "LSTM和GRU在学习企业中选取比学习题更晚的预测日，只查看日期检查误差来保留较好的轮次；不读取开发测试或区间确定企业的答案决定轮次。", "history_rule": "每题只读预测日及以前、28个银行营业日覆盖的日期；逐笔资料按日期顺序读入金额、交易后余额、用途独立标记和日期间隔，不读取生成器写入的小时、分钟、秒。树方法只读预测日以前逐笔资料的固定数字汇总。", "calendar_rule": "只读准备资料中已保存的银行日标记；本运行不依据预测日后的交易改变日历", "category_rule": "用途代码拆成九个各自独立的0或1标记；代码的字母和数字大小不参与金额或先后判断。", "range_rule": "每种方法、每种资料、三项预测分别只用区间确定组的绝对余额误差确定目标约80%的上下范围；开发测试组只接受已确定的范围。范围宽度和漏出范围扣分会与命中率一起保存。", "transaction_length_audit": transaction_length_audit((learning, calibration, development)), "not_done": ["未运行525份学习、113份区间确定、142份开发测试的正式全量比较" if args.mode == "smoke" else "尚未生成或评分全新最终大考企业", "未执行可选的中文词组和英文词组文字比较", "第一轮已训练LSTM历史参照尚未接入本入口"], "runs": details}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "mode": args.mode, "rows": len(all_rows), "runs": len(details)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
