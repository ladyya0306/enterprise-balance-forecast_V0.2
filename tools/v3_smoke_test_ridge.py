"""Small V3 route check: Ridge regression on each prepared material form.

This is not a model comparison.  It deliberately uses four fixed learning
enterprises and two fixed development-test enterprises only to prove that each
file can form a 28-bank-workday history and three future answers without reading
post-cutoff rows as inputs.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


VERSION = "v3_three_materials_ridge_smoke_20260923_1"
TRAIN_CUTOFF = "2025-06-30"
PREDICT_CUTOFF = "2025-09-30"
LOOKBACK_BANK_WORKDAYS = 28
USAGE_CODES = ("CUSTOMER_RECEIPT", "PROCUREMENT", "TAX", "OPERATING_SERVICE", "PREMISES_RENT", "PAYROLL", "UNKNOWN", "FINANCING", "EQUIPMENT")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def daily_windows(rows: list[dict[str, str]], cutoff: str):
    visible = [row for row in rows if row["calendar_date"] <= cutoff]
    history = [row for row in visible if row["is_bank_workday"] == "True"][-LOOKBACK_BANK_WORKDAYS:]
    future = [row for row in rows if row["calendar_date"] > cutoff and row["is_bank_workday"] == "True"][:30]
    if len(history) != LOOKBACK_BANK_WORKDAYS or len(future) != 30:
        raise ValueError(f"{rows[0]['sample_id']}在{cutoff}缺少28天历史或30天答案")
    return history, future


def targets(future: list[dict[str, str]]) -> np.ndarray:
    balances = np.asarray([float(row["ending_balance_cny"]) for row in future])
    return np.asarray([balances[0], balances[:10].mean(), balances.mean()])


def numeric_daily(history: list[dict[str, str]], include_usage: bool) -> np.ndarray:
    columns = ["ending_balance_cny", "inflow_cny", "outflow_cny", "transaction_count"]
    if include_usage:
        columns += [key for key in history[0] if key.endswith("_cny") or key.endswith("_transaction_count")]
        columns = list(dict.fromkeys(columns))
    return np.asarray([[float(row[column]) for column in columns] for row in history], dtype=float).reshape(-1)


def numeric_transaction(transactions: list[dict[str, str]], daily_rows: list[dict[str, str]], cutoff: str) -> np.ndarray:
    history, _ = daily_windows(daily_rows, cutoff)
    start = history[0]["calendar_date"]
    visible = [row for row in transactions if start <= row["booking_date"] <= cutoff]
    amounts = np.asarray([float(row["amount_cny"]) for row in visible], dtype=float)
    if amounts.size == 0:
        amounts = np.asarray([0.0])
    values = [float(history[-1]["ending_balance_cny"]), float(len(visible)), float(amounts.sum()), float(np.median(amounts)), float(np.percentile(amounts, 90))]
    for code in USAGE_CODES:
        subset = [row for row in visible if row["usage_code"] == code]
        values += [float(len(subset)), float(sum(float(row["amount_cny"]) for row in subset))]
    return np.asarray(values, dtype=float)


def select_ids(root: Path):
    rows = read_csv(root / "data/manifests/样本分组名单.csv")
    by_role = {role: sorted(row["sample_id"] for row in rows if row["v3_role"] == role) for role in ("学习", "开发测试")}
    return by_role["学习"][:4], by_role["开发测试"][:2]


def run_one(name: str, train_features, train_y, predict_features, predict_y, predict_ids, output: Path):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.vstack(train_features))
    X_predict = scaler.transform(np.vstack(predict_features))
    model = Ridge(alpha=1.0).fit(X_train, np.vstack(train_y))
    predicted = model.predict(X_predict)
    rows = []
    names = ("下一银行营业日的日末余额", "以后10个银行营业日的日均余额", "以后30个银行营业日的日均余额")
    for sample_id, actual, estimate in zip(predict_ids, predict_y, predicted):
        for horizon, actual_value, estimate_value in zip(names, actual, estimate):
            rows.append({"资料类型": name, "sample_id": sample_id, "预测日期": PREDICT_CUTOFF, "预测内容": horizon, "实际金额_cny": f"{actual_value:.2f}", "预测金额_cny": f"{estimate_value:.2f}", "绝对误差_cny": f"{abs(actual_value-estimate_value):.2f}"})
    with (output / f"{name}_逐项结果.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return {"资料类型": name, "训练行数": len(train_features), "预测行数": len(predict_features), "每行输入数字个数": int(X_train.shape[1]), "三项预测平均绝对误差_cny": f"{np.mean(np.abs(np.vstack(predict_y)-predicted)):.2f}"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    materials = root / "data/training_materials/v1"
    output = root / "runs/v3_三种资料岭回归小规模试运行_20260923"
    output.mkdir(parents=True, exist_ok=True)
    train_ids, predict_ids = select_ids(root)
    daily_basic = {role: {sid: [] for sid in train_ids + predict_ids} for role in ("学习", "开发测试")}
    daily_usage = {role: {sid: [] for sid in train_ids + predict_ids} for role in ("学习", "开发测试")}
    transactions = {role: {sid: [] for sid in train_ids + predict_ids} for role in ("学习", "开发测试")}
    for role in ("学习", "开发测试"):
        for row in read_csv(materials / "每天余额收入支出" / f"{role}.csv"):
            if row["sample_id"] in daily_basic[role]: daily_basic[role][row["sample_id"]].append(row)
        for row in read_csv(materials / "每天余额收入支出及用途" / f"{role}.csv"):
            if row["sample_id"] in daily_usage[role]: daily_usage[role][row["sample_id"]].append(row)
        for row in read_csv(materials / "逐笔金额余额用途" / f"{role}.csv"):
            if row["sample_id"] in transactions[role]: transactions[role][row["sample_id"]].append(row)
    checks = []
    for sample_id in train_ids + predict_ids:
        role, cutoff = ("学习", TRAIN_CUTOFF) if sample_id in train_ids else ("开发测试", PREDICT_CUTOFF)
        history, future = daily_windows(daily_basic[role][sample_id], cutoff)
        if max(row["calendar_date"] for row in history) > cutoff:
            raise ValueError("历史窗口混入预测日后资料")
        checks.append({"sample_id": sample_id, "v3_role": role, "预测日期": cutoff, "历史开始日期": history[0]["calendar_date"], "历史结束日期": history[-1]["calendar_date"], "答案开始日期": future[0]["calendar_date"], "答案结束日期": future[-1]["calendar_date"]})
    trials = []
    for name, feature in (("每天余额收入支出", lambda role, sid, cutoff: numeric_daily(daily_windows(daily_basic[role][sid], cutoff)[0], False)), ("每天余额收入支出及用途", lambda role, sid, cutoff: numeric_daily(daily_windows(daily_usage[role][sid], cutoff)[0], True)), ("逐笔金额余额用途", lambda role, sid, cutoff: numeric_transaction(transactions[role][sid], daily_basic[role][sid], cutoff))):
        train_y = [targets(daily_windows(daily_basic["学习"][sid], TRAIN_CUTOFF)[1]) for sid in train_ids]
        predict_y = [targets(daily_windows(daily_basic["开发测试"][sid], PREDICT_CUTOFF)[1]) for sid in predict_ids]
        trials.append(run_one(name, [feature("学习", sid, TRAIN_CUTOFF) for sid in train_ids], train_y, [feature("开发测试", sid, PREDICT_CUTOFF) for sid in predict_ids], predict_y, predict_ids, output))
    with (output / "小规模试运行记录.json").open("w", encoding="utf-8") as handle:
        json.dump({"version": VERSION, "purpose": "检查三种资料能被同一成熟岭回归完整读取、训练和预测；不是模型成绩比较", "training_sample_ids": train_ids, "prediction_sample_ids": predict_ids, "training_cutoff": TRAIN_CUTOFF, "prediction_cutoff": PREDICT_CUTOFF, "lookback_bank_workdays": LOOKBACK_BANK_WORKDAYS, "checks": checks, "trials": trials, "not_done": ["未运行全部八种方法", "未使用113份区间校准资料确定上下范围", "未对142份开发测试资料作正式比较", "未生成或读取最终大考新企业资料"]}, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), "trials": trials}, ensure_ascii=False))


if __name__ == "__main__":
    main()
