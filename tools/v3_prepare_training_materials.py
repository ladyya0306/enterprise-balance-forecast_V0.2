"""Prepare the three approved V3 training material tables without changing source samples.

The program only turns already checked transaction categories into stable codes.
It does not read a prediction result to decide categories, train a model, or create
new enterprises.  Rows keep enterprise and date/transaction keys for later checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path


VERSION = "v3_training_materials_20260923_1"
CATEGORY_CODES = {
    "客户回款": "CUSTOMER_RECEIPT",
    "采购材料外协": "PROCUREMENT",
    "税费": "TAX",
    "经营服务": "OPERATING_SERVICE",
    "租赁场地": "PREMISES_RENT",
    "工资": "PAYROLL",
    "未知": "UNKNOWN",
    "融资": "FINANCING",
    "设备资产": "EQUIPMENT",
}
ROLES = ("学习", "区间校准", "开发测试")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, columns: list[str], rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def usage_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = row["sample_id"], row["transaction_id"]
        if key in index:
            raise ValueError(f"用途清单交易编号重复：{key}")
        if row["usage_class"] not in CATEGORY_CODES:
            raise ValueError(f"用途清单出现未登记类别：{row['usage_class']}")
        index[key] = row
    return index


def validate_source(sample: dict[str, str], directory: Path, usage: dict[tuple[str, str], dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    transactions = read_csv(directory / "transactions_total.csv")
    notes = read_csv(directory / "cash_flow_review_notes.csv")
    daily = read_csv(directory / "account_daily_total.csv")
    transaction_ids = [row["transaction_id"] for row in transactions]
    if len(transaction_ids) != len(set(transaction_ids)):
        raise ValueError(f"{sample['sample_id']}逐笔交易编号重复")
    note_by_id = {row["transaction_id"]: row for row in notes}
    if len(notes) != len(transactions) or set(note_by_id) != set(transaction_ids):
        raise ValueError(f"{sample['sample_id']}逐笔交易和原备注不能一一对应")
    total_in = sum((Decimal(row["credit_cny"]) for row in transactions), Decimal("0"))
    total_out = sum((Decimal(row["debit_cny"]) for row in transactions), Decimal("0"))
    if total_in != sum((Decimal(row["inflow_cny"]) for row in daily), Decimal("0")) or total_out != sum((Decimal(row["outflow_cny"]) for row in daily), Decimal("0")):
        raise ValueError(f"{sample['sample_id']}逐笔与按日收支汇总不一致")
    if transactions and Decimal(transactions[-1]["post_transaction_balance_cny"]) != Decimal(daily[-1]["ending_balance_cny"]):
        raise ValueError(f"{sample['sample_id']}最终逐笔余额与日表余额不一致")
    transaction_by_day: dict[str, list[dict[str, str]]] = defaultdict(list)
    for transaction in transactions:
        transaction_by_day[transaction["booking_datetime"][:10]].append(transaction)
    previous: date | None = None
    for row in daily:
        current = date.fromisoformat(row["calendar_date"])
        if previous is not None and current.toordinal() != previous.toordinal() + 1:
            raise ValueError(f"{sample['sample_id']}日表日期不连续")
        previous = current
    for transaction in transactions:
        key = sample["sample_id"], transaction["transaction_id"]
        item = usage.get(key)
        if item is None:
            raise ValueError(f"{sample['sample_id']}交易缺少已核对用途：{transaction['transaction_id']}")
        note = note_by_id[transaction["transaction_id"]]
        amount = transaction["debit_cny"] if Decimal(transaction["debit_cny"]) > 0 else transaction["credit_cny"]
        direction = "流出" if Decimal(transaction["debit_cny"]) > 0 else "流入"
        if (item["booking_date"], item["direction"], Decimal(item["amount_cny"])) != (transaction["booking_datetime"][:10], direction, Decimal(amount)):
            raise ValueError(f"{sample['sample_id']}用途清单与逐笔交易不一致：{transaction['transaction_id']}")
        if (note["booking_datetime"], note["direction_cn"], Decimal(note["amount_cny"])) != (transaction["booking_datetime"], direction, Decimal(amount)):
            raise ValueError(f"{sample['sample_id']}原备注与逐笔交易不一致：{transaction['transaction_id']}")
    prior_balance: Decimal | None = None
    for day in daily:
        day_transactions = transaction_by_day[day["calendar_date"]]
        if Decimal(day["inflow_cny"]) != sum((Decimal(item["credit_cny"]) for item in day_transactions), Decimal("0")):
            raise ValueError(f"{sample['sample_id']}日表流入与当日逐笔交易不一致：{day['calendar_date']}")
        if Decimal(day["outflow_cny"]) != sum((Decimal(item["debit_cny"]) for item in day_transactions), Decimal("0")):
            raise ValueError(f"{sample['sample_id']}日表流出与当日逐笔交易不一致：{day['calendar_date']}")
        if int(day["transaction_count"]) != len(day_transactions):
            raise ValueError(f"{sample['sample_id']}日表交易笔数与当日逐笔交易不一致：{day['calendar_date']}")
        expected_balance = Decimal(day_transactions[-1]["post_transaction_balance_cny"]) if day_transactions else prior_balance
        if expected_balance is not None and Decimal(day["ending_balance_cny"]) != expected_balance:
            raise ValueError(f"{sample['sample_id']}日表日末余额与当日末笔余额不一致：{day['calendar_date']}")
        prior_balance = Decimal(day["ending_balance_cny"])
    return transactions, daily


def common(sample: dict[str, str]) -> dict[str, str]:
    return {"sample_id": sample["sample_id"], "family_id": sample["family_id"], "v3_role": sample["v3_role"]}


def prepare(root: Path, output: Path) -> dict[str, object]:
    samples = read_csv(root / "data/manifests/样本分组名单.csv")
    process_rows = json.loads((root / "data/manifests/样本处理清单.json").read_text(encoding="utf-8"))
    directories = {row["sample_id"]: row["v3_folder"] for row in process_rows}
    if set(directories) != {row["sample_id"] for row in samples}:
        raise ValueError("样本分组名单与样本处理清单企业编号不一致")
    usage_rows = read_csv(root / "data/manifests/历史用途输入.csv")
    usage = usage_index(usage_rows)
    by_role = {role: [row for row in samples if row["v3_role"] == role] for role in ROLES}
    if {role: len(by_role[role]) for role in ROLES} != {"学习": 525, "区间校准": 113, "开发测试": 142}:
        raise ValueError("样本分组不是525份学习、113份区间校准、142份开发测试")

    dictionary_rows = [{"usage_class_cn": name, "usage_code": code, "meaning": "用途名称；代码不是金额、大小或先后顺序"} for name, code in CATEGORY_CODES.items()]
    write_csv(output / "用途中文名称与固定代码对照表.csv", ["usage_class_cn", "usage_code", "meaning"], dictionary_rows)
    output_rows: dict[str, dict[str, int]] = defaultdict(dict)
    source_hashes: dict[str, dict[str, str]] = {}
    columns_basic = ["sample_id", "family_id", "v3_role", "calendar_date", "is_bank_workday", "business_step", "ending_balance_cny", "inflow_cny", "outflow_cny", "transaction_count"]
    usage_columns = [f"{code}_{suffix}" for code in CATEGORY_CODES.values() for suffix in ("inflow_cny", "outflow_cny", "transaction_count")]
    columns_daily_usage = columns_basic + usage_columns
    columns_transaction = ["sample_id", "family_id", "v3_role", "transaction_id", "booking_datetime", "booking_date", "direction_cn", "amount_cny", "debit_cny", "credit_cny", "post_transaction_balance_cny", "usage_class_cn", "usage_code", "source_note_type"]

    for role, role_samples in by_role.items():
        basic_rows, daily_usage_rows, transaction_rows = [], [], []
        for sample in role_samples:
            directory = root / directories[sample["sample_id"]]
            transactions, daily = validate_source(sample, directory, usage)
            source_hashes[sample["sample_id"]] = {name: sha256(directory / name) for name in ("account_daily_total.csv", "transactions_total.csv", "cash_flow_review_notes.csv")}
            per_day: dict[str, dict[str, Decimal | int]] = defaultdict(lambda: defaultdict(Decimal))
            for transaction in transactions:
                item = usage[(sample["sample_id"], transaction["transaction_id"])]
                code = CATEGORY_CODES[item["usage_class"]]
                direction = "流出" if Decimal(transaction["debit_cny"]) > 0 else "流入"
                amount = Decimal(transaction["debit_cny"] if direction == "流出" else transaction["credit_cny"])
                key = transaction["booking_datetime"][:10]
                per_day[key][f"{code}_{'outflow_cny' if direction == '流出' else 'inflow_cny'}"] += amount
                per_day[key][f"{code}_transaction_count"] += 1
                transaction_rows.append({**common(sample), "transaction_id": transaction["transaction_id"], "booking_datetime": transaction["booking_datetime"], "booking_date": key, "direction_cn": direction, "amount_cny": str(amount), "debit_cny": transaction["debit_cny"], "credit_cny": transaction["credit_cny"], "post_transaction_balance_cny": transaction["post_transaction_balance_cny"], "usage_class_cn": item["usage_class"], "usage_code": code, "source_note_type": item["source_note_type"]})
            for day in daily:
                base = {**common(sample), **{key: day[key] for key in columns_basic if key in day}}
                basic_rows.append(base)
                sums = per_day[day["calendar_date"]]
                daily_usage_rows.append({**base, **{key: str(sums[key]) if key.endswith("_cny") else str(sums[key]) for key in usage_columns}})
        output_rows[role]["每天余额收入支出"] = write_csv(output / "每天余额收入支出" / f"{role}.csv", columns_basic, basic_rows)
        output_rows[role]["每天余额收入支出及用途"] = write_csv(output / "每天余额收入支出及用途" / f"{role}.csv", columns_daily_usage, daily_usage_rows)
        output_rows[role]["逐笔金额余额用途"] = write_csv(output / "逐笔金额余额用途" / f"{role}.csv", columns_transaction, transaction_rows)

    output_files = []
    for path in sorted(output.rglob("*.csv")):
        output_files.append({"path": str(path.relative_to(output)).replace("\\\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = {"version": VERSION, "source_sample_count": len(samples), "split_counts": {role: len(by_role[role]) for role in ROLES}, "usage_row_count": len(usage_rows), "usage_categories": [{"usage_class_cn": name, "usage_code": code, "transaction_count": sum(row["usage_class"] == name for row in usage_rows)} for name, code in CATEGORY_CODES.items()], "output_rows": output_rows, "output_files": output_files, "source_hashes": source_hashes, "rules": ["原始余额、金额、日期和原备注文件未改动", "主比较输入使用usage_class_cn和usage_code；source_note_type仅供追溯，不作为主比较输入", "逐笔资料按28个银行工作日覆盖的日期范围截取，不能把28笔交易当作28天", "每次预测只能读取预测日当天及以前的行；本表保存全程资料，读取程序必须按截止日期过滤"], "source_manifest_sha256": sha256(root / "data/manifests/样本分组名单.csv"), "source_usage_sha256": sha256(root / "data/manifests/历史用途输入.csv")}
    (output / "资料准备核对报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="准备第三轮三种训练资料")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output or root / "data/training_materials/v1").resolve()
    report = prepare(root, output)
    print(json.dumps({"output": str(output), "output_rows": report["output_rows"], "usage_row_count": report["usage_row_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
