"""Close out the V3 sample-preparation review without changing raw records."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
CLUSTER = "F-V3-E1-STRONG-NEAR-COPY-CLUSTER"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    manifest_path = MANIFEST_DIR / "样本处理清单.json"
    manifest = read_json(manifest_path)
    pairs = read_json(MANIFEST_DIR / "近复制证据.json")["cross_family_strong_pairs"]
    trend_rows = read_json(MANIFEST_DIR / "实测走势与曲线重复.json")
    curve_review = [
        row for row in trend_rows
        if row.get("curve_repetition_evidence", {}).get("passed") is False
    ]

    cluster_ids = sorted({item[key] for item in pairs for key in ("left", "right")})
    rows_by_id = {row["sample_id"]: row for row in manifest}
    missing = sorted(set(cluster_ids) - set(rows_by_id))
    if missing:
        raise ValueError(f"near-copy sample ids absent from manifest: {missing}")

    original_families = sorted({
        rows_by_id[sample_id].get("original_family_id", rows_by_id[sample_id]["family_id"])
        for sample_id in cluster_ids
    })
    cluster_roles = {rows_by_id[sample_id]["v3_role"] for sample_id in cluster_ids}
    if len(cluster_roles) != 1:
        raise ValueError(f"near-copy cluster crosses roles: {cluster_roles}")
    cluster_role = cluster_roles.pop()
    for sample_id in cluster_ids:
        row = rows_by_id[sample_id]
        row.setdefault("original_family_id", row["family_id"])
        row["family_id"] = CLUSTER
        row["near_copy_disposition"] = (
            "保留：17 对跨原族高相似证据构成同一 E1 同源机制簇；"
            f"仅作为{cluster_role}组中的一个已知样本族计数，原始记录不删除。"
        )

    curve_dispositions = {
        "matrix_t07_f04_c01_calendar01_revision01_revision01": (
            "保留：合同验收回款、备件采购、工资、租赁及服务费按月发生；"
            "交易日期、金额与事项不同，属正常经营周期而非机械复制。"
        ),
        "syn_b1_p0_233": (
            "保留：销售回款、采购付款、工资、水电及税费形成月度经营周期；"
            "交易时间、金额和事项并非静态重复序列。"
        ),
    }
    for sample_id, disposition in curve_dispositions.items():
        if sample_id not in rows_by_id:
            raise ValueError(f"curve-review sample absent from manifest: {sample_id}")
        rows_by_id[sample_id]["curve_repeat_disposition"] = disposition

    write_json(manifest_path, manifest)

    # Keep the group roster synchronized with the reviewed family identity.
    roster_path = MANIFEST_DIR / "样本分组名单.csv"
    with roster_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "family_id", "v3_role", "original_family_id"])
        writer.writeheader()
        for row in sorted(manifest, key=lambda item: item["sample_id"]):
            writer.writerow({
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "v3_role": row["v3_role"],
                "original_family_id": row.get("original_family_id", ""),
            })

    atlas_rows = {}
    with (MANIFEST_DIR / "图册CSV对应.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            atlas_rows[row["sample_id"]] = row
    index_path = MANIFEST_DIR / "按走势查看图册.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["sample_id", "measured_shape_v3", "atlas_page", "atlas_position", "daily_csv"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(trend_rows, key=lambda item: (item["measured_shape_v3"], item["sample_id"])):
            atlas = atlas_rows.get(row["sample_id"], {})
            writer.writerow({
                "sample_id": row["sample_id"],
                "measured_shape_v3": row["measured_shape_v3"],
                "atlas_page": atlas.get("page", ""),
                "atlas_position": atlas.get("position", ""),
                "daily_csv": atlas.get("daily_csv", ""),
            })

    family_counts = Counter(row["family_id"] for row in manifest)
    role_records = Counter(row["v3_role"] for row in manifest)
    role_families = Counter()
    for family, family_rows in __import__("itertools").groupby(
        sorted(manifest, key=lambda item: (item["family_id"], item["v3_role"])),
        key=lambda item: item["family_id"],
    ):
        roles = {item["v3_role"] for item in family_rows}
        if len(roles) != 1:
            raise ValueError(f"family crosses roles after review: {family}: {roles}")
        role_families[roles.pop()] += 1

    dispositions = {
        "review_scope": "仅处置既有 17 对跨族高相似线索和 2 户曲线重复线索；不生成新样本、不训练模型。",
        "near_copy_cluster": {
            "pair_count": len(pairs),
            "sample_ids": cluster_ids,
            "original_family_ids": original_families,
            "reviewed_family_id": CLUSTER,
            "v3_role": cluster_role,
            "disposition": f"同源近复制，保留为一个已知样本族；保留原因是用于{cluster_role}组中的机制覆盖，原始交易和CSV均不删除。",
        },
        "curve_repetition": [
            {
                "sample_id": item["sample_id"],
                "family_id": item["family_id"],
                "evidence": item["curve_repetition_evidence"],
                "disposition": curve_dispositions[item["sample_id"]],
            }
            for item in curve_review
        ],
        "post_review_counts": {
            "usable_records": len(manifest),
            "known_families": len(family_counts),
            "records_by_role": dict(sorted(role_records.items())),
            "families_by_role": dict(sorted(role_families.items())),
        },
    }
    write_json(MANIFEST_DIR / "近复制与曲线复核处置.json", dispositions)
    print(json.dumps(dispositions["post_review_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
