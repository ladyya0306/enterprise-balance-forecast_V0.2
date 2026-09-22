"""Verify the two alternating history-usage CSV chunks can reconstruct the source bytes.

The chunks deliberately each include a header and alternate source data rows.  A normal
concatenation is therefore wrong: it both repeats a header and loses the original order.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def split_header(handle):
    prefix = handle.read(3)
    if prefix != b"\xef\xbb\xbf":
        handle.seek(0)
        prefix = b""
    return prefix, handle.readline()


def digest_reassembled(part1: Path, part2: Path) -> str:
    digest = hashlib.sha256()
    with part1.open("rb") as left, part2.open("rb") as right:
        bom1, header1 = split_header(left)
        bom2, header2 = split_header(right)
        if header1 != header2:
            raise ValueError("usage chunk headers differ")
        digest.update(bom1 or bom2)
        digest.update(header1)
        while True:
            left_line = left.readline()
            right_line = right.readline()
            if not left_line and not right_line:
                break
            if left_line:
                digest.update(left_line)
            if right_line:
                digest.update(right_line)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--part2", type=Path, required=True)
    args = parser.parse_args()
    rebuilt = digest_reassembled(args.part1, args.part2)
    full = hashlib.sha256(args.full.read_bytes()).hexdigest().upper()
    print(f"reassembled_sha256={rebuilt}")
    print(f"full_sha256={full}")
    print(f"matches={rebuilt == full}")
    if rebuilt != full:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
