"""Reassemble or verify the two alternating history-usage CSV chunks.

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


def reassemble(part1: Path, part2: Path, output: Path | None = None) -> str:
    digest = hashlib.sha256()
    with part1.open("rb") as left, part2.open("rb") as right:
        bom1, header1 = split_header(left)
        bom2, header2 = split_header(right)
        if header1 != header2:
            raise ValueError("usage chunk headers differ")
        output_handle = output.open("wb") if output else None
        try:
            def emit(data: bytes) -> None:
                digest.update(data)
                if output_handle:
                    output_handle.write(data)

            emit(bom1 or bom2)
            emit(header1)
            while True:
                left_line = left.readline()
                right_line = right.readline()
                if not left_line and not right_line:
                    break
                if left_line:
                    emit(left_line)
                if right_line:
                    emit(right_line)
        finally:
            if output_handle:
                output_handle.close()
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, help="Optional local complete CSV for byte-level verification.")
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--part2", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write a complete CSV reconstructed from the two remote chunks.")
    args = parser.parse_args()
    if not args.full and not args.output:
        parser.error("provide --output to reassemble, --full to verify, or both")
    if args.output and args.output.resolve() in {args.part1.resolve(), args.part2.resolve()}:
        parser.error("--output must not overwrite a source chunk")
    if args.output and args.full and args.output.resolve() == args.full.resolve():
        parser.error("--output must differ from --full when performing verification")
    rebuilt = reassemble(args.part1, args.part2, args.output)
    print(f"reassembled_sha256={rebuilt}")
    if args.output:
        print(f"output={args.output}")
    if args.full:
        full = hashlib.sha256(args.full.read_bytes()).hexdigest().upper()
        print(f"full_sha256={full}")
        print(f"matches={rebuilt == full}")
        if rebuilt != full:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
