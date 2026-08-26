# /// script
# requires-python = ">=3.10"
# dependencies = ["zstandard"]
# ///
"""Print all scores from inspect-ai `.eval` files.

A `.eval` file is a zip archive whose members are zstd-compressed. Python's
stdlib `zipfile` only learned that compression method in 3.14, so members are
located from the central directory and decompressed by hand.

Usage:
    uv run --no-project read_eval_scores.py <path> [<path> ...]

Each <path> is a `.eval` file or a directory containing them.
"""

import json
import struct
import sys
import zipfile
from pathlib import Path

import zstandard

ZSTD_COMPRESS_TYPE = 93


def read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    info = zf.getinfo(name)
    if info.compress_type != ZSTD_COMPRESS_TYPE:
        return zf.read(name)
    with open(zf.filename, "rb") as f:
        # Local file header: 30 fixed bytes, with the name and extra field
        # lengths at offsets 26 and 28. Payload follows both.
        f.seek(info.header_offset + 26)
        name_len, extra_len = struct.unpack("<HH", f.read(4))
        f.seek(info.header_offset + 30 + name_len + extra_len)
        raw = f.read(info.compress_size)
    return zstandard.ZstdDecompressor().decompress(raw, max_output_size=info.file_size)


def report(path: Path) -> None:
    print(f"=== {path.name}")
    zf = zipfile.ZipFile(path)
    if "header.json" not in zf.namelist():
        print("  incomplete — no header.json (run was preempted or never finished)")
        return

    header = json.loads(read_member(zf, "header.json"))
    eval_spec = header.get("eval", {})
    print(f"  task    {eval_spec.get('task')}")
    print(f"  model   {eval_spec.get('model')}")
    print(f"  status  {header.get('status')}")
    print(f"  ended   {(header.get('stats') or {}).get('completed_at')}")

    results = header.get("results") or {}
    print(f"  samples {results.get('completed_samples')}/{results.get('total_samples')}")
    for score in results.get("scores", []):
        for metric, value in (score.get("metrics") or {}).items():
            print(f"  {score.get('name')}/{metric} = {value.get('value')}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        path = Path(arg)
        # Grouped pools (e.g. TASKS=ifeval+ifbench_test) nest one subdirectory
        # per task, so recurse rather than globbing a single level.
        targets = sorted(path.rglob("*.eval")) if path.is_dir() else [path]
        if not targets:
            print(f"=== {path}\n  no .eval files found")
        for target in targets:
            report(target)


if __name__ == "__main__":
    main()
