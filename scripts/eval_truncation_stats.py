"""
Report truncation rate and mean scores from an inspect-ai .eval file.

Truncation is defined as stop_reason == "max_tokens" on the first choice.
Reports mean score for all samples and for the non-truncated subset.

Usage:
    python eval_truncation_stats.py <path/to/file.eval> [<file.eval> ...]
"""

import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import zstandard as _zstd
    def _decompress_zstd(data: bytes) -> bytes:
        return _zstd.ZstdDecompressor().stream_reader(data).read()
except ImportError:
    def _decompress_zstd(data: bytes) -> bytes:
        result = subprocess.run(["zstd", "-d", "-", "--stdout"], input=data, capture_output=True, check=True)
        return result.stdout


def _read_raw_entry(zf: zipfile.ZipFile, name: str) -> bytes:
    info = zf.getinfo(name)
    zf.fp.seek(info.header_offset)
    header = zf.fp.read(30)
    fname_len, extra_len = struct.unpack_from("<HH", header, 26)
    zf.fp.seek(info.header_offset + 30 + fname_len + extra_len)
    compressed = zf.fp.read(info.compress_size)
    if info.compress_type == 93:  # zstd
        return _decompress_zstd(compressed)
    elif info.compress_type == 0:  # stored
        return compressed
    else:
        import zlib
        return zlib.decompress(compressed, -15)  # deflate


def _extract_scores(sample: dict) -> dict[str, float]:
    """Return a flat {metric_name: value} dict from a sample's scores field."""
    result = {}
    for scorer_data in (sample.get("scores") or {}).values():
        value = scorer_data.get("value")
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    result[k] = float(v)
        elif isinstance(value, (int, float)):
            result["score"] = float(value)
    return result


def truncation_stats(path: str) -> None:
    total = 0
    truncated = 0
    all_scores: dict[str, list[float]] = {}
    nontrunc_scores: dict[str, list[float]] = {}
    max_tokens = None

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

        if "_journal/start.json" in names:
            start = json.loads(_read_raw_entry(zf, "_journal/start.json"))
            max_tokens = start.get("plan", {}).get("config", {}).get("max_tokens")

        sample_names = [n for n in names if n.startswith("samples/") and n.endswith(".json")]
        for name in sample_names:
            data = json.loads(_read_raw_entry(zf, name))
            choices = (data.get("output") or {}).get("choices") or []
            if not choices:
                continue
            total += 1
            is_truncated = choices[0].get("stop_reason") == "max_tokens"
            if is_truncated:
                truncated += 1

            scores = _extract_scores(data)
            for k, v in scores.items():
                all_scores.setdefault(k, []).append(v)
                if not is_truncated:
                    nontrunc_scores.setdefault(k, []).append(v)

    pct = 100 * truncated / total if total else 0.0
    label = Path(path).parent.name or Path(path).name
    print(f"\n{label}")
    print(f"  max_tokens config : {max_tokens}")
    print(f"  total samples     : {total}")
    print(f"  truncated         : {truncated} ({pct:.1f}%)")

    if all_scores:
        print(f"  {'metric':<35}  {'all':>7}  {'non-truncated':>14}")
        print(f"  {'-'*35}  {'-'*7}  {'-'*14}")
        for k in sorted(all_scores):
            mean_all = sum(all_scores[k]) / len(all_scores[k])
            nt = nontrunc_scores.get(k, [])
            mean_nt = sum(nt) / len(nt) if nt else float("nan")
            print(f"  {k:<35}  {mean_all:>7.4f}  {mean_nt:>14.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        truncation_stats(p)
