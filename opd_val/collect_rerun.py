"""Build the results table for the post-fix re-run.

Reads the offline checkpoint-eval logs (`opd_val/logs/evalckpt_*.out`) rather
than the in-run evals, which are unreliable on this task: the orchestrator
builds eval clients as `openai_chat_completions` rather than the renderer, so
`enable_thinking = false` never applies and every arm reports ~95% truncation.

Usage: uv run python opd_val/collect_rerun.py [--prefix R-]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

LOGS = Path(__file__).parent / "logs"
HEADER = re.compile(r"^#{2,}\s*(\S+?)\s*/\s*step_(\d+)\s*#{2,}\s*$")
PASS = re.compile(r"overall pass rate\s*:\s*([0-9.]+)")
TRUNC = re.compile(r"truncated\s*:\s*\d+/\d+\s*\(([0-9.]+)%\)")


def collect(prefix: str) -> dict[str, dict[int, tuple[float, float]]]:
    """arm -> {step: (pass_rate, truncation_pct)}, latest log wins."""
    out: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for log in sorted(LOGS.glob("evalckpt_*.out"), key=lambda p: p.stat().st_mtime):
        run = step = None
        rate = None
        for line in log.read_text(errors="replace").splitlines():
            if m := HEADER.match(line.strip()):
                run, step = m.group(1), int(m.group(2))
                rate = None
            elif run and (m := PASS.search(line)):
                rate = float(m.group(1))
            elif run and rate is not None and (m := TRUNC.search(line)):
                arm = run.rsplit("_", 1)[0]  # strip the _<jobid> suffix
                if arm.startswith(prefix):
                    out[arm][step] = (rate, float(m.group(1)))
                rate = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="R-", help="arm-name prefix to include")
    args = ap.parse_args()

    data = collect(args.prefix)
    if not data:
        print(f"no completed evals matching prefix {args.prefix!r} yet")
        return

    steps = sorted({s for v in data.values() for s in v})
    w = max(len(a) for a in data) + 2
    print(f"GSM8K band (327 problems) | reference: student 0.588, teacher 0.893")
    print(f"post-fix re-run, FLASH_ATTENTION_TRITON_AMD_ENABLE unset, default max_norm\n")
    print(f"{'arm':<{w}}" + "".join(f"{'step ' + str(s):>16}" for s in steps))
    for arm in sorted(data):
        row = f"{arm:<{w}}"
        for s in steps:
            if s in data[arm]:
                rate, trunc = data[arm][s]
                row += f"{rate:>10.4f}{'(' + f'{trunc:.0f}' + '%)':>6}"
            else:
                row += f"{'-':>16}"
        print(row)
    print("\n(parenthesised = truncation rate; >50% means the arm stopped terminating)")


if __name__ == "__main__":
    main()
