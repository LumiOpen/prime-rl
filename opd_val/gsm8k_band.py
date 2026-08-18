"""Build the 20-80% competence band on GSM8K for a given student.

The single thing that made every reverse-text result uninterpretable was the
absence of a usable operating point: the base model scored ~0.04 (nothing to
learn from), the released SFT model ~0.72 against a 0.799 teacher (nothing left
to learn). Rather than hope a task provides one, construct it — the trick from
`examples/basic/hendrycks-sanity`, which filters MATH to problems the student
solves 20-80% of the time.

Samples `--rollouts` completions per problem from the student, scores them with
the same semantics as the env's reward (`gsm8k_v1/verify.py`: last `#### ...`
match, both sides wrapped in \\boxed{} for math-verify), and writes the
in-band problems to a HF dataset the taskset can load.

Also reports the completion-length and truncation distribution, so the training
token budget is sized from measurement rather than a guess — the mistake that
cost the reverse-text runs several days.
"""

import argparse
import asyncio
import json
import re
import statistics as st
from pathlib import Path

from datasets import Dataset, load_dataset
from math_verify import parse, verify
from openai import AsyncOpenAI
from transformers import AutoTokenizer

GSM8K_SYSTEM = (
    "Solve the grade-school math problem. Reason step by step, then give the final "
    "answer as a single number on the last line, prefixed with '#### ' (e.g. '#### 42')."
)
# math_env_v1's framing (deps/research-environments/.../math_env_v1/taskset.py).
MATH_INSTRUCTION = (
    "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
)


def score_gsm8k(gold: str, pred: str) -> float:
    """Mirror of gsm8k_v1/verify.py so the band matches the env's own reward."""
    matches = re.findall(r"####\s*(.+)", pred or "")
    prediction = matches[-1].strip() if matches else (pred or "")
    try:
        return 1.0 if verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + prediction + "}")) else 0.0
    except Exception:
        return 0.0


def score_math(gold: str, pred: str) -> float:
    """math_env_v1's scoring, minus the LLM reference-judge fallback — a judge
    would add a second moving part to a measurement whose whole purpose is to be
    unambiguous, so this is a strict lower bound on that env's reward."""
    try:
        return 1.0 if verify(parse("\\boxed{" + gold + "}"), parse(pred or "")) else 0.0
    except Exception:
        return 0.0


def load_problems(dataset: str, n: int):
    """Return ([(prompt, gold, question_to_store, answer_to_store)], scorer).

    ``gsm8k-boxed`` is the one to use for building a training band: GSM8K's
    questions under math_env_v1's instruction and \\boxed{} scoring, so the
    measured pass rate is produced by exactly the prompt and reward the training
    run will use. (``gsm8k`` uses the native `#### N` framing of gsm8k_v1, whose
    taskset hardcodes its dataset and so cannot consume a filtered set.)"""
    if dataset in ("gsm8k", "gsm8k-boxed"):
        rows = list(load_dataset("openai/gsm8k", "main", split="train"))[:n]
        gold = [r["answer"].split("####")[-1].strip() for r in rows]
        if dataset == "gsm8k":
            return [
                (f"{GSM8K_SYSTEM}\n\n{r['question']}", g, r["question"], r["answer"])
                for r, g in zip(rows, gold)
            ], score_gsm8k
        # Store the bare numeric answer so math_env_v1's answer_key works directly.
        return [
            (f"{MATH_INSTRUCTION}{r['question']}", g, r["question"], g) for r, g in zip(rows, gold)
        ], score_math
    rows = list(load_dataset("PrimeIntellect/Hendrycks-Math", split="train"))[:n]
    return [
        (f"{MATH_INSTRUCTION}{r['question']}", str(r["answer"]), r["question"], str(r["answer"])) for r in rows
    ], score_math


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-problems", type=int, default=512)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--label", type=str, default="student")
    ap.add_argument("--dataset", choices=["gsm8k", "gsm8k-boxed", "math", "local"], default="gsm8k")
    ap.add_argument("--local-path", type=str, default="", help="parquet dir for --dataset local")
    ap.add_argument("--no-thinking", action="store_true", help="Qwen3: enable_thinking=False")
    args = ap.parse_args()

    if args.dataset == "local":
        rows = list(load_dataset(args.local_path, "default", split="train"))[: args.num_problems]
        problems = [(f"{MATH_INSTRUCTION}{r['question']}", str(r["answer"]), r["question"], str(r["answer"])) for r in rows]
        score = score_math
    else:
        problems, score = load_problems(args.dataset, args.num_problems)
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    tok = AutoTokenizer.from_pretrained(args.model)
    sem = asyncio.Semaphore(args.concurrency)
    # Qwen3 defaults to thinking mode; on short-form math that spends the whole
    # budget on CoT and pins the length distribution against the cap, which
    # turns a capability measurement into a truncation measurement.
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_thinking else {}

    async def one(prompt: str):
        async with sem:
            r = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                extra_body=extra,
            )
            c = r.choices[0]
            text = c.message.content or ""
            reasoning = getattr(c.message, "reasoning_content", None) or ""
            return text, reasoning, c.finish_reason

    results = await asyncio.gather(*(one(p) for p, _, _, _ in problems for _ in range(args.rollouts)))

    kept, pass_rates, lengths, truncated = [], [], [], 0
    for i, (_, gold, question, answer) in enumerate(problems):
        chunk = results[i * args.rollouts : (i + 1) * args.rollouts]
        scores = [score(gold, text) for text, _, _ in chunk]
        rate = sum(scores) / len(scores)
        pass_rates.append(rate)
        for text, reasoning, finish in chunk:
            lengths.append(len(tok.encode(reasoning + text)))
            truncated += finish == "length"
        if args.lo <= rate <= args.hi:
            kept.append({"question": question, "answer": answer, "pass_rate": rate})

    rows = problems
    n_roll = len(rows) * args.rollouts
    print(f"\n=== {args.label}: {args.model} [{args.dataset}, thinking={not args.no_thinking}] ===")
    print(f"problems={len(rows)} rollouts/problem={args.rollouts} max_tokens={args.max_tokens}")
    print(f"overall pass rate : {st.mean(pass_rates):.4f}")
    print(f"truncated         : {truncated}/{n_roll} ({truncated / n_roll:.1%})")
    print(
        "completion tokens : mean %.0f  p50 %.0f  p90 %.0f  p99 %.0f  max %d"
        % (
            st.mean(lengths),
            st.quantiles(lengths, n=100)[49],
            st.quantiles(lengths, n=100)[89],
            st.quantiles(lengths, n=100)[98],
            max(lengths),
        )
    )
    buckets = {"0.00": 0, "(0,0.2)": 0, "[0.2,0.8]": 0, "(0.8,1)": 0, "1.00": 0}
    for r in pass_rates:
        key = "0.00" if r == 0 else "1.00" if r == 1 else "[0.2,0.8]" if args.lo <= r <= args.hi else ("(0,0.2)" if r < args.lo else "(0.8,1)")
        buckets[key] += 1
    print("pass-rate buckets :", json.dumps(buckets))
    print(f"IN BAND [{args.lo},{args.hi}] : {len(kept)}/{len(rows)} problems")

    if args.out and kept:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        Dataset.from_list(kept).to_parquet(f"{args.out}/train.parquet")
        print(f"wrote filtered dataset -> {args.out}/train.parquet ({len(kept)} problems)")


asyncio.run(main())
