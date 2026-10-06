"""Smoke-test repetition-env: dataset loads, and the reward separates looping from clean text.

uv run python scripts/check_repetition_env.py
"""

from repetition_env import load_environment
from repetition_env.repetition_env import RepetitionRubric

THINKING = (
    "Let me work through this. The user wants a short summary of the photosynthesis "
    "process. I should mention light absorption, water splitting, and carbon fixation, "
    "then keep the answer under five sentences."
)

ANSWER = (
    "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs "
    "photons, which drive the splitting of water and release oxygen. The resulting "
    "electrons and protons power the synthesis of ATP and NADPH. Those carriers then "
    "fuel the Calvin cycle, which fixes carbon dioxide into sugars."
)

LOOP = "Wait, let me reconsider the problem carefully.\n\n" * 31

# A loop buried in an otherwise healthy rollout: the binary detector sees the same
# thing as a pure loop, the continuous reward sees a partial one.
PARTIAL_LOOP = THINKING + "\n\n" + "Wait, let me reconsider the problem carefully.\n\n" * 12

# (name, message, output_tokens, reward bounds)
CASES = [
    ("clean", {"content": f"<think>{THINKING}</think>{ANSWER}"}, 120, (1.0, 1.0)),
    ("loop", {"content": f"<think>{LOOP}</think>The answer is 42."}, 400, (0.0, 0.05)),
    ("partial_loop", {"content": f"<think>{PARTIAL_LOOP}</think>The answer is 42."}, 400, (0.05, 0.5)),
    ("short", {"content": "<think></think>42"}, 5, (0.0, 0.0)),
    # Long enough to clear the token floor, but one side of </think> is empty.
    ("no_think", {"content": f"<think>  </think>{ANSWER}"}, 120, (0.0, 0.0)),
    ("no_answer", {"content": f"<think>{THINKING}</think>   "}, 120, (0.0, 0.0)),
    ("unclosed", {"content": THINKING}, 120, (0.0, 0.0)),
    # vLLM reasoning parser: thinking arrives out-of-band, never in `content`.
    ("parsed", {"reasoning_content": THINKING, "content": ANSWER}, 120, (1.0, 1.0)),
    ("parsed_loop", {"reasoning_content": LOOP, "content": "The answer is 42."}, 400, (0.0, 0.05)),
    ("parsed_unclosed", {"reasoning_content": LOOP, "content": None}, 400, (0.0, 0.0)),
]


def main():
    env = load_environment()
    ds = env.build_dataset()
    print(f"dataset rows: {len(ds)}")
    print(f"first prompt:  {ds[0]['prompt']}")

    (rubric,) = [r for r in getattr(env.rubric, "rubrics", [env.rubric]) if isinstance(r, RepetitionRubric)]
    print(f"funcs: {[f.__name__ for f in rubric.funcs]}")
    print(f"weights: {rubric.weights}")

    failures = []
    for name, message, usage, (lo, hi) in CASES:
        completion = [{"role": "assistant", **message}]
        state = {"usage": {"output_tokens": usage}, "is_truncated": False}
        reward = rubric.no_repetition(completion, state)
        print(
            f"{name:16s} reward={reward:.3f} (want [{lo:.2f}, {hi:.2f}])  "
            f"blockcov={rubric.block_coverage(completion, state):.3f}  "
            f"maxrep={rubric.max_block_repeats(completion, state):.0f}  "
            f"detected={rubric.repetition_detected(completion, state):.0f}  "
            f"think={rubric.has_reasoning(completion, state):.0f}  "
            f"answer={rubric.has_answer(completion, state):.0f}  "
            f"ngram_frac={rubric.repeated_ngram_fraction(completion, state):.3f}"
        )
        if not lo <= reward <= hi:
            failures.append(name)

    assert not failures, f"unexpected reward for: {failures}"
    print("OK")


if __name__ == "__main__":
    main()
