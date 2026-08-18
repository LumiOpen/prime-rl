"""Compare the OPD student and teacher on reverse-text, then measure the
per-token reverse KL the OPD loss actually consumes.

Behavioural comparison tells us whether the two models differ on the task.
The KL measurement tells us whether they differ *on the student's own
trajectories* — which is the only thing OPD's gradient can see.
"""

import asyncio
import statistics as st
import sys
from difflib import SequenceMatcher

from datasets import load_dataset
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from prime_rl.utils.client import prefill_logprobs

STUDENT = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
TEACHER = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL"
SYSTEM = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."
N = 64
MAX_TOKENS = 128

s_port, t_port = sys.argv[1], sys.argv[2]


def lcs(response: str, answer: str) -> float:
    import re

    m = re.search(r"<reversed_text>(.*?)</reversed_text>", response or "", re.DOTALL)
    return SequenceMatcher(None, m.group(1).strip() if m else "", answer).ratio()


async def sample(client, model, prompt):
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=1.0,
    )
    msg = r.choices[0].message
    return (msg.content or ""), (getattr(msg, "reasoning_content", None) or ""), r.choices[0].finish_reason


async def profile(client, model, rows, label):
    outs = await asyncio.gather(*(sample(client, model, r["prompt"]) for r in rows))
    rew = [lcs(c, r["prompt"][::-1]) for (c, _, _), r in zip(outs, rows)]
    thinking = sum(1 for _, rc, _ in outs if rc.strip())
    closed = sum(1 for c, _, _ in outs if "</reversed_text>" in c)
    trunc = sum(1 for _, _, f in outs if f == "length")
    print(f"  {label:<8} reward={st.mean(rew):.3f}  thinking={thinking}/{len(outs)}  "
          f"closing-tag={closed}/{len(outs)}  truncated={trunc}/{len(outs)}")
    return outs, rew


async def main():
    rows = list(load_dataset("PrimeIntellect/Reverse-Text-RL", split="train"))[:N]
    sc = AsyncOpenAI(base_url=f"http://127.0.0.1:{s_port}/v1", api_key="EMPTY")
    tc = AsyncOpenAI(base_url=f"http://127.0.0.1:{t_port}/v1", api_key="EMPTY")

    print(f"\n=== Behaviour on {N} reverse-text prompts (max_tokens={MAX_TOKENS}, T=1.0) ===")
    s_outs, _ = await profile(sc, STUDENT, rows, "student")
    await profile(tc, TEACHER, rows, "teacher")

    # The KL OPD actually optimizes: teacher scores the STUDENT's own tokens.
    print("\n=== Per-token reverse KL on the student's own trajectories ===")
    print("    (this is exactly what ref_kl_loss_fn consumes)")
    tok = AutoTokenizer.from_pretrained(STUDENT)
    kls, per_seq = [], []
    for (content, reasoning, _), row in list(zip(s_outs, rows))[:24]:
        reply = (f"<think>{reasoning}</think>" if reasoning.strip() else "") + content
        ids = list(
            tok.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": reply},
                ],
                tokenize=True,
                return_dict=False,
            )
        )
        s_lp = await prefill_logprobs(sc, STUDENT, ids)
        t_lp = await prefill_logprobs(tc, TEACHER, ids)
        # ref_kl = log pi_ref - log pi_theta, over the completion tail only.
        tail = max(1, len(ids) - 40)
        d = [t - s for t, s in zip(t_lp[tail:], s_lp[tail:])]
        kls.extend(d)
        per_seq.append(st.mean(d))

    print(f"  tokens measured : {len(kls)}")
    print(f"  mean ref_kl     : {st.mean(kls):+.4f}   (0 => teacher and student agree, no gradient)")
    print(f"  median ref_kl   : {st.median(kls):+.4f}")
    print(f"  stdev ref_kl    : {st.pstdev(kls):.4f}")
    print(f"  |ref_kl| mean   : {st.mean([abs(x) for x in kls]):.4f}")
    print(f"  per-seq mean    : min={min(per_seq):+.3f} max={max(per_seq):+.3f}")
    big = sum(1 for x in kls if abs(x) > 1.0)
    print(f"  |ref_kl| > 1.0  : {big}/{len(kls)} ({big / len(kls):.1%})")


asyncio.run(main())
