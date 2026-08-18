"""Generate the matched reverse-text configs from base_512.toml.

Two families, each with three arms. Keeping them generated rather than
hand-edited guarantees the only difference between arms is the algorithm block,
which is the whole point of running grpo and sft as controls for opd.

  family "512"  — student = the reverse-text SFT model (max_completion_tokens 512)
                  Result: no headroom. The student already scores ~0.72-0.75 vs
                  the teacher's 0.799, so all three arms sit flat. Kept for the
                  record; not worth re-running.

  family "base" — student = PrimeIntellect/Qwen3-0.6B, the base checkpoint the
                  SFT model was fine-tuned from. Verified to share the teacher's
                  tokenizer, vocab (151669) and chat template exactly, so the
                  teacher scores in-distribution token sequences and there is no
                  renderer confound. This is the family with real headroom:
                  the base model has never seen the reverse-text task.
                  1024 completion tokens because an untuned thinking model
                  rambles far more than the SFT one.
"""

from pathlib import Path

HERE = Path(__file__).parent
BASE = (HERE / "base_512.toml").read_text()
TEACHER = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL"
SFT_STUDENT = "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT"
BASE_STUDENT = "PrimeIntellect/Qwen3-0.6B"

# On-policy distillation: policy samples, per-token reverse KL to the frozen
# teacher. group_size only fans out sampling (no group-relative credit).
OPD = f"""
[orchestrator.algo]
type = "opd"

[orchestrator.algo.teacher]
name = "{TEACHER}"
base_url = ["http://127.0.0.1:8001/v1"]
"""

# Control: same data, group-relative credit instead of the teacher KL.
GRPO = """
[orchestrator.algo]
type = "grpo"
"""

# Upper-bound reference: the teacher generates the rollouts and the student
# trains with plain CE on them. Hard distillation.
SFT = f"""
[orchestrator.algo]
type = "sft"

# sft's teacher IS the model it samples from.
[orchestrator.algo.sampling.source]
name = "{TEACHER}"
base_url = ["http://127.0.0.1:8001/v1"]
"""

ARMS = {"opd": (OPD, 16), "grpo": (GRPO, 16), "sft": (SFT, 4)}

FAMILIES = {
    "512": dict(student=SFT_STUDENT, max_tokens=512, steps=60, seq_len=2048),
    "base": dict(student=BASE_STUDENT, max_tokens=1024, steps=100, seq_len=2048),
}

for fam, f in FAMILIES.items():
    for arm, (block, group_size) in ARMS.items():
        cfg = BASE
        cfg = cfg.replace(f'name = "{SFT_STUDENT}"', f'name = "{f["student"]}"', 1)
        cfg = cfg.replace(
            "group_size = 16  # replaced per-arm by make_512_configs.py",
            f"group_size = {group_size}",
        )
        cfg = cfg.replace("max_completion_tokens = 512", f"max_completion_tokens = {f['max_tokens']}")
        cfg = cfg.replace("max_steps = 60", f"max_steps = {f['steps']}")
        cfg = cfg.replace("seq_len = 2048", f"seq_len = {f['seq_len']}")
        name = f"{arm}_{fam}.toml"
        (HERE / name).write_text(cfg + block)
        print(f"wrote {name:20s} student={f['student'].split('/')[-1]:28s} "
              f"steps={f['steps']} max_tokens={f['max_tokens']} group_size={group_size}")
