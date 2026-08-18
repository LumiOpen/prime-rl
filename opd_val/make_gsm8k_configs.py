"""Generate the matched GSM8K arms from base_gsm8k.toml.

Same discipline as the reverse-text families: one shared base, the algorithm
block appended, so the ONLY difference between arms is the algorithm. That is
what made the earlier comparisons readable and what turned "opd is broken" into
"the trainer is broken" when the grpo control reproduced the failure.
"""

from pathlib import Path

HERE = Path(__file__).parent
BASE = (HERE / "base_gsm8k.toml").read_text()
TEACHER = "Qwen/Qwen3-8B"
TEACHER_URL = '["http://127.0.0.1:8001/v1"]'

OPD = f"""
[orchestrator.algo]
type = "opd"

# 0 = the sampled-token score-function estimator. The top-k support machinery is
# implemented and verified (see FINDINGS.md §5) but did not rescue opd on
# reverse-text, so start from the default here and sweep k only if the baseline
# arm shows life.
teacher_top_k = 0

[orchestrator.algo.teacher]
name = "{TEACHER}"
base_url = {TEACHER_URL}
"""

GRPO = """
[orchestrator.algo]
type = "grpo"
"""

SFT = f"""
[orchestrator.algo]
type = "sft"

# sft's teacher IS the model it samples from — the empirical ceiling for how
# much of the teacher's advantage is transferable at this budget.
[orchestrator.algo.sampling.source]
name = "{TEACHER}"
base_url = {TEACHER_URL}
"""

# grpo needs a group to compute a baseline over; sft trains on teacher tokens
# and gains nothing from fan-out.
ARMS = {"opd": (OPD, 16), "grpo": (GRPO, 16), "sft": (SFT, 4)}

for arm, (block, group_size) in ARMS.items():
    cfg = BASE.replace(
        "group_size = 16  # replaced per-arm by make_gsm8k_configs.py",
        f"group_size = {group_size}",
    )
    name = f"gsm8k_{arm}.toml"
    (HERE / name).write_text(cfg + block)
    print(f"wrote {name:20s} group_size={group_size}")
