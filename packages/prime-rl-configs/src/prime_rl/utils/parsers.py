import os
import re

# (regex, parser_name) — first match wins.
TOOL_CALL_PARSER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^deepseek-ai/DeepSeek-V3\.2"), "deepseek_v32"),
    (re.compile(r"^deepseek-ai/DeepSeek-V3\.1"), "deepseek_v31"),
    (re.compile(r"^zai-org/GLM-4\.5"), "glm45"),
    (re.compile(r"^zai-org/GLM-4\.7"), "glm47"),
    (re.compile(r"^zai-org/GLM-5"), "glm47"),
    (re.compile(r"^MiniMaxAI/MiniMax-M2"), "minimax_m2"),
    (re.compile(r"^PrimeIntellect/INTELLECT-3"), "qwen3_coder"),
    (re.compile(r"^nvidia/NVIDIA-Nemotron-3"), "qwen3_coder"),
    (re.compile(r"^stepfun-ai/Step-3\.5"), "step3p5"),
    # Qwen3.5 and Qwen3-Coder use qwen3_coder — must be before the Qwen3 catch-all.
    (re.compile(r"^Qwen/Qwen3\.5-"), "qwen3_coder"),
    (re.compile(r"^Qwen/Qwen3-Coder"), "qwen3_coder"),
    (re.compile(r"^Qwen/Qwen3-"), "hermes"),
]

REASONING_PARSER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^deepseek-ai/DeepSeek-V3\.[12]"), "deepseek_r1"),
    (re.compile(r"^zai-org/GLM-"), "glm45"),
    (re.compile(r"^MiniMaxAI/MiniMax-M2"), "minimax_m2_append_think"),
    (re.compile(r"^PrimeIntellect/INTELLECT-3"), "deepseek_r1"),
    (re.compile(r"^nvidia/NVIDIA-Nemotron-3-Super"), "nemotron_v3"),
    (re.compile(r"^nvidia/NVIDIA-Nemotron-3-Nano"), "nano_v3"),
    (re.compile(r"^stepfun-ai/Step-3\.5"), "step3p5"),
    # Only Qwen3 Thinking models reason — Instruct/base models do not.
    (re.compile(r"^Qwen/Qwen3-.*Thinking"), "deepseek_r1"),
    (re.compile(r"^Qwen/Qwen3\.5-"), "qwen3"),
]


def _normalize(model_name: str) -> str:
    """Return a canonical name that patterns can match against.

    Handles three local path forms:
    1. /scratch/models/Qwen3.5-35B-A3B  → "Qwen/Qwen3.5-35B-A3B"
    2. HF cache: .../models--zai-org--GLM-5.2-FP8/snapshots/<hash>
                                        → "zai-org/GLM-5.2-FP8"
    """
    if not model_name.startswith("/"):
        return model_name
    # Check all path components for HF cache "models--org--name" pattern.
    for part in model_name.split("/"):
        if part.startswith("models--"):
            segments = part[len("models--"):].split("--", 1)
            if len(segments) == 2:
                return f"{segments[0]}/{segments[1]}"
    basename = os.path.basename(model_name.rstrip("/"))
    # Map known basename prefixes to their HF org.
    _ORG_PREFIXES = [
        ("Qwen3.5-", "Qwen/"),
        ("Qwen3-", "Qwen/"),
        ("DeepSeek-", "deepseek-ai/"),
        ("GLM-", "zai-org/"),
        ("MiniMax-", "MiniMaxAI/"),
        ("NVIDIA-Nemotron-", "nvidia/"),
        ("INTELLECT-", "PrimeIntellect/"),
        ("Step-", "stepfun-ai/"),
    ]
    for prefix, org in _ORG_PREFIXES:
        if basename.startswith(prefix):
            return org + basename
    return basename


def _resolve(model_name: str, patterns: list[tuple[re.Pattern[str], str]]) -> str | None:
    normalized = _normalize(model_name)
    for pattern, parser_name in patterns:
        if pattern.search(normalized):
            return parser_name
    return None


def resolve_tool_call_parser(model_name: str) -> str | None:
    return _resolve(model_name, TOOL_CALL_PARSER_PATTERNS)


def resolve_reasoning_parser(model_name: str) -> str | None:
    return _resolve(model_name, REASONING_PARSER_PATTERNS)
