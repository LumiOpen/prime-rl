import json
import logging
import random

import verifiers as vf

from .registry import build_checker, score_response

logger = logging.getLogger("verifiers")

_LOG_SAMPLE_RATE = 0.01


def _strip_think_blocks(text: str) -> str:
    """Remove thinking content so scorers only see the final answer.

    Handles three cases:
    1. Full <think>...</think> block in completion — remove it
    2. Completion starts mid-think (chat template pre-fills '<think>', so the
       response begins inside the block with no opening tag): strip up to </think>
    3. No think tags at all (non-think model, or truncated mid-think with no </think>):
       return text as-is for non-think models; return "" only if truncated mid-think
    """
    import re
    # Case 1: full <think>...</think> blocks present — remove them
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Also handle unclosed opening tag (truncated mid-think)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()
    # Case 2: completion is the inside of a think block (no opening tag, has closing tag)
    # Chat template pre-fills '<think>' so response starts mid-reasoning
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
        return text.strip()
    # Case 3: no think tags at all — regular model output, return as-is
    return text.strip()


def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return _strip_think_blocks(completion)
    if isinstance(completion, list):
        parts = [msg.get("content") or "" for msg in completion if msg.get("role") == "assistant"]
        return _strip_think_blocks("\n".join(parts))
    return _strip_think_blocks(str(completion))


class IFEvalRubric(vf.Rubric):
    """Constraint-satisfaction reward for IFEval/IFEvalG examples."""

    def __init__(self):
        super().__init__()
        self.add_reward_func(self.ifeval_score)

    async def ifeval_score(self, completion, answer, info: dict, **kwargs) -> float:
        text = _extract_text(completion)
        raw_ids = info.get("instruction_ids", "[]")
        raw_kws = info.get("kwargs_list", "[]")
        instruction_ids: list[str] = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
        kwargs_list: list[dict | None] = json.loads(raw_kws) if isinstance(raw_kws, str) else raw_kws
        if not instruction_ids:
            return 0.0
        score = score_response(text, instruction_ids, kwargs_list)
        if random.random() < _LOG_SAMPLE_RATE:
            results = []
            for iid, kw in zip(instruction_ids, kwargs_list):
                try:
                    passed = bool(build_checker(iid, kw).check_following(text))
                except Exception:
                    passed = False
                results.append(f"{'PASS' if passed else 'FAIL'} {iid}")
            prompt = info.get("prompt_text", "")
            logger.info(
                "IFEval sample\n"
                f"  prompt ({len(prompt)} chars): {prompt[:300]!r}\n"
                f"  response ({len(text)} chars): {text[:300]!r}\n"
                f"  score: {score:.4f}  constraints: {sum(r.startswith('PASS') for r in results)}/{len(results)}\n"
                + "\n".join(f"    {r}" for r in results)
            )
        return score
