import json
import logging
import random

import verifiers as vf

from .registry import build_checker, score_response

logger = logging.getLogger("verifiers")

_LOG_SAMPLE_RATE = 0.01


def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = [msg.get("content") or "" for msg in completion if msg.get("role") == "assistant"]
        return "\n".join(parts)
    return str(completion)


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
