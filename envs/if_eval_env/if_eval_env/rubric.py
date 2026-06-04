import logging

import verifiers as vf

from .registry import score_response

logger = logging.getLogger(__name__)


def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = [msg.get("content", "") for msg in completion if isinstance(msg, dict) and msg.get("role") == "assistant"]
        return "\n".join(parts)
    return str(completion)


class IFEvalRubric(vf.Rubric):
    """Constraint-satisfaction reward for IFEval/IFEvalG examples."""

    def __init__(self):
        super().__init__()
        self.add_reward_func(self.ifeval_score)

    async def ifeval_score(self, completion, answer, info: dict, **kwargs) -> float:
        import json
        text = _extract_text(completion)
        raw_ids = info.get("instruction_ids", "[]")
        raw_kws = info.get("kwargs_list", "[]")
        instruction_ids: list[str] = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
        kwargs_list: list[dict | None] = json.loads(raw_kws) if isinstance(raw_kws, str) else raw_kws
        if not instruction_ids:
            return 0.0
        return score_response(text, instruction_ids, kwargs_list)
