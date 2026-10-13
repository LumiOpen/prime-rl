"""Language-consistency reward, shared by the instruction-following and judge envs.

For one rollout:

1. detect the language of the question,
2. detect the language of the model's text,
3. score the detection confidence when the two match, 0.0 when they do not.

The score is applied multiplicatively:

    reward *= (1 - weight) + weight * language_score

so ``weight = 0.2`` caps the penalty at 20% of the reward and ``weight = 0`` leaves
the reward untouched. The raw score is always written to ``state`` and exported as
the ``language_score`` metric, so a zero-weight run still measures the effect.

Detection uses fastText's ``lid.176.bin`` (126 MB), which is not in the repo. Fetch
it once with ``uv run python -m language_reward`` and point ``FASTTEXT_LID_PATH`` at
the result. Without the model every score is a neutral 1.0, so an env with a non-zero
weight calls :func:`resolve_language_reward` at startup and refuses to run rather than
training on a reward that silently does nothing.
"""

from __future__ import annotations

import logging
import os
import random
import threading
from pathlib import Path

logger = logging.getLogger("verifiers.v1")

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
# Repo root, matching the lid.176.bin entry in .gitignore.
DEFAULT_MODEL_PATH = os.environ.get(
    "FASTTEXT_LID_PATH", str(Path(__file__).resolve().parents[3] / "lid.176.bin")
)

# Key under which the raw score is staged in the rollout state, and the scopes an
# env may score: the final answer, or the whole completion including the reasoning.
LANGUAGE_SCORE_KEY = "language_score"
LANGUAGE_SCOPES = ("answer", "full")

_MAX_DETECT_CHARS = 500
_MIN_TEXT_LEN = 10
_CONF_THRESHOLD = 0.5
_LOG_SAMPLE_RATE = 0.01

# fastText models are not safe to load concurrently; keep one per worker thread.
_local = threading.local()
_load_lock = threading.Lock()
_warned_missing = False


def _load_model(model_path: str | None = None):
    """Return the fastText LID model, or None when the model file is absent."""
    global _warned_missing
    path = model_path or DEFAULT_MODEL_PATH
    if getattr(_local, "model_path", None) != path:
        if not os.path.exists(path):
            if not _warned_missing:
                _warned_missing = True
                logger.warning(
                    "fastText LID model not found at %s — every language score will be a "
                    "neutral 1.0. Fetch it with `uv run python -m language_reward` or point "
                    "FASTTEXT_LID_PATH at an existing copy.",
                    path,
                )
            return None
        import fasttext

        with _load_lock:
            _local.model = fasttext.load_model(path)
        _local.model_path = path
    return _local.model


def detect_language(text: str, model_path: str | None = None) -> tuple[str, float]:
    """Return (iso code, confidence) for the dominant language of the first 500 chars."""
    model = _load_model(model_path)
    if model is None:
        return "unknown", 0.0
    text = text.replace("\n", " ").strip()[:_MAX_DETECT_CHARS]
    if not text or not any(c.isalpha() for c in text):
        return "unknown", 0.0
    labels, probs = model.predict(text, k=1)
    return labels[0].removeprefix("__label__"), float(probs[0])


def _score_language(
    question: str, text: str, model_path: str | None = None
) -> tuple[float, tuple[str, float], tuple[str, float]]:
    """(score, question detection, text detection)."""
    if len(text.strip()) < _MIN_TEXT_LEN:
        return 1.0, ("short", 0.0), ("short", 0.0)
    q = detect_language(question, model_path)
    a = detect_language(text, model_path)
    if q[1] < _CONF_THRESHOLD or a[1] < _CONF_THRESHOLD:
        return 1.0, q, a
    return (a[1] if a[0] == q[0] else 0.0), q, a


def compute_language_score(question: str, text: str, model_path: str | None = None) -> float:
    """Score in [0, 1]: the detection confidence when *text* is in the question's
    language, 0.0 when it is not, and a neutral 1.0 when either side is too short or
    too uncertain to judge (or the LID model is missing).
    """
    return _score_language(question, text, model_path)[0]


def select_text(scope: str, answer_text: str, full_text: str) -> str:
    """Pick the text to score: the final answer, or the completion including reasoning."""
    if scope == "answer":
        return answer_text
    if scope == "full":
        return full_text
    raise ValueError(f"language_reward_scope must be one of {LANGUAGE_SCOPES}, got {scope!r}")


def resolve_language_reward(
    weight: float, scope: str | None, finnish: bool = False, model_path: str | None = None
) -> str:
    """Validate an env's language-reward config at startup and return the scope to score.

    Finnish arms default to "full" — the whole point of training on Finnish prompts is
    that the model thinks in Finnish too, and a trace reasoned in English before a
    Finnish answer passes "answer" untouched. English arms default to "answer", where
    the reasoning language is not what we are buying.
    """
    if scope is None:
        scope = "full" if finnish else "answer"
    if scope not in LANGUAGE_SCOPES:
        raise ValueError(f"language_reward_scope must be one of {LANGUAGE_SCOPES}, got {scope!r}")
    if weight > 0.0 and _load_model(model_path) is None:
        raise RuntimeError(
            f"language_reward_weight={weight} but the fastText LID model is missing at "
            f"{model_path or DEFAULT_MODEL_PATH}. Fetch it with `uv run python -m language_reward` "
            "or set FASTTEXT_LID_PATH; without it every language score is a neutral 1.0 and the "
            "reward would silently do nothing."
        )
    return scope


def apply_language_reward(
    score: float,
    question: str,
    text: str,
    weight: float,
    state: dict | None = None,
    model_path: str | None = None,
) -> float:
    """Stage the language score in *state* and scale *score* by it.

    ``weight = 0`` is monitoring only: the metric is still exported, the reward is
    returned unchanged.
    """
    lang_score, (q_lang, q_conf), (a_lang, a_conf) = _score_language(question, text, model_path)
    if state is not None:
        state[LANGUAGE_SCORE_KEY] = lang_score
    scaled = score if weight <= 0.0 else score * (1.0 - weight + weight * lang_score)
    if random.random() < _LOG_SAMPLE_RATE:
        logger.info(
            "Language sample\n"
            f"  question: {q_lang} ({q_conf:.2f})\n"
            f"  text:     {a_lang} ({a_conf:.2f})  {len(text)} chars\n"
            f"  language score: {lang_score:.2f}  reward: {score:.4f} -> {scaled:.4f}"
        )
    return scaled


def language_score(completion, answer, info, state=None, **kwargs) -> float:
    """Rubric metric (weight 0) exposing the raw language score of the last rollout."""
    return (state or {}).get(LANGUAGE_SCORE_KEY, 1.0)


