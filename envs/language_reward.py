"""
Language consistency reward using fastText lid.176.bin.

For each rollout:
  1. Detect the question's language.
  2. Check for an explicit target-language marker in the question
     (e.g. "in Finnish", "suomeksi", "translate to X") — if found, that
     overrides the detected question language.
  3. Detect the answer's language (after stripping think blocks).
  4. Return a confidence-weighted score: 1.0 if languages match with high
     confidence, graduating down toward 0.0 as confidence drops or languages
     diverge.

The fastText LID model (lid.176.bin) is downloaded automatically on first use
from https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin.
Override the path via FASTTEXT_LID_PATH env var.

If the model cannot be downloaded or loaded, language reward is silently
disabled — all functions return neutral values (1.0) and a one-time warning
is logged. This prevents a missing model from breaking an experiment.

Usage (as a zero-weight monitoring metric, add to a Rubric):
    from envs.language_reward import make_language_reward_func
    rubric.add_metric(make_language_reward_func(), weight=0.0)

Or as a multiplicative post-factor (apply to reward before advantage):
    lang_score = compute_language_score(question, answer)
    reward *= lang_score
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger("verifiers.v1")

_FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# Path to the fastText LID model — override via FASTTEXT_LID_PATH env var.
_DEFAULT_MODEL_PATH = Path(os.environ.get("FASTTEXT_LID_PATH", str(Path(__file__).parent.parent / "lid.176.bin")))

# Thread-local model instance (fasttext models are not thread-safe to load in parallel).
_local = threading.local()
_disable_lock = threading.Lock()
_disabled = False  # set to True permanently if model cannot be loaded after download attempt


def _warn_disabled(reason: str) -> None:
    logger.warning("Language reward disabled: %s — all scores will return 1.0 (no penalty).", reason)


def _try_download(path: str) -> bool:
    """Attempt to download lid.176.bin. Returns True on success."""
    import urllib.request
    logger.warning("fastText LID model not found at %s — downloading from %s", path, _FASTTEXT_URL)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(_FASTTEXT_URL, path)
        logger.info("Downloaded fastText LID model to %s (%d MB)", path, os.path.getsize(path) // 1_000_000)
        return True
    except Exception as exc:
        logger.warning("Download failed: %s", exc)
        return False


def _get_model(model_path: str | None = None):
    global _disabled
    if _disabled:
        return None

    path = model_path or str(_DEFAULT_MODEL_PATH)

    if not hasattr(_local, "model") or _local.model_path != path:
        # Ensure model file exists, downloading if needed.
        if not os.path.exists(path):
            with _disable_lock:
                if _disabled:
                    return None
                if not _try_download(path):
                    _disabled = True
                    _warn_disabled(f"could not download model to {path}")
                    return None

        try:
            import fasttext
            import numpy as np
            # Patch numpy 2.x incompatibility in fasttext.
            if not getattr(np, "_fasttext_patched", False):
                _orig = np.array
                np.array = lambda obj, *a, copy=True, **kw: (
                    np.asarray(obj, *a, **kw) if copy is False else _orig(obj, *a, copy=copy, **kw)
                )
                np._fasttext_patched = True
            _local.model = fasttext.load_model(path)
            _local.model_path = path
        except Exception as exc:
            with _disable_lock:
                _disabled = True
            _warn_disabled(f"failed to load {path}: {exc}")
            return None

    return _local.model


def detect_language(text: str, model_path: str | None = None) -> tuple[str, float]:
    """Return (iso_code, confidence) for the dominant language in text."""
    model = _get_model(model_path)
    if model is None:
        return "unknown", 0.0
    text = text.replace("\n", " ").strip()[:500]
    if not text or not any(c.isalpha() for c in text):
        return "unknown", 0.0
    labels, probs = model.predict(text, k=1)
    lang = labels[0].replace("__label__", "")
    conf = float(probs[0])
    return lang, conf


# Maps natural-language target markers to ISO 639-1 codes.
_TARGET_LANG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bin finnish\b|\bsuomeksi\b|\bsuomen kielell[äa]\b|\bkäännä suomeksi\b', re.I), "fi"),
    (re.compile(r'\bin english\b|\btranslate to english\b|\bin the english language\b', re.I), "en"),
    (re.compile(r'\bin french\b|\ben français\b|\btranslate to french\b', re.I), "fr"),
    (re.compile(r'\bin german\b|\bauf deutsch\b|\btranslate to german\b', re.I), "de"),
    (re.compile(r'\bin spanish\b|\ben español\b|\btranslate to spanish\b', re.I), "es"),
    (re.compile(r'\bin swedish\b|\bpå svenska\b|\btranslate to swedish\b', re.I), "sv"),
    (re.compile(r'\bin chinese\b|\b用中文\b|\btranslate to chinese\b', re.I), "zh"),
    (re.compile(r'\bin japanese\b|\b日本語で\b|\btranslate to japanese\b', re.I), "ja"),
]


def detect_target_language(question: str) -> str | None:
    """Return explicit target language ISO code if question specifies one, else None."""
    for pattern, lang in _TARGET_LANG_PATTERNS:
        if pattern.search(question):
            return lang
    return None


def compute_language_score(
    question: str,
    answer: str,
    model_path: str | None = None,
    min_answer_len: int = 10,
    conf_threshold: float = 0.5,
) -> float:
    """Return a [0, 1] score: 1.0 if answer language matches expected, 0.0 otherwise.

    Returns 1.0 (no penalty) when:
      - The model is unavailable (disabled after failed download/load).
      - The answer is too short to reliably detect language.
      - Detection confidence is below conf_threshold.
    """
    if _disabled:
        return 1.0

    answer = answer.strip()
    if len(answer) < min_answer_len:
        return 1.0

    explicit_target = detect_target_language(question)
    if explicit_target:
        expected_lang = explicit_target
    else:
        q_lang, q_conf = detect_language(question, model_path)
        if q_conf < conf_threshold:
            return 1.0
        expected_lang = q_lang

    a_lang, a_conf = detect_language(answer, model_path)
    if a_conf < conf_threshold:
        return 1.0

    return float(a_conf) if a_lang == expected_lang else 0.0


def make_language_reward_func(model_path: str | None = None, log_sample_rate: float = 0.05):
    """Return an individual reward function compatible with verifiers Rubric.

    Add as weight=0 metric for monitoring, or weight>0 to affect training.
    """
    def language_consistency_reward(completion, answer, info: dict, **kwargs) -> float:
        try:
            from rubric_judge_env.rubric import _extract_text
            answer_text = _extract_text(completion)
        except ImportError:
            if isinstance(completion, list):
                parts = [m.get("content", "") for m in completion if m.get("role") == "assistant"]
                answer_text = "\n".join(parts)
            else:
                answer_text = str(completion)

        question = info.get("prompt_text", "")
        if not question:
            prompt = kwargs.get("prompt", [])
            question = prompt[-1].get("content", "") if prompt else ""

        score = compute_language_score(question, answer_text, model_path=model_path)

        import random
        if random.random() < log_sample_rate:
            q_lang, q_conf = detect_language(question) if question else ("?", 0.0)
            a_lang, a_conf = detect_language(answer_text) if answer_text.strip() else ("?", 0.0)
            explicit = detect_target_language(question)
            logger.info(
                "Language reward sample\n"
                f"  question lang: {q_lang} ({q_conf:.2f}) | explicit target: {explicit}\n"
                f"  answer lang:   {a_lang} ({a_conf:.2f})\n"
                f"  score: {score:.2f}"
            )

        return score

    language_consistency_reward.__name__ = "language_consistency_reward"
    return language_consistency_reward