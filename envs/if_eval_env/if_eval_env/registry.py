# Maps IFEval/IFEvalG instruction_ids to (checker_factory, kwarg_keys) tuples.
# checker_factory(**kwargs) -> object with .check_following(response_text) -> bool
# For stateless checkers (no constructor args), checker_factory is the class itself
# and kwarg_keys is empty.

import re
import json
import langdetect

from . import if_functions as F
from . import ifeval_g as G


# ---------------------------------------------------------------------------
# Thin stateless wrappers so every checker has the same .check_following(text) API
# ---------------------------------------------------------------------------

class _StatelessChecker:
    def __init__(self, fn, **kwargs):
        self._fn = fn
        self._kwargs = kwargs

    def check_following(self, text: str) -> bool:
        result = self._fn(text, **self._kwargs)
        # Some original functions return tuple (bool, list) — normalise to bool
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)


def _make(fn, **kwargs):
    return _StatelessChecker(fn, **kwargs)


# ---------------------------------------------------------------------------
# Factory: given instruction_id + kwargs dict from ground_truth, return a checker
# ---------------------------------------------------------------------------

def _parse_ground_truth(gt) -> tuple[list[str], list[dict | None]]:
    """
    Parse the ground_truth field from Dolci-Think-RL-7B.
    gt is a numpy array whose first element is a Python-literal string.
    Returns (instruction_ids, kwargs_list).
    """
    import ast, logging
    logger = logging.getLogger(__name__)
    try:
        raw = gt[0] if hasattr(gt, "__getitem__") else gt
        parsed = ast.literal_eval(raw)
        ids, kws = [], []
        for entry in parsed:
            for iid, kw in zip(entry["instruction_id"], entry["kwargs"]):
                ids.append(iid)
                kws.append(kw)
        return ids, kws
    except Exception as e:
        logger.debug("Failed to parse ground_truth %r: %s", gt, e)
        return [], []


def build_checker(instruction_id: str, kwargs: dict | None):
    """Return a checker object for the given instruction_id and kwargs."""
    # Strip None-valued keys so checkers can use .get() safely
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None}
    iid = instruction_id

    # --- Original IFEval 25 constraints ---
    if iid == "keywords:existence":
        return _make(F.verify_keywords, keyword_list=kw["keywords"])
    if iid == "keywords:frequency":
        return _make(F.verify_keyword_frequency, word=kw["keyword"], N=kw["frequency"])
    if iid == "keywords:forbidden_words":
        return _make(F.validate_forbidden_words, forbidden_words=kw["forbidden_words"])
    if iid == "keywords:letter_frequency":
        return _make(F.verify_letter_frequency, letter=kw["letter"], N=kw["let_frequency"])
    if iid == "language:response_language":
        return _make(F.validate_response_language, language=kw["language"])
    if iid == "length_constraints:number_paragraphs":
        return _make(F.verify_paragraph_count, N=kw["num_paragraphs"])
    if iid == "length_constraints:number_words":
        return _make(F.validate_word_constraint, N=kw["num_words"], quantifier=kw["relation"])
    if iid == "length_constraints:number_sentences":
        return _make(F.verify_sentence_constraint, N=kw["num_sentences"], quantifier=kw["relation"])
    if iid == "length_constraints:nth_paragraph_first_word":
        return _make(F.validate_paragraphs,
                     N=kw["num_paragraphs"], first_word=kw["first_word"], i=kw["nth_paragraph"])
    if iid == "detectable_content:postscript":
        return _make(F.verify_postscript, postscript_marker=kw["postscript_marker"])
    if iid == "detectable_content:number_placeholders":
        return _make(F.validate_placeholders, N=kw["num_placeholders"])
    if iid == "detectable_format:number_bullet_lists":
        return _make(F.verify_bullet_points, N=kw["num_bullets"])
    if iid == "detectable_format:title":
        return _make(F.validate_title)
    if iid == "detectable_format:constrained_response":
        options = kw.get("constrained_responses", kw.get("options", []))
        return _make(F.validate_choice, options=options)
    if iid == "detectable_format:number_highlighted_sections":
        return _make(F.validate_highlighted_sections, N=kw["num_highlights"])
    if iid == "detectable_format:multiple_sections":
        return _make(F.validate_sections,
                     N=kw["num_sections"], section_splitter=kw["section_splitter"])
    if iid == "detectable_format:json_format":
        return _make(F.validate_json_format)
    if iid == "combination:repeat_prompt":
        return _make(F.validate_repeat_prompt, original_prompt=kw.get("original_prompt", ""))
    if iid == "combination:two_responses":
        return _make(F.validate_two_responses)
    if iid == "startend:end_checker":
        return _make(F.validate_end, end_phrase=kw["end_phrase"])
    if iid == "change_case:capital_word_frequency":
        return _make(F.validate_frequency_capital_words,
                     N=kw["capital_frequency"], quantifier=kw["relation"])
    if iid == "change_case:english_capital":
        return _make(F.validate_uppercase)
    if iid == "change_case:english_lowercase":
        return _make(F.validate_lowercase)
    if iid == "punctuation:no_comma":
        return _make(F.validate_no_commas)
    if iid == "startend:quotation":
        return _make(F.validate_quotation)

    # --- IFEvalG new constraints ---
    if iid == "detectable_format:bigram_wrapping":
        return G.BiGramWrappingChecker()
    if iid == "detectable_format:square_brackets":
        return G.SquareBracketChecker()
    if iid == "detectable_format:sentence_hyphens":
        return G.SentenceHyphenChecker()
    if iid == "first_word:first_word_sent":
        return G.FirstWordSentChecker(first_word=kw["first_word"])
    if iid == "first_word:first_word_answer":
        return G.FirstWordAnswerChecker(first_word=kw["first_word"])
    if iid == "last_word:last_word_sent":
        return G.LastWordSentChecker(last_word=kw["last_word"])
    if iid == "last_word:last_word_answer":
        return G.LastWordAnswerChecker(last_word=kw["last_word"])
    if iid == "keywords:word_once":
        return G.KeywordFrequencyOnceChecker(keyword=kw["keyword"])
    if iid == "keywords:exclude_word_harder":
        return G.ExcludeWordHarderChecker(keyword=kw["keyword"])
    if iid == "keywords:start_end":
        return G.StartEndChecker()
    if iid == "count:counting_composition":
        return G.CountingCompositionChecker(n_sent=kw["n_sent"], n_words=kw["n_words"])
    if iid == "keywords:palindrome":
        return G.PalindromeBasicChecker()
    if iid == "keywords:keyword_specific_position":
        return G.KeywordSpecificPositionChecker(
            keyword=kw["keyword"], n=kw["n"], m=kw["m"])
    if iid == "keywords:no_adjacent_consecutive":
        return G.AdjacentLetterChecker()
    if iid == "count:count_unique":
        return G.CountUniqueChecker()
    if iid == "count:count_increment_word":
        return G.CountIncrementWordChecker(keyword1=kw["keyword1"], keyword2=kw["keyword2"])
    if iid == "count:lowercase_counting":
        return G.LowercaseCountingChecker(N=kw["N"])
    if iid == "letters:letter_counting":
        return G.LetterCountingChecker(N=kw["N"], relation=kw["relation"])
    if iid == "letters:letter_counting2":
        # Reuses LetterFrequencyChecker logic = verify_letter_frequency
        return _make(F.verify_letter_frequency, letter=kw["letter"], N=kw["N"])
    if iid == "punctuation:punctuation_dot":
        return G.PunctuationDotChecker()
    if iid == "punctuation:punctuation_exclamation":
        return G.PunctuationExclamationChecker()
    if iid == "paragraphs:paragraphs":
        return G.ParagraphBasicChecker()
    if iid == "paragraphs:paragraphs2":
        return G.ParagraphBasicChecker2()
    if iid == "new:copy_span_idx":
        return G.CopySpanIdxChecker(
            prompt_to_repeat=kw["prompt_to_repeat"],
            n_start=kw["n_start"],
            n_end=kw["n_end"],
        )
    if iid == "keywords:word_count_different_numbers":
        return G.KeywordFrequencyCheckerDifferent(
            keyword1=kw["keyword1"], freq1=kw["freq1"],
            keyword2=kw["keyword2"], freq2=kw["freq2"],
        )
    # copy: constraints — these require verbatim copying; treat as pass-through
    if iid in ("copy:copy", "copy:copying_simple", "copy:copying_multiple", "copy:repeat_phrase"):
        text_to_copy = kw.get("text_to_copy", kw.get("phrase", ""))
        return _make(lambda text, s=text_to_copy: s.strip().lower() in text.strip().lower())

    raise ValueError(f"Unknown instruction_id: {instruction_id!r}")


def score_response(response: str, instruction_ids: list[str], kwargs_list: list[dict | None]) -> float:
    """
    Score a single response against a list of constraints.
    Returns fraction of constraints satisfied in [0.0, 1.0].
    """
    if not instruction_ids:
        return 0.0

    scores = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        try:
            checker = build_checker(iid, kw)
            scores.append(float(checker.check_following(response)))
        except Exception:
            # Unknown / malformed constraint: treat as 0 rather than crashing
            scores.append(0.0)

    return sum(scores) / len(scores)
