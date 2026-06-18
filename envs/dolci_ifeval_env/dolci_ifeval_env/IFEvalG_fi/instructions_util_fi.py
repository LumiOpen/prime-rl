# Finnish-adapted utilities for IFEvalG.
#
# All helpers are re-exported from the original instructions_util unchanged,
# except count_sentences, which is replaced with a version that uses
# NLTK's Finnish Punkt tokenizer (with fallback to the English one if the
# Finnish model is not installed).

import functools

import nltk

from ifeval_env.IFEvalG.instructions_util import (  # re-export unchanged helpers
    WORD_LIST,
    LANGUAGE_CODES,
    split_into_sentences,
    count_words,
    generate_keywords,
)

__all__ = [
    "WORD_LIST",
    "LANGUAGE_CODES",
    "split_into_sentences",
    "count_words",
    "generate_keywords",
    "count_sentences",
]


@functools.cache
def _get_sentence_tokenizer():
    try:
        return nltk.data.load("nltk:tokenizers/punkt/finnish.pickle")
    except LookupError:
        # Finnish punkt model not installed; fall back to English.
        return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def count_sentences(text):
    """Count sentences using the Finnish Punkt tokenizer."""
    tokenizer = _get_sentence_tokenizer()
    return len(tokenizer.tokenize(text))
