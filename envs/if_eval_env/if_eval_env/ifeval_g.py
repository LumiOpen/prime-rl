# Copyright 2024 The Google Research Authors. Apache License 2.0
# Ported from https://github.com/allenai/open-instruct/blob/main/open_instruct/IFEvalG/
# Covers the 29 new IFEvalG constraints used in Dolci IF-RLVR training.

import functools
import re


# ---------------------------------------------------------------------------
# Sentence / tokenization utilities (inline, no external IFEvalG dep)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _punkt_tokenizer():
    import nltk
    try:
        return nltk.data.load("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return nltk.data.load("tokenizers/punkt/english.pickle")


@functools.lru_cache(maxsize=1)
def _word_tokenizer():
    import nltk
    try:
        return nltk.tokenize.RegexpTokenizer(r"\w+")
    except Exception:
        nltk.download("punkt", quiet=True)
        return nltk.tokenize.RegexpTokenizer(r"\w+")


def split_into_sentences(text: str) -> list[str]:
    tokenizer = _punkt_tokenizer()
    return tokenizer.tokenize(text)


def word_tokenize(text: str) -> list[str]:
    return _word_tokenizer().tokenize(text)


# ---------------------------------------------------------------------------
# Checker classes (one per IFEvalG instruction type)
# ---------------------------------------------------------------------------

class BiGramWrappingChecker:
    def check_following(self, value):
        words = value.split()
        for i in range(0, len(words) - 1, 2):
            if i + 1 < len(words) and not (words[i].startswith("<<") and words[i + 1].endswith(">>")):
                return False
        return True


class SquareBracketChecker:
    def check_following(self, value):
        words = value.split()
        return all(word.startswith("[") and word.endswith("]") for word in words)


class SentenceHyphenChecker:
    def check_following(self, value):
        sentences_gold = re.sub("-", " ", value)
        sentences_gold = split_into_sentences(sentences_gold)
        sentences = value.split("-")
        for sentence, gold in zip(sentences, sentences_gold):
            if sentence.strip() != sentence or sentence != gold:
                return False
        return True


class FirstWordSentChecker:
    def __init__(self, first_word: str):
        self._first_word = first_word

    def check_following(self, value):
        sentences = split_into_sentences(value)
        for sentence in sentences:
            if not sentence.strip():
                return False
            first_word = sentence.split()[0].strip()
            if first_word.lower() != self._first_word.lower():
                return False
        return True


class FirstWordAnswerChecker:
    def __init__(self, first_word: str):
        self._first_word = first_word

    def check_following(self, value):
        if not value.strip() or len(value.split()) == 0:
            return False
        first_word = value.split()[0].strip()
        return first_word.lower() == self._first_word.lower()


class LastWordSentChecker:
    def __init__(self, last_word: str):
        self._last_word = last_word

    def check_following(self, value):
        sentences = split_into_sentences(value)
        for sentence in sentences:
            if not sentence.strip():
                return False
            last_word = sentence.split()[-1].strip()
            last_word = re.sub(r"[^\w\s]", "", last_word)
            if last_word.lower() != self._last_word.lower():
                return False
        return True


class LastWordAnswerChecker:
    def __init__(self, last_word: str):
        self._last_word = last_word

    def check_following(self, value):
        last_word = value.split()[-1].strip()
        last_word = re.sub(r"[^\w\s]", "", last_word)
        return last_word.lower() == self._last_word.lower()


class KeywordFrequencyOnceChecker:
    def __init__(self, keyword: str):
        self._keyword = keyword

    def check_following(self, value):
        actual = len(re.findall(self._keyword, value, flags=re.IGNORECASE))
        return actual == 1


class ExcludeWordHarderChecker:
    def __init__(self, keyword: str):
        self._keyword = keyword

    def check_following(self, value):
        return " " + self._keyword + " " not in value


class StartEndChecker:
    def check_following(self, value):
        words = word_tokenize(value)
        if len(words) < 2:
            return False
        return words[0].lower() == words[-1].lower()


class CountingCompositionChecker:
    def __init__(self, n_sent: int, n_words: int):
        self._n_sent = n_sent
        self._n_words = n_words

    def check_following(self, value):
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False
                continue
            sentences = split_into_sentences(paragraph)
            if len(sentences) != self._n_sent:
                return False
            for sentence in sentences:
                words = word_tokenize(sentence)
                if len(words) != self._n_words:
                    return False
        return num_paragraphs == 3


class PalindromeBasicChecker:
    def check_following(self, value):
        palindromes = [w for w in value.split() if w == w[::-1] and len(w) > 1]
        return len(palindromes) > 0


class KeywordSpecificPositionChecker:
    def __init__(self, keyword: str, n: int, m: int):
        self._keyword = keyword
        self._n = n
        self._m = m

    def check_following(self, value):
        sentences = split_into_sentences(value)
        if len(sentences) < self._n:
            return False
        words = word_tokenize(sentences[self._n - 1])
        if len(words) < self._m:
            return False
        return words[self._m - 1].lower() == self._keyword.lower()


class AdjacentLetterChecker:
    """No two adjacent words may start with consecutive letters of the alphabet."""
    def check_following(self, value):
        words = value.split()
        for i in range(len(words) - 1):
            if not words[i] or not words[i + 1]:
                continue
            a = words[i][0].lower()
            b = words[i + 1][0].lower()
            if ord(b) - ord(a) == 1:
                return False
        return True


class CountUniqueChecker:
    def check_following(self, value):
        words = word_tokenize(value)
        return len(words) == len(set(words))


class CountIncrementWordChecker:
    def __init__(self, keyword1: str, keyword2: str):
        self._keyword1 = keyword1
        self._keyword2 = keyword2

    def check_following(self, value):
        c1 = len(re.findall(self._keyword1, value, flags=re.IGNORECASE))
        c2 = len(re.findall(self._keyword2, value, flags=re.IGNORECASE))
        return c1 == 1 and c2 == 2


class LowercaseCountingChecker:
    def __init__(self, N: int):
        self._N = N

    def check_following(self, value):
        lowercase_words = re.findall(r"\b[a-z]+\b", value)
        return len(lowercase_words) <= self._N


class LetterCountingChecker:
    def __init__(self, N: int, relation: str):
        self._N = N
        self._relation = relation

    def check_following(self, value):
        letters = re.findall(r"[a-zA-Z]", value)
        if self._relation == "at least":
            return len(letters) >= self._N
        elif self._relation == "less than":
            return len(letters) < self._N
        return False


class PunctuationDotChecker:
    def check_following(self, value):
        return not re.search(r"\.", value)


class PunctuationExclamationChecker:
    def check_following(self, value):
        return not re.search(r"!", value)


class ParagraphBasicChecker:
    """Exactly 2 paragraphs separated by * * *."""
    def check_following(self, value):
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num = len(paragraphs)
        for i, p in enumerate(paragraphs):
            if not p.strip():
                if i == 0 or i == len(paragraphs) - 1:
                    num -= 1
                else:
                    return False
        return num == 2


class ParagraphBasicChecker2:
    """Exactly 2 paragraphs separated by double newline."""
    def check_following(self, value):
        paragraphs = re.split(r"\n\n", value)
        num = len(paragraphs)
        for i, p in enumerate(paragraphs):
            if not p.strip():
                if i == 0 or i == len(paragraphs) - 1:
                    num -= 1
                else:
                    return False
        return num == 2


class CopySpanIdxChecker:
    def __init__(self, prompt_to_repeat: str, n_start: int, n_end: int):
        self._prompt_to_repeat = prompt_to_repeat
        self._n_start = n_start
        self._n_end = n_end

    def check_following(self, value):
        expected = self._prompt_to_repeat[self._n_start:self._n_end].strip().lower()
        return value.strip().lower() == expected


class KeywordFrequencyCheckerDifferent:
    """Two different keywords each appearing a different specified number of times."""
    def __init__(self, keyword1: str, freq1: int, keyword2: str, freq2: int):
        self._keyword1 = keyword1
        self._freq1 = freq1
        self._keyword2 = keyword2
        self._freq2 = freq2

    def check_following(self, value):
        c1 = len(re.findall(self._keyword1, value, flags=re.IGNORECASE))
        c2 = len(re.findall(self._keyword2, value, flags=re.IGNORECASE))
        return c1 == self._freq1 and c2 == self._freq2
