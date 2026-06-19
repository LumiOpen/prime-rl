# Finnish instruction registry.
#
# Identical structure to ifeval_env.IFEvalG.instructions_registry, but
# language-sensitive checkers are replaced with their Finnish counterparts
# from instructions_fi.py.  All other checkers are re-used as-is.

from dolci_ifeval_env.IFEvalG_fi import instructions_fi as _fi

_PARAGRAPH = "paragraphs:"
_KEYWORD = "keywords:"
_LETTER = "letters:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_MULTITURN = "multi-turn:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"
_NEW = "new:"
_COPY = "copy:"
_BASIC = "basic:"
_FIRSTWORD = "first_word:"
_LASTWORD = "last_word:"
_COUNT = "count:"

INSTRUCTION_DICT = {
    # --- keywords ---
    _KEYWORD + "existence": _fi.KeywordFinnishChecker,
    _KEYWORD + "frequency": _fi.KeywordFrequencyFinnishChecker,
    _KEYWORD + "forbidden_words": _fi.ForbiddenWordsFinnishChecker,
    _KEYWORD + "letter_frequency": _fi.LetterFrequencyFinnishChecker,
    _KEYWORD + "no_adjacent_consecutive": _fi.AdjacentLetterFinnishChecker,
    _KEYWORD + "word_once": _fi.KeywordFrequencyOnceFinnishChecker,
    _KEYWORD + "word_count_different_numbers": _fi.KeywordFrequencyDifferentFinnishChecker,
    _KEYWORD + "exclude_word_harder": _fi.ExcludeWordHarderFinnishChecker,
    _KEYWORD + "palindrome": _fi.PalindromeBasicFinnishChecker,
    _KEYWORD + "keyword_specific_position": _fi.KeywordSpecificPositionFinnishChecker,
    _KEYWORD + "start_end": _fi.StartEndFinnishChecker,
    # --- language ---
    _LANGUAGE + "response_language": _fi.ResponseLanguageFinnishChecker,
    # --- length ---
    _LENGTH + "number_sentences": _fi.NumberOfSentencesFinnishChecker,
    _LENGTH + "number_paragraphs": _fi.ParagraphFinnishChecker,
    _LENGTH + "number_words": _fi.NumberOfWordsFinnishChecker,
    _LENGTH + "nth_paragraph_first_word": _fi.ParagraphFirstWordFinnishChecker,
    # --- detectable_content ---
    _CONTENT + "number_placeholders": _fi.PlaceholderFinnishChecker,
    _CONTENT + "postscript": _fi.PostscriptFinnishChecker,
    # --- detectable_format ---
    _FORMAT + "number_bullet_lists": _fi.BulletListFinnishChecker,
    _FORMAT + "constrained_response": _fi.ConstrainedResponseFinnishChecker,
    _FORMAT + "number_highlighted_sections": _fi.HighlightSectionFinnishChecker,
    _FORMAT + "multiple_sections": _fi.SectionFinnishChecker,
    _FORMAT + "json_format": _fi.JsonFormatFinnishChecker,
    _FORMAT + "title": _fi.TitleFinnishChecker,
    _FORMAT + "sentence_hyphens": _fi.SentenceHyphenFinnishChecker,
    _FORMAT + "square_brackets": _fi.SquareBracketFinnishChecker,
    _FORMAT + "bigram_wrapping": _fi.BiGramWrappingFinnishChecker,
    # --- combination ---
    _COMBINATION + "two_responses": _fi.TwoResponsesFinnishChecker,
    _COMBINATION + "repeat_prompt": _fi.RepeatPromptThenAnswerFinnishChecker,
    # --- startend ---
    _STARTEND + "end_checker": _fi.EndFinnishChecker,
    _STARTEND + "quotation": _fi.QuotationFinnishChecker,
    # --- change_case ---
    _CHANGE_CASES + "capital_word_frequency": _fi.CapitalWordFrequencyFinnishChecker,
    _CHANGE_CASES + "english_capital": _fi.CapitalLettersFinnishChecker,
    _CHANGE_CASES + "english_lowercase": _fi.LowercaseLettersFinnishChecker,
    # --- punctuation ---
    _PUNCTUATION + "no_comma": _fi.CommaFinnishChecker,
    _PUNCTUATION + "punctuation_dot": _fi.PunctuationDotFinnishChecker,
    _PUNCTUATION + "punctuation_exclamation": _fi.PunctuationExclamationFinnishChecker,
    # --- copy ---
    _COPY + "repeat_phrase": _fi.RepeatPhraseFinnishChecker,
    _COPY + "copy": _fi.CopyFinnishChecker,
    _COPY + "copying_simple": _fi.CopyingSimpleFinnishChecker,
    _COPY + "copying_multiple": _fi.CopyingMultipleFinnishChecker,
    # --- new ---
    _NEW + "copy_span_idx": _fi.CopySpanIdxFinnishChecker,
    # --- paragraphs ---
    _PARAGRAPH + "paragraphs": _fi.ParagraphBasicFinnishChecker,
    _PARAGRAPH + "paragraphs2": _fi.ParagraphBasic2FinnishChecker,
    # --- first_word ---
    _FIRSTWORD + "first_word_sent": _fi.FirstWordSentFinnishChecker,
    _FIRSTWORD + "first_word_answer": _fi.FirstWordAnswerFinnishChecker,
    # --- last_word ---
    _LASTWORD + "last_word_sent": _fi.LastWordSentFinnishChecker,
    _LASTWORD + "last_word_answer": _fi.LastWordAnswerFinnishChecker,
    # --- count ---
    _COUNT + "lowercase_counting": _fi.LowercaseCountingFinnishChecker,
    _COUNT + "counting_composition": _fi.CountingCompositionFinnishChecker,
    _COUNT + "count_unique": _fi.CountUniqueFinnishChecker,
    _COUNT + "count_increment_word": _fi.CountIncrementWordFinnishChecker,
    # --- letters ---
    _LETTER + "letter_counting": _fi.LetterCountingFinnishChecker,
    _LETTER + "letter_counting2": _fi.LetterFrequencyFinnishChecker,
}
