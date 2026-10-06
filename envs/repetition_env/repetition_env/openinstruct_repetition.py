"""Repetition detection, copied verbatim from allenai/open-instruct.

Source: scripts/data/filtering_and_updates/filter_ngram_repetitions.py
Everything below this docstring is byte-identical to upstream; only the CLI,
the `datasets`/`open_instruct` imports and the dataset-mapping helpers were
dropped, plus `find_consecutive_repetitions`, which upstream never calls.

Despite the upstream filename, `detect_exact_block_repetition` is an *exact
block* detector over paragraphs and sentences -- it does not use n-grams.
`find_ngram_repetitions` is present upstream but unreachable from the
detector; here it backs the `repeated_ngram_fraction` diagnostic metric only.
"""

import re


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, removing empty ones."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using basic punctuatisi."""
    # Simple sentence splitting - can be improved with nltk/spacy
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def is_math_or_code(text: str) -> bool:
    """Detect if text contains mathematical expressions or code patterns."""
    # Common math/code symbols and patterns
    math_symbols = [
        "=",
        "+",
        "-",
        "*",
        "/",
        "^",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
        "|",
        "&",
        "~",
        "!",
        "@",
        "#",
        "$",
        "%",
        "\\",
    ]

    # Check if text contains math symbols
    has_math_symbols = any(symbol in text for symbol in math_symbols)

    # Check for common code patterns
    code_patterns = [
        r"\w+\([^)]*\)",  # function calls like Array(n)
        r"\w+\s*=\s*\w+",  # assignments like s = 1/3600
        r"\d+/\d+",  # fractions like 1/3600
        r"[a-zA-Z_]\w*\s*[+\-*/]\s*[a-zA-Z_]\w*",  # variable operations
        r"[a-zA-Z_]\w*\s*[=<>!]+\s*[a-zA-Z_]\w*",  # comparisons
    ]

    has_code_patterns = any(re.search(pattern, text) for pattern in code_patterns)

    return has_math_symbols or has_code_patterns


def is_code_import_or_return(text: str) -> bool:
    """Detect if text contains import statements, return statements, or other code patterns."""
    # Common code patterns that should be ignored
    code_patterns = [
        r"^import\s+",  # import statements
        r"^from\s+\w+\s+import",  # from ... import statements
        r"^return\s+",  # return statements
        r"^return\s+\w+",  # return with value like "return None", "return True"
        r"^\w+\s*\+\s*=\s*\d+",  # increment operations like "next_node += 1"
        r"^\w+\s*-\s*=\s*\d+",  # decrement operations
        r"^\w+\s*\*\s*=\s*\d+",  # multiplication assignment
        r"^\w+\s*/\s*=\s*\d+",  # division assignment
        r"^for\s+",  # for loops
        r"^while\s+",  # while loops
        r"^if\s+",  # if statements
        r"^def\s+",  # function definitions
        r"^class\s+",  # class definitions
        r"^\w+\.\w+\(",  # method calls like "Image.open()"
        r"^\w+\s*=\s*\w+\(",  # function assignments
    ]

    # Check for single-sided brackets/parentheses (unmatched)
    single_bracket_patterns = [
        r"[\(\[\{][^\)\]\}]*$",  # Starts with opening bracket but no closing
        r"^[^\(\[\{]*[\)\]\}]",  # Ends with closing bracket but no opening
        r"[\(\[\{][^\(\[\{\)\]\}]*[\)\]\}]",  # Has both opening and closing brackets
    ]

    # If it has single-sided brackets, it's likely a code fragment
    has_single_brackets = any(re.search(pattern, text) for pattern in single_bracket_patterns[:2])
    has_matched_brackets = any(re.search(pattern, text) for pattern in single_bracket_patterns[2:])

    # If it has unmatched brackets, consider it code
    if has_single_brackets and not has_matched_brackets:
        return True

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in code_patterns)


def is_short_phrase(text: str) -> bool:
    """Detect if text is a short phrase that shouldn't count for overall repetition."""
    # Short phrases that are common and shouldn't trigger repetition detection
    short_phrases = [
        "not sure",
        "wait no",
        "i think",
        "maybe",
        "probably",
        "yes",
        "no",
        "ok",
        "okay",
        "right",
        "wrong",
        "correct",
        "incorrect",
        "true",
        "false",
        "good",
        "bad",
        "nice",
        "great",
        "awesome",
        "terrible",
        "wow",
        "oh",
        "hmm",
        "um",
        "uh",
        "well",
        "so",
        "now",
        "then",
        "here",
        "there",
        "this",
        "that",
        "these",
        "those",
        "it",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "can",
        "cannot",
        "cant",
        "dont",
        "doesnt",
        "isnt",
        "arent",
        "second < b",
        "we can use:",
        "we can use",
        "first < a",
        "third < c",
        "fourth < d",
        "a < b",
        "b < c",
        "c < d",
        "d < e",
        "e < f",
        "f < g",
        "g < h",
        "x < y",
        "y < z",
        "i < j",
        "j < k",
        "k < l",
        "l < m",
        "m < n",
        "n < o",
        "o < p",
        "p < q",
        "q < r",
        "r < s",
        "s < t",
        "t < u",
        "u < v",
        "v < w",
        "w < x",
        "x < y",
        "y < z",
        "return none",
        "return true",
        "return false",
        "return 0",
        "return 1",
        "return -1",
        "return null",
        "return undefined",
    ]

    # Check if the text (case insensitive) matches any short phrase
    text_lower = text.lower().strip()
    if any(phrase in text_lower for phrase in short_phrases):
        return True

    # Also check for simple comparison patterns (word < word or word > word)
    # But only if it's not a code pattern
    comparison_pattern = r"^\w+\s*[<>]\s*\w+$"
    if re.match(comparison_pattern, text.strip()):
        # Make sure it's not a code pattern like "import math"
        code_patterns = [
            "import",
            "from",
            "return",
            "def",
            "class",
            "if",
            "for",
            "while",
            "return none",
            "return true",
            "return false",
        ]
        if not any(code_pattern in text_lower for code_pattern in code_patterns):
            return True

    return False


def is_complex_math_expression(text: str) -> bool:
    """Detect complex mathematical expressions that should be ignored."""
    # Patterns for complex math expressions
    math_patterns = [
        r"math\.\w+\([^)]*\)",  # math functions like math.log()
        r"\w+\*\*[0-9]+",  # power operations
        r"[0-9]+\s*[+\-*/]\s*[0-9]+",  # arithmetic operations
        r"[a-zA-Z_]\w*\s*[+\-*/]\s*[a-zA-Z_]\w*",  # variable operations
        r"def\s+\w+\([^)]*\):",  # function definitions
        r"lambda\s+[^:]+:",  # lambda functions
        r"\[[^\]]*for[^\]]*\]",  # list comprehensions
        r"\{[^}]*for[^}]*\}",  # set comprehensions
        r"\([^)]*for[^)]*\)",  # generator expressions
    ]

    return any(re.search(pattern, text) for pattern in math_patterns)


def is_structured_list(text: str) -> bool:
    """Detect if text contains structured list patterns."""
    # Check for numbered lists (1., 2., 3., etc.)
    numbered_list_pattern = r"^\d+\.\s"

    # Check for bullet points (-, *, •, etc.)
    bullet_pattern = r"^[\-\*•]\s"

    # Check for structured content with consistent formatting
    lines = text.split("\n")
    if len(lines) < 2:
        return False

    # Check if multiple lines start with the same pattern
    numbered_count = sum(1 for line in lines if re.match(numbered_list_pattern, line.strip()))
    bullet_count = sum(1 for line in lines if re.match(bullet_pattern, line.strip()))

    # If more than 2 lines have the same structure, consider it a structured list
    return numbered_count >= 2 or bullet_count >= 2


def is_multi_line_paragraph(text: str) -> bool:
    """Detect if text is a multi-line paragraph (contains newlines)."""
    return "\n" in text.strip()


def is_common_transition_word(text: str) -> bool:
    """Detect if text is a common transition word that frequently appears and should be ignored."""
    # Common transition words that are frequently repeated and cause false positives
    transition_words = [
        "alternatively,",
        "therefore,",
        "however,",
        # 'furthermore,', 'moreover,',
        # 'consequently,', 'nevertheless,', 'nonetheless,', 'hence,', 'thus,',
        # 'meanwhile,', 'subsequently,', 'additionally,', 'likewise,', 'similarly,',
        # 'conversely,', 'on the other hand,', 'in contrast,', 'in conclusion,',
        # 'finally,', 'firstly,', 'secondly,', 'thirdly,', 'lastly,', 'initially,'
    ]

    # Check if the text (case insensitive) exactly matches a transition word
    text_lower = text.lower().strip()
    return text_lower in transition_words


def find_all_repetitions(items: list[str], block_type: str) -> dict[str, dict]:
    """
    Find all repetitions (both consecutive and non-consecutive) in a list of items.
    Returns dict mapping repeated items to repetition info.
    """
    repetition_info = {}

    # Count occurrences of each item
    item_counts = {}
    item_positions = {}

    for i, item in enumerate(items):
        clean_item = item.strip()
        if len(clean_item) < (5 if block_type == "line" else 10):  # Minimum 5 chars for sentences, 10 for paragraphs
            continue

        if clean_item not in item_counts:
            item_counts[clean_item] = 0
            item_positions[clean_item] = []

        item_counts[clean_item] += 1
        item_positions[clean_item].append(i)

    # Find consecutive runs for each repeated item
    for item, count in item_counts.items():
        if count > 1:  # Only consider items that appear more than once
            positions = item_positions[item]

            # Find the longest consecutive run
            max_consecutive = 1
            current_consecutive = 1

            for i in range(1, len(positions)):
                if positions[i] == positions[i - 1] + 1:
                    current_consecutive += 1
                else:
                    max_consecutive = max(max_consecutive, current_consecutive)
                    current_consecutive = 1
            max_consecutive = max(max_consecutive, current_consecutive)

            repetition_info[item] = {
                "type": f"total_{block_type}_repeated_{count}x",
                "total_count": count,
                "consecutive_count": max_consecutive,
                "positions": positions,
            }

    return repetition_info


def find_ngram_repetitions(text: str, n: int = 3, min_occurrences: int = 2) -> dict[str, list[int]]:
    """
    Find n-gram repetitions in text.
    Returns dict mapping n-grams to their positions.
    """
    words = text.lower().split()
    ngrams = {}

    # Generate all n-grams and their positions
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        if ngram not in ngrams:
            ngrams[ngram] = []
        ngrams[ngram].append(i)

    # Filter to only repeated n-grams
    repeated = {ngram: positions for ngram, positions in ngrams.items() if len(positions) >= min_occurrences}

    return repeated


def detect_exact_block_repetition(
    text: str,
    min_repetitions: int = 10,
    min_sentence_repetitions: int = 40,
    min_consecutive_sentence_repetitions: int = 4,
):
    """
    Detect exact block repetitions in text (paragraphs or lines).

    This function analyzes text for repetitive patterns at both paragraph and sentence levels,
    using different thresholds based on content type and repetition pattern.

    Args:
        text (str): The input text to analyze for repetitive patterns.
        min_repetitions (int, optional): Minimum number of repetitions for paragraphs.
            Defaults to 2.
        min_sentence_repetitions (int, optional): Minimum number of repetitions for
            non-consecutive sentences. Defaults to 20.
        min_consecutive_sentence_repetitions (int, optional): Minimum number of repetitions
            for consecutive sentences. Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - has_repetition (bool): True if repetitive patterns are detected, False otherwise
            - reason (str or None): Description of the repetition type if detected, None otherwise
            - details (dict or None): Detailed information about the repetition including:
                - block_type (str): Type of repeated block ('paragraph' or 'line')
                - total_count (int): Total number of repetitions
                - consecutive_count (int): Number of consecutive repetitions
                - positions (list): List of positions where repetitions occur
                - repeated_block (str): The actual repeated text block

    Thresholds Used:
        - Paragraphs: min_repetitions+ repetitions (default: 10+)
        - Multi-line paragraphs: 4x higher threshold (40+ repetitions)
        - Math/code paragraphs: 2x higher threshold (20+ repetitions)
        - Consecutive sentences: min_consecutive_sentence_repetitions+ repetitions (if ~10+ characters)
        - Non-consecutive sentences: min_sentence_repetitions+ repetitions
        - Math/code content: 2x higher thresholds
        - Structured lists: 3x higher thresholds
        - Short phrases: Ignored for non-consecutive repetitions
        - Short paragraphs: 20+ characters for non-consecutive repetitions
        - Code patterns: Completely ignored for both paragraphs and sentences
    """
    # Check if this is structured content (lists, etc.)
    is_structured = is_structured_list(text)

    # Check paragraph-level repetitions first (threshold: 2+)
    paragraphs = split_into_paragraphs(text)
    paragraph_repetitions = find_all_repetitions(paragraphs, "paragraph")

    # Check for any paragraph repeated min_repetitions or more times
    for paragraph, info in paragraph_repetitions.items():
        # Skip short paragraphs (less than 20 characters) for non-consecutive repetitions
        if info["consecutive_count"] == 1 and len(paragraph.strip()) < 20:
            continue

        # Skip code patterns entirely for paragraphs
        if is_code_import_or_return(paragraph.strip()):
            continue

        # Skip complex math expressions for paragraphs
        if is_complex_math_expression(paragraph.strip()):
            continue

        # Skip common transition words that are false positives
        if is_common_transition_word(paragraph.strip()):
            continue

        # Check if this is a multi-line paragraph
        is_multi_line = is_multi_line_paragraph(paragraph)

        # Check if this paragraph contains math/code content
        is_math_code = is_math_or_code(paragraph)

        # Use higher threshold for structured content, multi-line paragraphs, and math/code content
        structured_multiplier = 3 if is_structured else 1
        multi_line_multiplier = 4 if is_multi_line else 1  # Much higher threshold for multi-line
        math_code_multiplier = 2 if is_math_code else 1  # Higher threshold for math/code content
        effective_min_repetitions = min_repetitions * max(
            structured_multiplier, multi_line_multiplier, math_code_multiplier
        )

        # For non-consecutive paragraph repetitions, use higher threshold
        if info["consecutive_count"] == 1 and info["total_count"] > 1:
            # Non-consecutive repetitions need higher threshold, but cap it reasonably
            effective_min_repetitions = min(effective_min_repetitions * 2, 30)  # Increased cap for multi-line and code

        if info["total_count"] >= effective_min_repetitions:
            details = {
                "block_type": "paragraph",
                "total_count": info["total_count"],
                "consecutive_count": info["consecutive_count"],
                "positions": info["positions"],
                "repeated_block": paragraph,
            }
            return True, f"paragraph_repeated_{info['total_count']}x", details

    # Check line-level repetitions with different thresholds for consecutive vs non-consecutive
    sentences = split_into_sentences(text)
    sentence_repetitions = find_all_repetitions(sentences, "line")

    # Check for any sentence repeated
    for sentence, info in sentence_repetitions.items():
        # Skip single words entirely
        sentence_words = sentence.strip().split()
        if len(sentence_words) <= 1:
            continue

        # Skip short phrases for overall repetition (but not consecutive)
        if is_short_phrase(sentence.strip()):
            # For short phrases, only flag if they're consecutive
            if info["consecutive_count"] >= min_consecutive_sentence_repetitions * 2:
                details = {
                    "block_type": "line",
                    "total_count": info["total_count"],
                    "consecutive_count": info["consecutive_count"],
                    "positions": info["positions"],
                    "repeated_block": sentence,
                }
                return True, f"line_repeated_{info['total_count']}x_consecutive_short_phrase", details
            continue

        # Skip code patterns entirely
        if is_code_import_or_return(sentence.strip()):
            continue

        # Skip complex math expressions
        if is_complex_math_expression(sentence.strip()):
            continue

        # Skip common transition words that are false positives
        if is_common_transition_word(sentence.strip()):
            continue

        # Check if this sentence contains math/code content
        is_math_code = is_math_or_code(sentence)

        # Adjust thresholds for math/code content (2x higher) and structured content (3x higher)
        math_multiplier = 2 if is_math_code else 1
        structured_multiplier = 3 if is_structured else 1
        effective_min_consecutive = min_consecutive_sentence_repetitions * max(math_multiplier, structured_multiplier)
        effective_min_sentence = min_sentence_repetitions * max(math_multiplier, structured_multiplier)

        # For consecutive repetitions, use lower threshold (2+) if sentence is long enough
        if info["consecutive_count"] >= effective_min_consecutive and len(sentence.strip()) >= 5:
            details = {
                "block_type": "line",
                "total_count": info["total_count"],
                "consecutive_count": info["consecutive_count"],
                "positions": info["positions"],
                "repeated_block": sentence,
            }
            return True, f"line_repeated_{info['total_count']}x_consecutive", details

        # For non-consecutive repetitions, use higher threshold (5+) and require longer sentences
        elif info["total_count"] >= effective_min_sentence and len(sentence.strip()) >= 10:
            details = {
                "block_type": "line",
                "total_count": info["total_count"],
                "consecutive_count": info["consecutive_count"],
                "positions": info["positions"],
                "repeated_block": sentence,
            }
            return True, f"line_repeated_{info['total_count']}x", details

    return False, None, None
