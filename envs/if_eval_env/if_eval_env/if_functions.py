# Copyright 2024 The Google Research Authors. Apache License 2.0
# Ported from https://github.com/allenai/open-instruct/blob/main/open_instruct/if_functions.py
# Covers the 25 original IFEval constraints.

import json
import re

import langdetect


def verify_keywords(text, keyword_list):
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


def verify_keyword_frequency(text, word, N):
    text = text.lower()
    keyword = word.lower()
    words = re.findall(r"\b\w+\b", text)
    actual_count = sum(1 for w in words if w == keyword)
    return actual_count == N


def validate_forbidden_words(text, forbidden_words):
    text_lower = text.lower()
    found_words = [word for word in forbidden_words if word.lower() in text_lower]
    return len(found_words) == 0


def verify_letter_frequency(text: str, letter: str, N: int) -> bool:
    if len(letter) != 1:
        raise ValueError("Letter parameter must be a single character")
    return text.count(letter) == N


def validate_response_language(text, language):
    detected_language = langdetect.detect(text)
    return detected_language == language


def verify_paragraph_count(text: str, N: int) -> bool:
    text = "\n".join(line.strip() for line in text.splitlines()).strip()
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False
    return actual_count == N


def validate_word_constraint(text: str, N: int, quantifier: str) -> bool:
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    return False


def verify_sentence_constraint(text: str, N: int, quantifier: str) -> bool:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    actual_count = len(sentences)
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    return False


def validate_paragraphs(text, N, first_word, i):
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False
    return bool(paragraphs[i - 1].strip().startswith(first_word))


def verify_postscript(text, postscript_marker):
    if postscript_marker in text:
        marker_index = text.find(postscript_marker)
        remaining_text = text[marker_index:].strip()
        return len(remaining_text) > len(postscript_marker)
    return False


def validate_placeholders(text: str, N: int) -> bool:
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)
    return len(placeholders) >= N


def verify_bullet_points(text: str, N: int) -> bool:
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    return len(bullet_points) == N


def validate_title(text: str) -> bool:
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, text)
    return len(matches) > 0


def validate_choice(text: str, options: list) -> bool:
    return any(option in text for option in options)


def validate_highlighted_sections(text: str, N: int) -> bool:
    pattern = r"\*(.*?)\*"
    matches = re.findall(pattern, text)
    return len(matches) >= N


def validate_sections(text: str, N: int, section_splitter: str) -> bool:
    sections = text.split(section_splitter)
    if sections[0] == "":
        sections.pop(0)
    return len(sections) == N


def validate_json_format(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def validate_repeat_prompt(text: str, original_prompt: str) -> bool:
    return bool(text.startswith(original_prompt))


def validate_two_responses(text: str) -> bool:
    if text.count("******") == 1:
        parts = text.split("******")
        first, second = parts[0].strip(), parts[1].strip()
        if first != second:
            return True
    return False


def validate_uppercase(text: str) -> bool:
    return text == text.upper()


def validate_lowercase(text: str) -> bool:
    return text == text.lower()


def validate_frequency_capital_words(text: str, N: int, quantifier: str) -> bool:
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return abs(len(words) - N) <= max(round(N * 0.1), 1)
    elif quantifier == "at most":
        return len(words) <= N
    return False


def validate_end(text: str, end_phrase: str) -> bool:
    return bool(text.endswith(end_phrase))


def validate_quotation(text: str) -> bool:
    return bool(text.startswith('"') and text.endswith('"'))


def validate_no_commas(text: str) -> bool:
    return "," not in text
