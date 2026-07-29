import re


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks so reward functions only see the final answer.

    Handles three cases:
    1. Full <think>...</think> block — remove it (including truncated unclosed tag)
    2. Response starts inside a think block (chat template pre-fills '<think>',
       so generation begins mid-reasoning with no opening tag): strip up to </think>
    3. No think tags — return as-is
    """
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
        return text.strip()
    return text.strip()
