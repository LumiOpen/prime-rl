#!/usr/bin/env python3
"""Decode a list of token IDs using a model's tokenizer.

Token IDs are read from stdin (space or comma separated, or a Python list).

Usage:
    uv run scripts/decode_tokens.py <model_dir>   # then paste IDs, Ctrl+D
    echo "1 2 3" | uv run scripts/decode_tokens.py <model_dir>
"""
import sys
import re

from transformers import AutoTokenizer

if len(sys.argv) < 2:
    sys.exit(f"Usage: {sys.argv[0]} <model_dir>")

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])

raw = sys.stdin.read()
token_ids = [int(x) for x in re.findall(r"\d+", raw)]

print(f"Token count: {len(token_ids)}")
print("---")
print(tokenizer.decode(token_ids))
