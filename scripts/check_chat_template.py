"""
Tokenize a test message with a model's chat template and print the decoded result.

Usage:
    python check_chat_template.py <model_dir> [message]
"""

import sys
from transformers import AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "."
message = sys.argv[2] if len(sys.argv) > 2 else "Write a short poem about the sea."

tokenizer = AutoTokenizer.from_pretrained(model_dir)

messages = [{"role": "user", "content": message}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
