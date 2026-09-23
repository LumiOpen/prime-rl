"""Send one open-instruct-style request to a running judge server and dump the raw reply.

The env only logs a 200-char excerpt of a failed parse, which is not enough to tell
"the judge ignored the JSON instruction" from "the judge was cut off mid-answer".
This prints finish_reason and completion_tokens alongside the full text.

Usage (on the node serving the judge):
  python3 scripts/probe_judge_output.py --url http://127.0.0.1:8001 --model <path> [--max-tokens N]
"""

import argparse
import json
import urllib.request

PROMPT = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
- Your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
- If there is a system prompt, ensure the AI answer prioritizes following it.
- Begin your evaluation by providing a short explanation.
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
- Please adhere to the following format.

[Conversation History]
{input}

[AI Answer]
{output}

[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--no-thinking", action="store_true", default=True)
    args = ap.parse_args()

    cases = [
        ("User: What is the capital of France?", "Paris."),
        (
            "User: Explain why the sky is blue in two sentences.",
            "The sky looks blue because air scatters short wavelengths more than long ones. "
            "That scattered blue light reaches your eye from every direction.",
        ),
        ("User: Write a haiku about winter.", "asdkjh asdkjh asdkjh"),
    ][: args.n]

    for question, answer in cases:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPT.format(input=question, output=answer)}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        req = urllib.request.Request(
            f"{args.url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as resp:
            data = json.loads(resp.read())
        choice = data["choices"][0]
        print("=" * 70)
        print(f"question      : {question}")
        print(f"finish_reason : {choice['finish_reason']}")
        print(f"completion_tok: {data['usage']['completion_tokens']}")
        print(f"raw output    : {choice['message']['content']!r}")


if __name__ == "__main__":
    main()
