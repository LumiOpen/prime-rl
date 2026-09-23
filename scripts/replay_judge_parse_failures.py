"""Replay real rollouts through the judge and measure open-instruct format compliance.

The env logs only a 200-char excerpt of a failed parse, which cannot distinguish
"the judge ignored the JSON instruction", "the judge answered the conversation
instead of judging it", and "the judge was cut off at max_tokens". This replays
actual (prompt, completion) pairs from a run's traces and classifies every reply.

Usage (on the node serving the judge):
  python3 scripts/replay_judge_parse_failures.py \
      --traces <run>/rollouts/step_7/train/all/traces.jsonl \
      --model <judge path> --n 40 --max-tokens 2048
"""

import argparse
import json
import re
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

LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def extract_json_score(text: str):
    """Same logic as rubric._extract_json_score, so failures here mirror the env."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.replace("\r\n", "\n").replace("\n", "\\n")
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned).strip()
    try:
        return float(json.loads(cleaned)["SCORE"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        match = re.search(r'"SCORE"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', cleaned)
        return float(match.group(1)) if match else None


def classify(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("[Conversation History]"):
        return "echoed-the-template"
    if re.match(r"\[REASONING\]\s*[:\n]", stripped):
        return "bracket-style-not-json"
    if "SCORE" in text:
        return "json-ish"
    return "other"


def render_messages(messages) -> str:
    return "\n\n".join(
        f"{LABELS.get(m['role'], m['role'].title())}: {m['content']}"
        for m in messages
        if m.get("content")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    rows = []
    with open(args.traces) as f:
        for line in f:
            trace = json.loads(line)
            nodes = trace["nodes"]
            messages = [n["message"] for n in nodes]
            answer = next(
                (str(n["message"].get("content") or "") for n in reversed(nodes) if n.get("sampled")), ""
            )
            prompt_msgs = [m for m in messages if not (m.get("role") == "assistant" and not m.get("content"))]
            # everything before the sampled turn is what the judge sees as history
            sampled_at = max(i for i, n in enumerate(nodes) if n.get("sampled"))
            prompt_msgs = messages[:sampled_at]
            if not prompt_msgs or not answer:
                continue
            rows.append((render_messages(prompt_msgs), answer, prompt_msgs[0].get("role") == "system"))
            if len(rows) >= args.n:
                break

    stats = {}
    capped = parsed = 0
    shown = 0
    for question, answer, has_system in rows:
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
        with urllib.request.urlopen(req, timeout=1800) as resp:
            data = json.loads(resp.read())
        choice = data["choices"][0]
        text = choice["message"]["content"]
        kind = classify(text)
        score = extract_json_score(text)
        key = (kind, choice["finish_reason"], has_system, score is not None)
        stats[key] = stats.get(key, 0) + 1
        capped += choice["finish_reason"] == "length"
        parsed += score is not None
        if score is None and shown < 3:
            shown += 1
            print("-" * 70)
            print(f"FAILED  system_msg={has_system}  finish={choice['finish_reason']}  "
                  f"tok={data['usage']['completion_tokens']}  q_chars={len(question)}")
            print(f"  question head: {question[:200]!r}")
            print(f"  judge output : {text[:400]!r}")

    print("=" * 70)
    print(f"n={len(rows)}  parsed={parsed}  failed={len(rows) - parsed}  hit_max_tokens={capped}")
    print(f"{'kind':<24}{'finish':<10}{'sys':<6}{'parsed':<8}count")
    for (kind, finish, has_system, ok), count in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"{kind:<24}{finish:<10}{str(has_system):<6}{str(ok):<8}{count}")


if __name__ == "__main__":
    main()
