import logging
import random
import re

import verifiers as vf

logger = logging.getLogger("verifiers.v1")

_LOG_SAMPLE_RATE = 0.05  # log ~5% of scored examples

_GENERAL_CATEGORIES = {"general-quality", "general-quality_ref"}

# Default judge prompt for LLMJudgeRubric.
# Placeholders: {question}, {answer}.  Judge must emit "Score: N" (0-100).
# Reward is normalized to [0, 1] by dividing by 100.
# Used for "general-quality" examples (no reference answer).
_DEFAULT_JUDGE_PROMPT = """\
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.

IMPORTANT: If the [AI Answer] is marked as [EMPTY RESPONSE] or [DEGENERATE RESPONSE], the AI produced no meaningful content and must receive Score: 1.

You MUST respond in exactly this format:
Score: <integer from 0 to 100>
<your brief reasoning here>

[Query]
{question}

[AI Answer]
{answer}

[Your judgement]"""

# Used for "general-quality_ref" examples (reference answer available).
_JUDGE_PROMPT_WITH_REF = """\
### Task Description
Please act as an impartial judge and evaluate the quality of the answer provided by an
AI assistant to the conversation history leading up to the answer displayed below.
Judge whether the provided answer is good by comparing it to the reference answer.

Besides comparing to the reference answer, your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
Note that sometimes the reference answer is not the only answer. So any valid variation of the reference answer is also acceptable and can get a full score.

IMPORTANT: If the [AI Answer] is marked as [EMPTY RESPONSE] or [DEGENERATE RESPONSE], the AI produced no meaningful content and must receive Score: 1.

You MUST respond in exactly this format:
Score: <integer from 0 to 100>
<your brief reasoning here>

[Query]
{question}

[AI Answer]
{answer}

[Reference Gold Answer]
{reference}

[Your judgement]"""


def _strip_think_blocks(text: str) -> str:
    """Remove thinking content so scorers only see the final answer.

    Handles three cases:
    1. Full <think>...</think> block in completion — remove it
    2. Completion starts mid-think (chat template pre-fills '<think>', so the
       response begins inside the block with no opening tag): strip up to </think>
    3. No think tags at all (non-think model, or truncated mid-think with no </think>):
       return text as-is for non-think models; return "" only if truncated mid-think
    """
    import re
    # Case 1: full <think>...</think> blocks present — remove them
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Also handle unclosed opening tag (truncated mid-think)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()
    # Case 2: completion is the inside of a think block (no opening tag, has closing tag)
    # Chat template pre-fills '<think>' so response starts mid-reasoning
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
        return text.strip()
    # Case 3: plain-text thinking format (e.g. Qwen3.5-9B base outputs
    # "Thinking Process:\n\n1. Analyze..." without XML tags).
    # Extract only the final answer so the judge evaluates content quality.
    thinking_markers = [
        "Thinking Process:\n",
        "Here's a thinking process",
        "Here is a thinking process",
    ]
    for marker in thinking_markers:
        if text.startswith(marker) or f"\n{marker}" in text:
            # 1. Look for an explicit final answer header
            for answer_marker in [
                "\n**Final Answer",
                "\n**Answer",
                "\nFinal Answer:",
                "\nAnswer:",
                "\n---\n",
            ]:
                if answer_marker in text:
                    return text[text.index(answer_marker):].strip().lstrip("-").strip()
            # 2. Find the "drafting" section — where the model writes the actual response
            for draft_marker in [
                "\n**Drafting",
                "\n**Writing",
                "\n**Composing",
                "\n**Response",
                "\nDrafting",
            ]:
                if draft_marker in text:
                    return text[text.index(draft_marker):].strip()
            # 3. Find the last top-level numbered step content — the model's actual output
            # is typically in the last step after all analysis steps
            last_step = re.search(r'\n\d+\.\s+\*\*[^*]+\*\*[^#]*$', text, re.DOTALL)
            if last_step:
                candidate = text[last_step.start():].strip()
                # Only use if it looks like actual content (>50 chars), not a step header
                if len(candidate) > 50:
                    return candidate
            # 4. Nothing found — pass full text so judge can still evaluate
            return text.strip()
    # No think tags at all — regular model output, return as-is
    return text.strip()


def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return _strip_think_blocks(completion)
    if isinstance(completion, list):
        parts = [msg.get("content") or "" for msg in completion if msg.get("role") == "assistant"]
        return _strip_think_blocks("\n".join(parts))
    return _strip_think_blocks(str(completion))


class RubricJudgeRubric(vf.Rubric):
    """RM-based reward for general-quality examples.

    Supports two backends:
    - Local (default): loads the model via transformers on rm_device.
    - vLLM server: set rm_server_url to the base URL of a running vLLM instance
      (e.g. "http://localhost:8000"). Calls the /v1/score endpoint; rm_model_path
      must match the model name the server was started with.
    """

    def __init__(
        self,
        rm_model_path: str,
        rm_device: str = "cuda",
        rm_server_url: str | None = None,
        rm_max_response_chars: int | None = None,
    ):
        super().__init__()
        self._rm_model_path = rm_model_path
        self._rm_device = rm_device
        self._rm_server_url = rm_server_url.rstrip("/") if rm_server_url else None
        self._rm_max_response_chars = rm_max_response_chars
        self._rm = None
        self._rm_tokenizer = None
        self._session = None  # aiohttp session, created lazily
        self.add_reward_func(self.rm_score)

    # ------------------------------------------------------------------
    # Local backend
    # ------------------------------------------------------------------

    def _load_rm(self):
        if self._rm is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info(f"Loading reward model from {self._rm_model_path}")
        self._rm_tokenizer = AutoTokenizer.from_pretrained(self._rm_model_path)
        self._rm = AutoModelForSequenceClassification.from_pretrained(
            self._rm_model_path,
            dtype=torch.bfloat16,
        ).to(self._rm_device).eval()
        logger.info("Reward model loaded")

    def _run_rm_local(self, prompt: str, response: str) -> float:
        import torch
        self._load_rm()
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        out = self._rm_tokenizer.apply_chat_template(
            messages, return_tensors="pt", truncation=True, max_length=4096
        )
        input_ids = (out["input_ids"] if hasattr(out, "__getitem__") and not isinstance(out, torch.Tensor) else out)
        input_ids = input_ids.to(torch.int32).to(self._rm_device)  # int32: ROCm vectorized_gather_kernel bug with int64+bfloat16
        with torch.no_grad():
            logit = self._rm(input_ids).logits[0, 0]
        return torch.sigmoid(logit).item()

    # ------------------------------------------------------------------
    # vLLM server backend  (POST /pooling)
    # vLLM reward models use --task reward and are served via /pooling,
    # not /v1/score (which is for cross-encoder rerankers).
    # Response: {"data": [{"data": float | list[float], ...}]}
    # The server applies sigmoid internally; value is already in [0, 1].
    # ------------------------------------------------------------------

    async def _run_rm_server(self, prompt: str, response: str) -> float:
        import aiohttp
        if self._session is None:
            self._session = aiohttp.ClientSession()
        if self._rm_max_response_chars is not None and len(response) > self._rm_max_response_chars:
            response = response[:self._rm_max_response_chars]
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        url = f"{self._rm_server_url}/pooling"
        payload = {"model": self._rm_model_path, "messages": messages}
        async with self._session.post(url, json=payload) as resp:
            if not resp.ok:
                body = await resp.text()
                raise RuntimeError(f"RM server {url} returned {resp.status}: {body}")
            data = await resp.json()
        per_token = data["data"][0]["data"]
        # vLLM --task reward returns raw logits (no sigmoid applied server-side).
        # Causal reward models score on the last token, label index 0.
        logit = per_token[-1][0] if isinstance(per_token[0], list) else per_token[-1]
        return float(logit)

    # ------------------------------------------------------------------

    async def rm_score(self, completion, answer, info: dict, **kwargs) -> float:
        text = _extract_text(completion)
        prompt = info.get("prompt_text", "")
        if self._rm_server_url is not None:
            score = await self._run_rm_server(prompt, text)
        else:
            score = self._run_rm_local(prompt, text)
        if random.random() < _LOG_SAMPLE_RATE:
            logger.info(
                "RM sample\n"
                f"  prompt ({len(prompt)} chars): {prompt[:300]!r}\n"
                f"  response ({len(text)} chars): {text[:300]!r}\n"
                f"  score: {score:.4f}"
            )
        return score


class LLMJudgeRubric(vf.Rubric):
    """LLM-as-judge reward using a chat-capable model served via vLLM.

    Activated when ``custom_rm = true`` is passed in env args.  The judge model
    is expected to be served by the same vLLM instance configured under
    ``[rm_inference]`` (``judge_server_url`` is injected automatically from
    ``rm_server_url`` by ``load_environment`` when ``custom_rm=True``).

    The judge receives a structured prompt containing the original question and
    the policy's answer and must emit a line matching ``Score: N`` (N in 1-10).
    The reward is normalised to ``[0, 1]`` by dividing by 10.  On parse failure
    the reward defaults to 0.0 so that malformed judge output does not silently
    inflate rewards.
    """

    def __init__(
        self,
        judge_model_path: str,
        judge_server_url: str,
        judge_prompt_template: str = _DEFAULT_JUDGE_PROMPT,
        max_judge_tokens: int = 1024,
        judge_temperature: float = 0.0,
    ):
        super().__init__()
        self._judge_model = judge_model_path
        self._judge_url = judge_server_url.rstrip("/")
        self._judge_prompt_template = judge_prompt_template
        self._max_judge_tokens = max_judge_tokens
        self._judge_temperature = judge_temperature
        self._session = None
        self.add_reward_func(self.llm_judge_score)

    async def _call_judge(self, question: str, answer: str, reference: str = "", category: str = "") -> str:
        import aiohttp
        if category == "general-quality_ref" and reference:
            user_content = _JUDGE_PROMPT_WITH_REF.format(question=question, answer=answer, reference=reference)
        else:
            user_content = self._judge_prompt_template.format(question=question, answer=answer)
        payload = {
            "model": self._judge_model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": self._max_judge_tokens,
            "temperature": self._judge_temperature,
            # Disable Qwen3 thinking mode — without this the <think> block fills the
            # token budget before the model can emit "Score: N".
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = f"{self._judge_url}/v1/chat/completions"
        for attempt in range(3):
            if self._session is None or self._session.closed or self._session.connector is None or self._session.connector.closed:
                self._session = aiohttp.ClientSession()
            try:
                async with self._session.post(url, json=payload) as resp:
                    if not resp.ok:
                        body = await resp.text()
                        raise RuntimeError(f"Judge server {url} returned {resp.status}: {body}")
                    data = await resp.json()
                return data["choices"][0]["message"]["content"]
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectorError, RuntimeError) as e:
                if "Connector is closed" in str(e) or isinstance(e, (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectorError)):
                    await self._session.close()
                    self._session = None
                    if attempt == 2:
                        raise
                else:
                    raise
        raise RuntimeError("unreachable")

    @staticmethod
    def _parse_score(text: str) -> float:
        # 1. Score-first format: "Score: N" on the first line (primary)
        first_line = text.strip().split("\n")[0]
        match = re.match(r"[Ss]core\s*:\s*(\d+)", first_line)
        # 2. "Score: N" anywhere in text
        if not match:
            match = re.search(r"[Ss]core\s*:\s*(\d+)", text)
        # 3. open-instruct format: {"SCORE": "7"} or {"SCORE": 7}
        if not match:
            match = re.search(r'"SCORE"\s*:\s*"?(\d+)"?', text)
        # 4. "rating: N"
        if not match:
            match = re.search(r'[Rr]ating\s*:\s*(\d+)', text)
        # 5. Last resort: leading number on first line
        if not match:
            match = re.match(r'(\d{1,3})\b', first_line)
        if match:
            raw = int(match.group(1))
            return max(0, min(100, raw)) / 100.0
        logger.warning("LLM judge: could not parse score from output: %r", text[:200])
        return 0.0

    async def llm_judge_score(self, completion, answer, info: dict, state=None, **kwargs) -> float:
        response_text = _extract_text(completion)
        # If the stripped answer is empty or just an answer marker with no content,
        # return 0 immediately — the judge hallucinates a correct answer otherwise.
        stripped = response_text.strip().rstrip("*_: \n")
        if not stripped:
            logger.info("LLM judge: empty answer after stripping thinking blocks, returning 0.0")
            return 0.0
        question = info.get("prompt_text", "")
        reference = info.get("reference", "")
        category = info.get("category", "")
        judge_output = await self._call_judge(question, response_text, reference=reference, category=category)
        score = self._parse_score(judge_output)
        if random.random() < _LOG_SAMPLE_RATE:
            ref_line = f"  reference ({len(reference)} chars): {reference[:200]!r}\n" if reference else ""
            logger.info(
                "LLM judge sample\n"
                f"  question ({len(question)} chars): {question[:200]!r}\n"
                f"  answer ({len(response_text)} chars): {response_text[:200]!r}\n"
                + ref_line +
                f"  judge output: {judge_output[:300]!r}\n"
                f"  score: {score:.2f}"
            )
        return score
