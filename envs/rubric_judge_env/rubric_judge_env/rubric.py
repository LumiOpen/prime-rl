import asyncio
import json
import logging
import random
import re

import verifiers as vf

logger = logging.getLogger("verifiers.v1")

_LOG_SAMPLE_RATE = 0.01  # log ~1% of scored examples

# Every rollout in the batch hits the judge at once, so a single request can sit behind
# hundreds of others. aiohttp's default ClientTimeout is 300s total, which under load
# turned into asyncio.TimeoutError — whose str() is empty, so verifiers logged a bare
# "Error calling reward function llm_judge_score: " and scored the rollout 0.0.
_JUDGE_TIMEOUT_S = 1800.0
_JUDGE_MAX_ATTEMPTS = 3  # matches open_instruct.ground_truth_utils.LMJudgeVerifier
_JUDGE_RETRY_DELAY_S = 1.0
# Retrying an unparseable answer at temperature 0 replays it verbatim, so format retries
# have to sample. Transport retries reuse the configured temperature.
_JUDGE_RETRY_TEMPERATURE = 0.7

_GENERAL_CATEGORIES = {"general-quality", "general-quality_ref"}

# Default judge prompt for LLMJudgeRubric.
# Placeholders: {question}, {answer}.  Judge must emit "Score: N" (1-100).
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

# ---------------------------------------------------------------------------
# open-instruct judge prompts, copied verbatim from
# open_instruct/judge_utils.py (general_quality_template, general_quality_ref_template).
# Their {input}/{output}/{label} placeholders and the doubled braces around the
# JSON example are kept as-is so both frameworks send byte-identical text.
# Scores are 1-10 and arrive JSON-encoded, reasoning first.
# ---------------------------------------------------------------------------

_OI_JUDGE_PROMPT = """
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

_OI_JUDGE_PROMPT_WITH_REF = """
### Task Description
Please act as an impartial judge and evaluate the quality of the answer provided by an
AI assistant to the conversation history leading up to the answer displayed below.
Judge whether the provided answer is good by comparing it to the reference answer.

Notes:
- Besides comparing to the reference answer, your evaluation should consider factors such as the helpfulness, relevance, accuracy, creativity, appropriate level of detail, and how well the response satisfies the user's explicit constraints or accurately follows their instructions.
- Note that sometimes the reference answer is not the only answer. So any valid variation of the reference answer is also acceptable and can get a full score.
- If there is a system prompt, ensure the AI answer prioritizes following it.
- Begin your evaluation by providing a short explanation.
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
- Please adhere to the following format.

[Conversation History]
{input}

[AI Answer]
{output}

[Reference Gold Answer]
{label}

[Your judgement]
Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}"""

# Finnish variants — identical to the legacy prompts but with a language requirement added.
_DEFAULT_JUDGE_PROMPT_FI = """\
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

_JUDGE_PROMPT_WITH_REF_FI = """\
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

# style -> (no-reference prompt, with-reference prompt, score divisor)
_JUDGE_STYLES = {
    "legacy": (_DEFAULT_JUDGE_PROMPT, _JUDGE_PROMPT_WITH_REF, 100.0),
    "open-instruct": (_OI_JUDGE_PROMPT, _OI_JUDGE_PROMPT_WITH_REF, 10.0),
}


class JudgeError(RuntimeError):
    """A judgement we could not obtain. Assigned to ``state["error"]`` to drop the rollout."""


class JudgeRequestError(JudgeError):
    """Non-2xx response from the judge server.

    ``retryable`` separates transient overload (429, 5xx) from a request the server will
    reject however often we send it (e.g. an unsupported sampling argument).
    """

    def __init__(self, url: str, status: int, body: str):
        super().__init__(f"Judge server {url} returned {status}: {body[:200]}")
        self.status = status
        self.retryable = status == 429 or status >= 500


class JudgeTimeout(JudgeError):
    """The judge server did not answer within ``_JUDGE_TIMEOUT_S``."""


class JudgeParseFailure(JudgeError):
    """The judge answered, but with no score we can read."""


def _extract_json_score(text: str) -> float | None:
    """Port of open_instruct.judge_utils.extract_json_score_with_fallback.

    Returns the raw (unscaled) score, or None when nothing parseable is found.
    """
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


def _parse_score_legacy(text: str) -> float | None:
    """Raw 1-100 score from the score-first prompt, or None when nothing parses."""
    first_line = text.strip().split("\n")[0]
    # 1. Score-first format: "Score: N" on the first line (primary)
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
    return float(match.group(1)) if match else None


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


def _extract_text(completion, strip_think: bool = True) -> str:
    if isinstance(completion, str):
        raw = completion
    elif isinstance(completion, list):
        if strip_think:
            parts = [msg.get("content") or "" for msg in completion if msg.get("role") == "assistant"]
        else:
            # Include reasoning_content (separate field on AssistantMessage) before content.
            parts = []
            for msg in completion:
                if msg.get("role") != "assistant":
                    continue
                reasoning = msg.get("reasoning_content") or ""
                content = msg.get("content") or ""
                parts.append("\n".join(filter(None, [reasoning, content])))
        raw = "\n".join(parts)
    else:
        raw = str(completion)
    return _strip_think_blocks(raw) if strip_think else raw


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

    Two prompt styles are available, selected with ``judge_prompt_style``:

    - ``"legacy"`` (default): our own score-first plain-text prompt.  The judge
      emits ``Score: N`` with N in 1-100 and the reward is N / 100.
    - ``"open-instruct"``: the verbatim open-instruct templates.  The judge
      reasons first and answers in JSON with N in 1-10, so the reward is N / 10.
      Reasoning-first needs room: give ``max_judge_tokens`` at least ~256.

    A judgement that cannot be obtained — the server timed out, was overloaded, or
    answered in a format we cannot parse after ``_JUDGE_MAX_ATTEMPTS`` tries — marks the
    rollout errored instead of scoring it 0.0. A fabricated 0.0 is indistinguishable from
    a genuinely bad answer, and on the 1-10 scale it is not even reachable (the floor is
    0.1), so it drags the whole GRPO group baseline down and manufactures advantage.
    prime-rl drops errored rollouts from the batch and reports the rate as
    ``all/has_error/mean`` plus ``all/error/<type>`` counts.
    """

    def __init__(
        self,
        judge_model_path: str,
        judge_server_url: str,
        judge_prompt_template: str | None = None,
        max_judge_tokens: int = 512,
        judge_temperature: float = 0.0,
        judge_prompt_style: str = "legacy",
        use_finnish: bool = False,
        language_reward_weight: float = 0.0,
    ):
        super().__init__()
        if judge_prompt_style not in _JUDGE_STYLES:
            raise ValueError(
                f"Unknown judge_prompt_style={judge_prompt_style!r}. Choose one of {sorted(_JUDGE_STYLES)}."
            )
        default_prompt, ref_prompt, scale = _JUDGE_STYLES[judge_prompt_style]
        self._judge_model = judge_model_path
        self._judge_url = judge_server_url.rstrip("/")
        self._judge_prompt_style = judge_prompt_style
        self._use_finnish = use_finnish
        self._judge_prompt_template = _DEFAULT_JUDGE_PROMPT_FI if use_finnish else (judge_prompt_template or default_prompt)
        self._judge_ref_template = _JUDGE_PROMPT_WITH_REF_FI if use_finnish else ref_prompt
        self._score_scale = scale
        self._max_judge_tokens = max_judge_tokens
        self._judge_temperature = judge_temperature
        self._language_reward_weight = language_reward_weight
        self._session = None
        if judge_prompt_style == "open-instruct" and max_judge_tokens < 256:
            logger.warning(
                "judge_prompt_style='open-instruct' reasons before scoring, but max_judge_tokens=%d. "
                "The budget will likely run out before the JSON score is emitted.",
                max_judge_tokens,
            )
        self.add_reward_func(self.llm_judge_score)
        self.add_metric(self.language_score_metric)

    def _render_prompt(self, question: str, answer: str, reference: str, category: str) -> str:
        use_ref = category == "general-quality_ref" and reference
        if self._judge_prompt_style == "open-instruct":
            if use_ref:
                return self._judge_ref_template.format(input=question, output=answer, label=reference)
            return self._judge_prompt_template.format(input=question, output=answer)
        if use_ref:
            return self._judge_ref_template.format(question=question, answer=answer, reference=reference)
        return self._judge_prompt_template.format(question=question, answer=answer)

    async def _call_judge(self, question: str, answer: str, reference: str, category: str, temperature: float) -> str:
        import aiohttp
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=_JUDGE_TIMEOUT_S)
            )
        user_content = self._render_prompt(question, answer, reference, category)
        payload = {
            "model": self._judge_model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": self._max_judge_tokens,
            "temperature": temperature,
            # Disable Qwen3 thinking mode — without this the <think> block fills the
            # token budget before the model can emit "Score: N".
            "chat_template_kwargs": {"enable_thinking": False},
        }
        url = f"{self._judge_url}/v1/chat/completions"
        async with self._session.post(url, json=payload) as resp:
            if not resp.ok:
                raise JudgeRequestError(url, resp.status, await resp.text())
            data = await resp.json()
        return data["choices"][0]["message"]["content"]

    async def _score_with_retries(
        self, question: str, answer: str, reference: str, category: str
    ) -> tuple[float | None, str, JudgeError | None]:
        """Judge one response, retrying transport failures and format drift alike.

        Returns ``(score, last judge output, error)``. ``score`` is None exactly when
        ``error`` is set — the exception to assign to ``state["error"]``. Decoding
        stays unconstrained (no guided JSON, no schema), matching open-instruct.
        """
        import aiohttp
        last_output = ""
        error: JudgeError | None = None
        temperature = self._judge_temperature
        for attempt in range(_JUDGE_MAX_ATTEMPTS):
            try:
                last_output = await self._call_judge(question, answer, reference, category, temperature)
            except JudgeRequestError as e:
                error = e
                if not e.retryable:
                    # The server will reject the same request however often we send it.
                    break
            except (TimeoutError, aiohttp.ClientError) as e:
                # asyncio.TimeoutError stringifies to "", so name the class explicitly.
                error = JudgeTimeout(f"{type(e).__name__}: {e}")
            else:
                score = self._parse_score(last_output)
                if score is not None:
                    return score, last_output, None
                error = JudgeParseFailure(f"unparseable judge output: {last_output[:200]!r}")
                # A greedy judge replays the same unparseable answer verbatim, so a format
                # retry has to sample. Transport retries keep the configured temperature.
                if not temperature:
                    temperature = _JUDGE_RETRY_TEMPERATURE
            if attempt < _JUDGE_MAX_ATTEMPTS - 1:
                logger.warning(
                    "LLM judge attempt %d/%d failed (%s); retrying",
                    attempt + 1, _JUDGE_MAX_ATTEMPTS, str(error)[:200],
                )
                await asyncio.sleep(_JUDGE_RETRY_DELAY_S * 2**attempt)
        return None, last_output, error

    def _parse_score(self, text: str) -> float | None:
        """Reward in [0, 1], or None when the judge's answer carries no usable score."""
        if self._judge_prompt_style == "open-instruct":
            raw = _extract_json_score(text)
        else:
            raw = _parse_score_legacy(text)
        if raw is None:
            return None
        if not 0.0 <= raw <= self._score_scale:
            logger.warning(
                "LLM judge: raw score %s is outside the 0-%g scale, clamping", raw, self._score_scale
            )
            raw = max(0.0, min(self._score_scale, raw))
        return raw / self._score_scale

    async def llm_judge_score(self, completion, answer, info: dict, state: dict, **kwargs) -> float:
        response_text = _extract_text(completion)
        question = info.get("prompt_text", "")
        reference = info.get("reference", "")
        category = info.get("category", "")
        score, judge_output, error = await self._score_with_retries(
            question, response_text, reference, category
        )
        if score is None:
            logger.error(
                "LLM judge gave no usable score after %d attempts (%s) — dropping rollout",
                _JUDGE_MAX_ATTEMPTS, str(error)[:300],
            )
            # prime-rl excludes errored rollouts from the batch and from the group's
            # advantage baseline; the rate shows up as all/has_error/mean in wandb. The
            # per-type split does not: verifiers' rollout_output_to_trace reads the "type"
            # key while error_data writes "error", so every error lands under all/error/Error.
            state["error"] = error
            return 0.0

        try:
            from language_reward import compute_language_score
            lang_text = _extract_text(completion, strip_think=False) if self._use_finnish else response_text
            lang_score = compute_language_score(question, lang_text)
            if state is not None:
                state["_lang_score"] = lang_score
            if self._language_reward_weight > 0.0:
                score = score * (1.0 - self._language_reward_weight + self._language_reward_weight * lang_score)
        except Exception:
            pass

        if random.random() < _LOG_SAMPLE_RATE:
            if isinstance(completion, list):
                reasoning_parts = [
                    msg.get("reasoning_content") or ""
                    for msg in completion
                    if msg.get("role") == "assistant"
                ]
                reasoning_text = "\n".join(filter(None, reasoning_parts))
            else:
                reasoning_text = ""
            ref_line = f"  reference ({len(reference)} chars): {reference!r}\n" if reference else ""
            reasoning_line = f"  reasoning ({len(reasoning_text)} chars): {reasoning_text!r}\n" if reasoning_text else ""
            logger.info(
                "LLM judge sample\n"
                f"  question ({len(question)} chars): {question!r}\n"
                + reasoning_line +
                f"  answer ({len(response_text)} chars): {response_text!r}\n"
                + ref_line +
                f"  judge output: {judge_output!r}\n"
                f"  score: {score:.2f}"
            )
        return score

    def language_score_metric(self, completion, answer, info, state, **kwargs) -> float:
        return state.get("_lang_score", 1.0)
