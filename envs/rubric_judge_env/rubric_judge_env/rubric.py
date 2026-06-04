import logging
import random

import verifiers as vf

logger = logging.getLogger("verifiers")

_LOG_SAMPLE_RATE = 0.01  # log ~1% of scored examples

_GENERAL_CATEGORIES = {"general-quality", "general-quality_ref"}


def _extract_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = [msg.get("content", "") for msg in completion if isinstance(msg, dict) and msg.get("role") == "assistant"]
        return "\n".join(parts)
    return str(completion)


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
    ):
        super().__init__()
        self._rm_model_path = rm_model_path
        self._rm_device = rm_device
        self._rm_server_url = rm_server_url.rstrip("/") if rm_server_url else None
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
