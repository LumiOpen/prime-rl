"""Monkey-patch verifiers' TITO client to support PrefixRL exact token-prefix continuation.

The standard TITO client falls back to message-based inference on the first turn
(trajectory length 0). For PrefixRL, we need first-turn inference to use exact
pre-tokenized prefix_tokens from the dataset.

This patch checks for `prefix_tokens` in the RolloutInput. When present, it
sends those tokens directly to /chat/completions/tokens, bypassing the
trajectory-matching logic entirely.

Import this module early (e.g., in orchestrator.py) to apply the patch.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_prefix_token_patch() -> None:
    """Patch OpenAIChatCompletionsTokenClient.get_native_response to handle prefix_tokens."""
    global _PATCHED
    if _PATCHED:
        return

    from verifiers.clients.openai_chat_completions_token_client import (
        OpenAIChatCompletionsTokenClient,
    )

    _original_get_native_response = OpenAIChatCompletionsTokenClient.get_native_response

    async def _patched_get_native_response(
        self,
        prompt,
        model: str,
        sampling_args: dict[str, Any],
        tools=None,
        **kwargs,
    ):
        state = kwargs.get("state")
        if state is not None:
            rollout_input = state.get("input")
            prefix_tokens = rollout_input.get("prefix_tokens") if rollout_input else None
        else:
            prefix_tokens = None

        if prefix_tokens is not None and isinstance(prefix_tokens, list) and len(prefix_tokens) > 0:
            # PrefixRL path: send exact prefix tokens to /chat/completions/tokens
            sampling_args = dict(sampling_args)
            if "max_tokens" in sampling_args:
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
            sampling_args["logprobs"] = True
            extra_body = dict(return_token_ids=True)
            if "extra_body" in sampling_args:
                sampling_args["extra_body"] = {
                    **sampling_args["extra_body"],
                    **extra_body,
                }
            else:
                sampling_args["extra_body"] = extra_body
            sampling_args = {k: v for k, v in sampling_args.items() if v is not None}

            extra_body = sampling_args.pop("extra_body", {})
            extra_headers = kwargs.pop("extra_headers", None)

            body = dict(
                model=model,
                messages=prompt,
                tools=tools,
                tokens=prefix_tokens,
                **sampling_args,
                **extra_body,
            )

            return await self.client.post(
                "/chat/completions/tokens",
                body=body,
                cast_to=ChatCompletion,
                options={"headers": extra_headers} if extra_headers else {},
            )

        # No prefix_tokens — use original behavior
        return await _original_get_native_response(self, prompt, model, sampling_args, tools, **kwargs)

    OpenAIChatCompletionsTokenClient.get_native_response = _patched_get_native_response
    _PATCHED = True
    logger.info("Applied PrefixRL exact token-prefix patch to TITO client")
