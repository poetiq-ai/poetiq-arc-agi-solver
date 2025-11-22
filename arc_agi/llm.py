import asyncio
from typing import Any

import litellm
from asynciolimiter import Limiter
from litellm import acompletion
from litellm import exceptions as litellm_exceptions

from arc_agi.types import Models

# Silence unnecessary litellm logs.
litellm.suppress_debug_info = True

RETRIES = 3
RETRY_DELAY_SEC = 5

limiters: dict[Models, Limiter] = {
    "groq/openai/gpt-oss-120b": Limiter(1.0),
    "openai/gpt-5": Limiter(1.0),
    "openai/gpt-5.1": Limiter(1.0),
    "xai/grok-4-fast": Limiter(1.0),
    "xai/grok-4": Limiter(1.0),
    "anthropic/claude-sonnet-4-5": Limiter(1.0),
    "anthropic/claude-haiku-4-5": Limiter(1.0),
    "gemini/gemini-2.5-pro": Limiter(2.0),
    "gemini/gemini-3-pro-preview": Limiter(1.0),
}

props: dict[Models, dict] = {
    "groq/openai/gpt-oss-120b": {},
    "openai/gpt-5": {"reasoning_effort": "high"},
    "openai/gpt-5.1": {"reasoning_effort": "high"},
    "xai/grok-4-fast": {},
    "xai/grok-4": {},
    "anthropic/claude-sonnet-4-5": {"thinking": {"type": "enabled", "budget_tokens": 32_000}},
    "anthropic/claude-haiku-4-5": {"thinking": {"type": "enabled", "budget_tokens": 32_000}},
    "gemini/gemini-2.5-pro": {"thinking": {"type": "enabled", "budget_tokens": 16_000}},
    "gemini/gemini-3-pro-preview": {},
}


async def llm(
    model: Models,
    message: str,
    temperature,
    request_timeout: int | None,
    max_remaining_time: float | None,
    max_remaining_timeouts: int | None,
    problem_id: str | None = None,
    retries: int = RETRIES,
) -> tuple[str, float, float | None, int | None]:
    attempt = 1
    while attempt <= retries:
        await limiters[model].wait()

        current_request_timeout = request_timeout or 15 * 60
        if max_remaining_time is not None:
            current_request_timeout = min(current_request_timeout, max_remaining_time)

        start_time = asyncio.get_event_loop().time()
        try:
            resp: Any = await acompletion(
                model=model,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                timeout=current_request_timeout,
                num_retries=0,
                **props[model],
            )
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            if max_remaining_time is not None:
                max_remaining_time -= duration
            return (
                resp["choices"][0]["message"]["content"].strip(),
                duration,
                max_remaining_time,
                max_remaining_timeouts,
            )

        except (
            litellm_exceptions.RateLimitError,
            litellm_exceptions.InternalServerError,
            litellm_exceptions.ServiceUnavailableError,
            litellm_exceptions.APIConnectionError,
            litellm_exceptions.APIError,
            litellm.RouterRateLimitError,
            litellm.RouterRateLimitErrorBasic,
        ) as e:
            # None of these exceptions should prevent the problem from being solved, so don't let them count against the allotted retries.
            print(f"{problem_id or ''} Ignoring {type(e).__name__} and retrying attempt {attempt}: {e}")
            await asyncio.sleep(RETRY_DELAY_SEC)
            continue

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            if max_remaining_time is not None:
                max_remaining_time -= duration

            if "Timeout" in str(e):
                if max_remaining_timeouts is not None:
                    max_remaining_timeouts -= 1
                    print(
                        f"{problem_id or ''} Timed out. Remaining timeouts: {max_remaining_timeouts}"
                    )
                if max_remaining_timeouts is not None and max_remaining_timeouts <= 0:
                    raise RuntimeError("Exceeded timeouts allotted to the request")

                if attempt == retries:
                    return (
                        "Timeout",
                        duration,
                        max_remaining_time,
                        max_remaining_timeouts,
                    )
            if max_remaining_time is not None and max_remaining_time <= 0:
                raise RuntimeError("Exceeded time allotted to the request")

            if attempt == retries:
                print(f"{problem_id or ''} Max retry limit reached. Last exception during call:")
                print(str(e))
                raise e

            print(str(e))
            print(f"Exception during request for problem: {problem_id or ''}. Retry number {attempt}.")
            await asyncio.sleep(RETRY_DELAY_SEC)

            # Increment attempt at the end of the loop.
            attempt += 1

    raise RuntimeError("Retries exceeded")
