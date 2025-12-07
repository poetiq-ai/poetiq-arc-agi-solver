import asyncio
import os
from typing import Any

import httpx
import litellm
from asynciolimiter import Limiter
from litellm import acompletion
from litellm import exceptions as litellm_exceptions

from arc_agi.types import Models

# Silence unnecessary litellm logs.
litellm.suppress_debug_info = True

RETRIES = 3
RETRY_DELAY_SEC = 5

DIRECT_OPENAI_MODELS = {"gpt-5.1"}

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
    "gpt-5.1": Limiter(1.0),
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
    "gpt-5.1": {"reasoning": {"effort": "high"}},
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
) -> tuple[str, float, float | None, int | None, int, int]:
    attempt = 1
    while attempt <= retries:
        await limiters[model].wait()

        current_request_timeout = request_timeout or 15 * 60
        if max_remaining_time is not None:
            current_request_timeout = min(current_request_timeout, max_remaining_time)

        start_time = asyncio.get_event_loop().time()
        try:
            if model in DIRECT_OPENAI_MODELS:
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json",
                }
                use_background = current_request_timeout > 15 * 60
                payload = {
                    "model": model,
                    "input": message,
                    "temperature": temperature,
                    "background": use_background,
                    "store": True,
                    **props[model],
                }
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/responses",
                        headers=headers,
                        json=payload,
                        timeout=current_request_timeout,
                    )
                    resp.raise_for_status()
                    resp_json = resp.json()

                    if use_background:
                        resp_id = resp_json["id"]
                        status = resp_json["status"]

                        while status in {"queued", "in_progress"}:
                            await asyncio.sleep(30)
                            poll_start = asyncio.get_event_loop().time()

                            try:
                                poll_resp = await client.get(
                                    f"https://api.openai.com/v1/responses/{resp_id}",
                                    headers=headers,
                                    timeout=min(60, current_request_timeout),
                                )
                                poll_resp.raise_for_status()
                            except Exception as e:
                                print('Polling failed', str(e), 'retrying.')
                                continue
                            
                            resp_json = poll_resp.json()
                            status = resp_json["status"]

                            poll_end = asyncio.get_event_loop().time()
                            duration = poll_end - poll_start
                            if max_remaining_time is not None:
                                max_remaining_time -= duration
                                if max_remaining_time <= 0:
                                    raise RuntimeError("Exceeded Timeout allotted to the request") # catch as a timeout instead of nothing

                end_time = asyncio.get_event_loop().time()
                duration = end_time - start_time
                if max_remaining_time is not None:
                    max_remaining_time -= duration

                if not resp_json:
                    raise litellm_exceptions.InternalServerError("Empty response from server", model.split("/")[0], model.split("/")[-1])
                
                prompt_tokens = resp_json["usage"]["input_tokens"]
                completion_tokens = resp_json["usage"]["output_tokens"]

                text = extract_text_from_response(resp_json)

                return (
                    text,
                    duration,
                    max_remaining_time,
                    max_remaining_timeouts,
                    prompt_tokens,
                    completion_tokens,
                )

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

            prompt_tokens = resp.model_extra.get("usage").prompt_tokens
            completion_tokens = resp.model_extra.get("usage").completion_tokens

            return (
                resp["choices"][0]["message"]["content"].strip(),
                duration,
                max_remaining_time,
                max_remaining_timeouts,
                prompt_tokens,
                completion_tokens,
            )

        except (
            litellm_exceptions.RateLimitError,
            litellm_exceptions.InternalServerError,
            litellm_exceptions.ServiceUnavailableError,
            litellm_exceptions.APIConnectionError,
            litellm_exceptions.APIError,
            litellm.RouterRateLimitError,
            litellm.RouterRateLimitErrorBasic,
            httpx.HTTPStatusError,
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

            if "Timeout" in str(e) or isinstance(e, httpx.TimeoutException):
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
                        0,
                        0,
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


def extract_text_from_response(resp_json: dict) -> str:
    """
    Extract assistant text from a Responses API JSON payload.
    """
    output_items = resp_json.get("output", [])
    for item in output_items:
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    text = block.get("text", "")
                    if text is not None:
                        return text.strip()

    raise RuntimeError(f"No assistant message with output_text found in response: {resp_json}")
