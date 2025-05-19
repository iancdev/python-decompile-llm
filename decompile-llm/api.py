import os
import sys
import io
import openai
from .constants import (
    PROVIDER_ALIASES,
    GREEN,
    PROVIDER_OPENAI,
    PROVIDER_GOOGLE,
    LLM_REQUEST_TIMEOUT_SECONDS,
    YELLOW,
    RED,
    RESET,
)
from .token_utils import get_token_count


def get_api_key(args_provider: str, args_key: str):
    provider = PROVIDER_ALIASES.get(args_provider.lower(), PROVIDER_OPENAI)

    if args_key:
        print(f"{GREEN}A key was provided in args. Not checking environment variable.{RESET}")

    if provider == PROVIDER_GOOGLE:
        key = args_key if args_key else os.environ.get("GEMINI_API_KEY")
        return key
    else:
        key = args_key if args_key else os.environ.get("OPENAI_API_KEY")
        return key


def call_llm(
    api_key: str,
    model: str,
    system_message: str,
    prompt: str,
    provider: str,
    stream_output: bool,
    output_file_handle: io.TextIOWrapper = None,
    timeout: int = LLM_REQUEST_TIMEOUT_SECONDS,
    temperature: float = 0.5,
    top_p: float = 1.0,
    reasoning_effort: str = None,
):
    """Calls the LLM to decompile."""
    if not all(isinstance(arg, str) for arg in [api_key, model, system_message, prompt, provider]):
        return None, "Error: All string arguments to call_llm must be strings."
    if not api_key:
        return None, f"Error: API key for {provider} cannot be empty."
    if not model:
        return None, "Error: LLM model name cannot be empty."

    provider = PROVIDER_ALIASES.get(provider.lower(), PROVIDER_OPENAI)

    client_config = {}

    if provider == PROVIDER_GOOGLE:
        client_config["base_url"] = "https://generativelanguage.googleapis.com/v1beta"
        gemini_key = os.environ.get("GEMINI_API_KEY") or api_key
        if not gemini_key:
            return None, "Error: GEMINI_API_KEY not found for Google provider."
        api_key = gemini_key
        if not model.startswith("models/"):
            model = f"models/{model}"

    try:
        client = openai.OpenAI(api_key=api_key, **client_config)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        llm_params = {"model": model, "messages": messages, "timeout": float(timeout)}

        if temperature is not None:
            llm_params["temperature"] = temperature
        if top_p is not None:
            llm_params["top_p"] = top_p

        if reasoning_effort and reasoning_effort.lower() != "none":
            llm_params["reasoning_effort"] = reasoning_effort.lower()

        if stream_output:
            try:
                full_content = []
                stream_call_params = llm_params.copy()
                stream_call_params["stream"] = True
                print("[LLM] Sending request...")
                stream = client.chat.completions.create(**stream_call_params)
                print("[LLM] Received response.")
                if output_file_handle:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            output_file_handle.write(content)
                            output_file_handle.flush()
                            full_content.append(content)
                else:
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            full_content.append(content)
                    sys.stdout.write("\n")

                result = "".join(full_content)
                if not result.strip():
                    return None, "LLM streamed response was empty after stripping."
                return result.strip(), None
            except Exception as e:
                return None, f"LLM API stream error ({provider} - {model}): {e!r}"
        else:
            non_stream_param = llm_params.copy()
            non_stream_param["stream"] = False
            print("[LLM] Sending request...")
            completion = client.chat.completions.create(**non_stream_param)
            print("[LLM] Received response.")
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content is not None:
                content = completion.choices[0].message.content
                if content.startswith("```python\n"):
                    content = content[len("```python\n") :]
                elif content.startswith("```\n"):
                    content = content[len("```\n") :]
                if content.endswith("\n```"):
                    content = content[: -len("\n```")]
                elif content.endswith("```"):
                    content = content[: -len("```")]

                stripped_content = content.strip()
                if not stripped_content:
                    err_msg = "LLM response was empty after stripping markdown and whitespace."
                    if completion.choices[0].finish_reason:
                        err_msg += f" Finish reason: {completion.choices[0].finish_reason}."
                    return None, err_msg
                return stripped_content, None
            else:
                err_msg = "LLM response structure was unexpected or content was missing."
                if completion.choices and completion.choices[0].finish_reason:
                    err_msg += f" Finish reason: {completion.choices[0].finish_reason}."
                elif not completion.choices:
                    err_msg += " No choices returned by LLM."
                return None, err_msg

    except openai.AuthenticationError as e:
        return None, f"{provider} API Authentication Error: {e}. Check your API key."
    except openai.RateLimitError as e:
        return None, f"{provider} API Rate Limit Exceeded: {e}."
    except openai.NotFoundError as e:
        return None, f"{provider} API Error: Model '{model}' not found or API endpoint issue: {e}."
    except openai.BadRequestError as e:
        prompt_tokens = get_token_count(prompt, model, provider)
        system_tokens = get_token_count(system_message, model, provider)
        est_tokens = prompt_tokens + system_tokens
        return None, (
            f"{provider} API Bad Request Error: {e}. "
            f"(Est. prompt tokens: {prompt_tokens}, system: {system_tokens}, total: {est_tokens} "
            f"for model '{model}')"
        )
    except openai.APITimeoutError:
        return None, f"{provider} API request timed out after {timeout}s."
    except openai.APIConnectionError as e:
        return None, f"{provider} API Connection Error: {e}. Check network."
    except openai.APIError as e:
        return None, f"{provider} API Error ({type(e).__name__}): {e}."
    except Exception as e:
        return None, f"An unexpected error occurred during the LLM API call for {provider}: {e!r}"
