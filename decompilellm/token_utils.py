import sys
import tiktoken
from .constants import (
    PROVIDER_ALIASES,
    PROVIDER_OPENAI,
    PROVIDER_GOOGLE,
    YELLOW,
    RED,
    RESET,
)

def get_token_count(text: str, model_name: str, provider: str) -> int:
    """Estimates the number of tokens in a given text string based on the model."""
    if not text:
        return 0

    token_model = model_name
    provider = PROVIDER_ALIASES.get(provider.lower(), PROVIDER_OPENAI)

    if provider == PROVIDER_GOOGLE:
        print(
            f"{YELLOW}WARNING! Gemini models do not formally support token estimation via tiktoken. "
            f"Defaulting to 'gpt-4o' for token estimation... "
            f"(This may cause issues with auto splitting).{RESET}",
            file=sys.stderr,
        )
        token_model = "gpt-4o"
    print(
        f"[Token Count] Estimating token count for model '{model_name}' (effective: '{token_model}')...",
        file=sys.stderr,
    )
    try:
        encoding = tiktoken.encoding_for_model(token_model)
        token_count = len(encoding.encode(text))
        print(f"[Token Count] Estimated token count: {token_count}", file=sys.stderr)
        return token_count
    except Exception as e:
        if provider == PROVIDER_GOOGLE:
            pass
        else:
            print(
                f"{YELLOW}Warning: Could not get tiktoken encoding for model '{model_name}' (effective: '{token_model}'). "
                f"Error: {e}. Falling back to 'cl100k_base' encoding.{RESET}",
                file=sys.stderr,
            )
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(text))
            print(f"[Token Count] Estimated token count: {token_count}", file=sys.stderr)
            return token_count
        except Exception as e_fallback:
            print(
                f"{RED}Error: Fallback tiktoken encoding 'cl100k_base' also failed: {e_fallback}. "
                f"Reverting to character-based token estimation.{RESET}",
                file=sys.stderr,
            )

    token_count = len(text) // 4  # fallback
    print(f"[Token Count] Estimated token count: {token_count}", file=sys.stderr)
    return token_count
