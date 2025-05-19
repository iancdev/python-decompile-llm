import argparse
import sys
import os
import time
from .constants import (
    DEFAULT_LLM_MODEL_OPENAI,
    DEFAULT_LLM_MODEL_GOOGLE,
    DEFAULT_SYSTEM_MESSAGE,
    PROVIDER_ALIASES,
    PROVIDER_GOOGLE,
    PROVIDER_OPENAI,
    GREEN,
    YELLOW,
    RED,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_WORKERS_FOR_ITERATIONS,
)
from .api import get_api_key
from .decompiler import decompile
from .utils import verify
from .constants import RESET

def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM powered Python decompiler\n"
            "Specify your API key in environment variables or use --key."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("pyc_file", help="Path to the .pyc file to decompile.")
    parser.add_argument(
        "--model", default=None,
        help=f"LLM model (default for OpenAI: {DEFAULT_LLM_MODEL_OPENAI}, for Google: {DEFAULT_LLM_MODEL_GOOGLE})."
    )
    parser.add_argument(
        "--key", default=None,
        help="API key for the provider. Uses OPENAI_API_KEY/GEMINI_API_KEY from env if not set."
    )
    parser.add_argument(
        "--systemmsg", "--system-message", default=DEFAULT_SYSTEM_MESSAGE,
        help=f"Custom system message for decompiler LLM."
    )
    parser.add_argument(
        "--iter", type=int, default=1,
        help="Number of iterations (default: 1). Runs LLM multiple times (per chunk if split) and picks best."
    )
    parser.add_argument(
        "--verify", choices=['yes', 'no'], default='yes',
        help="Verify Python syntax of the decompiled code (default: yes)."
    )
    parser.add_argument(
        "--retry", type=int, default=0,
        help="Number of retries if decompilation or verification fails (default: 0)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path. Prints to console (stdout) if not provided."
    )
    
    parser.add_argument(
        '--stream', action=argparse.BooleanOptionalAction, default=None,
        help="Enable streaming output (default: True for CLI, False for file output)."
    )
    parser.add_argument(
        '--multithreaded', action=argparse.BooleanOptionalAction, default=True,
        help="Enable multithreading for iterations (default: True)."
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of threads for iterations. Defaults to --iter count (capped) if > 1, else 1."
    )
    parser.add_argument(
        "--provider", default="openai", type=str.lower,
        choices=list(PROVIDER_ALIASES.keys()) + ["openai", "google"],
        help=f"LLM provider (default: openai). Options: {', '.join(sorted(list(set(PROVIDER_ALIASES.keys()))))}."
    )
    parser.add_argument(
        "--split", type=int, default=0,
        help="Manually split bytecode into N chunks by char length (default: 0). Overrides --auto-split."
    )
    parser.add_argument(
        "--auto-split", action="store_true",
        help="Automatically split large disassembly based on --max-tokens if --split is not used."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens for a disassembly chunk when --auto-split is active (default: {DEFAULT_MAX_TOKENS}). "
             f"Requires tiktoken library."
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS,
        help=f"Max chars for a chunk if token-based splitting is not possible (default: {DEFAULT_MAX_CHARS}). "
             f"Primarily for fallback if tiktoken is unavailable."
    )
    parser.add_argument(
        "--temp", "--temperature", type=float, default=0.5,
        help="Optional: Model temperature (e.g., 0.0-2.0 for OpenAI, 0.0-1.0 for Gemini). Lower is more deterministic. Default: 0.5"
    )
    parser.add_argument(
        "--topp", "--top-p", type=float, default=1.0,
        help="Optional: Model top_p (e.g., 0.0-1.0). Nucleus sampling. Default: 1.0."
    )
    parser.add_argument(
        "--effort",
        type=str,
        choices=['low', 'medium', 'high', 'none'],
        default='none',
        help="Reasoning effort for the LLM (optional, options: low, medium, high, none. Defaults to none). \n"
             "If 'none' or not specified, reasoning effort will not be set.\n"
             "This disables reasoning on Gemini models."
    )

    args = parser.parse_args()

    if args.stream is None:
        args.stream = args.output is None 
    # auto resolve provider to google if gemini is in model name, in case the user forgets to specify the provider
    if args.provider == None and args.model != None and 'gemini' in args.model.lower():
        provider = PROVIDER_GOOGLE
    else:
        provider = PROVIDER_ALIASES.get(args.provider.lower(), PROVIDER_OPENAI)
    if args.model is None:
        args.model = DEFAULT_LLM_MODEL_GOOGLE if provider == PROVIDER_GOOGLE else DEFAULT_LLM_MODEL_OPENAI
        print(f"Info: Using default model for {provider}: {args.model}", file=sys.stderr)

    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be a positive integer.")
    if args.threads is None and args.multithreaded and args.iter > 1:
        args.threads = min(args.iter, DEFAULT_MAX_WORKERS_FOR_ITERATIONS)
    elif not args.multithreaded or args.iter == 1:
        args.threads = 1

    if args.iter < 1: parser.error("--iter must be a positive integer.")
    if args.retry < 0: parser.error("--retry must be a non-negative integer.")
    if args.max_tokens <= 0: parser.error("--max-tokens must be positive.")
    if args.max_chars <= 0: parser.error("--max-chars must be positive.")
    if args.split < 0: parser.error("--split must be non-negative.")

    if not (0.0 <= args.temp <= 2.0):
        print(f"{YELLOW}Warning: --temp value {args.temp} for {provider} is outside the typical range [0.0, 2.0].{RESET}", file=sys.stderr)
    if not (0.0 <= args.topp <= 1.0):
        print(f"{YELLOW}Warning: --topp value {args.topp} for {provider} is outside the typical range [0.0, 1.0].{RESET}", file=sys.stderr)
    
    if args.effort != None:
        if args.effort.lower() not in ["low", "medium", "high", "none"]:
            parser.error("--effort must be 'low', 'medium', 'high', or 'none'.")
        args.effort = args.effort.lower()
        if args.effort != 'none': print(f"{GREEN}Effort set to {args.effort.upper()}. This may get costly depending on your model provider!{RESET}", file=sys.stderr)

    key = get_api_key(args.provider, args.key)
    if not key:
        env_key = "GEMINI_API_KEY" if provider == PROVIDER_GOOGLE else "OPENAI_API_KEY"
        parser.error(
            f"{RED}{provider.capitalize()} API key not found. Provide via --key or {env_key} env var.{RESET}"
        )
    
    final_code = None
    error_msg = "Decompilation did not yield a result."

    for attempt_num in range(args.retry + 1):
        is_last_attempt = (attempt_num == args.retry)
        if attempt_num > 0:
            print(f"\n--- Retry Attempt {attempt_num}/{args.retry} ---", file=sys.stderr)

        print(
            f"Running decompilation for '{args.pyc_file}' (Provider: {args.provider}, Model: {args.model}, "
            f"Iter: {args.iter}, Threads: {args.threads}, Stream: {args.stream}, "
            f"Verify: {args.verify}, Split: {'manual '+str(args.split) if args.split > 0 else ('auto' if args.auto_split else 'none')}, "
            f"Max Tokens: {args.max_tokens if args.auto_split and args.split==0 else 'N/A'}, "
            f"Attempt: {attempt_num + 1}/{args.retry + 1})...",
            file=sys.stderr
        )

        current_code, error = decompile(args, key)

        if error:
            error_msg = f"Decompilation attempt {attempt_num + 1} failed: {error}"
            print(f"{RED}{error_msg}{RESET}", file=sys.stderr)
            if is_last_attempt: break
            print("Proceeding to next retry if available.", file=sys.stderr)
            time.sleep(1)
            continue

        if current_code == "__STREAMED_TO_FILE__":
            print("Decompilation streamed to file successfully.", file=sys.stderr)
            if args.verify.lower() == 'yes':
                print(f"{YELLOW}Verification of file content (when streamed to '{args.output}') "
                      f"is recommended manually or use '--verify no'.{RESET}", file=sys.stderr)
            final_code = "__STREAMED_TO_FILE__"
            break 

        if current_code is None:
            error_msg = f"Decompilation attempt {attempt_num + 1} returned no code."
            print(f"{RED}{error_msg}{RESET}", file=sys.stderr)
            if is_last_attempt: break
            print("Proceeding to next retry if available.", file=sys.stderr)
            time.sleep(1)
            continue
        
        if args.verify.lower() == 'yes':
            print("[Verify] Verifying syntax of the decompiled code...", file=sys.stderr)
            verified, verify_msg = verify(current_code)
            if verified:
                print("[Verify] Syntax verification successful.", file=sys.stderr)
                final_code = current_code
                break
            else:
                error_msg = f"[Verify] Syntax verification failed (attempt {attempt_num + 1}): {verify_msg}"
                print(f"{RED}{error_msg}{RESET}", file=sys.stderr)
                if is_last_attempt:
                    final_code = current_code 
                    break
        else:
            final_code = current_code
            break

    if final_code is None:
        print(f"\n{RED}Failed to produce decompiled code for '{args.pyc_file}'.", file=sys.stderr)
        print(f"Last error: {error_msg}{RESET}", file=sys.stderr)
        sys.exit(1)

    if final_code == "__STREAMED_TO_FILE__":
        sys.exit(0)

    verified = True 
    verify_msg = ""
    if args.verify.lower() == 'yes': 
        verified, verify_msg = verify(final_code)
        if not verified:
            print(f"\n{YELLOW}Warning: Final decompiled code is syntactically invalid. Error: {verify_msg}{RESET}", file=sys.stderr)
    
    elif args.verify.lower() == 'no' and not verify(final_code)[0]:
         print(f"\n{YELLOW}Warning: Final decompiled code (verification was 'no') appears syntactically invalid.{RESET}", file=sys.stderr)


    if args.output:
        if args.verify.lower() == 'yes' and not verified:
            print(f"{RED}Error: Syntactically invalid code will not be written to '{args.output}'. "
                  f"Fix or use '--verify no'.{RESET}", file=sys.stderr)
            if len(final_code) < 5000:
                 print("\n--- Invalid Code (Not Written to File) ---", file=sys.stderr)
                 sys.stderr.write(final_code + "\n")
                 sys.stderr.flush()
            sys.exit(1)
        try:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(final_code)
            print(f"Decompiled code written to '{args.output}'.", file=sys.stderr)
        except IOError as e:
            print(f"{RED}Error: Could not write to output file '{args.output}': {e}{RESET}", file=sys.stderr)
            print("\n--- Decompiled Code (Fallback to Console Output) ---", file=sys.stderr)
            sys.stdout.write(final_code)
            sys.stdout.flush()
            if args.verify.lower() == 'yes' and not verified:
                 print(f"\n{YELLOW}(Warning: The code printed above (fallback) is syntactically invalid).{RESET}", file=sys.stderr)
            sys.exit(1)
    else: 
        if not args.stream:
            sys.stdout.write(final_code) 
            if not final_code.endswith('\n'):
                sys.stdout.write("\n")
            sys.stdout.flush()
    if args.verify.lower() == 'yes' and not verified:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
