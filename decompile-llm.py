#!/usr/bin/env python3
"""
Decompiles .pyc files using an LLM via the OpenAI API or other compatible providers.
"""

import argparse
import ast
import io
import os
import sys
import types 
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from difflib import SequenceMatcher
import time

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
current_import = ''
try:
    current_import = 'xdis'
    import xdis.load
    import dis as std_dis
    current_import = 'OpenAI'
    import openai
    current_import = 'tqdm'
    import tqdm
    current_import = 'tiktoken'
    import tiktoken
except ImportError as exc:
    print(
        f"{RED}Error: The '{current_import}' library is required but could not be imported."
        f"({exc}). Did you install from requirements.txt? Install/upgrade with:  pip install -r requirements.txt{RESET}",
        file=sys.stderr,
    )
    sys.exit(1)

DEFAULT_LLM_MODEL_OPENAI = "gpt-4.1" 
DEFAULT_LLM_MODEL_GOOGLE = "gemini-2.5-flash"
DEFAULT_SYSTEM_MESSAGE = (
    "You are a Python decompiler. Given the following Python bytecode "
    "disassembly, please provide the corresponding Python source code. "
    "Output only the raw Python code. Do not include any explanations, "
    "comments about the process, or markdown code block delimiters."
)
LLM_REQUEST_TIMEOUT_SECONDS = 180 
THREAD_COMPLETION_TIMEOUT_SECONDS = LLM_REQUEST_TIMEOUT_SECONDS + 60 
DEFAULT_MAX_CHARS = 50000  
DEFAULT_MAX_TOKENS = 10000 
DEFAULT_MAX_WORKERS_FOR_ITERATIONS = 10 

PROVIDER_OPENAI = "openai"
PROVIDER_GOOGLE = "google"
PROVIDER_ALIASES = {
    "chatgpt": PROVIDER_OPENAI,
    "gpt": PROVIDER_OPENAI,
    "gemini": PROVIDER_GOOGLE,
    "google": PROVIDER_GOOGLE,
    "openai": PROVIDER_OPENAI,
}



def get_token_count(text: str, model_name: str, provider: str) -> int:
    """
    Estimates the number of tokens in a given text string based on the model.
    """
    if not text:
        return 0

    token_model = model_name
    provider = PROVIDER_ALIASES.get(provider.lower(), PROVIDER_OPENAI)

    if provider == PROVIDER_GOOGLE:
        print(
            f"{YELLOW}WARNING! Gemini models do not formally support token estimation via tiktoken. "
            f"Defaulting to 'gpt-4o' for token estimation... "
            f"(This may cause issues with auto splitting).{RESET}",
            file=sys.stderr
        )
        token_model = "gpt-4o"
    print(f"[Token Count] Estimating token count for model '{model_name}' (effective: '{token_model}')...", file=sys.stderr)
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
                file=sys.stderr
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
                file=sys.stderr
            )

    token_count = len(text) // 4 #fallback
    print(f"[Token Count] Estimated token count: {token_count}", file=sys.stderr)
    return token_count


def disassemble(pyc_file: str):
    """
    Disassembles provided .pyc file to readable bytecode.
    """
    if not isinstance(pyc_file, str) or not pyc_file:
        return None, "Error: .pyc file path must be a non-empty string."
    if not os.path.exists(pyc_file):
        return None, f"Error: File '{pyc_file}' not found."
    if not os.path.isfile(pyc_file):
        return None, f"Error: Path '{pyc_file}' is not a file."
    if os.path.getsize(pyc_file) == 0:
        return None, f"Error: File '{pyc_file}' is empty."

    try:
        print(f"[Disassembler] Disassembling... This might take a while...", file=sys.stderr)
        result = xdis.load.load_module(pyc_file)
        code = None
        if isinstance(result, tuple):
            if len(result) > 3 and isinstance(result[3], types.CodeType):
                 code = result[3]
            elif len(result) > 4 and isinstance(result[4], types.CodeType):
                 code = result[4]
            else:
                for item in result:
                    if isinstance(item, types.CodeType):
                        code = item
                        break

        elif isinstance(result, types.CodeType):
            code = result

        if code is None:
            return None, (
                f"Error: xdis failed to extract a valid code object from "
                f"'{pyc_file}'. Received type: {type(result)}. "
                f"Ensure xdis version compatibility and file integrity. Result: {result}"
            )

        string_io = io.StringIO()
        std_dis.dis(code, file=string_io)
        disassembly = string_io.getvalue()

        if not disassembly.strip():
            return None, (
                f"Error: Bytecode disassembly for '{pyc_file}' "
                "resulted in empty output. The .pyc might be trivial or corrupted."
            )
        print(f"[Disassembler] Disassembled successfully.", file=sys.stderr)
        return disassembly, None
    except PermissionError:
        return None, f"Error: Permission denied when trying to read '{pyc_file}'."
    except IOError as e:
        return None, (
            f"IOError during .pyc file processing for '{pyc_file}': {e}"
        )
    except Exception as e:
        return None, (
            f"An unexpected error occurred while reading or disassembling "
            f"'{pyc_file}': {e!r}"
        )


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
    reasoning_effort: str = None
):
    """
    Calls the LLM to decompile.
    """
    if not all(isinstance(arg, str) for arg in
               [api_key, model, system_message, prompt, provider]):
        return None, "Error: All string arguments to call_llm must be strings."
    if not api_key:
        return None, f"Error: API key for {provider} cannot be empty."
    if not model:
        return None, "Error: LLM model name cannot be empty."

    provider = PROVIDER_ALIASES.get(provider.lower(), PROVIDER_OPENAI)
    
    client_config = {}

    if provider == PROVIDER_GOOGLE:
        client_config['base_url'] = "https://generativelanguage.googleapis.com/v1beta"
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
            {"role": "user", "content": prompt}
        ]

        llm_params = {
            "model": model,
            "messages": messages,
            "timeout": float(timeout)
        }

        if temperature is not None:
            llm_params['temperature'] = temperature
        if top_p is not None:
            llm_params['top_p'] = top_p
        
        if reasoning_effort and reasoning_effort.lower() != 'none':
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
            if completion.choices and completion.choices[0].message and \
               completion.choices[0].message.content is not None:
                content = completion.choices[0].message.content
                # strip markdown in case llm generates them
                if content.startswith("```python\n"): content = content[len("```python\n"):]
                elif content.startswith("```\n"): content = content[len("```\n"):]
                if content.endswith("\n```"): content = content[:-len("\n```")]
                elif content.endswith("```"): content = content[:-len("```")]
                
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


def verify(code: str):
    """
    Checks code for syntax errors
    """
    if not code.strip():
        return False, "SyntaxError: Code is empty or contains only whitespace."
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        error_detail = f"Error: {e.msg}"
        if e.lineno is not None: error_detail += f" on line {e.lineno}"
        if e.offset is not None: error_detail += f", column {e.offset}"
        if e.text:
            problem_line = e.text.splitlines()[0] if isinstance(e.text, str) else ""
            error_detail += f". Problematic line: '{problem_line.strip()}'"
        return False, f"SyntaxError: {error_detail}"
    except RecursionError:
        return False, "Verification Error: Max recursion depth during AST parsing."
    except ValueError as e:
        return False, f"Verification Error during AST parsing (ValueError): {e}"
    except Exception as e:
        return False, f"Unexpected verification error: {e!r}"


def check_similarity(code1: str, code2: str) -> float:
    if not isinstance(code1, str) or not isinstance(code2, str): return 0.0
    if not code1 and not code2: return 1.0
    if not code1 or not code2: return 0.0
    return SequenceMatcher(None, code1, code2).ratio()


def split_manual(disassembled: str, num_chunks: int):
    """
    Splits by cahracter length as fallback
    """
    if num_chunks <= 0:
        return [disassembled]
    text_len = len(disassembled)
    chunk_size = (text_len + num_chunks - 1) // num_chunks
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, text_len)
        if start < end:
            chunks.append(disassembled[start:end])
    return chunks


def split_auto(
    disassembled: str,
    max_tokens: int,
    model_name: str,
    provider: str
):
    """
    Split by calculated token limit (preferred)
    """
    if not disassembled: return []
    if max_tokens <= 0:
        print(f"{YELLOW}Warning: max_tokens for splitting is <= 0. Ignoring split and returning as one chunk.{RESET}", file=sys.stderr)
        return [disassembled]

    lines = disassembled.splitlines(keepends=True)
    chunks = []
    current_chunks = []
    total_chunks = 0

    for line_num, line in enumerate(lines):
        current_count = get_token_count(line, model_name, provider)

        if current_count > max_tokens:
            if current_chunks:
                chunks.append("".join(current_chunks))
                current_chunks = []
                total_chunks = 0
            
            chunks.append(line)
            print(
                f"{YELLOW}Warning: Single line {line_num+1} ({current_count} tokens) exceeds max_tokens ({max_tokens}). "
                f"It will be processed as a separate, potentially oversized chunk.{RESET}",
                file=sys.stderr
            )
            continue

        if total_chunks + current_count > max_tokens and current_chunks:
            chunks.append("".join(current_chunks))
            current_chunks = [line]
            total_chunks = current_count
        else:
            current_chunks.append(line)
            total_chunks += current_count

    if current_chunks:
        chunks.append("".join(current_chunks))
    
    if not chunks and disassembled:
        return [disassembled]
        
    return chunks


def _decompile_llm(
    prompt: str,
    args: argparse.Namespace,
    api_key: str,
    out_file: io.TextIOWrapper = None,
    progress_desc: str = "LLM Call",
    iter_count: int = 0
):
    """
    Helper method, send to LLM and handle iterations
    """
    if args.iter < 1:
        return None, "Error: Number of iterations must be at least 1."

    if args.iter == 1:
        if not args.stream and not out_file:
            if 'tqdm' in globals() and callable(globals()['tqdm']):
                pbar = tqdm(total=1, desc=f"{progress_desc} (1 iter, no stream)")
            else:
                print(f"{progress_desc} (1 iter, no stream)...", file=sys.stderr)
        print(f"[LLM] Running decompilation for (Provider: {args.provider}, Model: {args.model}, Iter: {iter_count + 1}, Threads: {args.threads}, Stream: {args.stream}, Verify: {args.verify}, Split: {args.split}, Tokens: N/A, Attempt: {iter_count + 1}/{args.iter})...", file=sys.stderr)
        decompiled, error = call_llm(
            api_key, args.model, args.systemmsg,
            prompt, args.provider,
            args.stream,
            output_file_handle=out_file if args.stream else None,
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            temperature=args.temp,
            top_p=args.topp,
            reasoning_effort=args.effort
        )
        
        if not args.stream and not out_file:
            if 'pbar' in locals(): pbar.update(1); pbar.close()
            elif not ('tqdm' in globals() and callable(globals()['tqdm'])): print(f"{progress_desc} done.", file=sys.stderr)

        if error: return None, f"LLM call failed: {error}"
        if not decompiled and not (args.stream and out_file):
            return None, "LLM call returned no substantive code content."
        return decompiled, None
    else: 
        success = []
        
        worker_count = args.threads if args.multithreaded and args.threads else (args.iter if args.multithreaded else 1)
        worker_count = min(worker_count, args.iter, DEFAULT_MAX_WORKERS_FOR_ITERATIONS)
        if not args.multithreaded: worker_count = 1
            
        if args.stream:
            print(f"{YELLOW}Info: Streaming is disabled for multi-iteration (--iter > 1). Final result will be shown.{RESET}", file=sys.stderr)
        
        stream = False # we don't stream for iter > 1

        futures_list = []
        print(f"[LLM] Running decompilation for '{prompt}' (Provider: {args.provider}, Model: {args.model}, Iter: {iter_count + 1}, Threads: {worker_count}, Stream: {stream}, Verify: {args.verify}, Split: {args.split}, Tokens: N/A, Attempt: {iter_count + 1}/{args.iter})...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for i in range(args.iter):
                futures_list.append(
                    executor.submit(
                        call_llm, api_key, args.model, args.systemmsg,
                        prompt, args.provider,
                        stream,
                        None,
                        LLM_REQUEST_TIMEOUT_SECONDS,
                        temperature=args.temp,
                        top_p=args.topp,
                        reasoning_effort=args.effort
                    )
                )
            
            progress = range(args.iter)
            if 'tqdm' in globals() and callable(globals()['tqdm']):
                progress = tqdm(as_completed(futures_list), total=args.iter, desc=f"{progress_desc} ({args.iter} iters, {worker_count} threads)")
            else:
                print(f"{progress_desc} ({args.iter} iters, {worker_count} threads)...", file=sys.stderr)

            for i, future in enumerate(progress if isinstance(progress, tqdm) else as_completed(futures_list)):
                try:
                    result_code, error_msg = future.result(timeout=THREAD_COMPLETION_TIMEOUT_SECONDS)
                    if error_msg:
                        print(f"{YELLOW}LLM iteration {i+1}/{args.iter} failed: {error_msg}{RESET}", file=sys.stderr)
                    elif result_code:
                        success.append(result_code)
                    else:
                        print(f"{YELLOW}LLM iteration {i+1}/{args.iter} returned no code.{RESET}", file=sys.stderr)
                except FutureTimeoutError:
                    print(f"{RED}LLM iteration {i+1}/{args.iter} timed out after {THREAD_COMPLETION_TIMEOUT_SECONDS}s.{RESET}", file=sys.stderr)
                except Exception as e:
                    print(f"{RED}LLM iteration {i+1}/{args.iter} failed with an error: {e!r}{RESET}", file=sys.stderr)
                
                if not isinstance(progress, tqdm) and not ('tqdm' in globals() and callable(globals()['tqdm'])):
                    if (i + 1) % (args.iter // 10 + 1) == 0 or i + 1 == args.iter :
                        print(f"Completed iteration {i+1}/{args.iter}", file=sys.stderr)
        
        if isinstance(progress, tqdm): progress.close()

        if not success:
            return None, "All LLM calls failed or returned no code in multi-iteration."

        if len(success) == 1:
            return success[0], None
        else:
            best = -1.0
            chosen_code = success[0]
            avg_sim = [0.0] * len(success)
            for i in range(len(success)):
                current = 0.0
                for j in range(len(success)):
                    if i == j: continue
                    current += check_similarity(success[i], success[j])
                avg_sim[i] = current / (len(success) -1) if len(success) > 1 else 1.0
            
            avg_index = avg_sim.index(max(avg_sim))
            chosen_code = success[avg_index]
            best = avg_sim[avg_index]
            
            print(f"Selected code from {len(success)} successful iterations (best avg similarity: {best:.3f}).", file=sys.stderr)
            return chosen_code, None

def decompile(
    args: argparse.Namespace,
    api_key: str
):
    """
    Performs one full decompilation attempt. Handles splitting if enabled.
    """
    disassembled, error = disassemble(args.pyc_file)
    if error: return None, error
    if not disassembled: return None, "Bytecode disassembly empty."

    chunks = []
    est_tokens = 0 

    if args.split > 0: 
        print(f"Manually splitting bytecode into {args.split} chunks (character-based).", file=sys.stderr)
        chunks = split_manual(disassembled, args.split)
    elif args.auto_split: 
        est_tokens = get_token_count(disassembled, args.model, args.provider)
        print(f"Total estimated tokens for disassembly: {est_tokens}", file=sys.stderr)
        if est_tokens > args.max_tokens:
            print(f"Bytecode (est. {est_tokens} tokens) > limit ({args.max_tokens}). Auto-splitting by token limit.", file=sys.stderr)
            chunks = split_auto(disassembled, args.max_tokens, args.model, args.provider)
            if not chunks: 
                 print(f"{YELLOW}Warning: Token-based splitting resulted in no chunks. Falling back to character split.{RESET}", file=sys.stderr)
                 if len(disassembled) > args.max_chars:
                     chunks = split_auto(disassembled, args.max_chars // 4, args.model, args.provider)
                 else:
                     chunks = [disassembled]
            if not chunks:
                return None, "Auto-splitting failed to produce any chunks."

        else:
            print(f"Bytecode (est. {est_tokens} tokens) is within token limit ({args.max_tokens}). No auto-split needed.", file=sys.stderr)
            chunks = [disassembled]
    else: 
        chunks = [disassembled]

    if not chunks:
        return None, "Splitting (or lack thereof) resulted in no chunks."

    decompiled = []
    total_chunks = len(chunks)

    out_file = None
    if args.stream and args.output:
        try:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            out_file = open(args.output, 'w', encoding='utf-8')
            print(f"Streaming output directly to file: {args.output}", file=sys.stderr)
        except IOError as e:
            return None, f"Error opening output file {args.output} for streaming: {e}"

    for i, chunk_content in enumerate(chunks):
        chunk_token_est = get_token_count(chunk_content, args.model, args.provider)
        chunk_progress_desc = f"Chunk {i+1}/{total_chunks}"
        
        if not (args.stream and out_file is None and total_chunks == 1):
             print(f"\n--- Processing {chunk_progress_desc} (est. {chunk_token_est} tokens) ---", file=sys.stderr)

        prompt = chunk_content
        if total_chunks > 1:
            prompt = (
                f"This is part {i+1} of {total_chunks} of a Python bytecode "
                f"disassembly. Please decompile THIS PART. Output only the raw Python code for this part.\n\n{chunk_content}"
            )



        part, error_msg = _decompile_llm(
            prompt,
            args,
            api_key,
            out_file=out_file,
            progress_desc=chunk_progress_desc,
            iter_count=i
        )

        if error_msg:
            if out_file: out_file.close()
            return None, f"Failed to decompile chunk {i+1}/{total_chunks}: {error_msg}"
        
        if part:
            decompiled.append(part)
        elif not (args.stream and out_file and args.iter == 1): 
            if out_file: out_file.close()
            return None, f"Decompilation of chunk {i+1}/{total_chunks} returned no code (and not streaming to file or iter > 1)."

        if not (args.stream and out_file is None and total_chunks == 1):
             print(f"--- {chunk_progress_desc} processed. ---", file=sys.stderr)
        
        if args.stream and out_file and total_chunks > 1 and i < total_chunks -1 and args.iter == 1:
            out_file.write(f"\n\n# --- Decompiler Auto-Split Boundary ({i+1}/{total_chunks}) ---\n\n")
            out_file.flush()


    if out_file:
        if args.iter > 1 and decompiled:
            separator = f"\n\n# --- Decompiler Auto-Split Boundary ({total_chunks} chunks processed, multi-iter) ---\n\n"
            combined = separator.join(decompiled) if total_chunks > 1 else decompiled[0]
            out_file.write(combined)
            out_file.flush()

        out_file.close()
        print(f"\nAll {total_chunks} chunks processed. Output "
              f"{'streamed' if args.iter == 1 else 'written'} to '{args.output}'.", file=sys.stderr)
        return "__STREAMED_TO_FILE__", None 

    if not decompiled and total_chunks > 0 :
        return None, "No parts were successfully decompiled." # just in case

    if total_chunks > 1:
        separator = f"\n\n# --- Decompiler Auto-Split Boundary ({total_chunks} chunks processed) ---\n\n"
        final_code = separator.join(decompiled)
        print(f"\nAll {total_chunks} chunks processed and combined.", file=sys.stderr)
        return final_code, None
    elif decompiled:
        return decompiled[0], None
    else:
        return None, "Decompilation attempt yielded no combined code."


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
        print(f"{GREEN}Effort set to {args.effort.upper()}. This may get costly depending on your model provider!{RESET}", file=sys.stderr)

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