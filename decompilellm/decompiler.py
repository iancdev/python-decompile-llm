import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from .api import call_llm
from .disassembler import disassemble
from .token_utils import get_token_count
from .utils import check_similarity, split_manual, split_auto, verify
from tqdm import tqdm
from .constants import (
    DEFAULT_MAX_WORKERS_FOR_ITERATIONS,
    LLM_REQUEST_TIMEOUT_SECONDS,
    THREAD_COMPLETION_TIMEOUT_SECONDS,
    YELLOW,
    RED,
    RESET,
)


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
        verifieds = []
        
        worker_count = args.threads if args.multithreaded and args.threads else (args.iter if args.multithreaded else 1)
        worker_count = min(worker_count, args.iter, DEFAULT_MAX_WORKERS_FOR_ITERATIONS)
        if not args.multithreaded: worker_count = 1
            
        if args.stream:
            print(f"{YELLOW}Info: Streaming is disabled for multi-iteration (--iter > 1). Final result will be shown.{RESET}", file=sys.stderr)
        
        stream = False # we don't stream for iter > 1

        futures_list = []
        print(f"[LLM] Running decompilation... (Provider: {args.provider}, Model: {args.model}, Iter: {iter_count + 1}, Threads: {worker_count}, Stream: {stream}, Verify: {args.verify}, Split: {args.split}, Tokens: N/A, Attempt: {iter_count + 1}/{args.iter})...", file=sys.stderr)
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
                        if args.verify.lower() == 'yes':
                            ok, verr = verify(result_code)
                            if ok:
                                verifieds.append(result_code)
                            else:
                                print(f"{YELLOW}LLM iteration {i+1}/{args.iter} failed verification: {verr}{RESET}", file=sys.stderr)
                        else:
                            verifieds.append(result_code)
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

        results = verifieds if args.verify.lower() == 'yes' else success

        if not results:
            return None, "No syntactically valid output from iterations." if args.verify.lower() == 'yes' else "All LLM calls failed or returned no code in multi-iteration."

        if len(results) == 1:
            if args.verify.lower() == 'yes' and args.iter > 1:
                print(f"{YELLOW}Warning: Only one syntactically valid output was generated.{RESET}", file=sys.stderr)
            return results[0], None

        best = -1.0
        chosen_code = results[0]
        avg_sim = [0.0] * len(results)
        try:
            for i in range(len(results)):
                current = 0.0
                for j in range(len(results)):
                    if i == j:
                        continue
                    current += check_similarity(results[i], results[j])
                avg_sim[i] = current / (len(results) - 1) if len(results) > 1 else 1.0

            avg_index = avg_sim.index(max(avg_sim))
            chosen_code = results[avg_index]
            best = avg_sim[avg_index]
        except Exception as e:
            print(f"{YELLOW}Warning: Similarity comparison failed ({e}). Picking first valid result.{RESET}", file=sys.stderr)
            return results[0], None

        print(f"Selected code from {len(results)} successful iterations (best avg similarity: {best:.3f}).", file=sys.stderr)
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


