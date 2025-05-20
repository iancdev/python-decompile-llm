import sys
from difflib import SequenceMatcher


def _normalize(code: str) -> str:
    """Return code stripped of leading/trailing whitespace and blank lines."""
    if not isinstance(code, str):
        return ""
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    return "\n".join(lines)
import ast
from .token_utils import get_token_count
from .constants import YELLOW, RESET


def verify(code: str):
    """Checks code for syntax errors"""
    if not code.strip():
        return False, "SyntaxError: Code is empty or contains only whitespace."
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        error_detail = f"Error: {e.msg}"
        if e.lineno is not None:
            error_detail += f" on line {e.lineno}"
        if e.offset is not None:
            error_detail += f", column {e.offset}"
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
    """Return a ratio representing how similar two code snippets are."""
    if not isinstance(code1, str) or not isinstance(code2, str):
        return 0.0
    if not code1 and not code2:
        return 1.0
    if not code1 or not code2:
        return 0.0

    code1_norm = _normalize(code1)
    code2_norm = _normalize(code2)
    return SequenceMatcher(None, code1_norm, code2_norm).ratio()


def split_manual(disassembled: str, num_chunks: int):
    """Splits by character length as fallback"""
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


def split_auto(disassembled: str, max_tokens: int, model_name: str, provider: str):
    """Split by calculated token limit (preferred)"""
    if not disassembled:
        return []
    if max_tokens <= 0:
        print(
            f"{YELLOW}Warning: max_tokens for splitting is <= 0. Ignoring split and returning as one chunk.{RESET}",
            file=sys.stderr,
        )
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
                file=sys.stderr,
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
