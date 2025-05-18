import io
import os
import sys
import types
import xdis.load
import dis as std_dis


def disassemble(pyc_file: str):
    """Disassembles provided .pyc file to readable bytecode."""
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
                f"Error: xdis failed to extract a valid code object from '{pyc_file}'. Received type: {type(result)}. "
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
        return None, f"IOError during .pyc file processing for '{pyc_file}': {e}"
    except Exception as e:
        return None, (
            f"An unexpected error occurred while reading or disassembling '{pyc_file}': {e!r}"
        )
