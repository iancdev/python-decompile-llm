import py_compile
from decompilellm.disassembler import disassemble


def test_disassemble(tmp_path):
    src = tmp_path / "sample.py"
    src.write_text("def foo():\n    return 42\n")
    pyc_path = py_compile.compile(str(src), cfile=str(tmp_path / "sample.pyc"))
    disasm, err = disassemble(pyc_path)
    assert err is None
    assert "LOAD_CONST" in disasm
