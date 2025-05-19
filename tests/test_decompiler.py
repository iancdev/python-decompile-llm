import argparse
from decompilellm.decompiler import decompile


def test_decompile(monkeypatch):
    disasm = "l1\nl2\nl3\nl4\n"
    monkeypatch.setattr("decompile_llm.decompiler.disassemble", lambda path: (disasm, None))

    calls = []

    def fake_call_llm(api_key, model, system_message, prompt, provider, stream_output, output_file_handle=None, timeout=None, temperature=0.5, top_p=1.0, reasoning_effort=None):
        calls.append(prompt)
        return f"code{len(calls)}", None

    monkeypatch.setattr("decompile_llm.decompiler.call_llm", fake_call_llm)
    monkeypatch.setattr("decompile_llm.decompiler.get_token_count", lambda text, model, provider: 1)

    args = argparse.Namespace(
        pyc_file="dummy.pyc",
        iter=1,
        stream=False,
        output=None,
        provider="openai",
        model="gpt-4",
        split=0,
        auto_split=True,
        max_tokens=2,
        max_chars=100,
        temp=0.5,
        topp=1.0,
        effort="none",
        threads=1,
        multithreaded=False,
        systemmsg="sys",
        verify="no",
    )

    result, err = decompile(args, "key")
    assert err is None
    expected = "code1\n\n# --- Decompiler Auto-Split Boundary (2 chunks processed) ---\n\ncode2"
    assert result == expected
