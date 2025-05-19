from decompilellm.utils import verify, check_similarity, split_manual, split_auto


def test_verify_valid_invalid():
    ok, err = verify("print(1)")
    assert ok and err is None
    bad_code = "def oops(:\n    pass"
    ok, err = verify(bad_code)
    assert not ok and "SyntaxError" in err


def test_check_similarity():
    assert check_similarity("abc", "abc") == 1.0
    assert check_similarity("abc", "xyz") < 1.0


def test_split_manual():
    text = "abcdefghij"
    chunks = split_manual(text, 3)
    assert "".join(chunks) == text
    assert len(chunks) == 3
    assert [len(c) for c in chunks] == [4, 4, 2]


def test_split_auto(monkeypatch):
    disasm = "a\nb\nc\nd\n"

    def fake_count(text, model, provider):
        return 1

    monkeypatch.setattr("decompilellm.utils.get_token_count", fake_count)
    chunks = split_auto(disasm, 2, "model", "openai")
    assert chunks == ["a\nb\n", "c\nd\n"]
