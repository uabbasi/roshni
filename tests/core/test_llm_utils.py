"""Tests for core.llm.utils — response text extraction."""

from roshni.core.llm.utils import extract_text_from_response


class TestExtractText:
    def test_string_passthrough(self):
        assert extract_text_from_response("hello") == "hello"

    def test_empty_string(self):
        assert extract_text_from_response("") == ""

    def test_none(self):
        assert extract_text_from_response(None) == ""

    def test_content_block_list(self):
        blocks = [{"type": "text", "text": "first "}, {"type": "text", "text": "second"}]
        assert extract_text_from_response(blocks) == "first second"

    def test_skips_tool_use(self):
        blocks = [
            {"type": "text", "text": "answer"},
            {"type": "tool_use", "name": "search", "input": {}},
        ]
        assert extract_text_from_response(blocks) == "answer"

    def test_skips_thinking(self):
        blocks = [
            {"type": "thinking", "text": "internal"},
            {"type": "text", "text": "visible"},
        ]
        assert extract_text_from_response(blocks) == "visible"

    def test_mixed_list_strings(self):
        assert extract_text_from_response(["a", "b", "c"]) == "abc"

    def test_object_with_text_attr(self):
        class FakeContent:
            text = "from attribute"

        assert extract_text_from_response(FakeContent()) == "from attribute"

    def test_fallback_to_str(self):
        assert extract_text_from_response(42) == "42"
