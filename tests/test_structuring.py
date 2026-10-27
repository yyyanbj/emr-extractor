from types import SimpleNamespace

import pytest

from emr_extractor.structuring import RecordStructurer, parse_json_response


def test_parse_json_response_accepts_fences_and_thinking_block():
    content = '<THINK>synthetic reasoning</THINK>\n```json\n{"name": "synthetic"}\n```'
    assert parse_json_response(content) == {"name": "synthetic"}


def test_parse_json_response_rejects_non_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_json_response("not json")


def test_record_structurer_uses_injected_client():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"result": "ok"}'))]
    )

    class Completions:
        def create(self, **kwargs):
            assert kwargs["messages"][1]["content"] == "synthetic record"
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    structurer = RecordStructurer(client=client)

    assert structurer.structure_text("synthetic record") == {"result": "ok"}
