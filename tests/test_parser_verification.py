from pydantic import BaseModel, Field

from chllm.parser import RobustLLMParser


class SampleItem(BaseModel):
    id: str
    translated_text: str
    verification_prefix: str = Field(default="", alias="v")


def test_parser_extracts_verification_prefix():
    parser = RobustLLMParser()
    raw_json = '{"id": "h1", "v": "Original T", "translated_text": "Перевод"}'

    result = parser.parse(raw_json, validation_model=SampleItem)

    assert len(result["batch"]) == 1
    item = result["batch"][0]
    assert item.id == "h1"
    assert item.verification_prefix == "Original T"
    assert item.translated_text == "Перевод"


def test_parser_handles_full_field_name_and_alias():
    parser = RobustLLMParser()
    # Проверяем и 'v' и 'verification_prefix'
    raw_json = """
    {"id": "h1", "v": "Prefix 1", "translated_text": "T1"}
    {"id": "h2", "verification_prefix": "Prefix 2", "translated_text": "T2"}
    """
    result = parser.parse(raw_json, validation_model=SampleItem)

    assert result["batch"][0].verification_prefix == "Prefix 1"
    assert result["batch"][1].verification_prefix == "Prefix 2"
