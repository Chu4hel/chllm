from chllm.builder import PromptBuilder


def test_prompt_builder_supports_multiple_instructions():
    """Проверка возможности передать список инструкций, которые объединятся в системный блок."""
    builder = PromptBuilder(template="Global instruction")

    # Мы хотим передавать список дополнительных инструкций
    custom_instr = ["Project rules", "File rules", "Speaker rules"]

    # Вызов с расширенным параметром
    prompt = builder.build(input_data={"test": 1}, instruction=custom_instr)

    # Проверяем, что все инструкции попали в промпт
    assert "Global instruction" in prompt
    assert "Project rules" in prompt
    assert "File rules" in prompt
    assert "Speaker rules" in prompt
    # Проверяем порядок (от общего к частному или наоборот - решим в реализации)
    # Обычно приоритетнее должны быть последние в списке
