from chllm.masking import ContentMasker


def test_mask_simple_variables():
    # Тест маскировки простых переменных вида [name]
    masker = ContentMasker(patterns=[r"\[.+?\]"])
    text = "Hello [name], welcome to [city]!"

    result = masker.mask(text)

    # Проверяем, что в тексте появились плейсхолдеры
    assert "[[[VAR_0]]]" in result.masked_text
    assert "[[[VAR_1]]]" in result.masked_text
    assert "[name]" not in result.masked_text

    # Проверяем маппинг
    assert result.mapping["[[[VAR_0]]]"] == "[name]"
    assert result.mapping["[[[VAR_1]]]"] == "[city]"


def test_demask_text():
    masker = ContentMasker()
    masked_text = "Hi [[[VAR_0]]], how is [[[VAR_1]]]?"
    mapping = {"[[[VAR_0]]]": "John", "[[[VAR_1]]]": "London"}

    demasked = masker.demask(masked_text, mapping)

    assert demasked == "Hi John, how is London?"


def test_mask_with_custom_prefix():
    # Тест с кастомным префиксом плейсхолдера
    masker = ContentMasker(patterns=[r"\{.+?\}"], placeholder_prefix="TAG")
    text = "Color {color=red}red{/color}"

    result = masker.mask(text)

    assert "[[[TAG_0]]]" in result.masked_text
    assert result.mapping["[[[TAG_0]]]"] == "{color=red}"


def test_mask_newline_and_literals():
    # Тест маскировки спецсимволов (как мы делали в основном проекте)
    # Используем конкретные строки вместо регулярок для этого случая
    masker = ContentMasker(patterns=[r"\n", r"\\n"])
    text = "Line 1\nLine 2\\nLiteral"

    result = masker.mask(text)

    # Важно: \n длиннее чем \\n в плане поиска, или наоборот?
    # В реализации нужно будет учесть порядок.
    assert "[[[VAR_0]]]" in result.masked_text
    assert result.mapping["[[[VAR_0]]]"] == "\n"


TEST_MASKING_PATTERNS = [
    r"\[.+?\]",
    r"\{.+?\}",
    r"%\([a-zA-Z0-9_]+\)[a-zA-Z%]|%[a-zA-Z%]",
    r"\n",
    r"\\n",
]


def test_mask_python_interpolation():
    masker = ContentMasker(patterns=TEST_MASKING_PATTERNS)
    text = "Pleased to meet you, %(player_name)s! You have %d coins, %s."

    result = masker.mask(text)

    assert "%(player_name)s" not in result.masked_text
    assert "%d" not in result.masked_text
    assert "%s" not in result.masked_text
    assert result.mapping["[[[VAR_0]]]"] == "%(player_name)s"
    assert result.mapping["[[[VAR_1]]]"] == "%d"
    assert result.mapping["[[[VAR_2]]]"] == "%s"

    demasked = masker.demask(result.masked_text, result.mapping)
    assert demasked == text


def test_mask_with_excluded_values():
    masker = ContentMasker(patterns=TEST_MASKING_PATTERNS)
    text = "Hello [player_name], you have {b}5{/b} coins and %d items."

    result = masker.mask(text, excluded_values={"[player_name]"})

    assert "[player_name]" in result.masked_text
    assert "{b}" not in result.masked_text
    assert "%d" not in result.masked_text
    assert "[player_name]" not in result.mapping.values()
    assert "{b}" in result.mapping.values()
    assert "%d" in result.mapping.values()
