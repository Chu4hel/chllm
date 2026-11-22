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


def test_masking_presets_individual():
    """Тест отдельных стандартных пресетов MaskingPresets."""
    from chllm.masking import MaskingPresets

    # BRACKETS
    m_brackets = ContentMasker(patterns=MaskingPresets.BRACKETS)
    res_b = m_brackets.mask("Hero [hero_name] level [lvl]")
    assert "[hero_name]" not in res_b.masked_text
    assert m_brackets.demask(res_b.masked_text, res_b.mapping) == "Hero [hero_name] level [lvl]"

    # BRACES
    m_braces = ContentMasker(patterns=MaskingPresets.BRACES)
    res_br = m_braces.mask("{b}Warning{/b}: {color=#f00}Danger{/color}")
    assert "{b}" not in res_br.masked_text
    assert "{color=#f00}" not in res_br.masked_text
    assert m_braces.demask(res_br.masked_text, res_br.mapping) == "{b}Warning{/b}: {color=#f00}Danger{/color}"

    # PRINTF
    m_printf = ContentMasker(patterns=MaskingPresets.PRINTF)
    res_p = m_printf.mask("Values: %s, %d, %(score)d")
    assert "%s" not in res_p.masked_text
    assert "%d" not in res_p.masked_text
    assert "%(score)d" not in res_p.masked_text
    assert m_printf.demask(res_p.masked_text, res_p.mapping) == "Values: %s, %d, %(score)d"

    # HTML
    m_html = ContentMasker(patterns=MaskingPresets.HTML)
    html_sample = '<p class="intro">Hello &nbsp; <span>world</span> <!-- note --></p>'
    res_html = m_html.mask(html_sample)
    assert '<p class="intro">' not in res_html.masked_text
    assert "&nbsp;" not in res_html.masked_text
    assert "<!-- note -->" not in res_html.masked_text
    assert m_html.demask(res_html.masked_text, res_html.mapping) == html_sample


def test_masking_presets_bitwise_or_composition():
    """Тест композиции пресетов через оператор | и пресета RENPY."""
    from chllm.masking import MaskingPresets

    # Проверяем синтаксис как в R.A.T.: MaskingPresets.RENPY | MaskingPresets.NEWLINES
    combined_preset = MaskingPresets.RENPY | MaskingPresets.NEWLINES
    masker = ContentMasker(patterns=combined_preset)

    text = "{b}[player_name]{/b}:\nLine 1\\nLine 2 with %(points)d pts."
    res = masker.mask(text)

    # Все ключевые элементы должны быть замаскированы
    assert "{b}" not in res.masked_text
    assert "[player_name]" not in res.masked_text
    assert "{/b}" not in res.masked_text
    assert "\n" not in res.masked_text
    assert "\\n" not in res.masked_text
    assert "%(points)d" not in res.masked_text

    # Обратное восстановление
    demasked = masker.demask(res.masked_text, res.mapping)
    assert demasked == text


def test_masking_presets_with_lists():
    """Тест взаимодействия пресетов с обычными списками строк."""
    from chllm.masking import MaskingPresets

    preset = MaskingPresets.BRACKETS | [r"CUSTOM_\w+"]
    masker = ContentMasker(patterns=preset)

    text = "Find [target] with CUSTOM_ID_123"
    res = masker.mask(text)
    assert "[target]" not in res.masked_text
    assert "CUSTOM_ID_123" not in res.masked_text
    assert masker.demask(res.masked_text, res.mapping) == text

    # Проверка правостороннего объединения [list] | MaskingPreset
    r_preset = [r"FOO_\w+"] | MaskingPresets.BRACES
    r_masker = ContentMasker(patterns=r_preset)
    r_text = "Check FOO_BAR and {i}italic{/i}"
    r_res = r_masker.mask(r_text)
    assert "FOO_BAR" not in r_res.masked_text
    assert "{i}" not in r_res.masked_text
    assert r_masker.demask(r_res.masked_text, r_res.mapping) == r_text


def test_content_masker_presets_attribute():
    """Проверка доступности пресетов через атрибут класса ContentMasker.Presets."""
    masker = ContentMasker(patterns=ContentMasker.Presets.RENPY)
    text = "{color=#00f}[hero]{/color}"
    res = masker.mask(text)
    assert "{color=#00f}" not in res.masked_text
    assert "[hero]" not in res.masked_text
    assert masker.demask(res.masked_text, res.mapping) == text
