from salesdash.styles import CSS


def test_css_braces_are_balanced():
    assert CSS.count("{") == CSS.count("}")


def test_css_styles_the_main_components():
    for selector in ("stMetric", "stSidebar", "stDataFrame", "page-header", "baseweb=\"tab\""):
        assert selector in CSS


def test_css_has_a_mobile_breakpoint():
    assert "@media (max-width: 640px)" in CSS


def test_css_does_not_override_fonts_globally():
    # a blanket font-family rule can break Streamlit's icon font (arrows would show as text)
    assert "font-family" not in CSS


def test_css_never_lets_metric_text_overflow_its_card():
    # overflowing text spilled over the neighbouring KPI card (long model names)
    assert "overflow: visible" not in CSS
    assert "overflow-wrap" in CSS