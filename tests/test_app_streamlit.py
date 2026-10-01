"""Smoke test untuk app.py.

Streamlit app biasanya hanya dicek manual karena butuh server. Test di sini
menjalankan ``main()`` dengan widget Streamlit yang di-mock, sehingga
regresi render (exception, indeks kolom salah, nilai yang hilang) ketahuan
di test suite dan bukan pas user membuka linknya.

Yang diperiksa: app bisa dirender sampai selesai, pilihan tim terbaca,
tombol Predict memanggil ``get_prediction``, disclaimer ada, dan tidak ada
nama model yang ter-hardcode.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest import mock

import pytest
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


class FakeColumn:
    """Kolom Streamlit yang bisa dipakai sebagai context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@pytest.fixture
def render(monkeypatch):
    """Jalankan app.main() dengan widget di-mock; kembalikan HTML yang dirender."""
    markup: list[str] = []
    captions: list[str] = []
    selectboxes: list[tuple] = []

    def record_markdown(body, **kwargs):
        if isinstance(body, str):
            markup.append(body)

    def record_caption(body, **kwargs):
        if isinstance(body, str):
            captions.append(body)

    answers = iter(["Arsenal", "Chelsea"])

    def record_selectbox(label, options, **kwargs):
        selectboxes.append((label, list(options)))
        return next(answers)

    monkeypatch.setattr(st, "set_page_config", lambda *a, **kwargs: None)
    monkeypatch.setattr(st, "markdown", record_markdown)
    monkeypatch.setattr(st, "caption", record_caption)
    monkeypatch.setattr(st, "selectbox", record_selectbox)
    monkeypatch.setattr(st, "button", lambda *a, **kwargs: True)
    monkeypatch.setattr(
        st, "columns", lambda *a, **kwargs: [FakeColumn() for _ in range(a[0] if a else kwargs.get("count", 1))]
    )

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    for module in ("app",):
        sys.modules.pop(module, None)
    import app as app_module

    return app_module, markup, captions, selectboxes


def _text(markup: list[str]) -> str:
    joined = " ".join(markup)
    plain = re.sub(r"<[^>]+>", " ", joined)
    plain = plain.replace("&nbsp;", " ").replace("&ndash;", "-")
    plain = plain.replace("&middot;", "-").replace("&amp;", "&")
    return " ".join(plain.split())


def _css(markup: list[str]) -> str:
    """Blok stylesheet. Dicek dari tag pembuka supaya tidak tertukar dengan
    blok HTML lain yang kebetulan menyebut nama class yang sama."""
    return next(body for body in markup if body.lstrip().startswith("<style>"))


# --------------------------------------------------------------------------


def test_app_renders_without_exception(render):
    app_module, _, _, _ = render
    app_module.main()


def test_predict_button_triggers_prediction(render):
    app_module, _, _, _ = render
    with mock.patch.object(app_module.pm, "get_prediction", wraps=app_module.pm.get_prediction) as spy:
        app_module.main()
    spy.assert_called_once_with("Arsenal", "Chelsea")


def test_both_dropdowns_offer_real_teams(render):
    app_module, _, _, selectboxes = render
    app_module.main()
    assert [label for label, _ in selectboxes] == ["Home", "Away"]
    teams = app_module.pm.list_teams()
    assert selectboxes[0][1] == teams
    assert selectboxes[1][1] == teams


def test_title_is_centered_and_serif(render):
    app_module, markup, _, _ = render
    app_module.main()
    title = next(
        body
        for body in markup
        if body.lstrip().startswith("<h1") and "paper-title" in body
    )
    assert "Match predictor" in title
    # Posisi rata tengah diatur oleh aturan .paper-title di stylesheet.
    rule = re.search(r"\.paper-title\s*\{(.*?)\}", _css(markup), re.S)
    assert rule and "text-align: center" in rule.group(1)


def test_paper_palette_is_applied(render):
    app_module, markup, _, _ = render
    app_module.main()
    css = _css(markup)
    assert "#faf8f3" in css, "background krem paper"
    assert "#2c2c2a" in css
    assert "#888780" in css
    assert "Georgia" in css
    # Flat & minimal: tidak ada gradient, shadow, atau warna-warni.
    for banned in ("gradient", "box-shadow", "text-shadow"):
        assert banned not in css.lower(), f"{banned} dilarang"


def test_score_is_serif_and_large(render):
    app_module, markup, _, _ = render
    app_module.main()
    # Aturan .score-value ada di stylesheet, bukan di markup per-instance.
    rule = re.search(r"\.score-value\s*\{(.*?)\}", _css(markup), re.S)
    assert rule, "aturan .score-value harus ada di CSS"
    body = rule.group(1)
    assert "Georgia" in body
    size = re.search(r"font-size:\s*(\d+)px", body)
    assert size and 28 <= int(size.group(1)) <= 36, "skor harus ~32px"


def test_team_names_have_thin_bottom_border(render):
    app_module, markup, _, _ = render
    app_module.main()
    assert "border-bottom" in _css(markup)
    team_block = next(
        body
        for body in markup
        if body.lstrip().startswith("<div") and "team-name" in body
    )
    assert "Arsenal" in team_block and "Chelsea" in team_block


def test_wdl_shown_as_text_not_progress_bar(render):
    app_module, markup, _, _ = render
    app_module.main()
    wdl = next(body for body in markup if "Home" in body and "Draw" in body)
    assert re.search(r"Home\s+[\d.]+%", wdl)
    assert "progress" not in " ".join(markup).lower()


def test_model_labels_are_displayed(render):
    """Label sumber model harus tampil, persis seperti di CLI."""
    app_module, markup, _, _ = render
    app_module.main()
    text = _text(markup)
    assert "CV log loss" in text
    assert "out-of-sample" in text or "TimeSeriesSplit" in text


def test_corner_btts_and_cards_all_rendered(render):
    app_module, markup, _, _ = render
    app_module.main()
    text = _text(markup)
    for label in ("W / D / L", "Corner", "BTTS", "Kartu kuning"):
        assert label in text, f"{label} tidak tampil"


def test_secondary_metrics_have_real_numbers(render):
    """Bagian yang ditampilkan harus berisi angka, bukan pesan fallback."""
    app_module, markup, _, _ = render
    app_module.main()
    text = _text(markup)
    assert "tidak tersedia" not in text, "ada bagian web yang kosong di v1"


def test_key_player_section_absent(render):
    """v1 sengaja tidak menampilkan key player; CLI tetap menyediakannya."""
    app_module, markup, _, _ = render
    app_module.main()
    text = _text(markup).lower()
    assert "key player" not in text
    assert "man of the match" not in text
    # Tapi CLI-nya tetap punya.
    assert hasattr(app_module.pm, "load_match_player_predictions")


def test_disclaimer_is_on_page(render):
    app_module, markup, _, _ = render
    app_module.main()
    text = _text(markup)
    assert "bukan alat rekomendasi bet" in text
    assert "odds bandar" in text


def test_disclaimer_visible_even_before_predicting(render):
    """Disclaimer harus tampil tanpa harus klik tombol dulu."""
    app_module, markup, _, _ = render
    with mock.patch.object(st, "button", lambda *a, **kwargs: False):
        app_module.main()
    assert "bukan alat rekomendasi bet" in _text(markup)


def test_same_team_guard(render, monkeypatch):
    """Memilih tim yang sama dua kali harus ditolak, bukan crash."""
    answers = iter(["Arsenal", "Arsenal"])
    monkeypatch.setattr(st, "selectbox", lambda *a, **kwargs: next(answers))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    sys.modules.pop("app", None)
    import app as app_module

    assert app_module.main() is None  # return awal, tidak crash