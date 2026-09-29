"""Regression test: evaluate.py harus mengikuti konstanta di config.py.

Guard AST di ``tests/test_config.py`` hanya mencari assignment module-level,
jadi keyword argument yang ditulis sebagai literal (misal ``k=20.0``) lolos
dari sana. Modul ini mengunci celah itu: kalau evaluate.py kembali menulis
nilai Elo atau MAX_GOALS langsung, test ini gagal.
"""

import inspect

import penaltyblog as pb
import pytest

import config
import evaluate

SOURCE_PATH = inspect.getsourcefile(evaluate)

HARD_CODED_PATTERNS = (
    "k=20.0",
    "home_field_advantage=100.0",
    "max_goals=15",
)


def test_evaluate_source_has_no_hardcoded_config_values():
    source = open(SOURCE_PATH, encoding="utf-8").read()
    found = [pattern for pattern in HARD_CODED_PATTERNS if pattern in source]
    assert found == [], (
        "evaluate.py tidak boleh menulis konstanta config sebagai literal: "
        f"{found}. Ambil dari config.py."
    )


def test_new_elo_follows_config(monkeypatch):
    monkeypatch.setattr(config, "ELO_K", 33.0)
    elo = evaluate._new_elo()
    assert elo.k == pytest.approx(33.0)


def test_new_elo_home_advantage_follows_config(monkeypatch):
    monkeypatch.setattr(config, "ELO_HOME_ADVANTAGE", 77.0)
    elo = evaluate._new_elo()
    assert elo.hfa == pytest.approx(77.0)


def test_new_elo_matches_production_default():
    """Nilai default harus sama dengan yang dipakai statistical_models.py."""
    elo = evaluate._new_elo()
    reference = pb.ratings.Elo(
        k=config.ELO_K, home_field_advantage=config.ELO_HOME_ADVANTAGE
    )
    assert elo.k == reference.k
    assert elo.hfa == reference.hfa


def test_score_grid_uses_config_max_goals(monkeypatch):
    """max_goals harus dibaca dari config, bukan angka tetap."""
    captured = {}

    class FakeModel:
        def predict(self, home, away, max_goals):
            captured["max_goals"] = max_goals

            class Grid:
                home_draw_away = (0.5, 0.25, 0.25)

            return Grid()

    frame = evaluate._train_rows().head(1)
    monkeypatch.setattr(config, "MAX_GOALS", 9)
    evaluate._score_grid_probabilities(FakeModel(), frame)
    assert captured["max_goals"] == 9


def test_elo_row_probabilities_follows_config(monkeypatch):
    """Helper probabilitas Elo harus ikut berubah saat config Elo diubah."""
    row = {"team": "Arsenal", "opponent": "Chelsea",
           "elo_pre": 1600.0, "opponent_elo_pre": 1500.0}

    with_hfa = evaluate._elo_row_probabilities(dict(row))
    assert sum(with_hfa) == pytest.approx(1.0, abs=1e-9)

    # HFA config 100 bikin tim advantage jauh lebih unggul.
    assert with_hfa[0] > with_hfa[2]

    # HFA 0 bikin tim dengan rating sama/imbalance jadi simetris.
    monkeypatch.setattr(config, "ELO_HOME_ADVANTAGE", 0.0)
    neutral = evaluate._elo_row_probabilities(dict(row))
    assert sum(neutral) == pytest.approx(1.0, abs=1e-9)
    assert neutral[0] < with_hfa[0], "HFA 0 harus mengikis advantage kandangan"
    assert neutral[0] > neutral[2], "rating 1600 vs 1500 tetap membuat home menang"
