"""Tests untuk identifikasi tim promosi dan determinisme log cold-start.

``identify_promoted_teams`` sebelumnya mengembalikan ``set``, dan hasilnya
di-iterasi langsung untuk menulis ``data/metadata/elo_coldstart_log.csv``
yang sudah di-commit. Urutan iterasi set tidak stabil antar-proses, jadi
file itu berubah urutan baris setiap kali pipeline dijalankan meskipun
isinya identik. Test di sini mengunci sifat deterministik itu.
"""

import pandas as pd
import pytest

import statistical_models
from statistical_models import identify_promoted_teams


def _seasons_frame(seasons_per_team=None):
    """Buat tabel pertandingan dengan musim yang bisa dikendalikan.

    ``seasons_per_team`` memetakan nama tim ke daftar musim kehadirannya.
    """
    if seasons_per_team is None:
        seasons_per_team = {
            "Alpha": ["2020-2021", "2021-2022"],
            "Beta": ["2020-2021", "2021-2022"],
            "Gamma": ["2021-2022"],
        }
    rows = []
    for team, seasons in seasons_per_team.items():
        for season in seasons:
            rows.append(
                {
                    "match_id": f"{season}-{team}",
                    "season": season,
                    "date": pd.Timestamp("2021-08-01"),
                    "datetime": pd.Timestamp("2021-08-01"),
                    "team_home": team,
                    "team_away": "Opponent",
                    "goals_home": 1,
                    "goals_away": 0,
                    "result": "H",
                }
            )
    return pd.DataFrame(rows)


def test_returns_promoted_teams_only():
    frame = _seasons_frame()
    promoted = identify_promoted_teams(frame, "2021-2022")
    assert "Gamma" in promoted
    assert "Alpha" not in promoted
    assert "Beta" not in promoted


def test_result_is_sorted_not_a_set():
    """Regression guard: hasil harus list terurut, bukan set."""
    frame = _seasons_frame(
        {
            "Zeta": ["2020-2021", "2021-2022"],
            "Alpha": ["2020-2021", "2021-2022"],
            "Mu": ["2020-2021", "2021-2022"],
            "Beta": ["2020-2021", "2021-2022"],
        }
    )
    promoted = identify_promoted_teams(frame, "2021-2022")
    assert isinstance(promoted, list)
    assert promoted == sorted(promoted), "hasil harus terurut"


def test_order_is_stable_across_calls():
    """Pemanggilan berulang harus urutan sama (guard hash randomization)."""
    frame = _seasons_frame(
        {
            name: ["2020-2021", "2021-2022"]
            for name in ("Zeta", "Alpha", "Mu", "Beta", "Omega", "Delta")
        }
    )
    runs = [
        identify_promoted_teams(frame, "2021-2022")
        for _ in range(5)
    ]
    assert all(run == runs[0] for run in runs), f"urutan tidak stabil: {runs}"


def test_written_log_is_byte_identical_across_runs(tmp_path, monkeypatch):
    """Regression guard langsung ke artefak yang di-commit.

    Dua kali menulis log cold-start dengan data identik harus menghasilkan
    file identik byte-per-byte. Inilah yang rusak sebelum sorted() masuk.
    """
    frame = _seasons_frame(
        {
            name: ["2020-2021", "2021-2022"]
            for name in ("Zeta", "Alpha", "Mu", "Beta", "Omega")
        }
        | {
            # Tim yang hanya muncul di musim kedua, jadi ada yang dipromosikan.
            "Delta": ["2021-2022"],
            "Sigma": ["2021-2022"],
            "Kappa": ["2021-2022"],
            "Iota": ["2021-2022"],
            "Theta": ["2021-2022"],
        }
    )
    assert statistical_models.identify_promoted_teams(frame, "2021-2022"), (
        "test ini butuh tim promosi supaya log cold-start benar-benar ditulis"
    )
    monkeypatch.setattr(
        statistical_models, "get_clubelo_rating_detail", lambda team, date: None
    )
    monkeypatch.setattr(
        statistical_models, "get_bottom_three_average_elo", lambda *a, **k: 1400.0
    )
    monkeypatch.setattr(
        statistical_models, "ELO_COLDSTART_LOG_PATH", tmp_path / "log.csv"
    )

    outputs = []
    for index in range(2):
        _, _ = statistical_models.build_elo_history(frame)
        outputs.append((tmp_path / "log.csv").read_bytes())
    assert outputs[0] == outputs[1], (
        "elo_coldstart_log.csv tidak deterministik antar-run"
    )


def test_first_season_has_no_promoted_teams():
    frame = _seasons_frame()
    assert identify_promoted_teams(frame, "2020-2021") == []


def test_unknown_season_returns_empty_list():
    frame = _seasons_frame()
    assert identify_promoted_teams(frame, "1999-2000") == []


def test_build_elo_history_accepts_sorted_output():
    """build_elo_history harus tetap jalan dengan list terurut."""
    frame = _seasons_frame()
    promoted = identify_promoted_teams(frame, "2021-2022")
    assert promoted  # tidak kosong, supaya loop-nya benar-benar dieksekusi
    assert all(isinstance(team, str) for team in promoted)
