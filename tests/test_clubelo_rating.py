"""Tests untuk fetcher ClubElo di statistical_models (tanpa jaringan)."""

import pandas as pd

import statistical_models
from config import ELO_CLUBELO_INTERCEPT, ELO_CLUBELO_SLOPE
from statistical_models import get_clubelo_rating


def _rescaled(raw):
    return ELO_CLUBELO_INTERCEPT + ELO_CLUBELO_SLOPE * raw


def test_get_clubelo_rating_returns_prematch_rating(monkeypatch):
    # Rating pada 2023-08-11 adalah hasil match HARI ITU; cold-start butuh
    # rating sebelum tanggal tersebut (2023-05-07).
    history = (("2023-05-07", 1600.0), ("2023-08-11", 1580.0))
    monkeypatch.setattr(
        statistical_models, "_clubelo_history", lambda name: ("clubelo_api", history)
    )
    result = get_clubelo_rating("Burnley", pd.Timestamp("2023-08-11"))
    assert result == (_rescaled(1600.0), "clubelo_api")


def test_get_clubelo_rating_rescales_to_internal_scale(monkeypatch):
    # Rating ClubElo mentah ~1600-2100; yang di-inject harus versi rescaled
    # ke skala Elo internal, bukan nilai mentahnya.
    history = (("2020-01-01", 2000.0),)
    monkeypatch.setattr(
        statistical_models, "_clubelo_history", lambda name: ("clubelo_api", history)
    )
    rating, source = get_clubelo_rating("Everton", pd.Timestamp("2023-01-01"))
    assert source == "clubelo_api"
    assert rating == _rescaled(2000.0)
    assert rating != 2000.0


def test_get_clubelo_rating_none_when_history_starts_after_date(monkeypatch):
    # Chart halaman klub cuma menutup ~4 tahun terakhir; tim promosi musim
    # lebih lama tidak punya titik data sebelum tanggal yang diminta.
    history = (("2022-10-01", 1500.0),)
    monkeypatch.setattr(
        statistical_models, "_clubelo_history", lambda name: ("clubelo_chart", history)
    )
    assert get_clubelo_rating("Bournemouth", pd.Timestamp("2022-08-05")) is None


def test_get_clubelo_rating_none_when_club_unavailable(monkeypatch):
    monkeypatch.setattr(
        statistical_models, "_clubelo_history", lambda name: None
    )
    assert get_clubelo_rating("Huddersfield", pd.Timestamp("2017-08-11")) is None


def test_get_clubelo_rating_uses_mapping(monkeypatch):
    seen = {}

    def fake_history(clubelo_name):
        seen["name"] = clubelo_name
        return ("clubelo_api", (("2017-05-07", 1459.0),))

    monkeypatch.setattr(statistical_models, "_clubelo_history", fake_history)
    result = get_clubelo_rating("Huddersfield", pd.Timestamp("2017-08-11"))
    assert result == (_rescaled(1459.0), "clubelo_api")
    assert seen["name"] == "Huddersfield"

    result = get_clubelo_rating("Wolverhampton Wanderers", pd.Timestamp("2018-08-10"))
    assert result == (_rescaled(1459.0), "clubelo_api")
    assert seen["name"] == "Wolves"
