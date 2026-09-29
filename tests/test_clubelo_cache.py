"""Tests untuk cache disk ClubElo & log cold-start (tanpa jaringan)."""

import json

import pandas as pd
import pytest

import statistical_models
from config import CLUBELO_CACHE_MAX_AGE_DAYS, ELO_CLUBELO_INTERCEPT, ELO_CLUBELO_SLOPE
from statistical_models import (
    _cache_entry_is_fresh,
    _clubelo_history,
    _load_clubelo_cache,
    get_clubelo_rating_detail,
    write_coldstart_log,
)


@pytest.fixture()
def cache_path(tmp_path, monkeypatch):
    """Arahkan cache ClubElo ke file sementara."""
    path = tmp_path / "clubelo_cache.json"
    monkeypatch.setattr(statistical_models, "CLUBELO_CACHE_PATH", path)
    statistical_models._clubelo_history.cache_clear()
    yield path
    statistical_models._clubelo_history.cache_clear()


def _stub_network(monkeypatch, api=None, chart=None):
    """Stub fetcher ClubElo. api/chart boleh berupa value atau callable."""

    def _api(name):
        return api(name) if callable(api) else api

    def _chart(slug):
        return chart(slug) if callable(chart) else chart

    monkeypatch.setattr(statistical_models, "_fetch_clubelo_api_history", _api)
    monkeypatch.setattr(statistical_models, "_fetch_clubelo_chart_history", _chart)


def test_cache_miss_then_network_then_cache_hit(cache_path, monkeypatch):
    """Run pertama ambil dari jaringan & tulis cache; run kedua pakai cache."""
    calls = {"api": 0, "chart": 0}

    def fake_api(name):
        calls["api"] += 1
        return (("2020-01-01", 1700.0),)

    def fake_chart(slug):
        calls["chart"] += 1
        return None

    _stub_network(monkeypatch, api=fake_api, chart=fake_chart)

    first = _clubelo_history("Leeds")
    assert first == ("clubelo_api", (("2020-01-01", 1700.0),))
    assert calls["api"] == 1
    assert cache_path.exists()

    # Run berikutnya: cache harus dipakai tanpa menyentuh jaringan.
    statistical_models._clubelo_history.cache_clear()
    second = _clubelo_history("Leeds")
    assert second == first
    assert calls["api"] == 1, "jaringan tidak boleh dipanggil ulang saat cache fresh"


def test_stale_cache_triggers_refetch(cache_path, monkeypatch):
    """Cache yang lebih tua dari CLUBELO_CACHE_MAX_AGE_DAYS di-refetch."""
    payload = {
        "Leeds": {
            "source": "clubelo_chart",
            "fetched_at": 0.0,
            "history": [["2020-01-01", 1500.0]],
        }
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    _stub_network(monkeypatch, api=None, chart=(("2024-01-01", 1800.0),))
    result = _clubelo_history("Leeds")
    assert result == ("clubelo_chart", (("2024-01-01", 1800.0),))


def test_fetch_failure_is_not_cached(cache_path, monkeypatch):
    """Entry gagal tidak di-cache, supaya run berikutnya tetap mencoba jaringan."""
    _stub_network(monkeypatch, api=None, chart=None)
    assert _clubelo_history("Ghost Town") is None

    payload = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    assert "Ghost Town" not in payload

    # Jaringan pulih -> harus berhasil tanpa perlu hapus cache.
    statistical_models._clubelo_history.cache_clear()
    _stub_network(monkeypatch, api=None, chart=(("2024-01-01", 1700.0),))
    assert _clubelo_history("Ghost Town") == ("clubelo_chart", (("2024-01-01", 1700.0),))


def test_corrupt_cache_is_ignored(cache_path, monkeypatch):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{bukan json", encoding="utf-8")
    _stub_network(monkeypatch, api=(("2024-01-01", 1700.0),), chart=None)
    assert _clubelo_history("Arsenal") == ("clubelo_api", (("2024-01-01", 1700.0),))


def test_cache_freshness_window():
    now = 1_000_000_000.0
    fresh = {"fetched_at": now - 10}
    stale = {"fetched_at": now - (CLUBELO_CACHE_MAX_AGE_DAYS + 1) * 86400}
    assert _cache_entry_is_fresh(fresh, now)
    assert not _cache_entry_is_fresh(stale, now)
    assert not _cache_entry_is_fresh({}, now)


def test_load_cache_missing_file_returns_empty(cache_path):
    assert _load_clubelo_cache() == {}


def test_get_clubelo_rating_detail_reports_raw_and_rescaled(monkeypatch):
    history = (("2023-05-07", 1600.0), ("2023-08-11", 1580.0))
    monkeypatch.setattr(
        statistical_models, "_clubelo_history", lambda name: ("clubelo_chart", history)
    )
    detail = get_clubelo_rating_detail("Burnley", pd.Timestamp("2023-08-11"))
    assert detail["source"] == "clubelo_chart"
    assert detail["rating_date"] == "2023-05-07"
    assert detail["raw_clubelo"] == 1600.0
    assert detail["rescaled"] == pytest.approx(
        ELO_CLUBELO_INTERCEPT + ELO_CLUBELO_SLOPE * 1600.0
    )


def test_write_coldstart_log_columns(tmp_path, monkeypatch):
    path = tmp_path / "meta" / "elo_coldstart_log.csv"
    monkeypatch.setattr(statistical_models, "ELO_COLDSTART_LOG_PATH", path)
    frame = pd.DataFrame(
        [
            {
                "season": "2025-2026",
                "team": "Leeds United",
                "source": "clubelo_chart",
                "rating_date": "2025-05-03",
                "raw_clubelo": 1721.16,
                "rescaled": 1405.69,
                "elo_rating": 1405.69,
            },
            {
                "season": "2025-2026",
                "team": "Burnley",
                "source": "fallback_bottom3",
                "rating_date": None,
                "raw_clubelo": None,
                "rescaled": None,
                "elo_rating": 1405.69,
            },
        ]
    )
    write_coldstart_log(frame)
    written = pd.read_csv(path)
    assert list(written["source"]) == ["clubelo_chart", "fallback_bottom3"]
    # Fallback tidak boleh punya raw_clubelo -> mencegah salah label seperti kasus Leeds.
    assert pd.isna(written.loc[1, "raw_clubelo"])
    assert written.loc[0, "raw_clubelo"] == 1721.16
