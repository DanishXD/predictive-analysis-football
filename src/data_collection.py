"""Collect and clean EPL match data for the 2016/17 to 2025/26 seasons."""

from __future__ import annotations

import io
import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

try:
    import penaltyblog as pb
except ImportError:
    pb = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COMPETITION = "ENG Premier League"
SEASONS = (
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
)

from team_mapping import TEAM_NAME_MAPPING

COLUMN_ALIASES = {
    "Date": "date",
    "Time": "time",
    "HomeTeam": "team_home",
    "AwayTeam": "team_away",
    "FTHG": "goals_home",
    "FTAG": "goals_away",
    "FTR": "result",
    "HTHG": "half_time_goals_home",
    "HTAG": "half_time_goals_away",
    "HTR": "half_time_result",
    "HS": "shots_home",
    "AS": "shots_away",
    "HST": "shots_on_target_home",
    "AST": "shots_on_target_away",
    "HC": "corners_home",
    "AC": "corners_away",
    "HY": "yellow_cards_home",
    "AY": "yellow_cards_away",
    "HR": "red_cards_home",
    "AR": "red_cards_away",
    "B365H": "odds_b365_home",
    "B365D": "odds_b365_draw",
    "B365A": "odds_b365_away",
    "AvgH": "odds_avg_home",
    "AvgD": "odds_avg_draw",
    "AvgA": "odds_avg_away",
    "MaxH": "odds_max_home",
    "MaxD": "odds_max_draw",
    "MaxA": "odds_max_away",
}

OUTPUT_COLUMNS = (
    "match_id",
    "competition",
    "season",
    "datetime",
    "date",
    "team_home",
    "team_away",
    "goals_home",
    "goals_away",
    "result",
    "half_time_goals_home",
    "half_time_goals_away",
    "half_time_result",
    "shots_home",
    "shots_away",
    "shots_on_target_home",
    "shots_on_target_away",
    "corners_home",
    "corners_away",
    "yellow_cards_home",
    "yellow_cards_away",
    "red_cards_home",
    "red_cards_away",
    "odds_b365_home",
    "odds_b365_draw",
    "odds_b365_away",
    "odds_avg_home",
    "odds_avg_draw",
    "odds_avg_away",
    "odds_max_home",
    "odds_max_draw",
    "odds_max_away",
)


def fetch_with_penaltyblog(season: str) -> pd.DataFrame:
    """Fetch one season through penaltyblog's FootballData scraper."""
    if pb is None:
        raise RuntimeError("penaltyblog is not installed")

    scraper = pb.scrapers.FootballData(COMPETITION, season)
    return scraper.get_fixtures().reset_index()


def fetch_manually(season: str) -> pd.DataFrame:
    """Fetch one season directly from football-data.co.uk."""
    start_year, end_year = season.split("-")
    season_code = f"{start_year[-2:]}{end_year[-2:]}"
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}
    request = Request(url, headers=headers)

    try:
        with urlopen(request, timeout=30) as response:
            content = response.read().decode("utf-8-sig")
    except URLError:
        # Some networks block football-data.co.uk or receive its invalid TLS
        # certificate. Jina is used only as a transport proxy for the same URL.
        proxy_url = f"https://r.jina.ai/{url}"
        proxy_request = Request(proxy_url, headers=headers)
        with urlopen(proxy_request, timeout=90) as response:
            content = response.read().decode("utf-8-sig")
        marker = "Markdown Content:\n"
        if marker not in content:
            raise ValueError("Respons proxy tidak berisi CSV football-data")
        content = content.split(marker, maxsplit=1)[1]
        print("  download manual lewat transport proxy karena akses langsung diblokir")

    return pd.read_csv(io.StringIO(content))


def fetch_season(season: str) -> tuple[pd.DataFrame, str]:
    """Use penaltyblog first, falling back to the source CSV on failure."""
    try:
        return fetch_with_penaltyblog(season), "penaltyblog"
    except Exception as exc:
        print(f"  penaltyblog gagal ({exc}); mencoba download manual")
        return fetch_manually(season), "manual"


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize both penaltyblog and original football-data column names."""
    frame = frame.rename(columns=COLUMN_ALIASES).copy()
    frame.columns = [
        re.sub(
            r"[^a-z0-9]+",
            "_",
            column.lower().replace(">", "_over_").replace("<", "_under_"),
        ).strip("_")
        for column in frame.columns
    ]

    penaltyblog_aliases = {
        "fthg": "goals_home",
        "ftag": "goals_away",
        "ftr": "result",
        "hthg": "half_time_goals_home",
        "htag": "half_time_goals_away",
        "htr": "half_time_result",
        "hs": "shots_home",
        "as": "shots_away",
        "hst": "shots_on_target_home",
        "ast": "shots_on_target_away",
        "hc": "corners_home",
        "ac": "corners_away",
        "hy": "yellow_cards_home",
        "ay": "yellow_cards_away",
        "hr": "red_cards_home",
        "ar": "red_cards_away",
        "b365_h": "odds_b365_home",
        "b365_d": "odds_b365_draw",
        "b365_a": "odds_b365_away",
        "avg_h": "odds_avg_home",
        "avg_d": "odds_avg_draw",
        "avg_a": "odds_avg_away",
        "max_h": "odds_max_home",
        "max_d": "odds_max_draw",
        "max_a": "odds_max_away",
    }
    return frame.rename(columns=penaltyblog_aliases)


def clean_matches(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine seasons and produce the canonical clean match table."""
    matches = pd.concat(frames, ignore_index=True, sort=False)

    missing_columns = [column for column in OUTPUT_COLUMNS if column not in matches.columns]
    if missing_columns:
        empty_columns = pd.DataFrame(pd.NA, index=matches.index, columns=missing_columns)
        matches = pd.concat([matches, empty_columns], axis=1)

    matches["date"] = pd.to_datetime(matches["date"], dayfirst=True, errors="coerce")
    parsed_datetime = pd.to_datetime(matches["datetime"], errors="coerce")
    if "time" in matches.columns:
        date_and_time = pd.to_datetime(
            matches["date"].dt.strftime("%Y-%m-%d")
            + " "
            + matches["time"].astype("string"),
            errors="coerce",
        )
        parsed_datetime = parsed_datetime.fillna(date_and_time)
    matches["datetime"] = parsed_datetime.fillna(matches["date"])

    for column in ("team_home", "team_away"):
        matches[column] = (
            matches[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .replace(TEAM_NAME_MAPPING)
        )

    for column in ("goals_home", "goals_away"):
        matches[column] = pd.to_numeric(matches[column], errors="coerce")

    completed = matches[["date", "team_home", "team_away", "goals_home", "goals_away"]].notna().all(axis=1)
    matches = matches.loc[completed].copy()
    matches[["goals_home", "goals_away"]] = matches[["goals_home", "goals_away"]].astype("int64")

    expected_result = pd.Series("D", index=matches.index)
    expected_result.loc[matches["goals_home"] > matches["goals_away"]] = "H"
    expected_result.loc[matches["goals_home"] < matches["goals_away"]] = "A"
    matches["result"] = expected_result

    matches["competition"] = COMPETITION
    matches["match_id"] = (
        matches["date"].dt.strftime("%Y%m%d")
        + "-"
        + matches["team_home"].str.lower().str.replace(r"[^a-z0-9]+", "-", regex=True).str.strip("-")
        + "-"
        + matches["team_away"].str.lower().str.replace(r"[^a-z0-9]+", "-", regex=True).str.strip("-")
    )

    matches = matches.loc[:, OUTPUT_COLUMNS]
    return matches.sort_values(["datetime", "match_id"]).reset_index(drop=True)


def validate_matches(matches: pd.DataFrame) -> list[str]:
    """Return data quality warnings; raise for invalid core match data."""
    errors = []
    warnings = []

    if matches["match_id"].duplicated().any():
        errors.append("match_id duplikat ditemukan")
    if (matches["team_home"] == matches["team_away"]).any():
        errors.append("tim kandang sama dengan tim tandang")
    if (matches[["goals_home", "goals_away"]] < 0).any().any():
        errors.append("nilai gol negatif ditemukan")
    if matches[["date", "team_home", "team_away", "result"]].isna().any().any():
        errors.append("missing value ditemukan pada kolom inti")

    for season in SEASONS:
        season_matches = matches.loc[matches["season"] == season]
        teams = set(season_matches["team_home"]) | set(season_matches["team_away"])
        if len(season_matches) != 380:
            warnings.append(f"{season}: {len(season_matches)} pertandingan, seharusnya 380")
        if len(teams) != 20:
            warnings.append(f"{season}: {len(teams)} tim, seharusnya 20")

    if errors:
        raise ValueError("; ".join(errors))
    return warnings


def main() -> None:
    """Run the complete Phase 1 collection and cleaning pipeline."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for season in SEASONS:
        print(f"Mengambil EPL {season}...")
        raw, source = fetch_season(season)
        raw.to_csv(RAW_DIR / f"epl_{season}.csv", index=False)

        normalized = normalize_columns(raw)
        normalized["season"] = season
        normalized["competition"] = COMPETITION
        frames.append(normalized)
        print(f"  {len(raw)} baris dari {source}")

    matches = clean_matches(frames)
    warnings = validate_matches(matches)
    output_path = PROCESSED_DIR / "matches_clean.csv"
    matches.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print(f"\nTersimpan: {output_path}")
    print(f"Total: {len(matches)} pertandingan, {len(matches.columns)} kolom")
    if warnings:
        print("Peringatan kualitas data:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Validasi kualitas data: tidak ada masalah")


if __name__ == "__main__":
    main()
