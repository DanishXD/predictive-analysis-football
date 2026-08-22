"""Per-match player predictions: SOT range and MOTM candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from player_stats import (
    SOCCERDATA_HINT,
    fetch_player_stats,
    get_key_players,
    FBREF_TO_CANONICAL,
    DISCLAIMER as PLAYER_DISCLAIMER,
)

try:
    import soccerdata as sd
except ImportError:
    sd = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "matches_clean.csv"
FBREF_SEASON = "2526"
RECENT_MATCHES = 5

SOT_DISCLAIMER = (
    "Estimasi SOT berdasarkan rata-rata beberapa pertandingan terakhir dan strength "
    "defense lawan. Varians per match tinggi, angka pasti misleading."
)
MOTM_DISCLAIMER = (
    "Kandidat MOTM berdasarkan proyeksi kontribusi gol (goals+assists musim ini). "
    "Keputusan MOTM aktual bisa berbeda karena faktor subjektif (momen magis, "
    "penyelamatan, dll) yang tidak tercakup. Juga tidak mencakup rotasi/cedera/"
    "keputusan taktik menit terakhir."
)

# Reverse mapping: canonical name -> FBref name (for schedule lookup)
CANONICAL_TO_FBREF = {v: k for k, v in FBREF_TO_CANONICAL.items()}


def _team_sot_conceded() -> dict[str, float]:
    """Average SOT conceded per game per team, from historical data."""
    d = pd.read_csv(MATCHES_PATH)
    home_def = d.groupby("team_home")["shots_on_target_away"].mean()
    away_def = d.groupby("team_away")["shots_on_target_home"].mean()
    avg = (home_def.add(away_def, fill_value=0) / 2).to_dict()
    league_mean = np.mean(list(avg.values()))
    # Fill any missing team with league average
    return {team: avg.get(team, league_mean) for team in
            set(d["team_home"]) | set(d["team_away"])}


def _fbref_team_name(team: str) -> str:
    """Convert canonical name back to FBref format for schedule lookup."""
    return CANONICAL_TO_FBREF.get(team, team)


def _recent_team_matches(team: str) -> pd.DataFrame:
    """Fetch match-level player stats for recent games of a specific team."""
    if sd is None:
        raise RuntimeError(SOCCERDATA_HINT)
    fbref = sd.FBref(leagues="ENG-Premier League", seasons=FBREF_SEASON)
    schedule = fbref.read_schedule()

    fbref_team = _fbref_team_name(team)
    team_games = schedule[
        (schedule["home_team"] == fbref_team) | (schedule["away_team"] == fbref_team)
    ].dropna(subset=["game_id"]).tail(RECENT_MATCHES)

    if team_games.empty:
        return pd.DataFrame()

    games = team_games["game_id"].tolist()
    all_frames = []
    for game in games:
        try:
            match_stats = fbref.read_player_match_stats(
                stat_type="summary", match_id=game
            )
            # Flatten: extract SoT + identify which team
            sot = match_stats[[("Performance", "SoT")]].copy()
            sot.columns = ["sot"]
            sot = sot.reset_index()
            # Keep only players from the requested team
            canon_team = FBREF_TO_CANONICAL.get(sot.iloc[0].get("team", ""), sot.iloc[0].get("team", ""))
            if canon_team == team:
                all_frames.append(sot)
            else:
                # The team might be in a different row format - check
                for row in sot.itertuples():
                    t = FBREF_TO_CANONICAL.get(row.team, row.team)
                    if t == team:
                        all_frames.append(sot)
                        break
        except Exception:
            continue

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def predict_player_sot(player_name: str, opponent_team: str) -> dict:
    """Estimate SOT range for a player based on recent form and opponent defense.

    Returns a dict with 'player', 'team', 'avg_sot', 'sot_range', 'matches_used',
    'disclaimer'.
    """
    sot_conceded = _team_sot_conceded()
    opp_defense = sot_conceded.get(opponent_team, np.mean(list(sot_conceded.values())))

    # Get the player's team from season-level data
    season_data = fetch_player_stats(FBREF_SEASON)
    player_row = season_data[season_data["player"] == player_name]
    if player_row.empty:
        return {
            "player": player_name,
            "avg_sot": None,
            "sot_range": None,
            "matches_used": 0,
            "error": f"Pemain '{player_name}' tidak ditemukan di data FBref.",
            "disclaimer": SOT_DISCLAIMER,
        }

    player_team = player_row.iloc[0]["team"]
    recent = _recent_team_matches(player_team)

    if recent.empty or "sot" not in recent.columns:
        return {
            "player": player_name,
            "player_team": player_team,
            "avg_sot": None,
            "sot_range": None,
            "matches_used": 0,
            "error": "Data pertandingan terakhir tidak tersedia.",
            "disclaimer": SOT_DISCLAIMER,
        }

    player_matches = recent[recent["player"] == player_name]
    if player_matches.empty:
        return {
            "player": player_name,
            "player_team": player_team,
            "avg_sot": None,
            "sot_range": None,
            "matches_used": 0,
            "error": f"Tidak ada data SOT untuk '{player_name}' dalam {RECENT_MATCHES} pertandingan terakhir.",
            "disclaimer": SOT_DISCLAIMER,
        }

    sot_values = player_matches["sot"].dropna().astype(float).values
    if len(sot_values) == 0:
        return {
            "player": player_name,
            "player_team": player_team,
            "avg_sot": None,
            "sot_range": None,
            "matches_used": 0,
            "error": "Nilai SOT tidak tersedia.",
            "disclaimer": SOT_DISCLAIMER,
        }

    avg_sot = float(np.mean(sot_values))
    std_sot = float(np.std(sot_values)) if len(sot_values) > 1 else 0.3
    league_avg = np.mean(list(sot_conceded.values()))

    # Adjust for opponent defense: stronger defense (lower SOT conceded) reduces
    # expected SOT proportionally
    adjustment_factor = opp_defense / league_avg
    adjusted_avg = avg_sot * adjustment_factor

    low = max(0, round(adjusted_avg - std_sot, 1))
    high = round(adjusted_avg + std_sot, 1)

    return {
        "player": player_name,
        "player_team": player_team,
        "avg_sot_raw": round(avg_sot, 2),
        "avg_sot_adjusted": round(adjusted_avg, 2),
        "sot_range": f"{low}–{high}",
        "sot_range_low": low,
        "sot_range_high": high,
        "matches_used": len(sot_values),
        "opponent_defense_avg_sot_conceded": round(opp_defense, 2),
        "disclaimer": SOT_DISCLAIMER,
    }


def predict_motm_candidate(team_a: str, team_b: str) -> dict:
    """Rank players by goal contribution (goals+assists) to suggest MOTM candidates.

    Returns a dict with 'home_candidates', 'away_candidates', 'disclaimer'.
    """
    season_data = fetch_player_stats(FBREF_SEASON)
    players_a = get_key_players(team_a, season_data, top_n=3)
    players_b = get_key_players(team_b, season_data, top_n=3)

    def format_candidate(df: pd.DataFrame) -> list[dict]:
        candidates = []
        for _, row in df.iterrows():
            candidates.append({
                "player": row.get("player", ""),
                "goals": int(row.get("goals", 0)),
                "assists": int(row.get("assists", 0)),
                "goal_contribution": int(row.get("goals", 0) + row.get("assists", 0)),
                "shots_on_target": int(row.get("shots_on_target", 0)),
                "expected_goals": (
                    round(row.get("expected_goals", 0), 1)
                    if pd.notna(row.get("expected_goals"))
                    else None
                ),
            })
        return candidates

    return {
        "home_team": team_a,
        "away_team": team_b,
        "home_candidates": format_candidate(players_a),
        "away_candidates": format_candidate(players_b),
        "label": "Kandidat MOTM paling mungkin (berdasarkan proyeksi kontribusi gol)",
        "disclaimer": MOTM_DISCLAIMER,
    }
