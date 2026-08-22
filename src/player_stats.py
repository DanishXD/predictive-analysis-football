"""Fetch key player statistics from FBref for contextual squad information."""

from __future__ import annotations

import argparse
from functools import lru_cache

import pandas as pd

from config import PROCESSED_DIR
from team_mapping import TEAM_NAME_MAPPING

try:
    import soccerdata as sd
except ImportError:
    sd = None

SOCCERDATA_HINT = (
    "soccerdata tidak terpasang. Fitur data pemain dinonaktifkan. "
    "Pasang opsional dengan: pip install soccerdata"
)

MATCHES_PATH = PROCESSED_DIR / "matches_clean.csv"

# FBref team names -> canonical EPL names (derived from single source of truth)
# FBref format is similar to football-data.co.uk short names, so we reuse the mapping
FBREF_TO_CANONICAL = TEAM_NAME_MAPPING.copy()

# FBref has some additional variations not in football-data.co.uk
FBREF_TO_CANONICAL.update({
    "Manchester Utd": "Manchester United",  # FBref uses "Utd" instead of "United"
    "Nott'ham Forest": "Nottingham Forest",  # FBref apostrophe variation
})

DISCLAIMER = (
    "Data pemain ini hanya konteks tambahan. TIDAK dipakai sebagai fitur di model "
    "prediksi skor/W-D-L dan tidak mencakup info cedera/suspensi."
)


@lru_cache(maxsize=4)
def fetch_player_stats(season: str) -> pd.DataFrame:
    """Fetch and merge standard and shooting stats from FBref.

    Hasil di-cache per season selama proses berjalan supaya pemanggil
    berulang (mis. CLI yang butuh data season + match-level) tidak scrape
    FBref berkali-kali.
    """
    if sd is None:
        raise RuntimeError(SOCCERDATA_HINT)
    print(f"Mengambil data pemain FBref musim {season}...")
    fbref = sd.FBref(leagues="ENG-Premier League", seasons=season)
    
    standard = fbref.read_player_season_stats(stat_type="standard")
    shooting = fbref.read_player_season_stats(stat_type="shooting")
    
    # Extract key columns from MultiIndex DataFrames
    # Standard: ('Performance', 'Gls'), ('Performance', 'Ast')
    standard_clean = standard[[("Performance", "Gls"), ("Performance", "Ast")]].copy()
    standard_clean.columns = ["Gls", "Ast"]
    
    # Shooting: ('Standard', 'Sh'), ('Standard', 'SoT'), ('Expected', 'xG') if available
    shooting_cols = [("Standard", "Sh"), ("Standard", "SoT")]
    # Try to get xG but it might not be available in all stat types
    if ("Expected", "xG") in shooting.columns:
        shooting_cols.append(("Expected", "xG"))
    elif ("Standard", "xG") in shooting.columns:
        shooting_cols.append(("Standard", "xG"))
    
    shooting_clean = shooting[shooting_cols].copy()
    shooting_clean.columns = ["Sh", "SoT", "xG"] if len(shooting_cols) == 3 else ["Sh", "SoT"]
    
    # Merge standard and shooting stats
    merged = standard_clean.merge(
        shooting_clean,
        left_index=True,
        right_index=True,
        how="left",
    )
    
    # Reset index to access season, team, player
    merged = merged.reset_index()
    
    # Standardize team names
    if "team" in merged.columns:
        merged["team"] = merged["team"].replace(FBREF_TO_CANONICAL)

    return merged.copy()


def get_key_players(
    team_name: str,
    player_data: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Return top N players by goals+assists with relevant stats."""
    team_data = player_data.loc[player_data["team"] == team_name].copy()
    
    if team_data.empty:
        return pd.DataFrame()
    
    # Calculate goal contribution and sort
    team_data["goal_contribution"] = team_data["Gls"] + team_data["Ast"]
    team_data = team_data.sort_values(
        ["goal_contribution", "Gls", "Ast"],
        ascending=False,
    )
    
    # Select top N and relevant columns
    columns = ["player", "Gls", "Ast", "Sh", "SoT", "xG", "goal_contribution"]
    available_columns = [col for col in columns if col in team_data.columns]
    
    result = team_data.head(top_n)[available_columns].copy()
    result = result.rename(
        columns={
            "Gls": "goals",
            "Ast": "assists",
            "Sh": "shots",
            "SoT": "shots_on_target",
        }
    )
    if "xG" in result.columns:
        result = result.rename(columns={"xG": "expected_goals"})
    else:
        result["expected_goals"] = float("nan")
    
    return result.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse optional season and team arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch EPL player stats from FBref for context."
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2526",
        help="Season in FBref format, e.g., '2526' for 2025-2026.",
    )
    parser.add_argument(
        "--team",
        type=str,
        help="Optional: show top-3 players for a specific team.",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch player stats and optionally display key players for a team."""
    args = parse_args()
    
    try:
        player_data = fetch_player_stats(args.season)
        print(f"Berhasil: {len(player_data)} pemain dari FBref musim {args.season}")
        
        if args.team:
            key_players = get_key_players(args.team, player_data)
            if key_players.empty:
                print(f"\nTim '{args.team}' tidak ditemukan di data FBref.")
                print("Coba nama tim persis seperti di dataset:")
                teams = sorted(player_data["team"].unique())
                for idx, team in enumerate(teams, start=1):
                    print(f"  {idx}. {team}")
            else:
                print(f"\nTop-3 pemain {args.team} (goals + assists):")
                print(key_players.to_string(index=False))
        else:
            teams = sorted(player_data["team"].unique())
            print(f"\n{len(teams)} tim tersedia:")
            for idx, team in enumerate(teams, start=1):
                print(f"  {idx}. {team}")
            print("\nGunakan --team 'Nama Tim' untuk melihat top-3 pemain.")
        
        print(f"\n{DISCLAIMER}")
    
    except Exception as exc:
        print(f"Error: {exc}")
        print("Pastikan koneksi internet tersedia dan FBref tidak memblokir akses.")


if __name__ == "__main__":
    main()
