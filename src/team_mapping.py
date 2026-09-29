"""Centralised team name mappings for consistency across all modules."""

# football-data.co.uk short names -> canonical EPL names
TEAM_NAME_MAPPING = {
    "Brighton": "Brighton & Hove Albion",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Tottenham": "Tottenham Hotspur",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
}

# Reverse mapping: canonical -> FBref format (for player_stats.py)
CANONICAL_TO_FBREF = {value: key for key, value in TEAM_NAME_MAPPING.items()}

# Canonical EPL names -> ClubElo club names (api.clubelo.com / clubelo.com).
# Teams absent here are identical in both naming schemes.
# Catatan: tidak sama dengan format football-data — ClubElo memakai "Forest"
# (bukan "Nott'm Forest") dan "Sheffield United" (bukan "Sheffield Utd").
CANONICAL_TO_CLUBELO = {
    "Brighton & Hove Albion": "Brighton",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}
