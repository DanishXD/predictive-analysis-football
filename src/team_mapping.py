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
