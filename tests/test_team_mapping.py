from team_mapping import CANONICAL_TO_CLUBELO, CANONICAL_TO_FBREF, TEAM_NAME_MAPPING


def test_canonical_names_are_unique():
    assert len(set(TEAM_NAME_MAPPING.values())) == len(TEAM_NAME_MAPPING)


def test_reverse_mapping_is_complete_inverse():
    for short_name, canonical in TEAM_NAME_MAPPING.items():
        assert CANONICAL_TO_FBREF[canonical] == short_name


def test_both_mappings_same_size():
    assert len(CANONICAL_TO_FBREF) == len(TEAM_NAME_MAPPING)


def test_known_mappings():
    assert TEAM_NAME_MAPPING["Man City"] == "Manchester City"
    assert TEAM_NAME_MAPPING["Man United"] == "Manchester United"
    assert TEAM_NAME_MAPPING["Tottenham"] == "Tottenham Hotspur"


def test_clubelo_known_mappings():
    assert CANONICAL_TO_CLUBELO["Nottingham Forest"] == "Forest"
    assert CANONICAL_TO_CLUBELO["Brighton & Hove Albion"] == "Brighton"
    assert CANONICAL_TO_CLUBELO["Wolverhampton Wanderers"] == "Wolves"
    assert CANONICAL_TO_CLUBELO["West Bromwich Albion"] == "West Brom"
    assert CANONICAL_TO_CLUBELO["Newcastle United"] == "Newcastle"


def test_clubelo_mapping_values_unique_and_nonempty():
    assert all(CANONICAL_TO_CLUBELO.values())
    assert len(set(CANONICAL_TO_CLUBELO.values())) == len(CANONICAL_TO_CLUBELO)
