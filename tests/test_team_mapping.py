from team_mapping import CANONICAL_TO_FBREF, TEAM_NAME_MAPPING


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
