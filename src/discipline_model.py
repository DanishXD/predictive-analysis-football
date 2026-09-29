"""Poisson discipline model: yellow cards, plus optional red cards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import penaltyblog as pb

from config import (
    MODELS_DIR,
    PROCESSED_DIR,
    REFEREE_ENABLED,
    REFEREE_MIN_MATCHES,
    REFEREE_OTHER_LABEL,
    REFEREE_PRIOR_STRENGTH,
    TEST_SEASON,
    TIME_DECAY_XI,
    YELLOW_OVER_UNDER_DEFAULT,
)
from team_mapping import TEAM_NAME_MAPPING

INPUT_PATH = PROCESSED_DIR / "matches_clean.csv"
YELLOW_MODEL_PATH = MODELS_DIR / "yellow_card_model.pkl"
RED_MODEL_PATH = MODELS_DIR / "red_card_model.pkl"
REFEREE_FACTORS_PATH = PROCESSED_DIR / "referee_card_factors.csv"


def fit_discipline_model(
    train: pd.DataFrame, home_col: str, away_col: str, label: str
):
    """Fit a Poisson model on card counts with time-decay weights."""
    weights = pb.models.dixon_coles_weights(
        train["date"],
        xi=TIME_DECAY_XI,
        base_date=train["date"].max(),
    )
    model = pb.models.PoissonGoalsModel(
        goals_home=train[home_col],
        goals_away=train[away_col],
        teams_home=train["team_home"],
        teams_away=train["team_away"],
        weights=weights,
    )
    model.fit()
    return model


def predict_discipline_fixture(model, home: str, away: str, max_cards: int = 15):
    """Predict card outcome, using league-average for unseen teams."""
    known_teams = set(model.teams)
    cold_start = home not in known_teams or away not in known_teams

    if not cold_start:
        return model.predict(home, away, max_goals=max_cards), False

    params = model.params_array
    attacks = params[: model.n_teams]
    defenses = params[model.n_teams : 2 * model.n_teams]
    home_adv = float(params[-1])

    def strength(team):
        if team in known_teams:
            idx = model.team_to_idx[team]
            return (float(params[idx]), float(params[idx + model.n_teams]))
        return (float(attacks.mean()), float(defenses.mean()))

    home_attack, home_defense = strength(home)
    away_attack, away_defense = strength(away)

    expected_home = np.exp(home_adv + home_attack + away_defense)
    expected_away = np.exp(away_attack + home_defense)

    grid = pb.models.create_dixon_coles_grid(
        expected_home, expected_away, rho=0.0, max_goals=max_cards
    )
    return grid, True


def expected_cards(grid):
    """Extract expected home and away card counts from a probability grid."""
    return float(grid.home_goal_expectation), float(grid.away_goal_expectation)


def over_under_probability(grid, threshold: float):
    """Compute P(over threshold), P(under threshold) and push for total cards."""
    total_dist = grid.total_goals_distribution()
    min_total = int(np.floor(threshold))
    is_whole_number = threshold == min_total
    over = float(sum(total_dist[min_total + 1 :]))
    push = float(total_dist[min_total]) if is_whole_number else 0.0
    under = float(
        sum(total_dist[:min_total] if is_whole_number else total_dist[: min_total + 1])
    )
    return {"over": over, "under": under, "push": push, "threshold": threshold}


def first_card_probability(grid) -> dict:
    """Probability which team gets more cards (proxy for 'first card')."""
    home_more = float(np.sum(grid.grid[np.triu_indices_from(grid.grid, k=1)]))
    away_more = float(np.sum(grid.grid[np.tril_indices_from(grid.grid, k=-1)]))
    equal = float(np.sum(np.diag(grid.grid)))
    return {
        "home_more": home_more,
        "away_more": away_more,
        "equal": equal,
        "home_label": "home team gets more cards",
        "away_label": "away team gets more cards",
    }


def build_referee_card_factors(
    train: pd.DataFrame, home_col: str, away_col: str, label: str
) -> pd.DataFrame:
    """Hitung faktor pengali kartu per wasit dari data TRAINING saja.

    Anti-leakage: rate wasit dihitung hanya dari pertandingan training. Wasit
    pertandingan uji sudah diketahui sebelum kick-off, jadi memakai rate
    historisnya tetap sah. Wasit yang jarang (di bawah REFEREE_MIN_MATCHES)
    di-group jadi REFEREE_OTHER_LABEL dengan faktor 1.0 (setara rata-rata
    liga), dan sisanya di-shrink ke rata-rata liga dengan kekuatan prior
    REFEREE_PRIOR_STRENGTH supaya rate tipis tidak berlebihan.
    """
    cards = train[home_col].fillna(0) + train[away_col].fillna(0)
    league_rate = float(cards.sum() / len(train)) if len(train) else 0.0

    referees = train["referee"]
    # Kolom referee bisa kosong total kalau sumber data tidak menyediakannya
    # (mis. scraper penaltyblog). Kembalikan tabel kosong supaya fitur dinonaktifkan
    # dengan rapi, bukan membentuk satu kategori palsu bernama "nan".
    if referees.isna().all() or (referees.astype("string").str.strip() == "").all():
        return pd.DataFrame(
            columns=[
                "referee",
                "training_matches",
                "total_cards",
                "raw_rate",
                "category",
                "prior_weight",
                "shrunk_rate",
                "factor",
                "league_rate",
                "label",
            ]
        )

    grouped = (
        pd.DataFrame({"referee": referees, "cards": cards.astype(float)})
        .groupby("referee", dropna=False)
        .agg(matches=("cards", "size"), total_cards=("cards", "sum"))
        .reset_index()
    )
    grouped["raw_rate"] = np.where(
        grouped["matches"] > 0, grouped["total_cards"] / grouped["matches"], league_rate
    )
    grouped["category"] = np.where(
        grouped["matches"] < REFEREE_MIN_MATCHES, REFEREE_OTHER_LABEL, grouped["referee"]
    )
    # Untuk wasit yang di-group, shrunk_rate = league_rate (faktor 1.0).
    grouped["prior_weight"] = np.where(
        grouped["matches"] < REFEREE_MIN_MATCHES, 0.0, REFEREE_PRIOR_STRENGTH
    )
    grouped["shrunk_rate"] = (
        grouped["matches"] * grouped["raw_rate"] + grouped["prior_weight"] * league_rate
    ) / (grouped["matches"] + grouped["prior_weight"])
    grouped.loc[grouped["prior_weight"] == 0.0, "shrunk_rate"] = league_rate
    grouped["factor"] = np.where(
        league_rate > 0, grouped["shrunk_rate"] / league_rate, 1.0
    )
    grouped["league_rate"] = league_rate
    grouped["label"] = label
    grouped = grouped.rename(columns={"matches": "training_matches"})
    return grouped.sort_values("training_matches", ascending=False, ignore_index=True)


def referee_factor_for(referee: str, factors: pd.DataFrame) -> float:
    """Ambil faktor pengali kartu untuk satu wasit (default 1.0 bila tak dikenal)."""
    if factors.empty or referee is None or (isinstance(referee, float) and pd.isna(referee)):
        return 1.0
    match = factors.loc[factors["referee"] == referee, "factor"]
    if match.empty:
        return 1.0
    return float(match.iloc[0])


def apply_referee_factor(grid, factor: float, max_cards: int = 15):
    """Kalikan expected kartu home & away dengan faktor wasit, lalu bangun grid baru."""
    if factor == 1.0:
        return grid
    home, away = expected_cards(grid)
    return pb.models.create_dixon_coles_grid(
        home * factor, away * factor, rho=0.0, max_goals=max_cards
    )


def get_discipline_strengths(model, all_teams: list[str], label: str) -> pd.DataFrame:
    """Return discipline-attack and discipline-defense parameters per team."""
    rows = []
    for team in all_teams:
        if team in model.teams:
            idx = model.team_to_idx[team]
            attack = float(model.params_array[idx])
            defense = float(model.params_array[idx + model.n_teams])
            source = "fitted"
        else:
            attacks = model.params_array[: model.n_teams]
            defenses = model.params_array[model.n_teams : 2 * model.n_teams]
            attack = float(attacks.mean())
            defense = float(defenses.mean())
            source = "cold_start_league_average"
        rows.append(
            {
                "team": team,
                f"{label}_attack": attack,
                f"{label}_defense": defense,
                "source": source,
            }
        )
    return pd.DataFrame(rows).sort_values(
        f"{label}_attack", ascending=False, ignore_index=True
    )


def main() -> None:
    """Fit yellow card model, test set predictions, save outputs."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(INPUT_PATH, parse_dates=["datetime", "date"])
    train = matches.loc[matches["season"] != TEST_SEASON].copy()
    test = matches.loc[matches["season"] == TEST_SEASON].copy()
    all_teams = sorted(set(matches["team_home"]) | set(matches["team_away"]))

    # Save training metadata
    import json
    metadata = {
        "training_matches": len(train),
        "training_seasons": sorted(train["season"].unique().tolist()),
        "training_period_start": train["season"].min(),
        "training_period_end": train["season"].max(),
        "test_season": TEST_SEASON,
        "timestamp": pd.Timestamp.now().isoformat(),
        "models": ["yellow_card", "red_card"],
    }
    (YELLOW_MODEL_PATH.parent / "discipline_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # --- Yellow cards ---
    print(f"Fit yellow card Poisson pada {len(train)} pertandingan...")
    yellow_model = fit_discipline_model(
        train, "yellow_cards_home", "yellow_cards_away", "yellow"
    )
    yellow_model.save(str(YELLOW_MODEL_PATH))

    # Referee card factors, fitted on TRAIN only (anti-leakage).
    if "referee" not in train.columns:
        referee_factors = pd.DataFrame()
        print("\nPERINGATAN: kolom 'referee' tidak ada, fitur wasit dilewati.")
    elif not REFEREE_ENABLED:
        referee_factors = pd.DataFrame()
        print("\nFitur wasit dinonaktifkan lewat config.REFEREE_ENABLED = False.")
    else:
        referee_factors = build_referee_card_factors(
            train, "yellow_cards_home", "yellow_cards_away", "yellow"
        )

    if referee_factors.empty and REFEREE_ENABLED and "referee" in train.columns:
        print(
            "Fitur wasit dilewati: tidak ada nama wasit yang bisa dipakai "
            "(kolom kosong di data training)."
        )
    elif not referee_factors.empty:
        referee_factors.to_csv(REFEREE_FACTORS_PATH, index=False)
        n_grouped = int((referee_factors["category"] == REFEREE_OTHER_LABEL).sum())
        n_known = int((referee_factors["category"] != REFEREE_OTHER_LABEL).sum())
        print(
            f"\nFitur wasit: {n_known} wasit dengan >= {REFEREE_MIN_MATCHES} match "
            f"training, {n_grouped} di-group jadi '{REFEREE_OTHER_LABEL}'"
        )
        print(
            f"  Faktor kartu range: "
            f"{referee_factors['factor'].min():.3f} - {referee_factors['factor'].max():.3f} "
            f"(1.0 = rata-rata liga)"
        )
        print(f"  Tabel faktor: {REFEREE_FACTORS_PATH}")

    yellow_rows = []
    for match in test.sort_values(["datetime", "match_id"]).itertuples(index=False):
        grid, cold = predict_discipline_fixture(
            yellow_model, match.team_home, match.team_away
        )
        exp_home, exp_away = expected_cards(grid)
        factor = referee_factor_for(getattr(match, "referee", None), referee_factors)
        adjusted = apply_referee_factor(grid, factor)
        adj_home, adj_away = expected_cards(adjusted)
        ou = over_under_probability(adjusted, YELLOW_OVER_UNDER_DEFAULT)
        fc = first_card_probability(adjusted)
        yellow_rows.append(
            {
                "match_id": match.match_id,
                "season": match.season,
                "datetime": match.datetime,
                "team_home": match.team_home,
                "team_away": match.team_away,
                "referee": getattr(match, "referee", None),
                "referee_factor": factor,
                "actual_yellow_home": match.yellow_cards_home,
                "actual_yellow_away": match.yellow_cards_away,
                "expected_yellow_home": adj_home,
                "expected_yellow_away": adj_away,
                "expected_total_yellow": adj_home + adj_away,
                "expected_total_yellow_no_referee": exp_home + exp_away,
                "yellow_over_under_threshold": ou["threshold"],
                "prob_yellow_over": ou["over"],
                "prob_yellow_under": ou["under"],
                "prob_home_more_yellow": fc["home_more"],
                "prob_away_more_yellow": fc["away_more"],
                "prob_equal_yellow": fc["equal"],
                "cold_start": cold,
            }
        )

    yellow_test = pd.DataFrame(yellow_rows)
    yellow_test.to_csv(
        PROCESSED_DIR / "yellow_card_test_predictions.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )

    # Dampak fitur wasit: error absolut kartu kuning total, dengan vs tanpa.
    actual_total = yellow_test["actual_yellow_home"] + yellow_test["actual_yellow_away"]
    mae_ref = float((yellow_test["expected_total_yellow"] - actual_total).abs().mean())
    mae_no_ref = float(
        (yellow_test["expected_total_yellow_no_referee"] - actual_total).abs().mean()
    )
    n_adjusted = int((yellow_test["referee_factor"] != 1.0).sum())
    print(
        f"\nDampak fitur wasit (test {TEST_SEASON}, {len(yellow_test)} match):"
    )
    print(f"  MAE total kartu kuning tanpa fitur wasit : {mae_no_ref:.4f}")
    print(f"  MAE total kartu kuning dengan fitur wasit: {mae_ref:.4f}")
    print(f"  Selisih MAE                              : {mae_ref - mae_no_ref:+.4f}")
    print(f"  Match yang benar-benar disesuaikan        : {n_adjusted}/{len(yellow_test)}")

    yellow_strengths = get_discipline_strengths(yellow_model, all_teams, "yellow")
    yellow_strengths.to_csv(
        PROCESSED_DIR / "yellow_card_team_strengths.csv", index=False
    )

    best_yellow_attack = yellow_strengths.iloc[0]["team"]
    best_yellow_defense = yellow_strengths.sort_values(
        "yellow_defense", ascending=True
    ).iloc[0]["team"]

    print(
        f"Yellow card test predictions: {len(yellow_test)} match, "
        f"cold-start: {int(yellow_test['cold_start'].sum())}"
    )
    print(
        f"Expected yellow range: "
        f"{yellow_test['expected_total_yellow'].min():.2f} - "
        f"{yellow_test['expected_total_yellow'].max():.2f}"
    )
    print(f"Yellow-attack terkuat: {best_yellow_attack}")
    print(f"Yellow-defense terkuat: {best_yellow_defense}")
    print(f"Yellow model tersimpan: {YELLOW_MODEL_PATH}")

    # --- Red cards (opsional, sparse data) ---
    red_matches_with_cards = ((train["red_cards_home"] > 0) | (train["red_cards_away"] > 0)).sum()
    print(
        f"\nRed cards: {red_matches_with_cards}/{len(train)} train matches have red cards "
        f"({red_matches_with_cards / len(train) * 100:.1f}%)"
    )

    try:
        red_model = fit_discipline_model(
            train, "red_cards_home", "red_cards_away", "red"
        )
        red_model.save(str(RED_MODEL_PATH))
        red_strengths = get_discipline_strengths(red_model, all_teams, "red")
        red_strengths.to_csv(
            PROCESSED_DIR / "red_card_team_strengths.csv", index=False
        )
        print(f"Red model tersimpan: {RED_MODEL_PATH}")
        print(
            "PERINGATAN: data kartu merah sangat jarang (~10% pertandingan), "
            "model Poisson kurang stabil. Gunakan prediksi merah dengan hati-hati."
        )
    except Exception as exc:
        print(f"Red model gagal di-fit: {exc}")
        print("Red card model dilewati karena data terlalu sparse.")

    print("\nModel disiplin: arsitektur Poisson sama seperti model gol dan corner.")
    print("Akurasi model kartu kuning lebih rendah dari model gol karena kartu")
    print("dipengaruhi faktor situasional (wasit, tensi pertandingan, dll).")
    print("Model kartu merah sangat tidak stabil karena data sparse.")


if __name__ == "__main__":
    main()
