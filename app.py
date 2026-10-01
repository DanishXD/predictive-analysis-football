"""Web app prediksi match EPL — antarmuka Streamlit untuk logika yang sama
dengan CLI (``src/predict_match.py``).

Versi web ini sengaja hanya menampilkan prediksi yang punya nilai nyata:
skor, W/D/L, BTTS, corner, dan kartu kuning. Bagian key player dan MOTM
tidak ada karena keduanya butuh scraping FBref saat runtime yang hampir
selalu gagal di lingkungan server, jadi isinya akan permanent berupa
pesan "tidak tersedia". CLI tetap menyediakan keduanya seperti biasa.

Pemilihan model tidak di-hardcode di sini. Semua angka model diambil dari
fungsi yang sama dengan CLI, sehingga web dan CLI tidak mungkin memilih
model berbeda untuk input yang sama.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import predict_match as pm  # noqa: E402

# Snapshot ter-commit, bukan data/processed dan models/ yang gitignored.
pm.use_snapshot_directory(Path(__file__).resolve().parent / "deploy")

# Minimal Paper: krem/kertas, teks gelap, muted abu-abu. Tanpa warna,
# gradient, atau shadow. Judul dan skor serif, sisanya sans-serif.
PAPER_CSS = """
<style>
    .stApp {
        background: #faf8f3;
        color: #2c2c2a;
    }
    .stApp h1, .stApp h2, .stApp h3 {
        font-family: Georgia, 'Times New Roman', serif;
        color: #2c2c2a;
        font-weight: 400;
    }
    h1.paper-title {
        text-align: center;
        font-size: 2rem;
        letter-spacing: 0.02em;
        margin-bottom: 0.1rem;
    }
    p.paper-subtitle {
        text-align: center;
        color: #888780;
        font-family: Georgia, 'Times New Roman', serif;
        font-style: italic;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .stApp label, .stApp p, .stApp span, .stApp div {
        font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #2c2c2a;
    }
    .team-name {
        text-align: center;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.15rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #d8d4c8;
        margin-bottom: 0.4rem;
    }
    .score-value {
        text-align: center;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 32px;
        line-height: 1.2;
        margin: 0.6rem 0 0.2rem 0;
    }
    .score-caption {
        text-align: center;
        color: #888780;
        font-size: 0.78rem;
        margin-bottom: 0.4rem;
    }
    .section-label {
        color: #888780;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.15rem;
    }
    .section-value {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.05rem;
        margin-bottom: 0.1rem;
    }
    .section-note {
        color: #888780;
        font-size: 0.75rem;
        margin-bottom: 0.2rem;
    }
    .model-label {
        color: #888780;
        font-size: 0.78rem;
        margin: 0.15rem 0 0.9rem 0;
    }
    .paper-disclaimer {
        border-top: 1px solid #d8d4c8;
        margin-top: 2.2rem;
        padding-top: 0.8rem;
        color: #888780;
        font-size: 0.76rem;
        line-height: 1.6;
    }
    .stButton button {
        background: #faf8f3;
        color: #2c2c2a;
        border: 1px solid #2c2c2a;
        border-radius: 0;
        font-family: Georgia, 'Times New Roman', serif;
    }
</style>
"""


def _metric_block(label: str, value: str, note: str = "") -> None:
    """Satu metrik dengan label kecil abu-abu di atas dan nilai di bawahnya."""
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-value">{value}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(
            f'<div class="section-note">{note}</div>', unsafe_allow_html=True
        )


def _render_wdl(result: dict) -> None:
    st.markdown(
        f'<div class="section-label">W / D / L</div>', unsafe_allow_html=True
    )
    probabilities = result["ml_probabilities"]
    st.markdown(
        '<div class="section-value">'
        f'Home {probabilities["H"]:.1%} &nbsp;·&nbsp; '
        f'Draw {probabilities["D"]:.1%} &nbsp;·&nbsp; '
        f'Away {probabilities["A"]:.1%}'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-note">'
        f'{pm.CLASSIFICATION_MODEL_NAMES[result["classification_selection"]["model"]]} '
        f'({pm.describe_selection_metric(result["classification_selection"])})'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_corner(result: dict) -> None:
    grid = result["corner_grid"]
    if grid is None:
        _metric_block("Corner", "tidak tersedia")
        return
    home, away = pm.expected_corners(grid)
    over_under = pm.corner_over_under(grid, threshold=result["corner_threshold"])
    _metric_block(
        "Corner",
        f"{home:.1f} &nbsp;·&nbsp; {away:.1f}",
        note=(
            f'Total {home + away:.1f} &nbsp;|&nbsp; '
            f'Over {over_under["threshold"]:.0f} {over_under["over"]:.1%} '
            f'&nbsp;|&nbsp; Under {over_under["under"]:.1%}'
        ),
    )


def _render_btts(result: dict) -> None:
    from statistical_models import btts_probability

    probability = btts_probability(result["score_grid"])
    _metric_block("BTTS", f"{probability:.1%}", note="Both teams to score")


def _render_yellow(result: dict) -> None:
    grid = result["yellow_grid"]
    if grid is None:
        _metric_block("Kartu kuning", "tidak tersedia")
        return
    home, away = pm.expected_yellow_cards(grid)
    over_under = pm.yellow_over_under(grid, result["yellow_threshold"])
    _metric_block(
        "Kartu kuning",
        f"{home:.1f} &nbsp;·&nbsp; {away:.1f}",
        note=(
            f'Total {home + away:.1f} &nbsp;|&nbsp; '
            f'Over {over_under["threshold"]:.0f} {over_under["over"]:.1%} '
            f'&nbsp;|&nbsp; Under {over_under["under"]:.1%}'
        ),
    )


def main() -> None:
    st.set_page_config(page_title="Match predictor", layout="centered")
    st.markdown(PAPER_CSS, unsafe_allow_html=True)

    st.markdown(
        '<h1 class="paper-title">Match predictor</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="paper-subtitle">Probabilistic EPL match prediction</p>',
        unsafe_allow_html=True,
    )

    teams = pm.list_teams()
    left, right = st.columns(2)
    with left:
        home_team = st.selectbox("Home", teams, index=teams.index("Arsenal"))
    with right:
        default_away = teams.index("Chelsea") if "Chelsea" in teams else 1
        away_team = st.selectbox("Away", teams, index=default_away)

    if home_team == away_team:
        st.caption("Pilih dua tim yang berbeda.")
        return

    if st.button("Predict", use_container_width=True):
        result = pm.get_prediction(home_team, away_team)

        score = pm.top_scorelines(result["score_grid"])[0]
        st.markdown("---")
        st.markdown(
            f'<div class="team-name">{result["home_team"]} &nbsp;vs&nbsp; '
            f'{result["away_team"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="score-value">{score[0]} &ndash; {score[1]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="score-caption">'
            f'Probabilitas {score[2]:.1%} &nbsp;|&nbsp; '
            f'{pm.GOAL_MODEL_NAMES[result["goal_selection"]["model"]]} '
            f'({pm.describe_selection_metric(result["goal_selection"])})'
            "</div>",
            unsafe_allow_html=True,
        )

        _render_wdl(result)

        columns = st.columns(3)
        with columns[0]:
            _render_corner(result)
        with columns[1]:
            _render_btts(result)
        with columns[2]:
            _render_yellow(result)

        if result["warnings"]:
            st.markdown("---")
            for warning in result["warnings"]:
                st.caption(warning)

    st.markdown(
        '<div class="paper-disclaimer">'
        "Ini adalah exercise data science untuk belajar, bukan alat rekomendasi bet. "
        "Model prediksi bola jarang bisa konsisten mengalahkan odds bandar. "
        "Jangan mempertaruhkan uang berdasarkan output halaman ini."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()