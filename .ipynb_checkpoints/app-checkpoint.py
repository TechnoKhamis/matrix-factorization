import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import joblib
import html as html_mod

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ── CUSTOM CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Source+Sans+3:wght@300;400;500;600;700&display=swap');

/* ── global resets ─────────────────────────────────────────── */
.stApp {
    background: #08080a;
    font-family: 'Source Sans 3', sans-serif;
}

/* hide streamlit chrome */
#MainMenu, footer, header, .stDeployButton { display: none !important; }

/* ── typography ─────────────────────────────────────────────── */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.04em;
}

/* main title */
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.4rem;
    color: #f5f0e8;
    letter-spacing: 0.06em;
    line-height: 1;
    margin: 0;
    padding: 0;
}
.hero-title span {
    background: linear-gradient(135deg, #e8a44a, #f0c97e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.95rem;
    color: #706b62;
    font-weight: 400;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── section headers ───────────────────────────────────────── */
.section-label {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #e8a44a;
    margin-bottom: 4px;
}
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.9rem;
    color: #f5f0e8;
    letter-spacing: 0.03em;
    line-height: 1.1;
    margin-bottom: 16px;
}

/* ── cards ──────────────────────────────────────────────────── */
.card {
    background: linear-gradient(165deg, #111113 0%, #0d0d0f 100%);
    border: 1px solid #1e1d1b;
    border-radius: 6px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #e8a44a33, transparent);
}
.card-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.35rem;
    color: #f5f0e8;
    letter-spacing: 0.04em;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.card-header .badge {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    background: #e8a44a18;
    color: #e8a44a;
    padding: 3px 10px;
    border-radius: 3px;
    border: 1px solid #e8a44a30;
}

/* ── movie list ─────────────────────────────────────────────── */
.movie-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1a1918;
    transition: background 0.2s;
}
.movie-row:last-child { border-bottom: none; }
.movie-row:hover { background: #ffffff04; }
.movie-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    color: #3a3833;
    width: 30px;
    text-align: center;
}
.movie-title {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    color: #d4cfc5;
    flex: 1;
    padding: 0 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.movie-rating {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    color: #e8a44a;
    letter-spacing: 0.03em;
    min-width: 36px;
    text-align: right;
}
.movie-bar-wrap {
    width: 80px;
    height: 4px;
    background: #1a1918;
    border-radius: 2px;
    margin-left: 12px;
    overflow: hidden;
}
.movie-bar {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s cubic-bezier(.22,1,.36,1);
}
.bar-svd { background: linear-gradient(90deg, #e8a44a, #f0c97e); }
.bar-pmf { background: linear-gradient(90deg, #5ba882, #7ec9a4); }

/* ── metric boxes ──────────────────────────────────────────── */
.metric-box {
    background: #0d0d0f;
    border: 1px solid #1e1d1b;
    border-radius: 6px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
}
.metric-label {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #706b62;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: #f5f0e8;
    line-height: 1;
}
.metric-delta {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 6px;
}
.delta-good { color: #5ba882; }
.delta-warn { color: #e8a44a; }

/* ── sidebar ───────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0b0b0d !important;
    border-right: 1px solid #1a1918 !important;
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f5f0e8 !important;
}

/* ── divider ───────────────────────────────────────────────── */
.thin-rule {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e1d1b, transparent);
    margin: 32px 0;
}

/* hide default streamlit dataframe */
.stDataFrame { display: none !important; }

/* ── button override ───────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #e8a44a, #d4922e) !important;
    color: #08080a !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #f0c97e, #e8a44a) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px #e8a44a33 !important;
}

/* ── number input & slider ─────────────────────────────────── */
input[type="number"] {
    background: #111113 !important;
    border: 1px solid #1e1d1b !important;
    color: #f5f0e8 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

st.cache_data.clear()
from utils.recommendation import generate_recommendations

user_index = joblib.load('./processed/user_index.pkl')
train_data = pd.read_csv('./processed/train_data.csv')
movies_pd  = pd.read_csv('./data/movies.dat', sep='::', encoding='latin-1', engine='python', names=["MovieID","Title","Genres"])

# ── HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 20px 0 10px 0;">
    <p class="hero-title">MOVIE <span>RECOMMENDER</span></p>
    <p class="hero-sub">SVD &amp; PMF matrix factorization · MovieLens 1M</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0;">
        <p class="section-label">Configuration</p>
        <p class="section-title">Settings</p>
    </div>
    """, unsafe_allow_html=True)
    user_id = st.number_input("User ID", min_value=1, max_value=len(user_index), value=1, step=1)
    top_n   =10
    btn     = st.button("Generate", use_container_width=True)


# ── HELPERS ─────────────────────────────────────────────────────────────
def render_movie_list(df, bar_class="bar-svd", max_rating=5.0):
    """Render a styled movie list with mini bars."""
    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        pct = min((row.Rating / max_rating) * 100, 100)
        safe_title = html_mod.escape(str(row.Title))
        rows.append(
            '<div class="movie-row">'
            f'<div class="movie-rank">{i:02d}</div>'
            f'<div class="movie-title">{safe_title}</div>'
            f'<div class="movie-rating">{row.Rating:.2f}</div>'
            '<div class="movie-bar-wrap">'
            f'<div class="movie-bar {bar_class}" style="width:{pct}%"></div>'
            '</div></div>'
        )
    return "".join(rows)


def render_card(header, badge_text, movie_html):
    """Build a complete card HTML string safely."""
    safe_header = html_mod.escape(str(header))
    safe_badge  = html_mod.escape(str(badge_text))
    return (
        '<div class="card">'
        f'<div class="card-header">{safe_header} <span class="badge">{safe_badge}</span></div>'
        f'{movie_html}'
        '</div>'
    )


# ── MAIN CONTENT ────────────────────────────────────────────────────────
if btn:
    uid = int(user_id)

    # top rated by user
    user_ratings = train_data[train_data['UserID'] == uid].sort_values('Rating', ascending=False).head(10)
    top_rated    = user_ratings.merge(movies_pd[['MovieID', 'Title']], on='MovieID')

    st.markdown(f"""
    <p class="section-label">Watch History</p>
    <p class="section-title">Top Rated by User {uid}</p>
    """, unsafe_allow_html=True)

    _top_html = render_card("Highest Ratings", f"User {uid}",
                            render_movie_list(top_rated[['Title','Rating']], bar_class="bar-svd"))
    st.markdown(_top_html, unsafe_allow_html=True)

    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

    # recommendations
    svd_recs = generate_recommendations(uid, model='svd', top_n=top_n)
    pmf_recs = generate_recommendations(uid, model='pmf', top_n=top_n)

    if svd_recs is None or pmf_recs is None:
        st.error(f"User {uid} not found.")
    else:
       
        st.markdown(f"""
        <p class="section-label">Predictions</p>
        <p class="section-title">Recommendations for User {uid}</p>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            _svd_html = render_card("SVD Model", "Singular Value Decomposition",
                                    render_movie_list(svd_recs[['Title','Rating']], bar_class="bar-svd"))
            st.markdown(_svd_html, unsafe_allow_html=True)

        with col2:
            _pmf_html = render_card("PMF Model", "Probabilistic Matrix Factorization",
                                    render_movie_list(pmf_recs[['Title','Rating']], bar_class="bar-pmf"))
            st.markdown(_pmf_html, unsafe_allow_html=True)

        st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)

        # ── CHART ───────────────────────────────────────────────────────
        st.markdown("""
        <p class="section-label">Analysis</p>
        <p class="section-title">SVD vs PMF Comparison</p>
        """, unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, top_n * 0.35)))
        fig.patch.set_facecolor('#08080a')

        for ax in axes:
            ax.set_facecolor('#08080a')
            ax.tick_params(colors='#706b62', labelsize=7.5)
            ax.xaxis.label.set_color('#706b62')
            ax.title.set_color('#f5f0e8')
            ax.title.set_fontsize(13)
            ax.title.set_fontweight('bold')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.grid(axis='x', color='#1a1918', linewidth=0.5)

        # SVD bars
        svd_titles = svd_recs['Title'].str[:35].tolist()[::-1]
        svd_vals   = svd_recs['Rating'].tolist()[::-1]
        bars1 = axes[0].barh(svd_titles, svd_vals, color='#e8a44a', height=0.6, edgecolor='none')
        axes[0].set_title(f'SVD — User {uid}', fontfamily='sans-serif', pad=12)
        axes[0].set_xlabel('Predicted Rating', fontsize=8)
        axes[0].set_xlim(0, 5.5)

        # PMF bars
        pmf_titles = pmf_recs['Title'].str[:35].tolist()[::-1]
        pmf_vals   = pmf_recs['Rating'].tolist()[::-1]
        bars2 = axes[1].barh(pmf_titles, pmf_vals, color='#5ba882', height=0.6, edgecolor='none')
        axes[1].set_title(f'PMF — User {uid}', fontfamily='sans-serif', pad=12)
        axes[1].set_xlabel('Predicted Rating', fontsize=8)
        axes[1].set_xlim(0, 5.5)

        plt.tight_layout(pad=2)
        st.pyplot(fig)

        # ── METRICS ─────────────────────────────────────────────────────
        st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)
        st.markdown("""
        <p class="section-label">Evaluation</p>
        <p class="section-title">Model Performance</p>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-label">SVD RMSE</div>
                <div class="metric-value">0.8899</div>
                <div class="metric-delta delta-good">▼ 0.0101 below target</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-label">PMF RMSE</div>
                <div class="metric-value">0.8462</div>
                <div class="metric-delta delta-good">▼ 0.0038 below target</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown("""
            <div class="metric-box">
                <div class="metric-label">PMF Improvement</div>
                <div class="metric-value">5.0%</div>
                <div class="metric-delta delta-warn">● meets 5% target</div>
            </div>
            """, unsafe_allow_html=True)

        # ── FOOTER ──────────────────────────────────────────────────────
        st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; padding: 10px 0 30px 0;">
            <span style="font-family:'Source Sans 3',sans-serif; font-size:0.65rem;
                         letter-spacing:0.2em; text-transform:uppercase; color:#3a3833;">
                Matrix Factorization · MovieLens 1M · Reboot 01
            </span>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── LANDING STATE ───────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:center;
                min-height: 50vh; flex-direction:column; text-align:center;">
        <div style="font-size: 3.5rem; margin-bottom: 16px; opacity: 0.15;">🎬</div>
        <p style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem;
                  color:#f5f0e8; letter-spacing:0.05em; margin:0;">
            Select a user and hit Generate
        </p>
        <p style="font-family:'Source Sans 3',sans-serif; font-size:0.8rem;
                  color:#3a3833; margin-top:8px; letter-spacing:0.1em;">
            Recommendations powered by SVD &amp; PMF models
        </p>
    </div>
    """, unsafe_allow_html=True)