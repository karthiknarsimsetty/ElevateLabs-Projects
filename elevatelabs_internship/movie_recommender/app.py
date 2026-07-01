import streamlit as st
import pandas as pd
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from movies_data import get_movies_df

st.set_page_config(page_title="🎬 CineMatch AI", layout="wide", page_icon="🎬")

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64_image("bg.jpg")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
* {{ font-family: 'Poppins', sans-serif; }}
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
.stApp::before {{
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0,0,0,0.78);
    z-index: 0;
}}
header {{ visibility: hidden; }}
.hero {{
    text-align: center;
    padding: 40px 20px 20px;
    background: linear-gradient(180deg, rgba(99,102,241,0.15) 0%, transparent 100%);
    border-radius: 20px;
    margin-bottom: 30px;
    position: relative;
    z-index: 1;
}}
.hero h1 {{
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #f5c518, #ff6b6b, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}}
.hero p {{ color: #d1d5db; font-size: 16px; margin: 0; }}
.selected-card {{
    background: linear-gradient(135deg, rgba(245,197,24,0.15), rgba(168,85,247,0.15));
    border: 1px solid rgba(245,197,24,0.4);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    position: relative;
    z-index: 1;
}}
.selected-card h3 {{ color: #f5c518; font-size: 18px; margin-bottom: 6px; }}
.selected-card p  {{ color: #9ca3af; font-size: 13px; margin: 3px 0; }}
.big-rating {{ font-size: 36px; font-weight: 800; color: #f5c518; margin: 10px 0 4px; }}
.rec-card {{
    background: rgba(0,0,0,0.60);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 18px 20px;
    margin: 10px 0;
    position: relative;
    overflow: hidden;
    z-index: 1;
}}
.rec-card::before {{
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #f5c518, #a855f7);
    border-radius: 4px 0 0 4px;
}}
.rec-card h4 {{ color: #f1f5f9; font-size: 16px; margin: 0 0 6px; font-weight: 700; }}
.genre-tag {{
    display: inline-block;
    background: rgba(168,85,247,0.25);
    color: #c4b5fd;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
}}
.lang-tag {{
    display: inline-block;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
    margin: 2px;
}}
.lang-hindi   {{ background: rgba(255,107,107,0.25); color: #fca5a5; }}
.lang-telugu  {{ background: rgba(251,191,36,0.25);  color: #fcd34d; }}
.lang-english {{ background: rgba(59,130,246,0.25);  color: #93c5fd; }}
.match-row {{ display: flex; align-items: center; gap: 10px; margin-top: 10px; }}
.match-bar-bg {{
    flex: 1; background: rgba(255,255,255,0.1);
    border-radius: 20px; height: 8px; overflow: hidden;
}}
.match-bar-fill {{
    height: 100%; border-radius: 20px;
    background: linear-gradient(90deg, #f5c518, #a855f7);
}}
.match-pct {{ color: #f5c518; font-size: 13px; font-weight: 700; min-width: 45px; }}
.rating-badge {{
    background: rgba(245,197,24,0.2); color: #f5c518;
    border-radius: 8px; padding: 2px 10px;
    font-size: 13px; font-weight: 700;
}}
.medal {{ font-size: 22px; position: absolute; top: 14px; right: 16px; }}
.section-title {{
    color: #f1f5f9; font-size: 22px; font-weight: 700;
    margin: 30px 0 16px; position: relative; z-index: 1;
}}
.section-title span {{
    background: linear-gradient(90deg, #f5c518, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.stat-box {{
    background: rgba(0,0,0,0.55);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 16px;
    text-align: center; position: relative; z-index: 1;
}}
.stat-box h2 {{ color: #f5c518; font-size: 28px; font-weight: 800; margin: 0; }}
.stat-box p  {{ color: #9ca3af; font-size: 12px; margin: 4px 0 0; font-weight: 600; }}
.divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    margin: 30px 0; position: relative; z-index: 1;
}}
.filter-box {{
    background: rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    position: relative;
    z-index: 1;
}}
.filter-box h4 {{
    color: #f5c518;
    font-size: 14px;
    font-weight: 700;
    margin: 0 0 14px 0;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
.result-count {{
    background: rgba(245,197,24,0.1);
    border: 1px solid rgba(245,197,24,0.3);
    border-radius: 10px;
    padding: 10px 16px;
    color: #f5c518;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 16px;
    position: relative;
    z-index: 1;
}}
.stSelectbox label, .stSlider label, .stMultiSelect label, .stTextInput label {{
    color: #d1d5db !important; font-size: 13px !important; font-weight: 600 !important;
}}
div[data-baseweb="select"] > div {{
    background: rgba(0,0,0,0.5) !important;
    border-color: rgba(255,255,255,0.15) !important;
    border-radius: 12px !important; color: #f1f5f9 !important;
}}
.stTextInput > div > input {{
    background: rgba(0,0,0,0.5) !important;
    border-color: rgba(255,255,255,0.15) !important;
    border-radius: 12px !important; color: #f1f5f9 !important;
}}
.stButton > button {{
    background: linear-gradient(90deg, #f5c518, #f59e0b) !important;
    color: #000 !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 12px 32px !important; width: 100% !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(245,197,24,0.4) !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return get_movies_df()

@st.cache_data
def build_similarity(df):
    df = df.copy()
    df["combined"] = df["genres"] + " " + df["description"] + " " + df["language"]
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["combined"])
    return cosine_similarity(matrix)

movies     = load_data()
sim_matrix = build_similarity(movies)

def recommend(title, n=6):
    idx    = movies[movies["title"] == title].index[0]
    scores = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    result = movies.iloc[[i[0] for i in scores]].copy()
    result["match"] = [round(s[1] * 100, 1) for s in scores]
    return result, scores


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🎬 CineMatch AI</h1>
    <p>Discover your next favourite movie — Hollywood · Bollywood · Tollywood</p>
</div>
""", unsafe_allow_html=True)

# Stats
total         = len(movies)
english_count = len(movies[movies["language"] == "English"])
hindi_count   = len(movies[movies["language"] == "Hindi"])
telugu_count  = len(movies[movies["language"] == "Telugu"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='stat-box'><h2>{total}</h2><p>🎬 Total Movies</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='stat-box'><h2>{english_count}</h2><p>🇺🇸 English</p></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='stat-box'><h2>{hindi_count}</h2><p>🇮🇳 Hindi</p></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='stat-box'><h2>{telugu_count}</h2><p>🎭 Telugu</p></div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ── SECTION 1: FIND SIMILAR MOVIES ───────────────────────────────────────────
st.markdown("<div class='section-title'>🎯 <span>Find Similar Movies</span></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2])

with col_left:
    lang_filter = st.multiselect("🌐 Filter by Language:", ["English", "Hindi", "Telugu"], default=[])
    filtered    = movies[movies["language"].isin(lang_filter)] if lang_filter else movies
    movie_list  = ["-- Select a movie --"] + sorted(filtered["title"].tolist())
    selected    = st.selectbox("🎥 Pick a movie you like:", movie_list)
    n_recs      = st.slider("Number of Recommendations:", 3, 10, 6)

with col_right:
    if selected and selected != "-- Select a movie --":
        info       = movies[movies["title"] == selected].iloc[0]
        lang_color = {"Hindi": "#fca5a5", "Telugu": "#fcd34d", "English": "#93c5fd"}
        lc         = lang_color.get(info["language"], "#d1d5db")
        st.markdown(f"""
        <div class='selected-card'>
            <p style='color:{lc}; font-weight:700; font-size:12px; letter-spacing:2px;'>{info['language'].upper()}</p>
            <h3>{info['title']}</h3>
            <div class='big-rating'>⭐ {info['rating']}</div>
            <p>{info['genres']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='selected-card'>
            <p style='color:#9ca3af; font-size:14px; margin-top:20px;'>
                👈 Select a movie to see its details
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Find Similar Movies"):
    if selected == "-- Select a movie --":
        st.warning("⚠️ Please select a movie first!")
    else:
        recs, scores = recommend(selected, n_recs)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>🍿 <span>Because you liked — {selected}</span></div>", unsafe_allow_html=True)

        medals     = ["🥇", "🥈", "🥉"]
        lang_class = {"Hindi": "lang-hindi", "Telugu": "lang-telugu", "English": "lang-english"}

        for i, (_, row) in enumerate(recs.iterrows()):
            medal      = medals[i] if i < 3 else f"#{i+1}"
            lclass     = lang_class.get(row["language"], "lang-english")
            genre_tags = "".join([f"<span class='genre-tag'>{g.strip()}</span>" for g in row["genres"].split()[:3]])
            bar_w      = min(row["match"], 100)
            st.markdown(f"""
            <div class='rec-card'>
                <span class='medal'>{medal}</span>
                <h4>{row['title']}</h4>
                <span class='lang-tag {lclass}'>{row['language']}</span>
                {genre_tags}
                <span class='rating-badge' style='float:right; margin-top:-2px;'>⭐ {row['rating']}</span>
                <div class='match-row'>
                    <div class='match-bar-bg'>
                        <div class='match-bar-fill' style='width:{bar_w}%;'></div>
                    </div>
                    <span class='match-pct'>{row['match']}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── SECTION 2: EXPLORE WITH ADVANCED FILTERS ─────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📽️ <span>Explore All Movies</span></div>", unsafe_allow_html=True)

# All unique genres from dataset
all_genres = sorted(set(
    g.strip()
    for genres in movies["genres"]
    for g in genres.split()
    if len(g.strip()) > 2
))

st.markdown("<div class='filter-box'><h4>🔍 Search & Filter</h4>", unsafe_allow_html=True)

# Row 1: Search box
search = st.text_input("🔎 Search by title or keyword:", placeholder="e.g. Dangal, RRR, Spider-Man...")

# Row 2: Language + Genre filters
f1, f2 = st.columns(2)
with f1:
    genre_filter = st.multiselect(
        "🎭 Filter by Genre:",
        ["Action", "Romance", "Comedy", "Drama", "Thriller",
         "Horror", "Sci-Fi", "Biography", "Sport", "Crime",
         "Fantasy", "Adventure", "Mystery", "Animation", "History"],
        default=[]
    )
with f2:
    lang_explore = st.multiselect(
        "🌐 Filter by Language:",
        ["English", "Hindi", "Telugu"],
        default=[],
        key="explore_lang"
    )

# Row 3: Rating + Sort
f3, f4 = st.columns(2)
with f3:
    min_rating = st.slider("⭐ Minimum Rating:", 0.0, 10.0, 0.0, 0.1)
with f4:
    sort_by = st.selectbox(
        "📊 Sort By:",
        ["Rating (High to Low)", "Rating (Low to High)", "Title (A-Z)", "Title (Z-A)"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── Apply all filters ─────────────────────────────────────────────────────────
explore = movies.copy()

# Language filter
if lang_explore:
    explore = explore[explore["language"].isin(lang_explore)]

# Genre filter
if genre_filter:
    explore = explore[
        explore["genres"].apply(
            lambda g: any(genre.lower() in g.lower() for genre in genre_filter)
        )
    ]

# Search filter
if search:
    explore = explore[
        explore["title"].str.contains(search, case=False) |
        explore["description"].str.contains(search, case=False)
    ]

# Minimum rating filter
if min_rating > 0:
    explore = explore[explore["rating"] >= min_rating]

# Sort
if sort_by == "Rating (High to Low)":
    explore = explore.sort_values("rating", ascending=False)
elif sort_by == "Rating (Low to High)":
    explore = explore.sort_values("rating", ascending=True)
elif sort_by == "Title (A-Z)":
    explore = explore.sort_values("title", ascending=True)
elif sort_by == "Title (Z-A)":
    explore = explore.sort_values("title", ascending=False)

explore = explore.reset_index(drop=True)
explore.index += 1

# Result count
st.markdown(f"<div class='result-count'>🎬 Found <strong>{len(explore)}</strong> movies matching your filters</div>", unsafe_allow_html=True)

# Show table
st.dataframe(
    explore[["title", "language", "genres", "rating"]],
    use_container_width=True
)

st.caption(f"CineMatch AI · ElevateLabs Internship · {total} movies total")