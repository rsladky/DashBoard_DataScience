import streamlit as st
from pathlib import Path
from streamlit.components.v1 import html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="DashBoard — Présentation", layout="wide")
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "dashboard" / "data" / "raw"
NB_DIR = ROOT / "dashboard" / "notebooks"

st.title("🎛️ Dashboard Data Science — Présentation")

@st.cache_data
def read_csv_robust(path: Path, nrows=None):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, nrows=nrows, encoding=enc, engine="python")
        except Exception:
            continue
    return pd.read_csv(path, nrows=nrows, engine="python", encoding_errors="replace")

# Charger données neige
SNOW_CSV = DATA_DIR / "data_de_aus_ch.csv"
df_snow = read_csv_robust(SNOW_CSV)
df_snow.columns = [c.strip() for c in df_snow.columns]

# Conversions utiles
num_cols = ["Elevation (m)", "Temperature (°C)", "Precipitation (mm)", "Daily Snow Depth (cm)",
            "Monthly Snow Depth (cm)", "Mean Snow Depth (cm)", "Max Snow Depth (cm)", "Days where AVG Temp < 0C"]
for c in num_cols:
    if c in df_snow.columns:
        df_snow[c] = pd.to_numeric(df_snow[c], errors="coerce")
df_snow["Year"] = pd.to_numeric(df_snow.get("Year"), errors="coerce")
df_snow["Month"] = pd.to_numeric(df_snow.get("Month"), errors="coerce")

# Navigation (sections)
page = st.sidebar.radio(
    "Sections",
    ("Introduction", "Carte des stations", "Neige et froid (bar charts)", "Aperçu des données", "À propos"),
)

if page == "Introduction":
    st.subheader("Objectifs")
    st.markdown(
        "- Visualiser la carte des stations de ski.\n"
        "- Comparer les stations sur la hauteur moyenne de neige et les jours froids.\n"
        "- Parcourir rapidement les données sources."
    )
    st.subheader("Contenus")
    st.markdown(
        "- Carte Folium intégrée.\n"
        "- Deux diagrammes en barres: Mean Snow Depth (cm) et Days where AVG Temp < 0C.\n"
        "- Tables d’aperçu des CSV."
    )

elif page == "Carte des stations":
    st.subheader("Carte des stations (Folium)")
    map_html = NB_DIR / "ski_resorts_points_map.html"
    if map_html.exists():
        body = map_html.read_text(encoding="utf-8")
        html(body, height=800, scrolling=True)
    else:
        st.warning("Carte HTML introuvable. Ouvrez votre notebook MapResort.ipynb et générez ski_resorts_points_map.html.")

elif page == "Neige et froid (bar charts)":
    st.subheader("Comparaison par station")
    if not {"Region", "Mean Snow Depth (cm)", "Days where AVG Temp < 0C"}.issubset(df_snow.columns):
        st.error("Colonnes requises absentes dans data_de_aus_ch.csv.")
    else:
        # Filtre pays
        countries = sorted(df_snow["Country"].dropna().unique().tolist())
        country = st.selectbox("Filtrer par pays (optionnel)", ["Tous"] + countries)
        df_plot = df_snow.copy()
        if country != "Tous":
            df_plot = df_plot[df_plot["Country"] == country]

        # Agrégation par station
        agg = df_plot.groupby("Region", as_index=False).agg(
            mean_snow=("Mean Snow Depth (cm)", "mean"),
            mean_cold_days=("Days where AVG Temp < 0C", "mean"),
            obs_count=("Year", "count"),
        )
        # Filtrer stations avec assez d'observations
        min_obs = st.slider("Min observations par station", 1, 24, 5)
        agg = agg[agg["obs_count"] >= min_obs]

        # Top N par hauteur de neige
        top_n = st.slider("Nombre de stations à afficher", 5, 50, 20)
        top = agg.sort_values("mean_snow", ascending=False).head(top_n)

        col1, col2 = st.columns(2)

        # Bar chart Mean Snow Depth
        with col1:
            st.caption("Mean Snow Depth (cm)")
            fig1, ax1 = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
            sns.barplot(x="mean_snow", y="Region", data=top, ax=ax1, palette="Blues_r")
            ax1.set_xlabel("cm")
            ax1.set_ylabel("Region")
            ax1.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig1)

        # Bar chart Days where AVG Temp < 0C pour mêmes stations
        with col2:
            st.caption("Days where AVG Temp < 0C (moyenne)")
            top_cold = agg.set_index("Region").loc[top["Region"]].reset_index()
            fig2, ax2 = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
            sns.barplot(x="mean_cold_days", y="Region", data=top_cold, ax=ax2, palette="coolwarm")
            ax2.set_xlabel("jours")
            ax2.set_ylabel("")
            ax2.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)

        # Petites métriques synthétiques
        st.markdown("---")
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Stations affichées", len(top))
        with colB:
            st.metric("Neige moyenne (cm)", f"{top['mean_snow'].mean():.1f}" if not top.empty else "N/A")
        with colC:
            st.metric("Jours froids (moy.)", f"{top_cold['mean_cold_days'].mean():.1f}" if not top.empty else "N/A")

elif page == "Aperçu des données":
    st.subheader("CSV — aperçu (10 lignes)")
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        st.warning(f"Aucun CSV trouvé dans {DATA_DIR}")
    for p in csvs:
        st.markdown(f"#### {p.name}")
        df_head = read_csv_robust(p, nrows=10)
        st.dataframe(df_head, use_container_width=True)

elif page == "À propos":
    st.subheader("Notes")
    st.markdown(
        "- Cette interface n’affiche aucun code des notebooks, uniquement les visualisations.\n"
        "- La carte Folium est chargée depuis le HTML généré par MapResort.ipynb."
    )
    st.markdown("Commandes pour lancer:")
    st.code("pip install streamlit seaborn matplotlib pandas\nstreamlit run streamlit_app.py")
