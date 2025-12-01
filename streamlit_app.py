import streamlit as st
from pathlib import Path
from streamlit.components.v1 import html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# Charger données stations de ski
SKI_CSV = DATA_DIR / "ski-resorts.csv"
df_resorts = read_csv_robust(SKI_CSV)
df_resorts.columns = [c.strip() for c in df_resorts.columns]

# Conversions utiles pour data_de_aus_ch
num_cols = ["Elevation (m)", "Temperature (°C)", "Precipitation (mm)", "Daily Snow Depth (cm)",
            "Monthly Snow Depth (cm)", "Mean Snow Depth (cm)", "Max Snow Depth (cm)", "Days where AVG Temp < 0C"]
for c in num_cols:
    if c in df_snow.columns:
        df_snow[c] = pd.to_numeric(df_snow[c], errors="coerce")
df_snow["Year"] = pd.to_numeric(df_snow.get("Year"), errors="coerce")
df_snow["Month"] = pd.to_numeric(df_snow.get("Month"), errors="coerce")

# Conversions pour ski-resorts
resort_num_cols = ["elevation_top_m", "annual_snowfall_cm", "number_of_lifts", "number_of_slopes", "rank", "rating"]
for c in resort_num_cols:
    if c in df_resorts.columns:
        df_resorts[c] = pd.to_numeric(df_resorts[c], errors="coerce")

# Navigation (sections)
page = st.sidebar.radio(
    "Sections",
    ("Introduction", "Aperçu des données", "Graphs", "Carte des stations"),
)

if page == "Introduction":
    st.subheader("🎿 Problématique")
    st.markdown(
        "- Comment les stations de ski évoluent avec le dérèglement climatique ?\n"
        "- Quels facteurs influencent le succès d'une station de ski ?"
    )
    st.subheader("🎯 Objectifs")
    st.markdown(
        "- Présentation rapide des données sources.\n"
        "- Analyser la relation entre élévation, chute de neige et infrastructures.\n"
        "- Comparer les stations bien classées vs mal classées.\n"
        "- Visualiser la répartition géographique des stations."
    )
    st.subheader("📊 Contenus")
    st.markdown(
        "- Box plots de présentation des datasets.\n"
        "- Scatter plots: élévation vs neige, infrastructures vs conditions.\n"
        "- Comparaison stations top vs bottom rank.\n"
        "- Carte Folium intégrée.\n"
        "- Diagrammes en barres par station."
    )

elif page == "Aperçu des données":
    st.subheader("CSV — aperçu (10 lignes)")
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        st.warning(f"Aucun CSV trouvé dans {DATA_DIR}")
    for p in csvs:
        st.markdown(f"#### {p.name}")
        df_head = read_csv_robust(p, nrows=10)
        st.dataframe(df_head, use_container_width=True)

elif page == "Graphs":
    st.subheader("📊 Analyses graphiques")
    
    # 1. Box plots présentation datasets
    st.markdown("### 1. Présentation des datasets (Box Plots)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Ski Resorts - Variables clés")
        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
        
        vars_to_plot = ["elevation_top_m", "annual_snowfall_cm", "number_of_lifts", "number_of_slopes"]
        titles = ["Élévation sommet (m)", "Chute neige annuelle (cm)", "Nombre de remontées", "Nombre de pistes"]
        
        for idx, (var, title) in enumerate(zip(vars_to_plot, titles)):
            ax = axes1[idx // 2, idx % 2]
            if var in df_resorts.columns:
                data = df_resorts[var].dropna()
                ax.boxplot(data, vert=True)
                ax.set_title(title)
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{var}\nnon disponible', ha='center', va='center')
        
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        st.caption("Data DE/AUS/CH - Variables clés")
        fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
        
        vars_snow = ["Mean Snow Depth (cm)", "Days where AVG Temp < 0C", "Temperature (°C)", "Precipitation (mm)"]
        titles_snow = ["Hauteur moy. neige (cm)", "Jours < 0°C", "Température (°C)", "Précipitations (mm)"]
        
        for idx, (var, title) in enumerate(zip(vars_snow, titles_snow)):
            ax = axes2[idx // 2, idx % 2]
            if var in df_snow.columns:
                data = df_snow[var].dropna()
                if len(data) > 0:
                    ax.boxplot(data, vert=True)
                    ax.set_title(title)
                    ax.grid(axis='y', alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'Pas de données\npour {var}', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, f'{var}\nnon disponible', ha='center', va='center')
        
        plt.tight_layout()
        st.pyplot(fig2)
    
    st.markdown("---")
    
    # 2. Scatter plot: Élévation vs Annual Snowfall
    st.markdown("### 2. Plus haut = Plus de neige ?")
    st.caption("Relation entre élévation au sommet et chute de neige annuelle")
    
    df_scatter1 = df_resorts.dropna(subset=["elevation_top_m", "annual_snowfall_cm"])
    
    if not df_scatter1.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        
        scatter = ax3.scatter(df_scatter1["elevation_top_m"], 
                             df_scatter1["annual_snowfall_cm"],
                             alpha=0.5, s=80, c=df_scatter1["annual_snowfall_cm"], 
                             cmap="YlGnBu")
        
        # Ligne de tendance
        z = np.polyfit(df_scatter1["elevation_top_m"], df_scatter1["annual_snowfall_cm"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_scatter1["elevation_top_m"].min(), df_scatter1["elevation_top_m"].max(), 100)
        ax3.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
                label=f"Tendance: y={z[0]:.3f}x+{z[1]:.1f}")
        
        ax3.set_xlabel("Élévation au sommet (m)", fontsize=12)
        ax3.set_ylabel("Chute de neige annuelle (cm)", fontsize=12)
        ax3.set_title("Impact de l'altitude sur l'enneigement", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        plt.colorbar(scatter, ax=ax3, label="Neige (cm)")
        plt.tight_layout()
        st.pyplot(fig3)
        
        corr = df_scatter1[["elevation_top_m", "annual_snowfall_cm"]].corr().iloc[0, 1]
        st.info(f"📈 Corrélation: **{corr:.3f}** — {'Forte' if abs(corr) > 0.7 else 'Modérée' if abs(corr) > 0.4 else 'Faible'} corrélation positive" if corr > 0 else "Corrélation négative")
    else:
        st.warning("Données insuffisantes pour ce graphique.")
    
    st.markdown("---")
    
    # 3. Infrastructure vs Conditions (neige + élévation)
    st.markdown("### 3. Bonne neige = Plus d'infrastructures ?")
    st.caption("Relation entre conditions (neige + altitude) et développement de la station")
    
    df_scatter2 = df_resorts.dropna(subset=["annual_snowfall_cm", "elevation_top_m", "number_of_lifts", "number_of_slopes"])
    
    if not df_scatter2.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            scatter4 = ax4.scatter(df_scatter2["elevation_top_m"], 
                                  df_scatter2["number_of_lifts"],
                                  alpha=0.6, s=100, 
                                  c=df_scatter2["annual_snowfall_cm"], cmap="Blues")
            
            # Ligne de tendance
            z4 = np.polyfit(df_scatter2["elevation_top_m"], df_scatter2["number_of_lifts"], 1)
            p4 = np.poly1d(z4)
            x_line4 = np.linspace(df_scatter2["elevation_top_m"].min(), 
                                 df_scatter2["elevation_top_m"].max(), 100)
            ax4.plot(x_line4, p4(x_line4), "r--", alpha=0.8, linewidth=2, 
                    label=f"Tendance: y={z4[0]:.3f}x+{z4[1]:.1f}")
            
            ax4.set_xlabel("Élévation sommet (m)")
            ax4.set_ylabel("Nombre de remontées")
            ax4.set_title("Altitude vs Remontées\n(couleur = neige)")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            cbar4 = plt.colorbar(scatter4, ax=ax4, label="Neige (cm)")
            plt.tight_layout()
            st.pyplot(fig4)
        
        with col2:
            fig5, ax5 = plt.subplots(figsize=(8, 6))
            scatter5 = ax5.scatter(df_scatter2["elevation_top_m"], 
                                  df_scatter2["number_of_slopes"],
                                  alpha=0.6, s=100, 
                                  c=df_scatter2["annual_snowfall_cm"], cmap="Blues")
            
            # Ligne de tendance
            z5 = np.polyfit(df_scatter2["elevation_top_m"], df_scatter2["number_of_slopes"], 1)
            p5 = np.poly1d(z5)
            x_line5 = np.linspace(df_scatter2["elevation_top_m"].min(), 
                                 df_scatter2["elevation_top_m"].max(), 100)
            ax5.plot(x_line5, p5(x_line5), "r--", alpha=0.8, linewidth=2, 
                    label=f"Tendance: y={z5[0]:.3f}x+{z5[1]:.1f}")
            
            ax5.set_xlabel("Élévation sommet (m)")
            ax5.set_ylabel("Nombre de pistes")
            ax5.set_title("Altitude vs Pistes\n(couleur = neige)")
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            cbar5 = plt.colorbar(scatter5, ax=ax5, label="Neige (cm)")
            plt.tight_layout()
            st.pyplot(fig5)
        
        # Afficher les statistiques pour vérifier les données
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.info(f"📊 Élévation: min={df_scatter2['elevation_top_m'].min():.0f}m, max={df_scatter2['elevation_top_m'].max():.0f}m, moyenne={df_scatter2['elevation_top_m'].mean():.0f}m")
        with col_stat2:
            st.info(f"❄️ Neige: min={df_scatter2['annual_snowfall_cm'].min():.0f}cm, max={df_scatter2['annual_snowfall_cm'].max():.0f}cm, moyenne={df_scatter2['annual_snowfall_cm'].mean():.0f}cm")
        
        st.success("💡 **Insight**: Les stations avec de bonnes conditions (altitude + neige) tendent à avoir plus d'infrastructures, indiquant des investissements plus importants et potentiellement plus de revenus.")
    else:
        st.warning("Données insuffisantes pour ce graphique.")
    
    st.markdown("---")
    
    # 4. Comparaison Top vs Bottom rank
    st.markdown("### 4. Impact du dérèglement climatique: Top vs Bottom stations")
    
    df_ranked = df_resorts.dropna(subset=["rank", "annual_snowfall_cm", "elevation_top_m"])
    
    if not df_ranked.empty and len(df_ranked) >= 20:
        top_n = st.slider("Nombre de stations top/bottom à comparer", 10, 50, 20)
        
        top_stations = df_ranked.nsmallest(top_n, "rank")  # rank 1 = meilleur
        bottom_stations = df_ranked.nlargest(top_n, "rank")
        
        fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6))
        
        # Comparaison neige
        ax6_0 = axes6[0]
        data_snow = [top_stations["annual_snowfall_cm"].dropna(), 
                     bottom_stations["annual_snowfall_cm"].dropna()]
        bp1 = ax6_0.boxplot(data_snow, labels=[f"Top {top_n}", f"Bottom {top_n}"], 
                            patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightgreen')
        bp1['boxes'][1].set_facecolor('lightcoral')
        ax6_0.set_ylabel("Chute de neige annuelle (cm)")
        ax6_0.set_title("Comparaison enneigement")
        ax6_0.grid(axis='y', alpha=0.3)
        
        # Comparaison altitude
        ax6_1 = axes6[1]
        data_elev = [top_stations["elevation_top_m"].dropna(), 
                     bottom_stations["elevation_top_m"].dropna()]
        bp2 = ax6_1.boxplot(data_elev, labels=[f"Top {top_n}", f"Bottom {top_n}"], 
                            patch_artist=True)
        bp2['boxes'][0].set_facecolor('lightgreen')
        bp2['boxes'][1].set_facecolor('lightcoral')
        ax6_1.set_ylabel("Élévation sommet (m)")
        ax6_1.set_title("Comparaison altitude")
        ax6_1.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig6)
        
        # Métriques comparatives
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Top - Neige moy.", f"{top_stations['annual_snowfall_cm'].mean():.0f} cm")
        with col2:
            st.metric("Bottom - Neige moy.", f"{bottom_stations['annual_snowfall_cm'].mean():.0f} cm")
        with col3:
            st.metric("Top - Alt. moy.", f"{top_stations['elevation_top_m'].mean():.0f} m")
        with col4:
            st.metric("Bottom - Alt. moy.", f"{bottom_stations['elevation_top_m'].mean():.0f} m")
        
        st.warning("⚠️ **Observation**: Les stations mieux classées ont tendance à avoir plus de neige et/ou une altitude plus élevée, facteurs de résilience face au réchauffement climatique.")
    else:
        st.warning("Données de ranking insuffisantes.")
    
    st.markdown("---")
    
    # 5. Comparaison Oberstdorf_Aistaig vs Zermatt
    st.markdown("### 5. Évolution climatique: Oberstdorf_Aistaig (Allemagne) vs Zermatt (Suisse)")
    st.caption("Tendances de l'enneigement et des jours de gel par année")
    
    # Filtrer les données pour les deux stations
    oberstdorf_data = df_snow[df_snow['Region'].str.contains('Oberstdorf', case=False, na=False)]
    zermatt_data = df_snow[df_snow['Region'].str.contains('Zermatt', case=False, na=False)]
    
    # Filtrer les lignes avec des valeurs > 0 pour Mean Snow Depth et Days where AVG Temp < 0C
    oberstdorf_data = oberstdorf_data[
        (oberstdorf_data['Mean Snow Depth (cm)'] > 0) | 
        (oberstdorf_data['Days where AVG Temp < 0C'] > 0)
    ]
    zermatt_data = zermatt_data[
        (zermatt_data['Mean Snow Depth (cm)'] > 0) | 
        (zermatt_data['Days where AVG Temp < 0C'] > 0)
    ]
    
    if not oberstdorf_data.empty and not zermatt_data.empty:
        # Agréger par année en excluant les valeurs 0
        oberstdorf_summary = oberstdorf_data[oberstdorf_data['Mean Snow Depth (cm)'] > 0].groupby('Year').agg({
            'Mean Snow Depth (cm)': 'mean'
        }).reset_index()
        
        oberstdorf_temp_summary = oberstdorf_data[oberstdorf_data['Days where AVG Temp < 0C'] > 0].groupby('Year').agg({
            'Days where AVG Temp < 0C': 'sum'
        }).reset_index()
        
        # Fusionner les deux dataframes
        oberstdorf_summary = oberstdorf_summary.merge(oberstdorf_temp_summary, on='Year', how='outer')
        
        zermatt_summary = zermatt_data[zermatt_data['Mean Snow Depth (cm)'] > 0].groupby('Year').agg({
            'Mean Snow Depth (cm)': 'mean'
        }).reset_index()
        
        zermatt_temp_summary = zermatt_data[zermatt_data['Days where AVG Temp < 0C'] > 0].groupby('Year').agg({
            'Days where AVG Temp < 0C': 'sum'
        }).reset_index()
        
        # Fusionner les deux dataframes
        zermatt_summary = zermatt_summary.merge(zermatt_temp_summary, on='Year', how='outer')
        
        # Créer les graphiques
        fig7, axes7 = plt.subplots(1, 2, figsize=(16, 6))
        
        # Graphique 1: Profondeur moyenne de la neige
        ax7_0 = axes7[0]
        
        # Filtrer les NaN pour le tracé
        oberstdorf_snow_plot = oberstdorf_summary.dropna(subset=['Mean Snow Depth (cm)'])
        zermatt_snow_plot = zermatt_summary.dropna(subset=['Mean Snow Depth (cm)'])
        
        if not oberstdorf_snow_plot.empty:
            ax7_0.plot(oberstdorf_snow_plot['Year'], oberstdorf_snow_plot['Mean Snow Depth (cm)'], 
                       marker='o', linewidth=2, label='Oberstdorf_Aistaig', color='#2E86AB')
            
            # Ajouter la ligne de tendance
            if len(oberstdorf_snow_plot) > 1:
                z_ob = np.polyfit(oberstdorf_snow_plot['Year'], oberstdorf_snow_plot['Mean Snow Depth (cm)'], 1)
                p_ob = np.poly1d(z_ob)
                ax7_0.plot(oberstdorf_snow_plot['Year'], p_ob(oberstdorf_snow_plot['Year']), 
                          '--', alpha=0.6, color='#2E86AB', linewidth=1.5)
        
        if not zermatt_snow_plot.empty:
            ax7_0.plot(zermatt_snow_plot['Year'], zermatt_snow_plot['Mean Snow Depth (cm)'], 
                       marker='s', linewidth=2, label='Zermatt', color='#A23B72')
            
            # Ajouter la ligne de tendance
            if len(zermatt_snow_plot) > 1:
                z_ze = np.polyfit(zermatt_snow_plot['Year'], zermatt_snow_plot['Mean Snow Depth (cm)'], 1)
                p_ze = np.poly1d(z_ze)
                ax7_0.plot(zermatt_snow_plot['Year'], p_ze(zermatt_snow_plot['Year']), 
                          '--', alpha=0.6, color='#A23B72', linewidth=1.5)
        
        ax7_0.set_xlabel('Année', fontsize=12)
        ax7_0.set_ylabel('Profondeur moyenne de neige (cm)', fontsize=12)
        ax7_0.set_title('Évolution de l\'enneigement', fontsize=14, fontweight='bold')
        ax7_0.legend(loc='best')
        ax7_0.grid(True, alpha=0.3)
        
        # Graphique 2: Jours où température < 0°C
        ax7_1 = axes7[1]
        
        # Filtrer les NaN pour le tracé
        oberstdorf_temp_plot = oberstdorf_summary.dropna(subset=['Days where AVG Temp < 0C'])
        zermatt_temp_plot = zermatt_summary.dropna(subset=['Days where AVG Temp < 0C'])
        
        if not oberstdorf_temp_plot.empty:
            ax7_1.plot(oberstdorf_temp_plot['Year'], oberstdorf_temp_plot['Days where AVG Temp < 0C'], 
                       marker='o', linewidth=2, label='Oberstdorf_Aistaig', color='#2E86AB')
            
            # Ajouter la ligne de tendance
            if len(oberstdorf_temp_plot) > 1:
                z_ob_temp = np.polyfit(oberstdorf_temp_plot['Year'], oberstdorf_temp_plot['Days where AVG Temp < 0C'], 1)
                p_ob_temp = np.poly1d(z_ob_temp)
                ax7_1.plot(oberstdorf_temp_plot['Year'], p_ob_temp(oberstdorf_temp_plot['Year']), 
                          '--', alpha=0.6, color='#2E86AB', linewidth=1.5)
        
        if not zermatt_temp_plot.empty:
            ax7_1.plot(zermatt_temp_plot['Year'], zermatt_temp_plot['Days where AVG Temp < 0C'], 
                       marker='s', linewidth=2, label='Zermatt', color='#A23B72')
            
            # Ajouter la ligne de tendance
            if len(zermatt_temp_plot) > 1:
                z_ze_temp = np.polyfit(zermatt_temp_plot['Year'], zermatt_temp_plot['Days where AVG Temp < 0C'], 1)
                p_ze_temp = np.poly1d(z_ze_temp)
                ax7_1.plot(zermatt_temp_plot['Year'], p_ze_temp(zermatt_temp_plot['Year']), 
                          '--', alpha=0.6, color='#A23B72', linewidth=1.5)
        
        ax7_1.set_xlabel('Année', fontsize=12)
        ax7_1.set_ylabel('Jours avec température < 0°C', fontsize=12)
        ax7_1.set_title('Évolution des jours de gel', fontsize=14, fontweight='bold')
        ax7_1.legend(loc='best')
        ax7_1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig7)
        
        # Statistiques comparatives (exclure les 0 pour les moyennes)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Oberstdorf - Neige moy.", 
                     f"{oberstdorf_snow_plot['Mean Snow Depth (cm)'].mean():.1f} cm" if not oberstdorf_snow_plot.empty else "N/A")
        with col2:
            st.metric("Zermatt - Neige moy.", 
                     f"{zermatt_snow_plot['Mean Snow Depth (cm)'].mean():.1f} cm" if not zermatt_snow_plot.empty else "N/A")
        with col3:
            st.metric("Oberstdorf - Jours gel moy.", 
                     f"{oberstdorf_temp_plot['Days where AVG Temp < 0C'].mean():.0f} jours/an" if not oberstdorf_temp_plot.empty else "N/A")
        with col4:
            st.metric("Zermatt - Jours gel moy.", 
                     f"{zermatt_temp_plot['Days where AVG Temp < 0C'].mean():.0f} jours/an" if not zermatt_temp_plot.empty else "N/A")
        
        # Analyse de tendance
        if not oberstdorf_snow_plot.empty and not zermatt_snow_plot.empty and len(oberstdorf_snow_plot) > 1 and len(zermatt_snow_plot) > 1:
            trend_ob_snow = "📉 Baisse" if z_ob[0] < 0 else "📈 Hausse"
            trend_ze_snow = "📉 Baisse" if z_ze[0] < 0 else "📈 Hausse"
            
            trend_info = f"""
            **Tendances observées (enneigement):**
            - **Oberstdorf** : {trend_ob_snow} de l'enneigement ({z_ob[0]:.2f} cm/an)
            - **Zermatt** : {trend_ze_snow} de l'enneigement ({z_ze[0]:.2f} cm/an)
            """
            
            if not oberstdorf_temp_plot.empty and not zermatt_temp_plot.empty and len(oberstdorf_temp_plot) > 1 and len(zermatt_temp_plot) > 1:
                trend_ob_temp = "📉 Baisse" if z_ob_temp[0] < 0 else "📈 Hausse"
                trend_ze_temp = "📉 Baisse" if z_ze_temp[0] < 0 else "📈 Hausse"
                trend_info += f"""
            
            **Tendances observées (jours de gel):**
            - **Oberstdorf** : {trend_ob_temp} des jours de gel ({z_ob_temp[0]:.2f} jours/an)
            - **Zermatt** : {trend_ze_temp} des jours de gel ({z_ze_temp[0]:.2f} jours/an)
            """
            
            st.info(trend_info)
        
        st.warning("⚠️ **Constat**: Cette comparaison illustre l'impact différencié du réchauffement climatique selon l'altitude et la localisation géographique des stations. Les valeurs nulles ont été exclues de l'analyse.")
    
    elif oberstdorf_data.empty:
        st.warning("⚠️ Aucune donnée trouvée pour Oberstdorf_Aistaig (après filtrage des valeurs à 0). Vérifiez le nom de la région dans le CSV.")
    elif zermatt_data.empty:
        st.warning("⚠️ Aucune donnée trouvée pour Zermatt (après filtrage des valeurs à 0). Vérifiez le nom de la région dans le CSV.")
    else:
        st.warning("⚠️ Données insuffisantes pour la comparaison Oberstdorf vs Zermatt (après filtrage des valeurs à 0).")

elif page == "Carte des stations":
    st.subheader("🗺️ Répartition géographique des stations")
    st.caption("Carte interactive - Taille des points: annual snowfall | Couleur: élévation")
    
    map_html = NB_DIR / "ski_resorts_points_map.html"
    if map_html.exists():
        body = map_html.read_text(encoding="utf-8")
        html(body, height=800, scrolling=True)
    else:
        st.warning("Carte HTML introuvable. Ouvrez votre notebook MapResort.ipynb et générez ski_resorts_points_map.html.")