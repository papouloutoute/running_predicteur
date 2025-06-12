#!/usr/bin/env python3
"""
Application Streamlit pour la prédiction de performances en course à pied
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.performance_predictor import RunningPerformancePredictor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="🏃‍♂️ Prédicteur de Performance Running",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_predictor():
    """Charge le prédicteur avec mise en cache"""
    predictor = RunningPerformancePredictor()
    if predictor.load_data():
        # Entraîner rapidement sur une distance pour avoir les encoders
        predictor.train_models('temps_5km')
        return predictor
    return None

@st.cache_data
def load_data():
    """Charge les données avec mise en cache"""
    try:
        df = pd.read_csv('data/processed/runners_profiles.csv')
        return df
    except:
        return None

def format_time(minutes):
    """Formate le temps en heures:minutes:secondes"""
    if pd.isna(minutes):
        return "N/A"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes % 1) * 60)
    
    if hours > 0:
        return f"{hours}h{mins:02d}m{secs:02d}s"
    else:
        return f"{mins}m{secs:02d}s"

def main():
    # En-tête principal
    st.markdown('<h1 class="main-header">🏃‍♂️ Prédicteur de Performance Running</h1>', unsafe_allow_html=True)
    
    # Chargement des données
    predictor = load_predictor()
    df = load_data()
    
    if predictor is None or df is None:
        st.error("❌ Impossible de charger les données. Assurez-vous que les fichiers sont présents.")
        return
    
    # Sidebar pour les paramètres
    st.sidebar.markdown("## 🎯 Paramètres du Coureur")
    
    # Informations personnelles
    st.sidebar.markdown("### 👤 Informations Personnelles")
    age = st.sidebar.slider("Âge", 18, 70, 30)
    poids = st.sidebar.slider("Poids (kg)", 45, 120, 70)
    sexe = st.sidebar.selectbox("Sexe", ["Homme", "Femme"])
    niveau = st.sidebar.selectbox("Niveau", ["Débutant", "Intermédiaire", "Avancé", "Expert"])
    
    # Entraînement
    st.sidebar.markdown("### 🏃‍♂️ Entraînement")
    km_semaine = st.sidebar.slider("Kilomètres par semaine", 5, 150, 40)
    seances_semaine = st.sidebar.slider("Séances par semaine", 1, 10, 4)
    type_parcours = st.sidebar.selectbox("Type de parcours préféré", ["Plat", "Vallonné", "Montagne"])
    
    # Données physiologiques
    st.sidebar.markdown("### ❤️ Données Physiologiques")
    fc_repos = st.sidebar.slider("Fréquence cardiaque au repos", 40, 100, 60)
    fc_moyenne = st.sidebar.slider("FC moyenne d'entraînement", 120, 200, 150)
    
    # Profil du coureur
    profil_coureur = {
        'age': age,
        'poids': poids,
        'sexe': sexe,
        'niveau': niveau,
        'km_semaine': km_semaine,
        'seances_semaine': seances_semaine,
        'type_parcours': type_parcours,
        'fc_repos': fc_repos,
        'fc_moyenne': fc_moyenne
    }
    
    # Layout principal avec colonnes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Prédictions de Performance")
        
        # Calculer les prédictions
        distances = ['temps_5km', 'temps_10km', 'temps_semi', 'temps_marathon']
        distance_names = ['5km', '10km', 'Semi-marathon', 'Marathon']
        
        predictions = {}
        for distance in distances:
            try:
                # Réentraîner rapidement le modèle si nécessaire
                if distance not in predictor.models:
                    predictor.train_models(distance)
                
                pred = predictor.predict_performance(profil_coureur, distance)
                predictions[distance] = pred
            except Exception as e:
                st.error(f"Erreur pour {distance}: {e}")
                predictions[distance] = None
        
        # Affichage des prédictions en cartes
        cols = st.columns(4)
        for i, (distance, name) in enumerate(zip(distances, distance_names)):
            with cols[i]:
                if predictions[distance] is not None:
                    time_str = format_time(predictions[distance])
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>{name}</h3>
                        <h2>{time_str}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Erreur {name}")
        
        # Graphique des prédictions
        if all(pred is not None for pred in predictions.values()):
            st.markdown("### 📊 Visualisation des Performances Prédites")
            
            # Créer le graphique
            fig = go.Figure()
            
            pred_values = [predictions[d] for d in distances]
            fig.add_trace(go.Bar(
                x=distance_names,
                y=pred_values,
                marker_color=['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5'],
                text=[format_time(p) for p in pred_values],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Temps Prédits par Distance",
                xaxis_title="Distance",
                yaxis_title="Temps (minutes)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("## 📊 Votre Profil")
        
        # Affichage du profil
        st.markdown("### 👤 Résumé")
        st.write(f"**Âge:** {age} ans")
        st.write(f"**Poids:** {poids} kg")
        st.write(f"**Sexe:** {sexe}")
        st.write(f"**Niveau:** {niveau}")
        
        st.markdown("### 🏃‍♂️ Entraînement")
        st.write(f"**Volume:** {km_semaine} km/semaine")
        st.write(f"**Fréquence:** {seances_semaine} séances/semaine")
        st.write(f"**Volume/séance:** {km_semaine/seances_semaine:.1f} km")
        st.write(f"**Terrain:** {type_parcours}")
        
        st.markdown("### ❤️ Physiologie")
        st.write(f"**FC repos:** {fc_repos} bpm")
        st.write(f"**FC entraînement:** {fc_moyenne} bpm")
        st.write(f"**Réserve FC:** {fc_moyenne - fc_repos} bpm")
        
        # Comparaison avec la population
        st.markdown("### 📈 Comparaison Population")
        
        # Statistiques de comparaison
        same_level = df[df['niveau'] == niveau]
        if len(same_level) > 0:
            avg_km = same_level['km_semaine'].mean()
            percentile_km = (df['km_semaine'] <= km_semaine).mean() * 100
            
            st.write(f"**Volume moyen {niveau}:** {avg_km:.1f} km/sem")
            st.write(f"**Votre percentile:** {percentile_km:.0f}%")
            
            if percentile_km > 75:
                st.success("🔥 Volume d'entraînement élevé!")
            elif percentile_km > 50:
                st.info("👍 Volume d'entraînement correct")
            else:
                st.warning("💪 Potentiel d'amélioration!")
    
    # Section d'analyse avancée
    st.markdown("---")
    st.markdown("## 🔍 Analyse Avancée")
    
    tab1, tab2, tab3 = st.tabs(["📊 Comparaisons", "🎯 Objectifs", "📈 Progression"])
    
    with tab1:
        st.markdown("### Comparaison avec des profils similaires")
        
        # Filtrer les coureurs similaires
        similar_runners = df[
            (df['niveau'] == niveau) & 
            (df['sexe'] == sexe) &
            (abs(df['age'] - age) <= 5)
        ]
        
        if len(similar_runners) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique de comparaison volume
                fig = px.histogram(similar_runners, x='km_semaine', 
                                 title=f"Distribution du volume - {niveau} {sexe}")
                fig.add_vline(x=km_semaine, line_dash="dash", line_color="red",
                             annotation_text="Votre volume")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Comparaison des temps
                if 'temps_5km' in similar_runners.columns:
                    avg_5km = similar_runners['temps_5km'].mean()
                    your_5km = predictions['temps_5km']
                    
                    comparison_data = pd.DataFrame({
                        'Catégorie': ['Moyenne groupe', 'Votre prédiction'],
                        'Temps_5km': [avg_5km, your_5km]
                    })
                    
                    fig = px.bar(comparison_data, x='Catégorie', y='Temps_5km',
                               title="Comparaison 5km")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas assez de coureurs similaires pour la comparaison")
    
    with tab2:
        st.markdown("### Définir vos objectifs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Objectif de temps")
            distance_obj = st.selectbox("Distance", distance_names)
            temps_objectif = st.number_input("Temps objectif (minutes)", 
                                           min_value=10.0, max_value=300.0, 
                                           value=30.0, step=0.5)
            
            distance_key = distances[distance_names.index(distance_obj)]
            temps_predit = predictions[distance_key]
            
            if temps_predit:
                diff = temps_objectif - temps_predit
                if diff < 0:
                    st.success(f"🎉 Objectif atteignable! Vous êtes {abs(diff):.1f} min plus rapide que prévu")
                else:
                    st.warning(f"💪 Défi à relever! Il faut améliorer de {diff:.1f} min")
        
        with col2:
            st.markdown("#### 📈 Recommandations")
            
            if km_semaine < 30:
                st.info("💡 Augmentez progressivement votre volume hebdomadaire")
            
            if seances_semaine < 3:
                st.info("💡 Ajoutez une séance d'entraînement par semaine")
            
            if fc_moyenne - fc_repos < 80:
                st.info("💡 Travaillez votre condition cardiovasculaire")
            
            st.info("💡 Variez vos types d'entraînement (fractionné, endurance, tempo)")
    
    with tab3:
        st.markdown("### Simulation de progression")
        
        st.markdown("#### 📊 Impact de l'augmentation du volume")
        
        # Simuler différents volumes
        volumes = range(max(5, km_semaine-20), min(150, km_semaine+40), 5)
        simulations = []
        
        for vol in volumes:
            profil_sim = profil_coureur.copy()
            profil_sim['km_semaine'] = vol
            
            try:
                pred_5km = predictor.predict_performance(profil_sim, 'temps_5km')
                simulations.append({'Volume': vol, 'Temps_5km': pred_5km})
            except:
                continue
        
        if simulations:
            sim_df = pd.DataFrame(simulations)
            
            fig = px.line(sim_df, x='Volume', y='Temps_5km',
                         title="Impact du volume sur le temps au 5km")
            fig.add_vline(x=km_semaine, line_dash="dash", line_color="red",
                         annotation_text="Volume actuel")
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommandation
            optimal_vol = sim_df.loc[sim_df['Temps_5km'].idxmin(), 'Volume']
            st.info(f"💡 Volume optimal simulé: {optimal_vol} km/semaine")

if __name__ == "__main__":
    main() 