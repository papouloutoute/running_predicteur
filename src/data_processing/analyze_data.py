#!/usr/bin/env python3
"""
Script d'analyse des données pour le prédicteur de performance en course
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Charge les données générées"""
    try:
        runners_df = pd.read_csv('data/processed/runners_profiles.csv')
        sessions_df = pd.read_csv('data/processed/training_sessions.csv')
        return runners_df, sessions_df
    except FileNotFoundError as e:
        print(f"❌ Erreur: {e}")
        print("Assurez-vous d'avoir généré les données avec generate_sample_data.py")
        return None, None

def minutes_to_time(minutes):
    """Convertit les minutes en format HH:MM:SS"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes % 1) * 60)
    if hours > 0:
        return f"{hours}h{mins:02d}m{secs:02d}s"
    else:
        return f"{mins}m{secs:02d}s"

def analyze_population(runners_df):
    """Analyse la population de coureurs"""
    print("👥 ANALYSE DE LA POPULATION")
    print("=" * 40)
    
    print(f"📊 Nombre total de coureurs: {len(runners_df)}")
    print(f"📊 Âge moyen: {runners_df['age'].mean():.1f} ans (écart-type: {runners_df['age'].std():.1f})")
    print(f"📊 Poids moyen: {runners_df['poids'].mean():.1f} kg (écart-type: {runners_df['poids'].std():.1f})")
    
    print("\n🚻 Répartition par sexe:")
    gender_counts = runners_df['sexe'].value_counts()
    for gender, count in gender_counts.items():
        pct = count/len(runners_df)*100
        print(f"   • {gender}: {count} ({pct:.1f}%)")
    
    print("\n🏆 Répartition par niveau:")
    level_counts = runners_df['niveau'].value_counts()
    for level, count in level_counts.items():
        pct = count/len(runners_df)*100
        print(f"   • {level}: {count} ({pct:.1f}%)")
    
    print("\n🗺️ Répartition par type de parcours:")
    terrain_counts = runners_df['type_parcours'].value_counts()
    for terrain, count in terrain_counts.items():
        pct = count/len(runners_df)*100
        print(f"   • {terrain}: {count} ({pct:.1f}%)")

def analyze_training(runners_df):
    """Analyse les habitudes d'entraînement"""
    print("\n🏃‍♂️ ANALYSE DE L'ENTRAÎNEMENT")
    print("=" * 40)
    
    print(f"📊 Volume moyen: {runners_df['km_semaine'].mean():.1f} km/semaine")
    print(f"📊 Fréquence moyenne: {runners_df['seances_semaine'].mean():.1f} séances/semaine")
    print(f"📊 FC repos moyenne: {runners_df['fc_repos'].mean():.0f} bpm")
    print(f"📊 FC entraînement moyenne: {runners_df['fc_moyenne'].mean():.0f} bpm")
    
    print("\n📈 Volume d'entraînement par niveau:")
    training_by_level = runners_df.groupby('niveau')[['km_semaine', 'seances_semaine']].mean()
    for level, data in training_by_level.iterrows():
        print(f"   • {level}: {data['km_semaine']:.1f} km/sem, {data['seances_semaine']:.1f} séances/sem")

def analyze_performances(runners_df):
    """Analyse les performances"""
    print("\n⏱️ ANALYSE DES PERFORMANCES")
    print("=" * 40)
    
    performance_vars = ['temps_5km', 'temps_10km', 'temps_semi', 'temps_marathon']
    
    print("📊 Temps moyens par distance:")
    for var in performance_vars:
        distance = var.replace('temps_', '').upper()
        mean_time = runners_df[var].mean()
        median_time = runners_df[var].median()
        min_time = runners_df[var].min()
        max_time = runners_df[var].max()
        
        print(f"\n🏃‍♂️ {distance}:")
        print(f"   Moyenne: {minutes_to_time(mean_time)}")
        print(f"   Médiane: {minutes_to_time(median_time)}")
        print(f"   Meilleur: {minutes_to_time(min_time)}")
        print(f"   Plus lent: {minutes_to_time(max_time)}")
    
    print("\n👫 Performances par sexe:")
    perf_by_gender = runners_df.groupby('sexe')[performance_vars].mean()
    for gender, data in perf_by_gender.iterrows():
        print(f"\n   {gender}:")
        for var in performance_vars:
            distance = var.replace('temps_', '').upper()
            print(f"     {distance}: {minutes_to_time(data[var])}")
    
    print("\n🏆 Performances par niveau:")
    perf_by_level = runners_df.groupby('niveau')[performance_vars].mean()
    for level, data in perf_by_level.iterrows():
        print(f"\n   {level}:")
        for var in performance_vars:
            distance = var.replace('temps_', '').upper()
            print(f"     {distance}: {minutes_to_time(data[var])}")

def analyze_correlations(runners_df):
    """Analyse les corrélations"""
    print("\n🔗 ANALYSE DES CORRÉLATIONS")
    print("=" * 40)
    
    performance_vars = ['temps_5km', 'temps_10km', 'temps_semi', 'temps_marathon']
    feature_vars = ['age', 'poids', 'km_semaine', 'seances_semaine', 'fc_repos', 'fc_moyenne']
    
    print("📊 Corrélations avec les performances:")
    
    for perf_var in performance_vars:
        distance = perf_var.replace('temps_', '').upper()
        print(f"\n⏱️ {distance}:")
        
        correlations = []
        for feature in feature_vars:
            corr = runners_df[feature].corr(runners_df[perf_var])
            correlations.append((feature, corr))
        
        # Trier par valeur absolue de corrélation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature, corr in correlations[:3]:  # Top 3
            direction = "positive" if corr > 0 else "négative"
            strength = "forte" if abs(corr) > 0.5 else "modérée" if abs(corr) > 0.3 else "faible"
            print(f"   • {feature}: {corr:.3f} (corrélation {direction} {strength})")

def analyze_sessions(sessions_df):
    """Analyse les séances d'entraînement"""
    if sessions_df is None:
        return
        
    print("\n🏋️‍♂️ ANALYSE DES SÉANCES D'ENTRAÎNEMENT")
    print("=" * 40)
    
    sessions_df['date'] = pd.to_datetime(sessions_df['date'])
    
    print(f"📊 Nombre total de séances: {len(sessions_df):,}")
    print(f"📊 Période: {sessions_df['date'].min().strftime('%d/%m/%Y')} - {sessions_df['date'].max().strftime('%d/%m/%Y')}")
    print(f"📊 Distance totale: {sessions_df['distance'].sum():.0f} km")
    print(f"📊 Durée totale: {sessions_df['duree'].sum()/60:.0f} heures")
    print(f"📊 Distance moyenne par séance: {sessions_df['distance'].mean():.1f} km")
    print(f"📊 Allure moyenne: {sessions_df['allure'].mean():.2f} min/km")
    
    print("\n🏃‍♂️ Répartition par type de séance:")
    session_types = sessions_df['type_seance'].value_counts()
    for session_type, count in session_types.items():
        pct = count/len(sessions_df)*100
        avg_distance = sessions_df[sessions_df['type_seance'] == session_type]['distance'].mean()
        avg_pace = sessions_df[sessions_df['type_seance'] == session_type]['allure'].mean()
        print(f"   • {session_type}: {count} séances ({pct:.1f}%) - {avg_distance:.1f}km - {avg_pace:.2f}min/km")

def create_summary_visualizations(runners_df):
    """Crée des visualisations de résumé"""
    print("\n📊 GÉNÉRATION DES GRAPHIQUES DE RÉSUMÉ")
    print("=" * 40)
    
    # Configuration
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure avec 4 sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution des niveaux
    runners_df['niveau'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Répartition par Niveau', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Niveau')
    axes[0,0].set_ylabel('Nombre de coureurs')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Volume d'entraînement vs Performance 5km
    sns.scatterplot(data=runners_df, x='km_semaine', y='temps_5km', 
                   hue='niveau', alpha=0.7, ax=axes[0,1])
    axes[0,1].set_title('Volume d\'Entraînement vs Performance 5km', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Km par semaine')
    axes[0,1].set_ylabel('Temps 5km (minutes)')
    
    # 3. Performances par sexe (5km)
    sns.boxplot(data=runners_df, x='sexe', y='temps_5km', ax=axes[1,0])
    axes[1,0].set_title('Performance 5km par Sexe', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Sexe')
    axes[1,0].set_ylabel('Temps 5km (minutes)')
    
    # 4. Corrélation âge vs performance
    sns.scatterplot(data=runners_df, x='age', y='temps_marathon', 
                   hue='sexe', alpha=0.7, ax=axes[1,1])
    axes[1,1].set_title('Âge vs Performance Marathon', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Âge (années)')
    axes[1,1].set_ylabel('Temps Marathon (minutes)')
    
    plt.tight_layout()
    plt.savefig('data/processed/analysis_summary.png', dpi=300, bbox_inches='tight')
    print("✅ Graphiques sauvegardés dans data/processed/analysis_summary.png")
    
    # Affichage si possible
    try:
        plt.show()
    except:
        print("ℹ️ Graphiques générés mais non affichés (mode non-interactif)")

def main():
    """Fonction principale d'analyse"""
    print("🏃‍♂️ ANALYSE DES DONNÉES - PRÉDICTEUR DE PERFORMANCE")
    print("=" * 60)
    
    # Chargement des données
    runners_df, sessions_df = load_data()
    if runners_df is None:
        return
    
    print(f"✅ Données chargées: {len(runners_df)} coureurs, {len(sessions_df) if sessions_df is not None else 0} séances")
    
    # Analyses
    analyze_population(runners_df)
    analyze_training(runners_df)
    analyze_performances(runners_df)
    analyze_correlations(runners_df)
    analyze_sessions(sessions_df)
    
    # Visualisations
    create_summary_visualizations(runners_df)

if __name__ == "__main__":
    main() 