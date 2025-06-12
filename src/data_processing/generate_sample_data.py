#!/usr/bin/env python3
"""
Générateur de données d'exemple pour le prédicteur de performance en course
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_runner_profiles(n_runners=200):
    """
    Génère des profils de coureurs avec des caractéristiques réalistes
    """
    np.random.seed(42)  # Pour la reproductibilité
    
    runners = []
    
    for i in range(n_runners):
        # Caractéristiques de base
        age = np.random.normal(35, 10)  # Âge moyen 35 ans
        age = max(18, min(70, age))  # Entre 18 et 70 ans
        
        sexe = np.random.choice(['Homme', 'Femme'], p=[0.6, 0.4])
        
        # Poids selon le sexe
        if sexe == 'Homme':
            poids = np.random.normal(75, 10)  # Hommes: 75kg ± 10
            poids = max(55, min(100, poids))
        else:
            poids = np.random.normal(62, 8)   # Femmes: 62kg ± 8
            poids = max(45, min(85, poids))
        
        # Niveau d'expérience (débutant, intermédiaire, avancé, expert)
        niveau = np.random.choice(['Débutant', 'Intermédiaire', 'Avancé', 'Expert'], 
                                 p=[0.3, 0.4, 0.25, 0.05])
        
        # Volume d'entraînement selon le niveau
        if niveau == 'Débutant':
            km_semaine = np.random.normal(15, 5)
            seances_semaine = np.random.choice([2, 3], p=[0.7, 0.3])
        elif niveau == 'Intermédiaire':
            km_semaine = np.random.normal(35, 10)
            seances_semaine = np.random.choice([3, 4], p=[0.6, 0.4])
        elif niveau == 'Avancé':
            km_semaine = np.random.normal(60, 15)
            seances_semaine = np.random.choice([4, 5, 6], p=[0.3, 0.5, 0.2])
        else:  # Expert
            km_semaine = np.random.normal(90, 20)
            seances_semaine = np.random.choice([5, 6, 7], p=[0.2, 0.5, 0.3])
        
        km_semaine = max(10, km_semaine)
        
        # Type de parcours préféré
        type_parcours = np.random.choice(['Plat', 'Vallonné', 'Montagneux'], 
                                        p=[0.5, 0.35, 0.15])
        
        # Fréquence cardiaque de repos (influence par l'âge et le niveau)
        fc_repos_base = 70 - (niveau == 'Expert') * 10 - (niveau == 'Avancé') * 5
        fc_repos = np.random.normal(fc_repos_base + age * 0.2, 8)
        fc_repos = max(45, min(90, fc_repos))
        
        # FC max théorique
        fc_max = 220 - age
        
        # FC moyenne d'entraînement (70-80% de la FC max)
        fc_moyenne = fc_max * np.random.uniform(0.70, 0.80)
        
        runners.append({
            'runner_id': f'R{i+1:03d}',
            'age': round(age),
            'sexe': sexe,
            'poids': round(poids, 1),
            'niveau': niveau,
            'km_semaine': round(km_semaine, 1),
            'seances_semaine': seances_semaine,
            'type_parcours': type_parcours,
            'fc_repos': round(fc_repos),
            'fc_max': round(fc_max),
            'fc_moyenne': round(fc_moyenne)
        })
    
    return pd.DataFrame(runners)

def calculate_performance_times(runners_df):
    """
    Calcule les temps de performance basés sur les caractéristiques des coureurs
    """
    performances = []
    
    for _, runner in runners_df.iterrows():
        # Facteurs influençant la performance
        
        # Facteur âge (pic vers 25-30 ans)
        age_factor = 1.0
        if runner['age'] < 25:
            age_factor = 1.05 + (25 - runner['age']) * 0.01
        elif runner['age'] > 35:
            age_factor = 1.0 + (runner['age'] - 35) * 0.015
        
        # Facteur sexe (différence physiologique)
        sexe_factor = 1.0 if runner['sexe'] == 'Homme' else 1.12
        
        # Facteur poids (optimal autour de l'IMC 20-22)
        taille_estimee = 1.70 if runner['sexe'] == 'Homme' else 1.62  # Estimation
        imc = runner['poids'] / (taille_estimee ** 2)
        if imc < 20:
            poids_factor = 1.02
        elif imc > 25:
            poids_factor = 1.0 + (imc - 25) * 0.02
        else:
            poids_factor = 1.0
        
        # Facteur entraînement (volume et fréquence)
        entrainement_factor = max(0.8, 1.2 - (runner['km_semaine'] / 100))
        entrainement_factor *= max(0.95, 1.1 - (runner['seances_semaine'] / 10))
        
        # Facteur niveau
        niveau_factors = {
            'Débutant': 1.3,
            'Intermédiaire': 1.1,
            'Avancé': 0.95,
            'Expert': 0.85
        }
        niveau_factor = niveau_factors[runner['niveau']]
        
        # Facteur parcours (impact sur les temps)
        parcours_factor = {
            'Plat': 1.0,
            'Vallonné': 1.05,
            'Montagneux': 1.12
        }[runner['type_parcours']]
        
        # Temps de base pour un coureur "standard" (homme, 30 ans, 70kg, niveau intermédiaire)
        temps_base = {
            '5km': 22.0,      # 22 minutes
            '10km': 46.0,     # 46 minutes  
            'semi': 100.0,    # 1h40
            'marathon': 210.0  # 3h30
        }
        
        # Calcul des temps avec variation aléatoire
        facteur_total = age_factor * sexe_factor * poids_factor * entrainement_factor * niveau_factor * parcours_factor
        
        temps_5km = temps_base['5km'] * facteur_total * np.random.uniform(0.95, 1.05)
        temps_10km = temps_base['10km'] * facteur_total * np.random.uniform(0.95, 1.05)
        temps_semi = temps_base['semi'] * facteur_total * np.random.uniform(0.95, 1.05)
        temps_marathon = temps_base['marathon'] * facteur_total * np.random.uniform(0.95, 1.05)
        
        performances.append({
            'runner_id': runner['runner_id'],
            'temps_5km': round(temps_5km, 1),
            'temps_10km': round(temps_10km, 1),
            'temps_semi': round(temps_semi, 1),
            'temps_marathon': round(temps_marathon, 1)
        })
    
    return pd.DataFrame(performances)

def generate_training_sessions(runners_df, n_sessions_per_runner=50):
    """
    Génère des séances d'entraînement pour chaque coureur
    """
    sessions = []
    
    for _, runner in runners_df.iterrows():
        # Nombre de séances basé sur le niveau
        if runner['niveau'] == 'Débutant':
            n_sessions = np.random.randint(30, 50)
        elif runner['niveau'] == 'Intermédiaire':
            n_sessions = np.random.randint(40, 70)
        elif runner['niveau'] == 'Avancé':
            n_sessions = np.random.randint(60, 100)
        else:  # Expert
            n_sessions = np.random.randint(80, 120)
        
        # Génération des séances sur les 6 derniers mois
        start_date = datetime.now() - timedelta(days=180)
        
        for i in range(n_sessions):
            # Date de la séance
            date_seance = start_date + timedelta(days=np.random.randint(0, 180))
            
            # Type de séance selon le niveau
            if runner['niveau'] in ['Débutant', 'Intermédiaire']:
                type_seance = np.random.choice(['Endurance', 'Tempo', 'Fractionné'], 
                                             p=[0.7, 0.2, 0.1])
            else:
                type_seance = np.random.choice(['Endurance', 'Tempo', 'Fractionné', 'Seuil'], 
                                             p=[0.5, 0.25, 0.15, 0.1])
            
            # Distance selon le type de séance et le niveau
            if type_seance == 'Endurance':
                distance_base = runner['km_semaine'] / runner['seances_semaine'] * 1.2
                distance = np.random.normal(distance_base, distance_base * 0.2)
            elif type_seance == 'Tempo':
                distance = np.random.uniform(8, 15)
            elif type_seance == 'Fractionné':
                distance = np.random.uniform(6, 12)
            else:  # Seuil
                distance = np.random.uniform(10, 18)
            
            distance = max(3, distance)
            
            # Durée estimée (basée sur l'allure d'entraînement)
            allure_base = 5.5  # min/km pour un coureur moyen
            if runner['niveau'] == 'Débutant':
                allure = allure_base * 1.3
            elif runner['niveau'] == 'Intermédiaire':
                allure = allure_base * 1.1
            elif runner['niveau'] == 'Avancé':
                allure = allure_base * 0.95
            else:  # Expert
                allure = allure_base * 0.85
            
            # Ajustement selon le type de séance
            if type_seance == 'Fractionné':
                allure *= 0.9  # Plus rapide
            elif type_seance == 'Endurance':
                allure *= 1.1  # Plus lent
            
            duree = distance * allure
            
            # FC moyenne de la séance
            fc_base = runner['fc_moyenne']
            if type_seance == 'Fractionné':
                fc_seance = fc_base * 1.1
            elif type_seance == 'Tempo':
                fc_seance = fc_base * 1.05
            else:
                fc_seance = fc_base * 0.95
            
            fc_seance = min(fc_seance, runner['fc_max'] * 0.95)
            
            sessions.append({
                'runner_id': runner['runner_id'],
                'date': date_seance.strftime('%Y-%m-%d'),
                'type_seance': type_seance,
                'distance': round(distance, 1),
                'duree': round(duree, 1),
                'allure': round(allure, 2),
                'fc_moyenne': round(fc_seance),
                'elevation': np.random.randint(0, 200) if runner['type_parcours'] != 'Plat' else np.random.randint(0, 50)
            })
    
    return pd.DataFrame(sessions)

def main():
    """Fonction principale pour générer toutes les données"""
    print("🏃‍♂️ Génération des données d'exemple...")
    
    # Génération des profils de coureurs
    print("📊 Génération des profils de coureurs...")
    runners_df = generate_runner_profiles(200)
    
    # Calcul des performances
    print("⏱️ Calcul des temps de performance...")
    performances_df = calculate_performance_times(runners_df)
    
    # Fusion des données
    complete_df = runners_df.merge(performances_df, on='runner_id')
    
    # Génération des séances d'entraînement
    print("🏋️‍♂️ Génération des séances d'entraînement...")
    sessions_df = generate_training_sessions(runners_df)
    
    # Sauvegarde
    print("💾 Sauvegarde des données...")
    complete_df.to_csv('data/processed/runners_profiles.csv', index=False)
    sessions_df.to_csv('data/processed/training_sessions.csv', index=False)
    
    print(f"✅ Données générées avec succès !")
    print(f"   - {len(complete_df)} profils de coureurs")
    print(f"   - {len(sessions_df)} séances d'entraînement")
    print(f"   - Fichiers sauvés dans data/processed/")
    
    # Affichage d'un aperçu
    print("\n📋 Aperçu des données générées:")
    print("\n🏃‍♂️ Profils de coureurs (5 premiers):")
    print(complete_df.head())
    
    print(f"\n📊 Statistiques des performances:")
    print(complete_df[['temps_5km', 'temps_10km', 'temps_semi', 'temps_marathon']].describe())

if __name__ == "__main__":
    main() 