# 🏃‍♂️ Prédicteur de Performance en Course à Pied

Un système complet de machine learning pour prédire les performances en course à pied basé sur le profil et l'entraînement des coureurs.

## 🎯 Objectif du Projet

Ce projet utilise des techniques de machine learning pour prédire les temps de course sur différentes distances (5km, 10km, semi-marathon, marathon) en fonction des caractéristiques personnelles et d'entraînement des coureurs.

## 🚀 Fonctionnalités

### 📊 Modèles de Machine Learning
- **7 algorithmes différents** : Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, SVR
- **Prédictions multi-distances** : 5km, 10km, semi-marathon, marathon
- **Feature engineering avancé** : IMC, volume par séance, réserve FC, etc.
- **Optimisation automatique** des hyperparamètres avec GridSearch
- **Validation croisée** pour évaluer la robustesse des modèles

### 🎨 Application Web Interactive
- **Interface Streamlit** moderne et intuitive
- **Prédictions en temps réel** basées sur votre profil
- **Comparaisons** avec des coureurs similaires
- **Simulations de progression** pour optimiser l'entraînement
- **Recommandations personnalisées** d'amélioration

### 📈 Analyse de Données
- **Génération de données réalistes** de 200 coureurs
- **Analyse statistique complète** des performances
- **Visualisations avancées** avec Matplotlib, Seaborn et Plotly
- **Corrélations** entre variables d'entraînement et performances

## 🏗️ Structure du Projet

```
running_predicteur/
├── data/
│   ├── raw/                    # Données brutes
│   └── processed/              # Données traitées
│       ├── runners_profiles.csv
│       └── training_sessions.csv
├── src/
│   ├── data_processing/
│   │   ├── generate_data.py    # Génération des données
│   │   └── analyze_data.py     # Analyse exploratoire
│   └── models/
│       └── performance_predictor.py  # Modèles ML
├── notebooks/
│   └── 01_exploration_donnees.ipynb  # Notebook d'exploration
├── models/                     # Modèles sauvegardés
├── app.py                      # Application Streamlit
├── test_setup.py              # Test des dépendances
├── test_models.py             # Test des modèles
└── requirements.txt           # Dépendances Python
```

## 🛠️ Installation et Configuration

### 1. Cloner le projet
```bash
git clone git@github.com:papouloutoute/running_predicteur.git
cd running_predicteur
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Tester l'installation
```bash
python test_setup.py
```

## 🚀 Utilisation

### 1. Générer les données d'entraînement
```bash
python src/data_processing/generate_data.py
```

### 2. Analyser les données
```bash
python src/data_processing/analyze_data.py
```

### 3. Entraîner les modèles
```bash
python src/models/performance_predictor.py
```

### 4. Lancer l'application web
```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

### 5. Explorer avec Jupyter
```bash
jupyter notebook notebooks/01_exploration_donnees.ipynb
```

## 📊 Résultats des Modèles

### Performances par Distance

| Modèle | 5km (R²) | 10km (R²) | Semi (R²) | Marathon (R²) |
|--------|----------|-----------|-----------|---------------|
| **Linear Regression** | **0.906** | **0.906** | **0.897** | **0.913** |
| Gradient Boosting | 0.900 | 0.861 | 0.887 | 0.879 |
| Ridge Regression | 0.877 | 0.881 | 0.877 | 0.887 |
| Random Forest | 0.832 | 0.804 | 0.812 | 0.820 |
| XGBoost | 0.825 | 0.806 | 0.837 | 0.825 |

### Features les Plus Importantes
1. **Volume hebdomadaire** (km_semaine) : ~76% d'importance
2. **Niveau du coureur** : ~7% d'importance
3. **Âge** : ~4% d'importance
4. **Données physiologiques** (FC) : ~3% d'importance

## 🎯 Fonctionnalités de l'Application

### Interface Principale
- **Paramètres personnalisables** : âge, poids, niveau, volume d'entraînement
- **Prédictions instantanées** pour toutes les distances
- **Visualisations interactives** des performances

### Analyses Avancées
- **Comparaisons** avec des coureurs de profil similaire
- **Définition d'objectifs** et évaluation de leur faisabilité
- **Simulations de progression** pour optimiser l'entraînement
- **Recommandations personnalisées** d'amélioration

## 🔬 Méthodologie Scientifique

### Génération des Données
- **Distributions réalistes** basées sur des données de course réelles
- **Corrélations cohérentes** entre variables (âge, niveau, volume, performance)
- **Variabilité contrôlée** pour simuler la diversité des coureurs

### Feature Engineering
- **Variables dérivées** : IMC, volume par séance, réserve FC
- **Encodage des variables catégorielles** : sexe, niveau, type de parcours
- **Normalisation** pour les modèles sensibles à l'échelle

### Validation des Modèles
- **Division train/test** (80/20)
- **Validation croisée** 5-fold
- **Métriques multiples** : MAE, RMSE, R²
- **Optimisation des hyperparamètres**

## 📈 Insights Clés

### Facteurs de Performance
1. **Volume d'entraînement** : Facteur le plus déterminant (corrélation négative forte)
2. **Niveau d'expérience** : Impact significatif sur toutes les distances
3. **Âge** : Relation non-linéaire avec les performances
4. **Fréquence cardiaque** : Indicateur de condition physique

### Recommandations d'Entraînement
- **Augmentation progressive** du volume hebdomadaire
- **Minimum 3 séances** par semaine pour des gains significatifs
- **Travail cardiovasculaire** pour améliorer la réserve FC
- **Variété des entraînements** : endurance, fractionné, tempo

## 🛠️ Technologies Utilisées

### Machine Learning
- **scikit-learn** : Modèles ML et preprocessing
- **XGBoost** : Gradient boosting avancé
- **pandas** : Manipulation des données
- **numpy** : Calculs numériques

### Visualisation
- **Streamlit** : Application web interactive
- **Plotly** : Graphiques interactifs
- **Matplotlib/Seaborn** : Visualisations statiques

### Développement
- **Jupyter** : Exploration et prototypage
- **Python 3.12** : Langage principal

## 🎯 Prochaines Étapes

### Améliorations Techniques
- [ ] Intégration de données réelles de courses
- [ ] Modèles de deep learning (réseaux de neurones)
- [ ] Prédictions de progression temporelle
- [ ] API REST pour intégration externe

### Fonctionnalités Utilisateur
- [ ] Historique des performances personnelles
- [ ] Plans d'entraînement personnalisés
- [ ] Notifications et rappels
- [ ] Communauté de coureurs

### Analyses Avancées
- [ ] Prédiction des risques de blessure
- [ ] Optimisation de la récupération
- [ ] Analyse des conditions météorologiques
- [ ] Recommandations nutritionnelles
