#!/usr/bin/env python3
"""
Modèles prédictifs pour les performances en course à pied
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class RunningPerformancePredictor:
    """
    Classe principale pour prédire les performances en course à pied
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.target_distances = ['temps_5km', 'temps_10km', 'temps_semi', 'temps_marathon']
        
    def load_data(self, filepath='data/processed/runners_profiles.csv'):
        """Charge et prépare les données"""
        try:
            self.df = pd.read_csv(filepath)
            print(f"✅ Données chargées: {len(self.df)} coureurs")
            return True
        except FileNotFoundError:
            print(f"❌ Fichier non trouvé: {filepath}")
            return False
    
    def prepare_features(self):
        """Prépare les features pour l'entraînement"""
        print("🔧 Préparation des features...")
        
        # Features numériques
        numeric_features = ['age', 'poids', 'km_semaine', 'seances_semaine', 'fc_repos', 'fc_moyenne']
        
        # Features catégorielles à encoder
        categorical_features = ['sexe', 'niveau', 'type_parcours']
        
        # Créer le DataFrame des features
        X = self.df[numeric_features].copy()
        
        # Encoder les variables catégorielles
        for feature in categorical_features:
            le = LabelEncoder()
            X[f'{feature}_encoded'] = le.fit_transform(self.df[feature])
            self.label_encoders[feature] = le
        
        # Features dérivées (feature engineering)
        X['imc'] = X['poids'] / (1.70 ** 2)  # IMC approximatif
        X['volume_par_seance'] = X['km_semaine'] / X['seances_semaine']
        X['fc_reserve'] = X['fc_moyenne'] - X['fc_repos']
        X['age_squared'] = X['age'] ** 2  # Relation non-linéaire avec l'âge
        
        self.feature_names = X.columns.tolist()
        print(f"✅ {len(self.feature_names)} features préparées")
        
        return X
    
    def train_models(self, target_distance='temps_5km'):
        """Entraîne plusieurs modèles pour une distance donnée"""
        print(f"\n🏃‍♂️ Entraînement des modèles pour {target_distance}")
        print("=" * 50)
        
        # Préparer les données
        X = self.prepare_features()
        y = self.df[target_distance]
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalisation des features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[target_distance] = scaler
        
        # Définition des modèles
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        
        results = {}
        
        # Entraîner chaque modèle
        for name, model in models_config.items():
            print(f"🔄 Entraînement {name}...")
            
            # Utiliser les données normalisées pour certains modèles
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculer les métriques
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Validation croisée
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'y_test': y_test
            }
            
            print(f"   MAE: {mae:.2f} min | RMSE: {rmse:.2f} min | R²: {r2:.3f} | CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        self.models[target_distance] = results
        return results
    
    def optimize_best_model(self, target_distance='temps_5km'):
        """Optimise le meilleur modèle avec GridSearch"""
        print(f"\n🎯 Optimisation du meilleur modèle pour {target_distance}")
        
        # Trouver le meilleur modèle
        best_model_name = max(self.models[target_distance].keys(), 
                             key=lambda x: self.models[target_distance][x]['r2'])
        print(f"🏆 Meilleur modèle: {best_model_name}")
        
        # Préparer les données
        X = self.prepare_features()
        y = self.df[target_distance]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Paramètres à optimiser selon le modèle
        if best_model_name == 'Random Forest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif best_model_name == 'XGBoost':
            model = xgb.XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            print("⚠️ Optimisation non implémentée pour ce modèle")
            return None
        
        # GridSearch
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        
        print("🔄 Recherche des meilleurs hyperparamètres...")
        grid_search.fit(X_train, y_train)
        
        # Évaluer le modèle optimisé
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Modèle optimisé:")
        print(f"   Meilleurs paramètres: {grid_search.best_params_}")
        print(f"   MAE: {mae:.2f} min | RMSE: {rmse:.2f} min | R²: {r2:.3f}")
        
        # Sauvegarder le modèle optimisé
        self.models[target_distance]['optimized'] = {
            'model': best_model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'params': grid_search.best_params_
        }
        
        return best_model
    
    def analyze_feature_importance(self, target_distance='temps_5km'):
        """Analyse l'importance des features"""
        print(f"\n📊 Analyse de l'importance des features pour {target_distance}")
        
        # Utiliser le Random Forest pour l'importance des features
        if 'Random Forest' in self.models[target_distance]:
            model = self.models[target_distance]['Random Forest']['model']
            importances = model.feature_importances_
            
            # Créer un DataFrame pour l'analyse
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("🔝 Top 10 des features les plus importantes:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"   {i+1:2d}. {row['feature']:20s}: {row['importance']:.3f}")
            
            # Graphique
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_importance.head(10), y='feature', x='importance')
            plt.title(f'Importance des Features - {target_distance}', fontsize=14, fontweight='bold')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'data/processed/feature_importance_{target_distance}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            return feature_importance
        else:
            print("❌ Random Forest non disponible pour l'analyse des features")
            return None
    
    def create_performance_comparison(self):
        """Compare les performances des modèles"""
        print("\n📊 COMPARAISON DES PERFORMANCES DES MODÈLES")
        print("=" * 60)
        
        comparison_data = []
        
        for distance in self.models.keys():
            for model_name, results in self.models[distance].items():
                if model_name != 'optimized':
                    comparison_data.append({
                        'Distance': distance,
                        'Modèle': model_name,
                        'MAE': results['mae'],
                        'RMSE': results['rmse'],
                        'R²': results['r2'],
                        'CV_Mean': results['cv_mean']
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Affichage du tableau
        print(comparison_df.round(3))
        
        # Graphique de comparaison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² par modèle
        sns.barplot(data=comparison_df, x='Modèle', y='R²', ax=axes[0,0])
        axes[0,0].set_title('R² par Modèle')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # MAE par modèle
        sns.barplot(data=comparison_df, x='Modèle', y='MAE', ax=axes[0,1])
        axes[0,1].set_title('MAE par Modèle (minutes)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # R² par distance
        sns.boxplot(data=comparison_df, x='Distance', y='R²', ax=axes[1,0])
        axes[1,0].set_title('R² par Distance')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Heatmap des performances
        pivot_r2 = comparison_df.pivot(index='Modèle', columns='Distance', values='R²')
        sns.heatmap(pivot_r2, annot=True, cmap='YlOrRd', ax=axes[1,1])
        axes[1,1].set_title('Heatmap R² (Modèle vs Distance)')
        
        plt.tight_layout()
        plt.savefig('data/processed/models_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def predict_performance(self, runner_profile, target_distance='temps_5km'):
        """Prédit la performance d'un coureur"""
        if target_distance not in self.models:
            print(f"❌ Modèle non entraîné pour {target_distance}")
            return None
        
        # Préparer les features du coureur
        features = self.prepare_single_profile(runner_profile)
        
        # Utiliser le meilleur modèle
        best_model_name = max(self.models[target_distance].keys(), 
                             key=lambda x: self.models[target_distance][x]['r2'] if x != 'optimized' else 0)
        
        model = self.models[target_distance][best_model_name]['model']
        
        # Normaliser si nécessaire
        if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'SVR']:
            features_scaled = self.scalers[target_distance].transform([features])
            prediction = model.predict(features_scaled)[0]
        else:
            prediction = model.predict([features])[0]
        
        return prediction
    
    def prepare_single_profile(self, profile):
        """Prépare les features d'un profil individuel"""
        # Features numériques
        features = [
            profile['age'], profile['poids'], profile['km_semaine'], 
            profile['seances_semaine'], profile['fc_repos'], profile['fc_moyenne']
        ]
        
        # Features catégorielles encodées
        for cat_feature in ['sexe', 'niveau', 'type_parcours']:
            encoded_value = self.label_encoders[cat_feature].transform([profile[cat_feature]])[0]
            features.append(encoded_value)
        
        # Features dérivées
        imc = profile['poids'] / (1.70 ** 2)
        volume_par_seance = profile['km_semaine'] / profile['seances_semaine']
        fc_reserve = profile['fc_moyenne'] - profile['fc_repos']
        age_squared = profile['age'] ** 2
        
        features.extend([imc, volume_par_seance, fc_reserve, age_squared])
        
        return features
    
    def save_models(self, filepath_prefix='models/running_predictor'):
        """Sauvegarde les modèles entraînés"""
        import os
        os.makedirs('models', exist_ok=True)
        
        for distance in self.models.keys():
            for model_name, results in self.models[distance].items():
                if model_name != 'optimized':
                    filename = f"{filepath_prefix}_{distance}_{model_name.replace(' ', '_')}.joblib"
                    joblib.dump(results['model'], filename)
        
        # Sauvegarder les scalers et encoders
        joblib.dump(self.scalers, f"{filepath_prefix}_scalers.joblib")
        joblib.dump(self.label_encoders, f"{filepath_prefix}_encoders.joblib")
        
        print(f"✅ Modèles sauvegardés dans le dossier models/")

def main():
    """Fonction principale pour entraîner tous les modèles"""
    print("🏃‍♂️ PRÉDICTEUR DE PERFORMANCE EN COURSE")
    print("=" * 50)
    
    # Initialiser le prédicteur
    predictor = RunningPerformancePredictor()
    
    # Charger les données
    if not predictor.load_data():
        return
    
    # Entraîner les modèles pour chaque distance
    distances = ['temps_5km', 'temps_10km', 'temps_semi', 'temps_marathon']
    
    for distance in distances:
        predictor.train_models(distance)
        predictor.analyze_feature_importance(distance)
        
        # Optimiser le meilleur modèle
        predictor.optimize_best_model(distance)
    
    # Comparaison globale
    comparison_df = predictor.create_performance_comparison()
    
    # Sauvegarder les modèles
    predictor.save_models()
    
    # Exemple de prédiction
    print("\n🎯 EXEMPLE DE PRÉDICTION")
    print("=" * 30)
    
    exemple_coureur = {
        'age': 30,
        'poids': 70,
        'sexe': 'Homme',
        'niveau': 'Intermédiaire',
        'km_semaine': 40,
        'seances_semaine': 4,
        'type_parcours': 'Plat',
        'fc_repos': 60,
        'fc_moyenne': 150
    }
    
    print("👤 Profil du coureur exemple:")
    for key, value in exemple_coureur.items():
        print(f"   {key}: {value}")
    
    print("\n⏱️ Prédictions:")
    for distance in distances:
        prediction = predictor.predict_performance(exemple_coureur, distance)
        distance_name = distance.replace('temps_', '').upper()
        
        # Conversion en format lisible
        hours = int(prediction // 60)
        mins = int(prediction % 60)
        if hours > 0:
            time_str = f"{hours}h{mins:02d}m"
        else:
            time_str = f"{mins}m{int((prediction % 1) * 60):02d}s"
        
        print(f"   {distance_name}: {time_str}")
    
    print("\n🎉 Entraînement terminé avec succès !")

if __name__ == "__main__":
    main() 