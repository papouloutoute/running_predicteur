#!/usr/bin/env python3
"""
Test rapide des modèles de prédiction
"""

from src.models.performance_predictor import RunningPerformancePredictor

def test_models():
    """Test rapide des modèles"""
    print("🧪 TEST DES MODÈLES DE PRÉDICTION")
    print("=" * 40)
    
    # Initialiser le prédicteur
    predictor = RunningPerformancePredictor()
    
    # Charger les données
    if not predictor.load_data():
        print("❌ Impossible de charger les données")
        return
    
    # Test sur une seule distance pour commencer
    print("\n🏃‍♂️ Test sur le 5km...")
    results = predictor.train_models('temps_5km')
    
    # Afficher les résultats
    print("\n📊 RÉSULTATS:")
    for model_name, metrics in results.items():
        print(f"   {model_name:20s}: R² = {metrics['r2']:.3f} | MAE = {metrics['mae']:.2f} min")
    
    # Test de prédiction
    print("\n🎯 Test de prédiction:")
    exemple_coureur = {
        'age': 25,
        'poids': 65,
        'sexe': 'Femme',
        'niveau': 'Débutant',
        'km_semaine': 20,
        'seances_semaine': 3,
        'type_parcours': 'Plat',
        'fc_repos': 65,
        'fc_moyenne': 160
    }
    
    prediction = predictor.predict_performance(exemple_coureur, 'temps_5km')
    print(f"   Temps prédit pour le 5km: {prediction:.1f} minutes")
    
    print("\n✅ Test terminé avec succès !")

if __name__ == "__main__":
    test_models() 