#!/usr/bin/env python3
"""
Script de test pour vérifier l'installation des dépendances
"""

def test_imports():
    """Teste l'import de toutes les dépendances principales"""
    
    try:
        import pandas as pd
        print("✅ Pandas:", pd.__version__)
    except ImportError as e:
        print("❌ Erreur Pandas:", e)
        return False
    
    try:
        import numpy as np
        print("✅ NumPy:", np.__version__)
    except ImportError as e:
        print("❌ Erreur NumPy:", e)
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn:", sklearn.__version__)
    except ImportError as e:
        print("❌ Erreur Scikit-learn:", e)
        return False
    
    try:
        import matplotlib
        print("✅ Matplotlib:", matplotlib.__version__)
    except ImportError as e:
        print("❌ Erreur Matplotlib:", e)
        return False
    
    try:
        import seaborn as sns
        print("✅ Seaborn:", sns.__version__)
    except ImportError as e:
        print("❌ Erreur Seaborn:", e)
        return False
    
    try:
        import xgboost as xgb
        print("✅ XGBoost:", xgb.__version__)
    except ImportError as e:
        print("❌ Erreur XGBoost:", e)
        return False
    
    try:
        import streamlit as st
        print("✅ Streamlit:", st.__version__)
    except ImportError as e:
        print("❌ Erreur Streamlit:", e)
        return False
    
    try:
        import jupyter_core
        print("✅ Jupyter:", jupyter_core.__version__)
    except ImportError as e:
        print("❌ Erreur Jupyter:", e)
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Test des dépendances du projet Running Predicteur")
    print("=" * 50)
    
    if test_imports():
        print("\n🎉 Toutes les dépendances sont installées correctement!")
        print("✨ Le projet est prêt pour la phase 2!")
    else:
        print("\n❌ Certaines dépendances ne sont pas installées correctement.")
        print("Vérifiez les erreurs ci-dessus.") 