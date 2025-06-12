# 🚀 Guide de Déploiement - Streamlit Cloud

## 📋 Prérequis
- Compte GitHub avec le repository public
- Compte Streamlit Community Cloud (gratuit)

## 🔧 Configuration du projet

### Fichiers nécessaires pour le déploiement :
- ✅ `app.py` - Application principale
- ✅ `requirements.txt` - Dépendances Python
- ✅ `.streamlit/config.toml` - Configuration Streamlit
- ✅ `packages.txt` - Dépendances système (si nécessaire)

## 🌐 Étapes de déploiement

### 1. Préparer le repository GitHub
```bash
git add .
git commit -m "Prêt pour déploiement Streamlit Cloud"
git push origin main
```

### 2. Accéder à Streamlit Community Cloud
- Aller sur : https://share.streamlit.io/
- Se connecter avec GitHub

### 3. Déployer l'application
- Cliquer sur "New app"
- Sélectionner le repository : `running_predicteur`
- Branch : `main`
- Main file path : `app.py`
- Cliquer sur "Deploy!"

### 4. URL de l'application
Votre app sera disponible à :
`https://[username]-running-predicteur-app-[hash].streamlit.app/`

## ⚙️ Configuration avancée

### Variables d'environnement (si nécessaire)
Dans l'interface Streamlit Cloud :
- Aller dans "Settings" > "Secrets"
- Ajouter les variables nécessaires

### Domaine personnalisé
- Disponible dans les paramètres avancés
- Format : `votre-nom.streamlit.app`

## 🔍 Monitoring et maintenance

### Logs et debugging
- Accessible via l'interface Streamlit Cloud
- Logs en temps réel disponibles

### Mise à jour automatique
- Chaque `git push` déclenche un redéploiement automatique
- Temps de déploiement : ~2-5 minutes

## 🎯 Optimisations pour la production

### Performance
- Les modèles sont déjà optimisés (32KB total)
- Cache Streamlit activé dans l'app
- Chargement des données optimisé

### Sécurité
- Pas de données sensibles dans le code
- Configuration CORS désactivée pour la sécurité

## 📊 Métriques d'usage
Streamlit Cloud fournit :
- Nombre de visiteurs
- Temps de réponse
- Utilisation des ressources

---

🎉 **Votre application sera accessible 24/7 gratuitement !** 