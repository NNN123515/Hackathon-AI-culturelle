# 🎭 Hackathon - IA Culturelle

Projet développé dans le cadre du Hackathon sur l'Intelligence Artificielle culturelle.

## 🚀 Installation

### 1. Créer l'environnement virtuel
À la racine du projet, exécutez la commande suivante pour créer un environnement virtuel nommé `env` :

```bash
python -m venv env
```
*(Ou `py -m venv env` sous Windows)*

### 2. Activer l'environnement virtuel

* **Sur Windows (PowerShell / CMD) :**
  ```bash
  .\env\Scripts\activate
  ```
* **Sur Linux / macOS :**
  ```bash
  source env/bin/activate
  ```

### 3. Installer les dépendances
Une fois l'environnement activé, installez les modules requis :

```bash
pip install -r requirements.txt
```

## 💻 Lancement de l'API

Démarrez le serveur local avec Uvicorn :

```bash
uvicorn app_fast_api:app --reload
```

Une fois lancée, l'API est accessible à l'adresse : [http://127.0.0.1:8000](http://127.0.0.1:8000)  
La documentation interactive (Swagger) est disponible sur : [http://127.0.0](http://127.0.0)
