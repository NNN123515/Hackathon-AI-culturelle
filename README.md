# Documentation du projet - Hackathon Intelligence Artificielle et Culture

## Présentation du système

Ce projet a été développé dans le cadre du Hackathon IA&Big Data de 3ème année de Telecom Nancy. Ce projet est dédié à l'Intelligence Artificielle appliquée au domaine culturel. Ce projet correspond au sujet 8 des défis de l'Open Data University : https://www.opendatauniversity.org/defis.

Il s'agit d'une interface de programmation (API) conçue pour interroger, analyser et évaluer les réponses de différents modèles de langage (LLMs) sur des thématiques patrimoniales. Le système permet de générer des réponses textuelles concernant le domaine de la culture, suite à une requête utilisateur.

Cette évaluation a pour but d'identifier les potentiels biais culturels, les hallucinations factuelles, ainsi que les omissions (notamment la négligence du patrimoine local au profit des monuments mondialement célèbres). L'API sert ainsi de moteur pour une interface de test et de sensibilisation aux biais des IA.

L'utilisateur peut tout d'abord changer le modèle permettant de générer une réponse sur le domaine de la culture. Ainsi, lorqu'il aura fourni une requête au système, il pourra comparer ses réponses selon différents types de modèles et, par la suite, indentifier à l'aide du modèle d'évaluation les différents biais trouvés dans les réponses.

## Procédure d'installation

### 1. Création de l'environnement virtuel
À la racine du projet, exécutez la commande suivante pour isoler les dépendances dans un environnement virtuel nommé `env` :

```bash
python -m venv env
```
*(Note : utilisez `py -m venv env` sous environnement Windows)*

### 2. Activation de l'environnement virtuel

- **Sous Windows (PowerShell / Invite de commandes) :**
  ```bash
  .\env\Scripts\activate
  ```
- **Sous environnements de type Unix (Linux / macOS) :**
  ```bash
  source env/bin/activate
  ```

### 3. Installation des dépendances
Une fois l'environnement activé, installez les modules requis figurant dans le fichier de configuration :

```bash
pip install -r requirements.txt
```

## Démarrage du service

Pour lancer l'interface en local, exécutez Uvicorn avec la commande suivante :

```bash
uvicorn app_fast_api:app --reload
```

Une fois le processus démarré, le service est opérationnel et accessible aux adresses suivantes :
- Point d'entrée de l'API : http://127.0.0.1:8000
- Documentation interactive (Swagger UI) : http://127.0.0.1:8000/docs