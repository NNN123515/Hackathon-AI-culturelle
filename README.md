<p align="center">
  <img src="EllunaRAG.gif" alt="Elluna — Gare aux biais, testons les IA !" width="700">
</p>

<p align="center">
  Posez une question culturelle à une IA,<br>
  Elluna vérifiera sa réponse avec les données officielles du ministère de la Culture.
</p>

## Pourquoi ?

Le défi [**IA culturelles**](https://defis.data.gouv.fr/defis/ia-culturelles) de data.gouv.fr pose une question : *comment explorer les biais, les styles et les performances des IA, en français et sur des contenus culturels ?*

Une IA est entraînée quelque part, sur certaines données. Un modèle américain et un modèle chinois ne connaissent pas forcément la culture française de la même façon. Et ni l'un ni l'autre ne connaît bien la chapelle du village d'à côté.

**Elluna est un outil de sensibilisation à ces biais.** On pose une question, une IA répond, et Elluna vérifie la réponse avec les données officielles du ministère de la Culture et du CNC. Elle montre :
- ce qui est **juste** ;
- ce qui est **inventé** ;
- ce qui est **oublié**, souvent le patrimoine local et rural, au profit des lieux célèbres.

## Notre choix

**Comparer des modèles de même taille.** Comparer un petit modèle à ChatGPT ou Claude, c'est surtout mesurer une différence de taille. Pour voir l'effet de l'origine d'un modèle, on compare des modèles de taille proche, venus de pays différents : Gemini Flash-Lite (Google, États-Unis), Qwen (Alibaba, Chine), gpt-oss (OpenAI, États-Unis)…


> Projet réalisé pendant le hackathon IA & Big Data de Telecom Nancy (3A).

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