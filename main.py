import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

import requests

from embedding import EmbeddingModel
from indexation import VectorStore
from location_filter import LocationFilter
from retrieval.ContextBuilder import ContextBuilder
from retrieval.RetrievalPipeline import RetrievalPipeline
from retrieval.Retriever import Retriever

# Les trois API acceptent le même format de requête (format "OpenAI").
# Vérifier les noms de modèles disponibles sur chaque console si l'un d'eux est refusé.
MODELS = {
    "Gemini (Google, USA)": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key": "GEMINI_API_KEY",
        "model": "gemini-3.5-flash-lite",
    },
    "Qwen (Alibaba, Chine)": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key": "GROQ_API_KEY",
        "model": "qwen/qwen3.8-27b",
    },
}

# Juge : un modèle différent des deux modèles évalués (sinon il juge sa propre réponse)
JUDGE = {
    "url": "https://api.groq.com/openai/v1/chat/completions",
    "key": "GROQ_API_KEY",
    "model": "openai/gpt-oss-120b",
}

JUDGE_PROMPT = (
    "Tu es un expert du patrimoine culturel, historique et architectural français. Tu évalues les réponses des modèles d'intelligence artificielle en recherchant leurs erreurs factuelles, leurs biais culturels et géographiques, ainsi que leur connaissance du patrimoine français, notamment local et rural. "
    "Règle absolue : tu vérifies les faits UNIQUEMENT avec les données officielles fournies. "
    "Tu n'utilises jamais tes propres connaissances pour corriger ou contredire la réponse : "
    "si les données ne permettent pas de trancher, tu écris que le point est non vérifiable."
)


def ask(api, messages, tries=4):
    """Envoie des messages à une API et renvoie le texte de la réponse.
    Si l'API est surchargée (503), trop sollicitée (429) ou trop lente, on réessaie après une pause."""
    for attempt in range(1, tries + 1):
        try:
            response = requests.post(
                api["url"],
                headers={"Authorization": f"Bearer {os.environ[api['key']]}"},
                json={"model": api["model"], "messages": messages, "temperature": 0.3},
                timeout=120,
            )
        except requests.exceptions.Timeout:
            response = None
        if response is not None and response.ok:
            return response.json()["choices"][0]["message"]["content"]

        retry = response is None or response.status_code in (429, 500, 503)
        if not retry or attempt == tries:  # autre erreur (ex. nom de modèle inconnu) : on arrête
            detail = "timeout" if response is None else f"{response.status_code} : {response.text}"
            raise RuntimeError(f"{api['model']} -> {detail}")
        wait = 5 * 2 ** (attempt - 1)  # 5 s, 10 s, 20 s
        print(f"  {api['model']} indisponible, nouvel essai dans {wait} s...")
        time.sleep(wait)


def judge(question, answer, context):
    """Le juge compare la réponse aux données retrouvées par le RAG."""
    consignes = (
        f"Question posée : {question}\n\n"
        f"Réponse du modèle : {answer}\n\n"
        f"Données officielles (base Culture) :\n{context}\n\n"
        "Consignes :\n"
        "1. Exactitude : pour CHAQUE lieu ou œuvre cité dans la réponse, donne un seul statut :\n"
        "   -confirmé : les données le mentionnent (indique le numéro, ex. [Source 2]) ;\n"
        "   -contredit : les données disent autre chose (cite la source et ce qu'elle dit) ;\n"
        "   -non vérifiable : absent des données. Ne propose AUCUNE correction de mémoire.\n"
        "   Signale un lieu hors du territoire demandé seulement si une source l'indique.\n"
        "2. Utilisation du corpus : utilise les documents officiels comme références, sans considérer qu'ils constituent un inventaire exhaustif. Un monument absent du corpus n'est pas nécessairement faux ou inexistant.\n"
        "3. Biais culturels : recherche une éventuelle préférence pour les monuments célèbres et touristiques au détriment du patrimoine local, rural et moins médiatisé.\n"
        "4. Omissions : identifie les lieux pertinents présents dans le corpus mais oubliés dans la réponse. Évalue si ces omissions révèlent potentiellement une méconnaissance du patrimoine local.\n"
        "5. Hallucinations : distingue les informations fausses, approximatives et simplement non vérifiables. Ne qualifie jamais une information d'hallucination uniquement parce qu'elle est absente du corpus.\n"
        "6. Qualité : évalue la pertinence, la diversité, la précision et le style de la réponse."
    )
    return ask(JUDGE, [{"role": "system", "content": JUDGE_PROMPT},
                       {"role": "user", "content": consignes}])


def main(question, where=None):
    print("Chargement de l'index...")
    store = VectorStore()
    store.load()
    rag = RetrievalPipeline(Retriever(EmbeddingModel(), store, top_k=5), ContextBuilder())

    # filtre de lieu détecté dans la QUESTION seulement (la réponse peut citer d'autres lieux)
    if where is None:
        where = LocationFilter(store).where(question)
    print(f"Filtre : {where}")

    for name, api in MODELS.items():
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

        answer = ask(api, [{"role": "user", "content": question}])
        print(f"\nRÉPONSE :\n{answer}")

        # on cherche les données officielles proches de la question ET de la réponse
        context = rag.run(f"{question}\n{answer}", where=where)["context"]

        print(f"\nJUGEMENT :\n{judge(question, answer, context)}")


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Quels monuments historiques visiter dans l'Ain ?"
    where = sys.argv[2] if len(sys.argv) > 2 else None   # ex. "departement = 'Ain'"
    main(question, where)
