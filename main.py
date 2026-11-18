import os
import sys
from dotenv import load_dotenv
load_dotenv()

import requests

from embedding import EmbeddingModel
from indexation import VectorStore
from retrieval.ContextBuilder import ContextBuilder
from retrieval.RetrievalPipeline import RetrievalPipeline
from retrieval.Retriever import Retriever

# Les trois API acceptent le même format de requête (format "OpenAI").
# Vérifier les noms de modèles disponibles sur chaque console si l'un d'eux est refusé.
MODELS = {
    "Gemini (Google, USA)": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key": "GEMINI_API_KEY",
        "model": "models/gemini-3.8-flash",
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
    "Tu es un expert rigoureux de la médiation culturelle et historique. "
    "Tu évalues la réponse d'un modèle d'IA à une question, en t'appuyant "
    "uniquement sur des données officielles du ministère de la Culture."
)


def ask(api, messages):
    """Envoie des messages à une API et renvoie le texte de la réponse."""
    response = requests.post(
        api["url"],
        headers={"Authorization": f"Bearer {os.environ[api['key']]}"},
        json={"model": api["model"], "messages": messages, "temperature": 0.3},
        timeout=120,
    )
    if not response.ok:  # affiche le message d'erreur de l'API (ex. nom de modèle inconnu)
        raise RuntimeError(f"{api['model']} -> {response.status_code} : {response.text}")
    return response.json()["choices"][0]["message"]["content"]


def judge(question, answer, context):
    """Le juge compare la réponse aux données retrouvées par le RAG."""
    consignes = (
        f"Question posée : {question}\n\n"
        f"Réponse du modèle : {answer}\n\n"
        f"Données officielles (base Culture) :\n{context}\n\n"
        "Consignes :\n"
        "1. Appuie-toi uniquement sur les données officielles ci-dessus.\n"
        "2. Signale les erreurs, hallucinations ou anachronismes.\n"
        "3. Signale les biais : lieux ou œuvres très connus mis en avant, "
        "d'autres oubliés alors qu'ils sont dans les données.\n"
        "4. Si les données ne permettent pas de vérifier un point, dis-le.\n"
        "5. Termine par une note sur 10 et une phrase de synthèse."
    )
    return ask(JUDGE, [{"role": "system", "content": JUDGE_PROMPT},
                       {"role": "user", "content": consignes}])


def main(question, where=None):
    print("Chargement de l'index...")
    store = VectorStore()
    store.load()
    rag = RetrievalPipeline(Retriever(EmbeddingModel(), store, top_k=5), ContextBuilder())

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
