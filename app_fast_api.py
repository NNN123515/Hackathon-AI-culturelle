from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse 
from typing import Optional
import json

from rag_app import RAGApplication
from model_generator.ModelGeneratorAPI import ModelGeneratorAPI # Assurez-vous que l'import correspond à votre structure

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API IA Culturelle",
    description="API FastAPI permettant d'évaluer, à l'aide d'un modèle d'IA, la pertinence et la véracité des réponses générées par un modèle d'IA par rapport à un contexte culturel et patrimonial fourni.",
    version="1.0.0"
)

# Initialisation globale de l'application RAG (charge le VectorStore et le modèle par défaut)
rag_app = RAGApplication(top_k=5)

# Initialisation d'un générateur fixe dédié uniquement à l'évaluation
EVALUATION_MODEL_NAME = "openai/gpt-oss-120b"
eval_generator = ModelGeneratorAPI(model_name=EVALUATION_MODEL_NAME)

LOCAL_MODELS_LIST = {
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct"
}

API_MODELS_LIST = {
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "nvidia/nemotron-3.5-lightning:free",
    "liquid/lfm-3-1.7b:free",
    "cohere/north-mini-code:free",
    "baidu/ernie-5.5-flash-preview:free"
}

AVAILABLE_MODELS = tuple(LOCAL_MODELS_LIST.union(API_MODELS_LIST))

# --- Routes de l'API ---

@app.get("/")
def read_root():
    """Route racine pour vérifier l'état de l'API."""
    return {
        "status": "online",
        "message": "Bienvenue sur l'API RAG et d'Évaluation.",
        "current_model": rag_app.generator.model_name
    }


@app.post("/generate")
async def generate_response(question: str, where: Optional[str] = None):
    """
    Reçoit une requête utilisateur et renvoie la réponse en streaming pur.
    """
    def token_generator():
        # Assurez-vous que votre méthode produit bien les chunks textuels
        for chunk in rag_app.answer_query(question=question, rag=False):
            print(f"{chunk}") 
            yield chunk

    return StreamingResponse(token_generator(), media_type="text/plain")

@app.post("/change_model")
def change_model(new_model: str):
    """
    Modifie le modèle utilisé.
    """
    try:
        # Déterminer le type cible (Local ou API)
        target_type = "Local" if new_model in LOCAL_MODELS_LIST else "API"
        
        # Déterminer le type actuellement actif
        current_type = "Local" if "Local" in rag_app.generator.__class__.__name__ else "API"

        # Si on passe d'API à Local (ou inversement), on change le type 
        # (L'idéal est de passer le nom du modèle pour l'initialiser directement)
        if target_type != current_type:
            # On vérifie si ta méthode supporte de prendre le nom du modèle
            if "model_name" in rag_app.change_model_type.__code__.co_varnames:
                rag_app.change_model_type(target_type, model_name=new_model)
                # Pas besoin de faire change_model() car il a été instancié avec le bon
            else:
                rag_app.change_model_type(target_type)
                rag_app.generator.change_model(new_model)
        else:
            # Si on reste dans le même type (ex: Local -> Local), on change juste les poids
            rag_app.generator.change_model(new_model)

        return {
            "success": True,
            "current_model": rag_app.generator.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_model")
def get_current_model():
    """
    Récupère le nom du modèle actuellement configuré.
    """
    return {"current_model": rag_app.generator.model_name}

@app.get("/models/local")
def list_local_models():
    """
    Renvoie simplement la liste des noms de modèles locaux disponibles.
    """
    return LOCAL_MODELS_LIST

@app.get("/models/api")
def list_api_models():
    """
    Renvoie simplement la liste des noms de modèles API disponibles.
    """
    return API_MODELS_LIST

@app.post("/evaluate")
async def evaluate_response(query: str, where: Optional[str] = None):
    """
    Applique le 'modèle d'évaluation' en streaming direct.
    """

    try:
        retrieval_result = rag_app.pipeline.run(query, where=where)
        context = retrieval_result.get("context", "")

        # Définition d'un générateur interne qui produit les chunks
        def token_generator():
            for chunk in eval_generator.generate_response_stream(query=query, context=context):
                yield chunk

        # On retourne une StreamingResponse avec un type texte/plain (ou text/event-stream)
        return StreamingResponse(token_generator(), media_type="text/plain")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-full-pipeline")
async def run_full_pipeline(question: str, where: Optional[str] = None):
    """
    Exécute l'enchaînement complet en streaming (SSE) : 
    1. Stream la génération de la réponse initiale (sans RAG).
    2. Récupère le contexte RAG.
    3. Stream le rapport d'évaluation des biais basé sur ce contexte.
    4. Envoie l'objet récapitulatif complet à la fin.
    """
    # Tield ("gen_chunk", chunk) ou ("evalèchunk", chunk)
    async def event_generator():
        try:
            # Génération de la réponse initiale en streaming ---
            yield f"data: {json.dumps({'status': 'start_generation'})}\n\n"
            generated_answer = ""
            
            # On s'assure d'itérer proprement sur le générateur
            for chunk in rag_app.generator.generate_response_stream(query=question, context=None):
                if chunk:
                    generated_answer += chunk
                    print(f"{chunk}", end="", flush=True)
                    yield f"data: {json.dumps({'type': 'gen_chunk', 'content': chunk})}\n\n"

            # Récupération du contexte RAG ---
            yield f"data: {json.dumps({'status': 'retrieving_rag'})}\n\n"
            retrieval_result = rag_app.pipeline.run(question, where=where)
            context = retrieval_result.get("context", "")
            
            yield f"data: {json.dumps({'type': 'rag_context', 'content': context})}\n\n"

            # Évaluation de la réponse générée en streaming ---
            yield f"data: {json.dumps({'status': 'start_evaluation'})}\n\n"
            evaluation_report = ""
            
            for chunk in eval_generator.generate_response_stream(query=generated_answer, context=context):
                if chunk:
                    evaluation_report += chunk
                    print(f"{chunk}", end="", flush=True)
                    yield f"data: {json.dumps({'type': 'eval_chunk', 'content': chunk})}\n\n"

            # Message de fin avec le résumé complet ---
            final_payload = {
                "status": "completed",
                "question": question,
                "model_used": rag_app.generator.model_name,
                "generated_response": generated_answer,
                "rag_context": context,
                "evaluation_report": evaluation_report
            }
            yield f"data: {json.dumps(final_payload)}\n\n"

        except Exception as e:
            # En cas d'erreur interne, on l'envoie proprement dans le flux SSE
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")