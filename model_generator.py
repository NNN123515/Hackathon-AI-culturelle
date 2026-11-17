from threading import Thread
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers import AutoModelForCausalLM
import torch


class ModelGenerator:
  """Classe responsable de la génération de réponses globales

  à partir de tout le contexte d'un seul coup, avec un affichage en streaming.
  """

  def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Initialise le modèle de génération Seq2Seq et son tokenizer."""
    self.model_name = model_name
    print(f"Chargement du modèle de génération : {model_name}...")

    # Chargement spécifique pour les modèles T5 / Seq2Seq
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Utilise le float16 pour économiser la mémoire VRAM/RAM
        device_map="auto",  
        trust_remote_code=True,
    )

  def generate_response_stream(self, query: str, context: str):
    """Prend tout le contexte global, l'injecte dans le prompt,
    puis génère et transmet la réponse en streaming.
    """
    if not context or not context.strip():
      yield "Aucun contexte disponible pour répondre."
      return

    # Construction du prompt global intégrant tout le contexte
    prompt = (
        f"Contexte documentaire (Patrimoine et Culture) :\n{context}\n\n"
        f"Requête de l'utilisateur : {query}\n\n"
        "Consignes pour l'évaluation et la génération :\n"
        "1. Agis en tant qu'expert rigoureux de la médiation culturelle et"
        " historique.\n"
        "2. Appuie-toi exclusivement sur le contexte fourni pour répondre et"
        " évaluer la véracité.\n"
        "3. Identifie et signale tout potentiel biais, anachronisme ou"
        " hallucination présent dans les éléments analysés.\n"
        "4. Évalue le style linguistique et le ton employé (registre de langue,"
        " adéquation du ton) et précise s'il est adapté au domaine"
        " culturel.\n"
        "5. Indique clairement si le contexte manque de preuves ou d'éléments"
        " pour répondre de manière exhaustive, sans jamais inventer d'informations.\n"
        "6. Si aucun biais n'est détecté, précise-le explicitement à"
        " l'utilisateur."
    )

    # Tokenisation du prompt
    inputs = self.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    )

    # Configuration du streamer pour intercepter les mots générés en direct
    streamer = TextIteratorStreamer(
        self.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Préparation des arguments pour la génération
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.5, #un peu d'originalité dans la réponse
        repetition_penalty=1.2  # Réduit la probabilité de répéter des phrases
    )

    # Lancement de la génération dans un thread séparé
    thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
    thread.start()

    # Transmission de chaque morceau de texte généré en temps réel
    for new_text in streamer:
      if new_text:
        yield new_text

if __name__ == "__main__":
    model_generator = ModelGenerator()
    context = "La capitale de la France est Paris. C'est une ville connue pour sa culture, son histoire et ses monuments emblématiques comme la Tour Eiffel et le Louvre."
    query = "Quelle est la capitale de la France ?"
    
    for response in model_generator.generate_response_stream(query, context):
        print(response, end="", flush=True)