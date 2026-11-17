from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class ModelGenerator:
  """Classe responsable de la génération de réponses globales

  à partir de tout le contexte d'un seul coup, avec un affichage en streaming.
  """

  def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Initialise le modèle Causal LM et son tokenizer."""
    self.model_name = model_name
    print(f"Chargement du modèle de génération : {model_name}...")

    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,  # Économise la mémoire VRAM/RAM
        device_map="auto",
        trust_remote_code=True,
    )

  def generate_response_stream(self, query: str, context: str):
    """Prend tout le contexte global, l'injecte dans le template de chat,

    puis génère et transmet la réponse en streaming.
    """

    # Utilisation du format de chat officiel de Qwen (recommandé pour les modèles Instruct)
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un expert rigoureux de la médiation culturelle et"
                " historique. Ton rôle est d'analyser la pertinence, la"
                " véracité, le style et les éventuels biais ou hallucinations"
                " des réponses de modèles d'IA fournies."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Contexte documentaire (Patrimoine et Culture) :\n{context}\n\n"
                f"Réponse d'un modèle : {query}\n\n"
                "Consignes d'évaluation :\n"
                "1. Appuie-toi exclusivement sur le contexte fourni issu d'une base de données culturelle.\n"
                "2. Identifie et signale tout potentiel biais, anachronisme ou"
                " hallucination dans la réponse.\n"
                "3. Évalue le style linguistique et le ton employé (précise"
                " s'il est adapté au domaine culturel).\n"
                "4. Indique clairement si le contexte manque de preuves ou"
                " d'éléments, sans jamais inventer d'informations.\n"
                "5. Si aucun biais ou problème n'est détecté, précise-le"
                " explicitement."
            ),
        },
    ]

    # Application du template de chat du modèle
    text_prompt = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenisation et envoi explicite sur le device du modèle (.to)
    inputs = self.tokenizer(
        text_prompt, return_tensors="pt", truncation=True, max_length=1536
    ).to(self.model.device)

    # Configuration du streamer pour intercepter les mots générés en direct
    streamer = TextIteratorStreamer(
        self.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # Préparation des arguments pour la génération
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.5,
        repetition_penalty=1.2,
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