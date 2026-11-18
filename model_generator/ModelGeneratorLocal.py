from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

class ModelGeneratorLocal:
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
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
  
  def change_model(self, new_model_name: str):
        """Change le modèle utilisé pour la génération de réponses."""

        self.model_name = new_model_name
        print(f"Chargement du nouveau modèle de génération : {new_model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            new_model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            new_model_name,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

  def generate_response_stream(self, query: str, context: str = None):
    """Prend tout le contexte global, l'injecte dans le template de chat,

    puis génère et transmet la réponse en streaming.
    """

    if context is None or not context.strip():
        # Cas où le contexte est vide ou None, c'est à dire que il s'agit d'une simple question dont le modèle doit répondre
        # On doit donc générer une réponse classique
        messages = [
            {"role": "user", "content": query}
        ]
    else:
        # Dans ce cas, on doit faire l'évaluation de la réponse d'un modèle par rapport à un contexte culturel et patrimonial fourni.
        # On va donc utiliser un template de chat avec un rôle système pour guider le modèle.
        messages = [
            {
                "role": "system",
                "content": (
                        "Tu es un expert rigoureux de la médiation culturelle et historique. "
                        "Ton rôle est d'analyser la pertinence, la véracité, le style et les "
                        "éventuels biais ou hallucinations des réponses fournies."
                        "Tu dois pour cela te baser sur une base de données culturelle et patrimoniale fournie afin d'évaluer la réponse d'un modèle d'IA."
                        "Les réponses que tu dois évaluer proviennent d'un modèle d'IA suite à une requête utilisateur (aucun contexte supplémentaire n'a été fourni)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Contexte documentaire (Patrimoine et Culture) :\n{context}\n\n"
                f"Réponse d'un modèle : {query}\n\n"
                "Consignes d'évaluation :\n"
                "1. Appuie-toi exclusivement sur le contexte fourni issu d'une"
                " base de données culturelle.\n"
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

    text_prompt = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = self.tokenizer(
        text_prompt, return_tensors="pt", truncation=True, max_length=1536
    ).to(self.model.device)

    streamer = TextIteratorStreamer(
        self.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.5,
        repetition_penalty=1.2,
    )

    thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
      if new_text:
        yield new_text

if __name__ == "__main__":
  model_generator = ModelGeneratorLocal()
  query = "Quelle est la capitale de l'Allemagne ?"

  for chunk in model_generator.generate_response_stream(query=query, context="L'Allemagne est un pays d'Europe centrale. Sa capitale est Berlin."):
    print(chunk, end="", flush=True)