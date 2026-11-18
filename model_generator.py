import queue
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
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

  def generate_response_stream(self, query: str, context: str):
    """Prend tout le contexte global, l'injecte dans le template de chat,

    puis génère et transmet la réponse en streaming.
    """
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

  def generate_dual_stream(
      self, query1: str, context1: str, query2: str, context2: str
  ):
    """Exécute les deux requêtes l'une après l'autre de manière sécurisée

    pour éviter les conflits de GPU (device mismatch avec accelerate).
    """
    if not context1 or not context1.strip():
      yield ("model1", "Aucun contexte disponible pour répondre.")
    else:
      for chunk in self.generate_response_stream(query1, context1):
        yield ("model1", chunk)

    if not context2 or not context2.strip():
      yield ("model2", "Aucun contexte disponible pour répondre.")
    else:
      for chunk in self.generate_response_stream(query2, context2):
        yield ("model2", chunk)

  def generate_dual_stream_threading(
      self, query1: str, context1: str, query2: str, context2: str
  ):
    """Permet de lancer deux requêtes en parallèle (attention aux risques de crash

    GPU avec accelerate selon la version, préférer generate_dual_stream si instable).
    """
    result_queue = queue.Queue()

    def worker(query, context, tag):
      try:
        for chunk in self.generate_response_stream(query, context):
          result_queue.put((tag, chunk))
      except Exception as e:
        result_queue.put((tag, f"[Erreur] {e}"))
      result_queue.put((tag, None))

    t1 = Thread(target=worker, args=(query1, context1, "model1"))
    t2 = Thread(target=worker, args=(query2, context2, "model2"))

    t1.start()
    t2.start()

    active_threads = 2
    while active_threads > 0:
      tag, chunk = result_queue.get()
      if chunk is None:
        active_threads -= 1
      else:
        yield (tag, chunk)

    t1.join()
    t2.join()


if __name__ == "__main__":
  model_generator = ModelGenerator()
  context1 = "La capitale de la France est Paris. C'est une ville connue pour sa culture, son histoire et ses monuments emblématiques comme la Tour Eiffel et le Louvre."
  query1 = "Quelle est la capitale de la France ?"

  context2 = "La capitale de l'Allemagne est Berlin. C'est une ville riche en histoire, notamment pour son rôle dans la Seconde Guerre mondiale et la Guerre froide."
  query2 = "Quelle est la capitale de l'Allemagne ?"

  # Si tu as des erreurs de device avec accelerate, tu pourras remplacer par .generate_dual_stream(...)
  for tag, chunk in model_generator.generate_dual_stream(
      query1, context1, query2, context2
  ):
    print(f"[{tag}] {chunk}", end="", flush=True)