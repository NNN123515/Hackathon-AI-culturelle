import os
from google import genai
from google.genai import types

class ModelGeneratorAPI:
  """Classe responsable de la génération de réponses globales
  à partir du contexte, en utilisant l'API Google GenAI (Gemini).
  """

  def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
    """Initialise le client Gemini.

    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("La variable d'environnement GEMINI_API_KEY est introuvable !")
    
    self.model_name = model_name
    print(f"Initialisation du client Google GenAI avec le modèle : {model_name}...")
    
    # Le client récupère automatiquement la clé dans os.environ["GEMINI_API_KEY"]
    self.client = genai.Client(api_key=api_key)

  def generate_response_stream(self, query: str, context: str):
    """Prend tout le contexte global, l'injecte dans le prompt,
    puis génère et transmet la réponse en streaming via l'API Gemini.
    """
    system_instruction = (
        "Tu es un expert rigoureux de la médiation culturelle et historique. "
        "Ton rôle est d'analyser la pertinence, la véracité, le style et les "
        "éventuels biais ou hallucinations des réponses fournies."
    )

    prompt = (
        f"Contexte documentaire (Patrimoine et Culture) :\n{context}\n\n"
        f"Réponse / Question à traiter : {query}\n\n"
        "Consignes d'évaluation :\n"
        "1. Appuie-toi exclusivement sur le contexte fourni issu d'une base de données culturelle.\n"
        "2. Identifie et signale tout potentiel biais, anachronisme ou hallucination.\n"
        "3. Évalue le style linguistique et le ton employé (précise s'il est adapté).\n"
        "4. Indique clairement si le contexte manque de preuves, sans jamais inventer d'informations.\n"
        "5. Si aucun biais ou problème n'est détecté, précise-le explicitement."
    )

    try:
      # Utilisation de l'API de streaming de Gemini
      response = self.client.models.generate_content_stream(
          model=self.model_name,
          contents=prompt,
          config=types.GenerateContentConfig(
              system_instruction=system_instruction,
              temperature=0.5,
              max_output_tokens=300,
          )
      )

      for chunk in response:
        if chunk.text:
          yield chunk.text

    except Exception as e:
      yield f"\n[Erreur API Gemini] : {e}"

  def generate_dual_stream(
      self, query1: str, context1: str, query2: str, context2: str
  ):
    """Exécute les deux requêtes l'une après l'autre."""

    for chunk in self.generate_response_stream(query1, context1):
        yield ("model1", chunk)

    for chunk in self.generate_response_stream(query2, context2):
        yield ("model2", chunk)


if __name__ == "__main__":
  
  model_generator = ModelGeneratorAPI()
  
  context1 = "La capitale de la France est Paris. C'est une ville connue pour sa culture et ses monuments."
  query1 = "Quelle est la capitale de la France ?"

  for tag, chunk in model_generator.generate_dual_stream(
      query1, context1, "Quelle est l'allemagne ?", "La capitale est Berlin."
  ):
    print(f"[{tag}] {chunk}", end="", flush=True)