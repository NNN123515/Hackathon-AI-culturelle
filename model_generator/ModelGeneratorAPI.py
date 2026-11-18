from logging import exception
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

  def change_model(self, new_model_name: str):
        """Change le modèle utilisé pour la génération de réponses."""

        self.model_name = new_model_name

  def generate_response_stream(self, query: str, context: str = None):
    """Prend tout le contexte global, l'injecte dans le prompt,
    puis génère et transmet la réponse en streaming via l'API Gemini.
    """

    if context is None or not context.strip():
        #Cas où le contexte est vide ou None, c'est à dire que il s'agit d'une simple question dont le modèle doit répondre
        #On doit donc générer une réponse classique
        system_instruction = None
        prompt = query

    else:
        #Dans ce cas, on doit faire l'évaluation de la réponse d'un modèle par rapport à un contexte culturel et patrimonial fourni.
        system_instruction = (
            "Tu es un expert rigoureux de la médiation culturelle et historique. "
            "Ton rôle est d'analyser la pertinence, la véracité, le style et les "
            "éventuels biais ou hallucinations des réponses fournies."
            "Tu dois pour cela te baser sur une base de données culturelle et patrimoniale fournie afin d'évaluer la réponse d'un modèle d'IA."
            "Les réponses que tu dois évaluer proviennent d'un modèle d'IA suite à une requête utilisateur (aucun contexte supplémentaire n'a été fourni)."
        )

        prompt = (
            f"Contexte documentaire (Patrimoine et Culture) :\n{context}\n\n"
            f"Réponse à traiter : {query}\n\n"
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
              temperature=0.5
          )
      )

      for chunk in response:
        if chunk.text:
          yield chunk.text

    except Exception as e:
      yield f"\n[Erreur API Gemini] : {e}"

if __name__ == "__main__":

  from dotenv import load_dotenv
  load_dotenv()  # Charger les variables d'environnement depuis le fichier .env
  
  model_generator = ModelGeneratorAPI()
  query =  """L'Allemagne est elle un pays ?"""

  for chunk in model_generator.generate_response_stream(query=query):
    print(f"{chunk}", end="", flush=True)


    