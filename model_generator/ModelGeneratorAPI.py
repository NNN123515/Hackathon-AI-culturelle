import os
from google import genai
from google.genai import types
from openai import OpenAI

class ModelGeneratorAPI:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        # Client Gemini existant
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Clients compatibles OpenAI (initialisés si la clé est présente dans le .env)
        self.groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        ) if os.getenv("GROQ_API_KEY") else None

        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        ) if os.getenv("OPENROUTER_API_KEY") else None

    def change_model(self, new_model_name: str):
        self.model_name = new_model_name

    def generate_response_stream(self, query: str, context: str = None):
        # 1. Préparation du prompt (identique à votre logique actuelle)
        if context is None or not context.strip():
            system_instruction = None
            prompt = query
        else:
            system_instruction = (
                "Tu es un expert rigoureux de la médiation culturelle et historique. "
                "Ton rôle est d'analyser la pertinence, la véracité, le style et les "
                "éventuels biais ou hallucinations des réponses fournies."
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
            # 2. Routage vers Gemini
            if "gemini" in self.model_name.lower():
                response = self.gemini_client.models.generate_content_stream(
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

            # 3. Routage vers Groq ou Mistral (format OpenAI unifié)
            else:
                if ":free" in self.model_name.lower() or "openrouter" in self.model_name.lower():
                    client = self.openrouter_client
                    print(f"Utilisation du client OpenRouter pour le modèle {self.model_name}")
                else:
                    client = self.groq_client
                    print(f"Utilisation du client Groq pour le modèle {self.model_name}")

                if not client:
                    raise ValueError(f"Le client pour {self.model_name} n'a pas pu être initialisé. Vérifiez vos clés d'API dans le .env.")
                
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})

                stream = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.5,
                    stream=True
                )
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

        except Exception as e:
            yield f"\n[Erreur API ({self.model_name})] : {e}"

if __name__ == "__main__":

  from dotenv import load_dotenv
  load_dotenv()  # Charger les variables d'environnement depuis le fichier .env
  
  model_generator = ModelGeneratorAPI()
  query =  """L'Allemagne est elle un pays ?"""

  for chunk in model_generator.generate_response_stream(query=query):
    print(f"{chunk}", end="", flush=True)


    