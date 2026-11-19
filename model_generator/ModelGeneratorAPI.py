import os
from google import genai
from google.genai import types
from openai import OpenAI

class ModelGeneratorAPI:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
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
            system_instruction = "Réponds à la question posée de manière concise (entre 200 et 300 mots). Préfères toujours un résumé qu'à un long résumé déatillé."
            prompt = query
        else:
            system_instruction = (
            "Tu es un expert du patrimoine culturel, historique et architectural français. Tu évalues les réponses des modèles d'intelligence artificielle en recherchant leurs erreurs factuelles, leurs biais culturels et géographiques, ainsi que leur connaissance du patrimoine français, notamment local et rural. "
            "Règle absolue : tu vérifies les faits UNIQUEMENT avec les données officielles fournies. "
            "Tu n'utilises jamais tes propres connaissances pour corriger ou contredire la réponse : "
            "si les données ne permettent pas de trancher, tu écris que le point est non vérifiable."
            )
            prompt = (
                f"Contexte documentaire (Patrimoine et Culture) :\n{context}\n\n"
                f"{query}\n\n"
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


    