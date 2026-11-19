from dotenv import load_dotenv
from embedding import EmbeddingModel
from indexation import VectorStore
from retrieval.Retriever import Retriever
from retrieval.ContextBuilder import ContextBuilder
from retrieval.RetrievalPipeline import RetrievalPipeline
from location_filter import LocationFilter
from model_generator.ModelGeneratorAPI import ModelGeneratorAPI
from model_generator.ModelGeneratorLocal import ModelGeneratorLocal

load_dotenv()  # Charger les variables d'environnement depuis le fichier .env

class RAGApplication:

    def __init__(self, top_k=5, model_type="API"):
        # Initialiser le modèle d'embedding et le VectorStore
        self.embedding_model = EmbeddingModel()
        self.store = VectorStore()
        self.store.load()

        # Configurer le pipeline de retrieval
        self.retriever = Retriever(self.embedding_model, self.store, top_k=top_k)
        self.context_builder = ContextBuilder()
        self.pipeline = RetrievalPipeline(self.retriever, self.context_builder,
                                          location_filter=LocationFilter(self.store))

        # Petit modèle qui liste les lieux cités dans une réponse (pour les vérifier un par un)
        self.extractor = ModelGeneratorAPI(model_name="gemini-3.5-flash-lite")

        # Initialiser le modèle de génération
        if model_type == "API":
            self.generator = ModelGeneratorAPI()
        else:
            self.generator = ModelGeneratorLocal()

    def change_model_type(self, target_type: str, model_name: str = None):
        if target_type == "Local":
            # S'il y a un modèle fourni, on l'initialise avec. Sinon, comportement par défaut.
            if model_name:
                self.generator = ModelGeneratorLocal(model_name=model_name)
            else:
                self.generator = ModelGeneratorLocal()
                
        elif target_type == "API":
            if model_name:
                self.generator = ModelGeneratorAPI(model_name=model_name)
            else:
                self.generator = ModelGeneratorAPI()
        
    def list_places(self, answer, max_places=15):
        """Demande au petit modèle la liste des lieux cités : [(nom, commune), ...]"""
        prompt = (
            "Liste les lieux, monuments, musées et œuvres cités dans le texte ci-dessous.\n"
            "Une ligne par élément, au format : nom | commune\n"
            "Laisse la commune vide si le texte ne la donne pas. N'écris rien d'autre.\n\n"
            f"Texte :\n{answer}"
        )
        text = "".join(self.extractor.generate_response_stream(query=prompt))
        places = []
        for line in text.splitlines():
            if "|" not in line or "[Erreur API" in line:
                continue
            name, commune = [part.strip(" *-•\t") for part in line.split("|", 1)]
            if name:
                places.append((name, commune))
        return places[:max_places]

    def verification_context(self, question, answer, where=None, per_place=2, max_chars=400):
        """Contexte pour le juge : sources générales + sources trouvées pour chaque lieu cité."""
        where = where or self.pipeline.location_filter.where(question)

        # 1. sources générales (servent aussi à repérer les oublis)
        chunks = self.pipeline.run(question, where=where)["chunks"]

        # 2. pour chaque lieu cité : d'abord dans sa commune, sinon dans le département/la région
        for name, commune in self.list_places(answer):
            vector = self.embedding_model.embed_query(f"{name} {commune}")
            found = []
            if commune:
                commune_sql = commune.replace("'", "''").lower()
                found = self.store.search(vector, top_k=per_place, where=f"lower(ville) = '{commune_sql}'")
            if not found:
                found = self.store.search(vector, top_k=per_place, where=where)
            chunks += found

        # 3. on enlève les doublons et on raccourcit les textes trop longs
        seen, parts = set(), []
        for chunk in chunks:
            if chunk.id not in seen:
                seen.add(chunk.id)
                parts.append(f"[Source {len(parts) + 1}]{chunk.text[:max_chars]}")
        return "\n\n".join(parts)

    def answer_query(self, question: str, where: str = None, rag=True):
        """Exécute la recherche (retriever) pour obtenir le contexte,
        puis génère la réponse en streaming classique.
        """
        if rag is True:        
            retrieval_result = self.pipeline.run(question, where=where)
            context = retrieval_result["context"]
        else:
            context = None

        print("\n--- Génération de la réponse (Streaming classique) ---")
        for chunk_text in self.generator.generate_response_stream(query=question, context=context):
            yield chunk_text

if __name__ == "__main__":
    # Instanciation de l'application RAG
    rag = RAGApplication(top_k=5, model_type="Local")  # Choisir "API" ou "Local" selon le modèle souhaité

    # Exemple de question avec un filtre optionnel sur le département
    question = "Quels sont les musées ou églises remarquables dans l'Ain ? Les musées et églises remarquables dans l'Ain sont le Musée de Brou à Bourg-en-Bresse, l'Abbaye d'Ambronay, la Basilique de Brou, le Château de Fléchères, et l'Église Saint-Nicolas à Nantua."

    # Exécution de la recherche et de la génération
    for chunk in rag.answer_query(question=question):
        print(chunk, end="", flush=True)
    