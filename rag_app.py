from dotenv import load_dotenv
from embedding import EmbeddingModel
from indexation import VectorStore
from retrieval.Retriever import Retriever
from retrieval.ContextBuilder import ContextBuilder
from retrieval.RetrievalPipeline import RetrievalPipeline
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
        self.pipeline = RetrievalPipeline(self.retriever, self.context_builder)

        # Initialiser le modèle de génération
        if model_type == "API":
            self.generator = ModelGeneratorAPI()
        else:
            self.generator = ModelGeneratorLocal()

    def change_model_type(self, new_model_type: str):
        """Change le type de modèle utilisé pour la génération de réponses."""
        if new_model_type == "API":
            self.generator = ModelGeneratorAPI()
        else:
            self.generator = ModelGeneratorLocal()
    
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
    