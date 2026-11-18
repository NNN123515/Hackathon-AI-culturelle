from dotenv import load_dotenv
from embedding import EmbeddingModel
from indexation import VectorStore
from retrieval.Retriever import Retriever
from retrieval.ContextBuilder import ContextBuilder
from retrieval.RetrievalPipeline import RetrievalPipeline
from model_generator.ModelGeneratorAPI import ModelGeneratorAPI

load_dotenv()  # Charger les variables d'environnement depuis le fichier .env

class RAGApplication:

    def __init__(self, top_k=5):
        # Initialiser le modèle d'embedding et le VectorStore
        self.embedding_model = EmbeddingModel()
        self.store = VectorStore()
        self.store.load()

        # Configurer le pipeline de retrieval
        self.retriever = Retriever(self.embedding_model, self.store, top_k=top_k)
        self.context_builder = ContextBuilder()
        self.pipeline = RetrievalPipeline(self.retriever, self.context_builder)

        # Initialiser le modèle de génération
        self.generator = ModelGeneratorAPI()

    def answer_query(self, question: str, where: str = None):
        """Exécute la recherche (retriever) pour obtenir le contexte,
        puis génère la réponse en streaming.
        """        
        # Récupérer le contexte via le pipeline
        retrieval_result = self.pipeline.run(question, where=where)
        context = retrieval_result["context"]

        if not context.strip():
            print("Aucun contexte trouvé dans la base de données.")
            return

        print("\n--- Génération de la réponse (Streaming) ---")
        # Passer le contexte et la question au générateur
        for chunk_text in self.generator.generate_response_stream(query=question, context=context):
            print(chunk_text, end="", flush=True)
        print("\n")
    
    def answer_query(self, question: str, where: str = None):
        """Exécute la recherche (retriever) pour obtenir le contexte,
        puis génère la réponse en streaming classique.
        """        
        retrieval_result = self.pipeline.run(question, where=where)
        context = retrieval_result["context"]

        if not context.strip():
            print("Aucun contexte trouvé dans la base de données.")
            return

        print("\n--- Génération de la réponse (Streaming classique) ---")
        for chunk_text in self.generator.generate_response_stream(query=question, context=context):
            print(chunk_text, end="", flush=True)
        print("\n")

    def answer_query_dual_stream(self, query1: str, query2: str, where: str = None):
        """Exécute deux recherches (sur query1 et query2),
        puis génère les deux réponses via generate_dual_stream.
        """
        #  Lancer la recherche pour la première requête
        result1 = self.pipeline.run(query1, where=where)
        context1 = result1.get("context", "")

        # Lancer la recherche pour la seconde requête
        result2 = self.pipeline.run(query2, where=where)
        context2 = result2.get("context", "")

        print("\n--- Génération du Dual Stream ---")
        
        # Appel du générateur en mode dual stream 
        for chunk in self.generator.generate_dual_stream(query1=query1, context1=context1, query2=query2, context2=context2):
            print(chunk, end="", flush=True)
            
        print("\n")

if __name__ == "__main__":
    # Instanciation de l'application RAG
    rag = RAGApplication(top_k=5)

    # Exemple de question avec un filtre optionnel sur le département
    question1 = "Quels sont les musées ou églises remarquables dans l'Ain ? Les musées et églises remarquables dans l'Ain sont le Musée de Brou à Bourg-en-Bresse, l'Abbaye d'Ambronay, la Basilique de Brou, le Château de Fléchères, et l'Église Saint-Nicolas à Nantua."
    question2 = "Quels sont les musées ou églises remarquables dans le Rhône ? Les musées et églises remarquables dans le Rhône sont le Musée des Beaux-Arts de Lyon, la Basilique Notre-Dame de Fourvière, l'Église Saint-Nizier, le Musée d'Art Contemporain de Lyon, et l'Église Saint-Paul."

    # Exécution de la recherche et de la génération
    rag.answer_query_dual_stream(
        query1=question1,
        query2=question2
    )