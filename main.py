from embedding import EmbeddingModel
from indexation import VectorStore
from retrieval.Retriever import Retriever
from retrieval.ContextBuilder import ContextBuilder
from retrieval.RetrievalPipeline import RetrievalPipeline
from model_generator import ModelGenerator

store = VectorStore()
store.load()
pipeline = RetrievalPipeline(Retriever(EmbeddingModel(), store, top_k=5), ContextBuilder())
generator = ModelGenerator()

question = "Quelles églises visiter dans l'Ain ?"
answer_a = "La basilique d'Ars-sur-Formans et l'église de Brou."
answer_b = "La cathédrale Notre-Dame de Paris."

context_a = pipeline.run(answer_a)["context"]
context_b = pipeline.run(answer_b)["context"]

for tag, morceau in generator.generate_dual_stream(
    f"Question : {question}\nRéponse : {answer_a}", context_a,
    f"Question : {question}\nRéponse : {answer_b}", context_b,
):
    print(morceau, end="", flush=True)
