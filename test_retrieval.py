from embedding import EmbeddingModel
from indexation import VectorStore
from retrieval.Retriever import Retriever
from retrieval.ContextBuilder import ContextBuilder
from retrieval.RetrievalPipeline import RetrievalPipeline

store = VectorStore()
store.load()

retriever = Retriever(EmbeddingModel(), store, top_k=3)
pipeline = RetrievalPipeline(retriever, ContextBuilder())

result = pipeline.run("églises historiques", where="departement = 'Ain'")
print(result["context"])
