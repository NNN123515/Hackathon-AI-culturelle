
from sentence_transformers import SentenceTransformer
from chunking import load_jsonl


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-small",
            device="cpu"
        )

    def embed_documents(self, chunks):
        texts = ["passage: " + chunk for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings

    def embed_query(self, question):
        text = "query: " + question

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding


if __name__ == "__main__":

    #chunks = [
    #    "Napoléon était petit.",
    #    "J'ai faim."
    #]

    chunks = list(load_jsonl("data/chunks/basilic.jsonl"))          
    print("Nombre de chunks :", len(chunks))

    embedding_model = EmbeddingModel()

    document_vectors = embedding_model.embed_documents(chunks)
    print("Dimensions des embeddings :", document_vectors.shape)
    
    question = "Qui a peint la Joconde ?"
    query_vector = embedding_model.embed_query(question)
    print("Dimensions de la question :", query_vector.shape)