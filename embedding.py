
from sentence_transformers import SentenceTransformer
from chunking import load_jsonl
import numpy as np


class EmbeddingModel:

    DIMENSION = 384

    def __init__(self, model=None):

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(
                "intfloat/multilingual-e5-small",
                device="cpu"
            )

    def embed_documents(self, chunks):

        if not chunks:
            return np.empty((0, self.DIMENSION), dtype=np.float32)

        texts = []

        for chunk in chunks:
            if not chunk.text or not chunk.text.strip():
                raise ValueError("Le texte du chunk ne peut pas être vide")

            texts.append("passage: " + chunk.text)

        embeddings = self.model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings

    def embed_query(self, question):

        if not question or not question.strip():
            raise ValueError("La question ne peut pas être vide")

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