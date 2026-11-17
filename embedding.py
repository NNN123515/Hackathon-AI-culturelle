
from sentence_transformers import SentenceTransformer
#from chunking import load_jsonl


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

    embedding_model = EmbeddingModel()

    chunks = [
        "Napoléon était petit.",
        "J'ai faim."
    ]

    #chunks = list(load_jsonl("data/chunks/basilic.jsonl"))          

    question = "Qui a peint la Joconde ?"

    document_vectors = embedding_model.embed_documents(chunks)
    question_vector = embedding_model.embed_query(question)

    print("Nombre de chunks :", len(chunks))
    print("Dimensions des embeddings :", document_vectors.shape)
    print("Dimensions de la question :", question_vector.shape)
