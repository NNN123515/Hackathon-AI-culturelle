class Retriever:

    def __init__(self, embedding_model, vector_store, top_k=5):

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question):

        query_embedding = self.embedding_model.embed_query(question)
        results = self.vector_store.search(query_embedding, top_k=self.top_k)

        return results