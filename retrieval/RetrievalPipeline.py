class RetrievalPipeline:

    def __init__(self, retriever, context_builder, location_filter=None):
        self.retriever = retriever
        self.context_builder = context_builder
        self.location_filter = location_filter  # optionnel : détecte le lieu dans la question

    def run(self, question, where=None):

        # pas de filtre donné : on cherche un département / une région dans la question
        if where is None and self.location_filter:
            where = self.location_filter.where(question)

        retrieved_chunks = self.retriever.retrieve(question, where=where)
        context = self.context_builder.build_context(retrieved_chunks)

        return {
            "question": question,
            "chunks": retrieved_chunks,
            "context": context
        }