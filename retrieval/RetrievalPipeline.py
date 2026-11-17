class RetrievalPipeline:

    def __init__(self, retriever, context_builder):
        self.retriever = retriever
        self.context_builder = context_builder

    def run(self, question):

        retrieved_chunks = self.retriever.retrieve(question)
        context = self.context_builder.build_context(retrieved_chunks)

        return {
            "question": question,
            "chunks": retrieved_chunks,
            "context": context
        }