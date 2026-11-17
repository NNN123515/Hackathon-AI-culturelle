class ContextBuilder:

    def build_context(self, chunks):

        if not chunks:
            return ""

        context_parts = []

        for i, chunk in enumerate(chunks):
            context_parts.append(f"""[Source {i + 1}]{chunk.text}""")

        return "\n\n".join(context_parts)