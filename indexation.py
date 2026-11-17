import json
import lancedb
from chunking import load_jsonl, Chunk

# noms possibles de chaque champ selon le dataset
REGION = ["Région", "region", "Region"]
DEPARTEMENT = ["Département", "departement", "Departement"]
VILLE = ["libelle_geographique", "ville", "Ville"]

def first_value(meta, keys):
    """Première valeur trouvée parmi keys, sinon ""."""
    for k in keys:
        if meta.get(k):
            return meta[k]
    return ""

def to_row(chunk, vector):
    m = chunk.metadata
    return {
        "id": chunk.id,
        "text": chunk.text,
        "vector": vector,
        "dataset": m.get("dataset", ""),
        "region": first_value(m, REGION),
        "departement": first_value(m, DEPARTEMENT),
        "ville": first_value(m, VILLE),
        "metadata": json.dumps(m, ensure_ascii=False),  # tout le reste, pour ne rien perdre
    }


class VectorStore:
    def __init__(self, path="data/index", table="chunks"):
        self.db= lancedb.connect(path)
        self.name=table
        self.table=None

    def build(self, chunks, embedding_model, batch_size=2000):
        for start in range(0,len(chunks), batch_size):
            batch=chunks[start:start+batch_size]
            vectors=embedding_model.embed_documents(batch)
            rows = [to_row(c, v.tolist())for c, v in zip(batch, vectors)]
            if self.table is None:
                self.table= self.db.create_table(self.name, data=rows, mode="overwrite")
            else:
                self.table.add(rows)
            print(f"{start + len(batch)}/{len(chunks)}")    


    def load(self):
        self.table=self.db.open_table(self.name)


    def search(self, query_embedding, top_k=5, where=None):
        query= self.table.search(query_embedding).limit(top_k)
        if where:
            query = query.where(where, prefilter=True)
        return [Chunk(r["id"], r["text"], json.loads(r["metadata"])) for r in query.to_list()]    


if __name__ == "__main__":
    import lancedb
    from embedding import EmbeddingModel

    chunks = load_jsonl("data/chunks/parser.jsonl")+ load_jsonl("data/chunks/films.jsonl")
    print(len(chunks),"chunks à indexer")
    model = EmbeddingModel()

    store= VectorStore()
    store.build(chunks, model)

    store2=VectorStore()
    store2.load()
    for c in store2.search(model.embed_query("église"), top_k=3, where="departement = 'Ain'"):
        print(c.metadata.get("Département"), "|", c.text[:90])

