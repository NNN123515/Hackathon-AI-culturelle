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

    def build(self, chunks, embedding_model, batch_size=2000, overwrite=True):
        if not overwrite:
            self.load()   # on rouvre la table existante : les paquets iront dans add()
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

    def add_jsonl(self, path, dataset, embedding_model, batch_size=2000):
        """Ajoute un gros fichier de chunks à la table existante, sans overwrite.
        Lu par paquets (pas tout en mémoire). Si on relance après un arrêt,
        on saute les chunks déjà indexés : pas de doublons."""
        self.load()
        done = self.table.count_rows(filter=f"dataset = '{dataset}'")
        print(f"{done} chunks '{dataset}' déjà indexés, on reprend à partir de là")
        batch = []
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f):
                if n < done:
                    continue
                batch.append(Chunk(**json.loads(line)))
                if len(batch) == batch_size:
                    self.build(batch, embedding_model, overwrite=False)
                    print(f"{n + 1} lignes traitées", flush=True)
                    batch = []
        if batch:
            self.build(batch, embedding_model, overwrite=False)
        print("terminé :", self.table.count_rows(), "lignes dans la table")


if __name__ == "__main__":
    from embedding import EmbeddingModel

    model = EmbeddingModel()
    store = VectorStore()
    store.add_jsonl("data/chunks/joconde.jsonl", "joconde", model)

    for c in store.search(model.embed_query("tableau impressionniste"), top_k=3, where="dataset = 'joconde'"):
        print(c.metadata.get("Ville"), "|", c.text[:90])

