"""
Chunking : données brutes -> chunks prêts pour embeddings.py

    1. ENTRÉE   : on lit les données (via parser.py, ou nos lecteurs pour ce qu'il ne gère pas)
                  -> documents {"content": dict (ligne de tableau) | str (texte), "metadata": {...}}
    2. CHUNKING : chaque document -> Chunk(id, text, metadata)
                  - ligne de tableau -> "Nom : X | Ville : Y"  (1 ligne = 1 chunk)
                  - texte (PDF)      -> découpé en morceaux de ~1000 caractères
                  - fréquentation    -> 1 chunk par musée, toutes années regroupées
    3. SORTIE   : chunks écrits en JSONL dans data/chunks/

"text" est embeddé ; "metadata" sert à filtrer et citer la source.
Lancer : env/bin/python chunking.py
"""
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

MAX_CHARS = 1000  # taille max d'un chunk (~250 tokens)
OVERLAP = 150     # chevauchement entre deux morceaux d'un même texte

# Pour chaque dataset : colonnes mises dans le texte, colonnes gardées en métadonnées,
# colonne identifiant. Dataset absent -> toutes les colonnes vont dans le texte.
DATASETS = {
    "basilic": {"id": "Identifiant origine",
                "text": ["Nom", "Type équipement ou lieu", "Label et appellation", "Domaine",
                         "Adresse postale", "libelle_geographique", "Département", "Région"],
                "meta": ["code_insee", "libelle_geographique", "Département", "Région", "Latitude", "Longitude"]},
    "joconde": {"id": "Reference",
                "text": ["Titre", "Denomination", "Auteur", "Domaine", "Epoque", "Sujet_Represente",
                         "Description", "Nom_officiel_musee", "Ville", "Region"],
                "meta": ["Code_Museofile", "Nom_officiel_musee", "Ville", "Departement", "Region"]},
    "frequentation": {"id": "IDMuseofile",
                      "text": ["nom_du_musee", "ville", "departement", "region"],
                      "meta": ["IDMuseofile", "ville", "departement", "region"]},
    "films": {"text": ["titre", "nationalité", "entrées"], "meta": []},
}


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ===========================================================================
# 1. ENTRÉE
# ===========================================================================

# parser.py renvoie {nom_fichier: contenu}. On retrouve le dataset grâce au nom du fichier.
FILE_TO_DATASET = {"base-des-lieux": "basilic", "entrees_et_categories": "frequentation",
                   "joconde": "joconde", "films": "films", ".pdf": "pdf"}

# parser.py lit les nombres comme des nombres -> zéros de tête perdus ("01800" -> "1800")
ZERO_PAD = {"Code Postal": 5, "code_insee": 5, "codeInseeCommune": 5}


def from_parser(parsed):
    """Sortie de parser.py -> documents.
    - CSV : un grand str, une ligne par ligne de tableau ("col: val | col: val")
    - PDF : un dict {"Page 1": texte, ...}
    """
    for filename, content in parsed.items():
        name = filename.lower()
        dataset = next((ds for key, ds in FILE_TO_DATASET.items() if key in name), name)
        meta = {"dataset": dataset, "source": filename}

        if isinstance(content, dict):
            for page, text in content.items():
                yield {"content": text, "metadata": {**meta, "page": page}}
        else:
            for line in content.splitlines():
                if line.strip():
                    yield {"content": line_to_dict(line), "metadata": meta}


def line_to_dict(line):
    """ "Nom: Musée X | Code Postal: 1800" -> {"Nom": "Musée X", "Code Postal": "01800"} """
    row = {}
    for part in line.split(" | "):
        col, _, val = part.partition(": ")
        col = col.strip()
        if col in ZERO_PAD and val.isdigit():
            val = val.zfill(ZERO_PAD[col])
        row[col] = val.removesuffix(".0")  # "5186.0" -> "5186"
    return row


def read_joconde(path):
    """Joconde (1,2 Go, séparateur '|') : parser.py ne sait pas le lire.
    Lecture ligne par ligne pour ne pas tout charger en mémoire."""
    csv.field_size_limit(10**8)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="|"):
            if is_empty(row.get("Titre")) and is_empty(row.get("Denomination")):
                continue  # ni titre ni dénomination : chunk sans intérêt
            yield {"content": row, "metadata": {"dataset": "joconde", "source": str(path)}}


def read_films(path):
    """XLSX du CNC : parser.py l'ignore.
    Un onglet par année ; le tableau commence après quelques lignes de titre."""
    import pandas as pd
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    for year, sheet in sheets.items():
        if not year.strip().isdigit():  # ignore "Sommaire" etc.
            continue
        cells = sheet.fillna("").apply(lambda col: col.str.strip().str.lower())
        header = next((i for i in range(min(15, len(sheet))) if "titre" in cells.iloc[i].values), None)
        if header is None:
            continue
        # "entrées (millions)" contient en fait le nombre brut d'entrées (ex. 6549396)
        columns = [c.replace(" (millions)", "") for c in cells.iloc[header]]
        for values in sheet.iloc[header + 1:].values.tolist():
            row = {c: v for c, v in zip(columns, values) if c}
            if not is_empty(row.get("titre")):
                yield {"content": row, "metadata": {"dataset": "films", "source": str(path), "annee": year}}


# ===========================================================================
# 2. CHUNKING
# ===========================================================================

def chunk_documents(docs):
    """Documents -> chunks. Les lignes de fréquentation sont regroupées par musée à la fin."""
    museums = defaultdict(list)
    for i, doc in enumerate(docs):
        content, meta = doc["content"], doc["metadata"]
        if isinstance(content, str):
            yield from chunk_text(content, meta, i)
        elif meta["dataset"] == "frequentation":
            # sans id (2 musées), on identifie le musée par son nom/ville/région
            museum = content.get("IDMuseofile") or row_to_text(content, DATASETS["frequentation"]["text"])
            museums[museum].append((content, meta))
        else:
            yield from chunk_row(content, meta, i)

    for museum, rows in museums.items():
        yield chunk_museum(museum, rows)


def chunk_text(text, meta, i):
    """Texte libre (page de PDF) -> morceaux de MAX_CHARS."""
    page = str(meta.get("page", i)).replace("Page ", "p")  # "Page 3" -> "p3"
    for k, piece in enumerate(split_text(text)):
        if len(re.findall(r"\.{4,}", piece)) >= 2:  # sommaire ("Titre ........ 3") : bruit
            continue
        yield Chunk(f"{meta['dataset']}-{page}-{k}", piece, {**meta, "part": k})


def chunk_row(row, meta, i):
    """Ligne de tableau -> 1 chunk (plusieurs seulement si la ligne est très longue)."""
    cfg = DATASETS.get(meta["dataset"], {})
    text = row_to_text(row, cfg.get("text"))
    if "annee" in meta:  # films : l'année vient du nom de l'onglet
        text = f"Année {meta['annee']} | {text}"
    meta = {**meta, **pick(row, cfg.get("meta", []))}
    # i rendu unique car certains identifiants sources sont dupliqués
    base_id = f"{meta['dataset']}-{row.get(cfg.get('id'), i)}-{i}"
    for k, piece in enumerate(split_text(text)):
        yield Chunk(f"{base_id}-{k}", piece, meta)


def chunk_museum(museum, rows):
    """Toutes les années d'un musée -> 1 chunk :
    "nom du musee : Musée de Brou | ... | Fréquentation : 2014 : 94648 entrées ; 2015 : ..." """
    rows.sort(key=lambda r: str(r[0].get("annee")))
    first, meta = rows[0]
    cfg = DATASETS["frequentation"]
    years = " ; ".join(f"{row.get('annee')} : {row.get('total')} entrées" for row, _ in rows)
    text = f"{row_to_text(first, cfg['text'])} | Fréquentation : {years}"
    return Chunk(f"frequentation-{museum}", text, {**meta, **pick(first, cfg["meta"])})


# --- utilitaires -----------------------------------------------------------

def is_empty(value):
    return value is None or str(value).strip().lower() in {"", "nan", "none"}


def pick(row, columns):
    """Garde les colonnes demandées qui ne sont pas vides."""
    return {c: row[c] for c in columns if not is_empty(row.get(c))}


def row_to_text(row, columns=None):
    """{"Nom": "Musée X", "Ville": "Lyon"} -> "Nom : Musée X | Ville : Lyon" """
    return " | ".join(f"{c.replace('_', ' ')} : {str(row[c]).strip()}"
                      for c in (columns or row) if not is_empty(row.get(c)))


def split_text(text, size=MAX_CHARS, overlap=OVERLAP):
    """Coupe aux fins de phrase en morceaux <= size.
    Chaque morceau reprend la fin du précédent (overlap) pour garder le contexte."""
    text = re.sub(r"\s+", " ", text).strip()
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        while len(sentence) > size:  # phrase trop longue : coupe brute
            pieces.append(sentence[:size])
            sentence = sentence[size - overlap:]
        if current and len(current) + len(sentence) + 1 > size:
            pieces.append(current)
            current = current[-overlap:].split(" ", 1)[-1] + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    return pieces + [current] if current else pieces


# ===========================================================================
# 3. SORTIE
# ===========================================================================

def save_jsonl(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_jsonl(path):
    """Relit les chunks (pour embeddings.py)."""
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]


if __name__ == "__main__":
    from pathlib import Path
    from parser import RAGParser

    # parser.py fichier par fichier (parse_all bloquerait sur Joconde)
    p = RAGParser("data")
    basilic = Path("data/datasets/basilic/base-des-lieux-et-des-equipements-culturels.csv")
    musees = Path("data/datasets/musees/ENTREES_ET_CATEGORIES_DE_PUBLIC.csv")
    pdf = Path("data/datasets/basilic/20appellation-1.pdf")
    parsed = {basilic.name: p._parse_csv(basilic), musees.name: p._parse_csv(musees), pdf.name: p._parse_pdf(pdf)}

    sources = {
        "parser": from_parser(parsed),
        "films": read_films(next(Path("data/datasets/films").glob("*.xlsx"))),
        "joconde": read_joconde("data/datasets/joconde/joconde.csv"),
    }
    Path("data/chunks").mkdir(exist_ok=True)
    for name, docs in sources.items():
        save_jsonl(chunk_documents(docs), f"data/chunks/{name}.jsonl")
        print(name, "ok")
