"""
Détecte un département ou une région dans une question et construit le filtre `where`.

    "Quelles églises visiter dans l'Ain ?"  ->  "departement IN ('Ain')"

Les noms viennent de l'index lui-même (pas de liste écrite à la main).
On compare sans accents, majuscules ni tirets, car l'index contient plusieurs
écritures d'une même région ("Grand Est" / "Grand-Est") : on les met toutes dans le filtre.
"""
import re
import unicodedata
from collections import defaultdict

# Départements qui sont aussi des mots courants ("le plus cher", "au nord") :
# on ne les accepte que précédés de "dans le/la/l'", "en", "du", "de la", "de l'"...
AMBIGUOUS = {"ain", "aube", "aude", "cher", "eure", "gard", "gers", "indre", "jura", "loir et cher",
             "lot", "manche", "marne", "meuse", "nord", "oise", "orne", "somme", "var", "vienne"}
PREFIX = r"(?:dans (?:le |la |l |les )?|en |du |des |de la |de l )"


def normalize(text):
    """ "Île-de-France" -> "ile de france" """
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # enlève les accents
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


class LocationFilter:

    def __init__(self, store):
        """Lit une fois les noms de départements et de régions présents dans l'index."""
        rows = store.table.search().select(["region", "departement"]).limit(store.table.count_rows()).to_pandas()
        self.names = {}  # (colonne, nom normalisé) -> écritures d'origine
        for column in ["departement", "region"]:
            spellings = defaultdict(set)
            for value in set(rows[column]):
                if value:
                    spellings[normalize(value)].add(value)
            for norm, originals in spellings.items():
                self.names[(column, norm)] = sorted(originals)

    def where(self, question):
        """Renvoie le filtre SQL pour le lieu trouvé dans la question, ou None."""
        q = " " + normalize(question) + " "
        # du plus long au plus court : "haute marne" doit gagner sur "marne"
        for (column, norm), originals in sorted(self.names.items(), key=lambda x: -len(x[0][1])):
            pattern = (PREFIX if norm in AMBIGUOUS else " ") + re.escape(norm) + " "
            if re.search(pattern, q):
                values = ", ".join("'" + v.replace("'", "''") + "'" for v in originals)
                return f"{column} IN ({values})"
        return None


if __name__ == "__main__":
    from indexation import VectorStore

    store = VectorStore()
    store.load()
    lf = LocationFilter(store)
    for q in ["Quelles églises visiter dans l'Ain ?",
              "Les musées du Grand-Est",
              "Que voir en Ile de France ?",
              "Châteaux de la Haute-Marne",
              "Quel est le musée le plus cher ?",
              "Musées dans le Nord",
              "Quels films ont fait plus d'un million d'entrées ?"]:
        print(f"{q!r:55} -> {lf.where(q)}")
