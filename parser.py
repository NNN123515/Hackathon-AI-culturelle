from pypdf import PdfReader
import pandas as pd
from pathlib import Path
import docx

class RAGParser:
    def __init__(self, data_dir: str):
        """
        Initialise le parser pour les données textuelles locales.
        Args:
          -Chemin vers le dossier contenant les fichiers téléchargés
        """
        self.data_dir = Path(data_dir)

    def parse_all(self):
        """Parcourt tous les fichiers du dossier et extrait les textes structurés.
        Args:
            -Une liste de chaîne de caractères où chaque chaîne de caractère correspond à un document
        """
        all_texts = {}
        if not self.data_dir.exists():
            return {}

        # Parcours de tous les fichiers du répertoire
        for file_path in self.data_dir.glob("**/*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                print(f"Traitement du fichier : {file_path.name}")
                
                try:
                    content = ""

                    #filtre selon l'extension du fichier
                    if ext in {".txt", ".md"}:
                        content = self._parse_txt(file_path)
                    elif ext== ".pdf":
                        content = self._parse_pdf(file_path)
                    elif ext==".parquet":
                        content = self._parse_parquet(file_path)
                    else:
                        print(f"Format ignoré pour le fichier {file_path.name} : {ext}")
                        continue

                    if content.strip():
                        filename = file_path.name
                        all_texts[filename] = content
                    
                except Exception as e:
                    print(f"Erreur lors de la lecture de {file_path.name}: {e}")
                    
        return all_texts

    def _parse_txt(self, file_path: Path) -> str:
            """Parse un fichier texte (.txt) ou markdown (.md)."""
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Erreur lecture texte {file_path.name}: {e}")
                return ""

    def _parse_pdf(self, file_path: Path) -> str:
            """Extrait le texte d'un fichier PDF."""
            try:
                reader = PdfReader(str(file_path))
                text_pages = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_pages.append(text.strip())
                return "\n".join(text_pages)
            
            except ImportError:
                print("La bibliothèque 'pypdf' n'est pas installée.")
                return ""
            except Exception as e:
                print(f"Erreur lecture PDF {file_path.name}: {e}")
                return ""

    def _parse_word(self, file_path: Path) -> str:
        """
        Extrait le texte et les tableaux d'un fichier Word (.docx) 
        et retourne un contenu fluide et textuel pour le RAG.
        """

        try:
            doc = docx.Document(file_path)
        except Exception as e:
            print(f"Erreur de lecture du fichier Word {file_path.name}: {e}")
            return ""

        list_txts = []

        # Extraction des paragraphes de texte classique
        for para in doc.paragraphs:
            txt = para.text.strip()
            if txt:
                list_txts.append(txt)

        # Extraction du contenu des tableaux 
        for table in doc.tables:
            for row in table.rows:
                # Récupère le texte non vide de chaque cellule de la ligne
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    # Transforme la ligne du tableau en une phrase fluide et lisible
                    list_txts.append(" : ".join(row_cells))

        # Retourne l'ensemble fusionné
        return "\n".join(list_txts)

    
    def _parse_parquet(self, file_path: Path) -> str:
        """Parse un fichier Parquet d'évaluation de LLM et génère

        un texte narratif centré sur les modèles, les biais et les défauts.
        """
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"   ❌ Erreur de lecture Parquet {file_path.name}: {e}")
            return ""

        if df.empty:
            return ""

        # Nettoyage des noms de colonnes pour éviter les erreurs de casse
        df.columns = [str(c).strip().lower() for c in df.columns]

        lines = []
        for _, row in df.iterrows():
            # Extraction des éléments essentiels (Modèles, Prompt, Vainqueur)
            prompt = str(row.get("opening_msg", "Prompt non spécifié")).strip()
            if not prompt or prompt.lower() in ["none", "nan"]:
                prompt = "Prompt non spécifié"

            model_a = str(row.get("model_a_name", "Modèle A")).strip()
            model_b = str(row.get("model_b_name", "Modèle B")).strip()

            # Recherche de la réponse du modèle A (gère plusieurs variantes de noms de colonnes)
            resp_a = ""
            for col in ["response_a", "output_a", "answer_a", "text_a"]:
                val = row.get(col)
                if (
                    pd.notna(val)
                    and str(val).strip()
                    and str(val).lower() not in ["none", "nan"]
                ):
                    resp_a = str(val).strip()
                    break

            # Recherche de la réponse du modèle B
            resp_b = ""
            for col in ["response_b", "output_b", "answer_b", "text_b"]:
                val = row.get(col)
                if (
                    pd.notna(val)
                    and str(val).strip()
                    and str(val).lower() not in ["none", "nan"]
                ):
                    resp_b = str(val).strip()
                    break

            # Gestion propre et naturelle du vainqueur (évite l'affichage brut de 'None' ou 'nan')
            raw_winner = row.get("chosen_model_name")
            if pd.isna(raw_winner) or str(raw_winner).strip().lower() in [
                "none",
                "nan",
                "",
            ]:
                winner = "Aucun vainqueur (égalité ou non défini)"
            else:
                winner = str(raw_winner).strip()

            # Construction d'une phrase fluide (Agencée logiquement pour un meilleur chunking)
            narrative = f"Évaluation d'un prompt : « {prompt} ». Modèles comparés : {model_a} et {model_b}."

            if resp_a:
                narrative += f" Réponse de {model_a} : « {resp_a} »."
            if resp_b:
                narrative += f" Réponse de {model_b} : « {resp_b} »."

            narrative += f" Vainqueur : {winner}."

            # Détection ciblée des défauts et biais (booléens d'évaluation)
            defauts = []
            for side, model_name in [("a", model_a), ("b", model_b)]:
                if row.get(f"conv_incorrect_{side}"):
                    defauts.append(f"{model_name} (erreur factuelle/incorrect)")
                if row.get(f"conv_instructions_not_followed_{side}"):
                    defauts.append(f"{model_name} (consignes non respectées)")
                if row.get(f"conv_superficial_{side}"):
                    defauts.append(f"{model_name} (biais superficiel)")

            if defauts:
                narrative += f" Défauts ou biais identifiés : {', '.join(defauts)}."

            # Intégration des commentaires qualitatifs des évaluateurs s'ils existent
            for side in ["a", "b"]:
                comment = row.get(f"conv_comments_{side}")
            if pd.notna(comment) and str(comment).strip():
                narrative += f" Remarque évaluateur ({side.upper()}) : {str(comment).strip()}.\n"

            lines.append(narrative)

        return "\n".join(lines)
    
if __name__ == "__main__":
    DOSSIER_DONNEES = "./data"
    
    parser = RAGParser(data_dir=DOSSIER_DONNEES)
    file_path = Path("./data/votes.parquet")
    print(parser._parse_parquet(file_path))