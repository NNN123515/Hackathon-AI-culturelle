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
        Renvoie un dictionnaire dont les clés sont les noms de fichiers et les valeurs sont le contenu extrait (contenu textuel ou dictionnaire pour les fichiers multi-pages comme les PDF ou Excel).
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
                    elif ext==".docx":
                        content = self._parse_word(file_path)
                    elif ext==".csv":
                        content = self._parse_csv(file_path)
                    else:
                        print(f"Format ignoré pour le fichier {file_path.name} : {ext}")
                        continue

                    if isinstance(content, dict):
                       all_texts[file_path.name] = content

                    elif content.strip():
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

    def _parse_pdf(self, file_path: Path) -> dict:
      """Extrait le texte d'un fichier PDF et retourne un dictionnaire

      dont les clés sont 'Page 1', 'Page 2', etc., et les valeurs le texte associé.
      """
      pages_dict = {}
      try:
        reader = PdfReader(str(file_path))

        for page_num, page in enumerate(reader.pages, start=1):
          text = page.extract_text()
          if text:
            cleaned_text = text.strip()
            if cleaned_text:
              pages_dict[f"Page {page_num}"] = cleaned_text

      except ImportError:
        print("La bibliothèque 'pypdf' n'est pas installée.")
      except Exception as e:
        print(f"Erreur lecture PDF {file_path.name}: {e}")

      return pages_dict

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
            print(f"Erreur de lecture Parquet {file_path.name}: {e}")
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

    def _parse_csv(self, file_path: Path) -> str:
      """Parse un fichier CSV de formatage et gérant les différents séparateurs (',', ';').
      """
      df = None
      # Tentative de lecture avec différents séparateurs et moteurs tolérants
      for sep in [",", ";", "\t"]:
        try:
          df = pd.read_csv(
              file_path,
              sep=sep,
              on_bad_lines="skip",  
              engine="python", 
              encoding="utf-8",
          )
          if len(df.columns) > 1:
            break
        except Exception:
          continue

      if df is None or df.empty:
        print(f"Impossible de lire le CSV : {file_path.name}")
        return ""

      # On filtre les collones avec trop peu de données---
      total_rows = len(df)
      threshold = total_rows * 0.05
      valid_cols = []

      #on parcourt le dataset une fois pour vérifier le nombre de valeurs valides par colonne
      for col in df.columns:
        valid_count = 0
        for val in df[col]:
          if pd.notna(val):
            val_str = str(val).strip().lower()
            if val_str and val_str not in {"none", "nan", "null", ""}:
              valid_count += 1
        # On ne conserve la colonne que si elle dépasse le seuil de 5%
        if valid_count >= threshold:
          valid_cols.append(col)

      # On restreint le DataFrame aux colonnes pertinentes
      df = df[valid_cols]

        # On parcourt le DataFrame pour formater chaque ligne en texte lisible
      formatted_lines = []
      for _, row in df.iterrows():
        row_blocks = []
        for col, val in row.items():
          if pd.notna(val) and str(val).strip():
            val_str  = str(val).strip()

            # Filtre les cellules vides, NaN, None ou textuelles vides
            if val_str and val_str.lower() not in {"none", "nan", "null", ""}:
                row_blocks.append(f"{col}: {str(val).strip()}")

        if row_blocks: #on sépare les colonnes par " | " pour une meilleure lisibilité
          formatted_lines.append(" | ".join(row_blocks))

      return "\n".join(formatted_lines)

    def _parse_excel(self, file_path: Path) -> str:
      """Parse toutes les feuilles d'un fichier Excel (.xlsx ou .xls)

      et transforme chaque ligne en un format textuel propre et lisible.
      """
      try:
        #  Ouvre le classeur Excel pour lister les feuilles
        xls = pd.ExcelFile(file_path)
      except Exception as e:
        print(f" Erreur d'ouverture du fichier Excel {file_path.name}: {e}")
        return ""

      all_sheets_text = {}

      # Parcours de chaque feuille du fichier Excel
      for sheet_name in xls.sheet_names:
        try:
          df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception:
          continue

        if df.empty:
          continue

        formatted_rows = []
        for _, row in df.iterrows():
          row_blocks = []
          for col, val in row.items():
            # Filtre les cellules vides, NaN, None ou textuelles vides

            if pd.notna(val):
              val_str = str(val).strip()
              if val_str and "Unnamed" in col:
                row_blocks.append(f"{val_str}")
              elif val_str and val_str.lower() not in {"none", "nan", "null"} :
                row_blocks.append(f"{col}: {val_str}")

          if row_blocks:
            formatted_rows.append(" | ".join(row_blocks))

        if formatted_rows:
          # Ajoute un titre pour indiquer la feuille et concatène les lignes formatées
          sheet_content = f"Feuille : {sheet_name}\n" + "\n".join(
              formatted_rows
          )
          all_sheets_text[sheet_name] = sheet_content

      return all_sheets_text

if __name__ == "__main__":
    DOSSIER_DONNEES = "./data"
    
    parser = RAGParser(data_dir=DOSSIER_DONNEES)
    parser.parse_all()