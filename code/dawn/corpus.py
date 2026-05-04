from __future__ import annotations

from pathlib import Path
import unicodedata


CORPUS_FOLDERS = {
    "anemie": "pediatrie/anemie",
    "paludisme": "pediatrie/paludisme",
    "accouchement": "gyneco_obstetrique/accouchement_travail",
    "grossesse": "gyneco_obstetrique/consultation_prenatale",
    "preeclampsie": "gyneco_obstetrique/hta_preeclampsie_eclampsie",
    "hemorragie": "gyneco_obstetrique/hemorragies_obstetricales",
    "respiratoire": "pediatrie/detresse_respiratoire",
    "deshydratation": "pediatrie/diarrhee_deshydratation",
    "malnutrition": "pediatrie/nutrition_malnutrition",
}

DEFAULT_PROTOCOL_FOLDER = "communs/protocoles_hospitaliers"
DEFAULT_GENERAL_FOLDER = "communs/recommandations_generales"
DEFAULT_TRIAGE_FOLDER = "communs/triage_signes_gravite"


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return normalized.encode("ascii", "ignore").decode("ascii")


def suggest_storage_folder(filename: str) -> str:
    normalized = normalize_text(filename)

    protocol_markers = {
        "protocole",
        "protocoles",
        "guideline",
        "guidelines",
        "guide",
        "recommandation",
        "recommandations",
        "prise_en_charge",
        "prise en charge",
    }
    if any(marker in normalized for marker in protocol_markers):
        return DEFAULT_PROTOCOL_FOLDER

    triage_markers = {
        "triage",
        "gravite",
        "urgence",
        "urgences",
        "risque",
    }
    if any(marker in normalized for marker in triage_markers):
        return DEFAULT_TRIAGE_FOLDER

    for topic, folder in CORPUS_FOLDERS.items():
        if topic in normalized:
            return folder

    return DEFAULT_GENERAL_FOLDER


def expected_corpus_directories(knowledge_path: Path) -> list[Path]:
    unique_dirs = {
        DEFAULT_GENERAL_FOLDER,
        DEFAULT_PROTOCOL_FOLDER,
        DEFAULT_TRIAGE_FOLDER,
        *CORPUS_FOLDERS.values(),
    }
    return [knowledge_path / relative_dir for relative_dir in sorted(unique_dirs)]
