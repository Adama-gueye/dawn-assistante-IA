from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_PATH = BASE_DIR / "data"

st.set_page_config(page_title="DAWN", page_icon="D", layout="wide")
st.title("DAWN")
st.caption("Assistant intelligent d'aide a la decision medicale base sur un RAG multimodal")


@st.cache_resource(show_spinner=True)
def load_assistant(knowledge_path: str, corpus_signature: str):
    from dawn.config import DawnConfig
    from dawn.generator import DawnAssistant

    config = DawnConfig(knowledge_path=Path(knowledge_path))
    return DawnAssistant(config)

def compute_corpus_signature(knowledge_path: Path) -> str:
    if knowledge_path.is_file():
        return f"{knowledge_path}:{knowledge_path.stat().st_mtime_ns}"

    if not knowledge_path.exists():
        return str(knowledge_path)

    pdf_files = sorted(knowledge_path.rglob("*.pdf"))
    parts = [f"{pdf}:{pdf.stat().st_mtime_ns}" for pdf in pdf_files]
    return "|".join(parts)


knowledge_path = st.text_input("Chemin du dossier ou PDF", value=str(DEFAULT_KNOWLEDGE_PATH))
mode = st.selectbox("Mode DAWN", options=["medecin", "patient"], index=0)
if st.button("Recharger la base documentaire"):
    load_assistant.clear()
    st.success("Cache vide. La base sera rechargee au prochain test.")
question = st.text_area(
    "Question",
    placeholder="Exemple : C'est quoi le paludisme ? ou Enfant avec fievre et vomissements depuis 2 jours.",
)

if st.button("Analyser", type="primary"):
    if not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        try:
            corpus_signature = compute_corpus_signature(Path(knowledge_path))
            assistant = load_assistant(knowledge_path, corpus_signature)
            result = assistant.answer(question, mode=mode)
        except Exception as exc:
            st.error(
                "Le backend IA ne peut pas demarrer. Verifiez la configuration du provider, les dependances et la variable d'environnement de la cle API."
            )
            st.exception(exc)
        else:
            st.subheader("Etat du corpus")
            col1, col2, col3 = st.columns(3)
            col1.metric("Documents charges", len(assistant.loaded_documents))
            col2.metric("Documents ignores", len(assistant.skipped_documents))
            col3.metric(
                "Pages indexees",
                sum(document["pages_loaded"] for document in assistant.loaded_documents),
            )

            if assistant.loaded_documents:
                st.caption("Documents exploites par le RAG")
                st.dataframe(
                    [
                        {
                            "Document": document["source_name"],
                            "Pages chargees": document["pages_loaded"],
                            "Chemin": document["source_path"],
                        }
                        for document in assistant.loaded_documents
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            if assistant.skipped_documents:
                st.warning("Certains PDF ont ete ignores car aucun texte exploitable n'a ete extrait.")
                st.caption("Documents ignores")
                st.dataframe(
                    [
                        {
                            "Document": document["source_name"],
                            "Raison": document["reason"],
                            "Chemin": document["source_path"],
                        }
                        for document in assistant.skipped_documents
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            st.subheader(f"Reponse mode {mode}")
            st.write(result["answer"])

            st.subheader("Sources retenues")
            for source in result["sources"]:
                st.write(f"{source['source_name']} | page {source['page']} | score={source['score']:.3f}")
