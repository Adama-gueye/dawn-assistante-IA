from pathlib import Path

import streamlit as st
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "data", "pediatrie.pdf")


st.set_page_config(page_title="DAWN", page_icon="D", layout="wide")
st.title("DAWN")
st.caption("Assistant intelligent d'aide a la decision medicale base sur un RAG multimodal")


@st.cache_resource(show_spinner=True)
def load_assistant(pdf_path: str):
    from dawn.config import DawnConfig
    from dawn.generator import DawnAssistant

    config = DawnConfig(pdf_path=Path(pdf_path))
    return DawnAssistant(config)


pdf_path = st.text_input("Chemin du document PDF", value="../data/pediatrie.pdf")
question = st.text_area(
    "Question medicale",
    placeholder="Exemple : Quand faut-il referer un enfant atteint d'anemie ?",
)

if st.button("Analyser", type="primary"):
    if not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        try:
            assistant = load_assistant(pdf_path)
            result = assistant.answer(question)
        except Exception as exc:
            st.error(
                "Le backend IA ne peut pas demarrer. Verifiez la configuration du provider, les dependances et la variable d'environnement de la cle API."
            )
            st.exception(exc)
        else:
            st.subheader("Reponse structuree")
            st.write(result["answer"])

            st.subheader("Sources retenues")
            for source in result["sources"]:
                st.write(f"page {source['page']} | score={source['score']:.3f}")
