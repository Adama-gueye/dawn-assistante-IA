from pathlib import Path

import streamlit as st

from dawn.corpus import expected_corpus_directories, suggest_storage_folder
from dawn.eval_questions import EVAL_QUESTIONS_FR

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_PATH = BASE_DIR / "data"
ASSISTANT_CACHE_VERSION = "2026-04-27-rag-corpus-fr-v5"

st.set_page_config(page_title="DAWN", page_icon="D", layout="wide")
st.title("DAWN")
st.caption("Assistant intelligent d'aide a la decision medicale base sur un RAG multimodal")


@st.cache_resource(show_spinner=True)
def load_assistant(
    knowledge_path: str,
    corpus_signature: str,
    cache_version: str,
    provider: str,
    generation_model: str,
    ollama_base_url: str,
):
    from dawn.config import DawnConfig
    from dawn.generator import DawnAssistant

    config = DawnConfig.from_env(knowledge_path=Path(knowledge_path))
    config.provider = provider
    config.generation_model = generation_model
    config.ollama_base_url = ollama_base_url
    return DawnAssistant(config)

def compute_corpus_signature(knowledge_path: Path) -> str:
    if knowledge_path.is_file():
        return f"{knowledge_path}:{knowledge_path.stat().st_mtime_ns}"

    if not knowledge_path.exists():
        return str(knowledge_path)

    corpus_files = sorted(
        [
            path
            for path in knowledge_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
        ]
    )
    parts = [f"{path}:{path.stat().st_mtime_ns}" for path in corpus_files]
    return "|".join(parts)


def render_pdf_downloads(title: str, pdf_paths: list[Path], base_path: Path) -> None:
    if not pdf_paths:
        return

    st.caption(title)
    for pdf_path in pdf_paths:
        relative_path = pdf_path.name if base_path.is_file() else str(pdf_path.relative_to(base_path))
        suggested_folder = suggest_storage_folder(pdf_path.name)
        st.markdown(f"**{pdf_path.name}**")
        st.caption(f"Dossier conseille : `{suggested_folder}`")
        st.caption(f"Emplacement actuel : `{relative_path}`")
        with pdf_path.open("rb") as pdf_stream:
            st.download_button(
                label=f"Telecharger {pdf_path.name}",
                data=pdf_stream.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
                key=f"download-{relative_path}",
            )


def render_analysis(result: dict, loaded_documents: list[dict], skipped_documents: list[dict]) -> None:
    st.subheader("Etat du corpus")
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents charges", len(loaded_documents))
    col2.metric("Documents ignores", len(skipped_documents))
    col3.metric(
        "Pages indexees",
        sum(document["pages_loaded"] for document in loaded_documents),
    )

    if loaded_documents:
        st.caption("Documents exploites par le RAG")
        st.dataframe(
            [
                {
                    "Document": document["source_name"],
                    "Langue": document.get("document_language", "unknown"),
                    "OCR": "oui" if document.get("ocr_used") else "non",
                    "Pages chargees": document["pages_loaded"],
                    "Chemin": document["source_path"],
                    "Dossier cible": document.get("storage_folder", "."),
                }
                for document in loaded_documents
            ],
            use_container_width=True,
            hide_index=True,
        )

    if skipped_documents:
        st.warning("Certains PDF ont ete ignores car aucun texte exploitable n'a ete extrait.")
        st.caption("Documents ignores")
        st.dataframe(
            [
                {
                    "Document": document["source_name"],
                    "Raison": document["reason"],
                    "Langue": document.get("document_language", "unknown"),
                    "Chemin": document["source_path"],
                    "Dossier cible": document.get("storage_folder", "."),
                }
                for document in skipped_documents
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(f"Reponse mode {result['mode']}")
    st.write(result["answer"])

    st.subheader("Sources retenues")
    for source in result["sources"]:
        st.write(
            f"{source['source_name']} | page {source['page']} | pertinence={source.get('relevance_score', 0.0):.3f} | score interne={source['score']:.3f} | dossier={source.get('storage_folder', '.')}"
        )

    with st.expander("Diagnostic RAG"):
        st.write(f"Mode: {result['mode']}")
        st.write(
            f"Type de question detecte: {result.get('question_kind', 'indisponible dans cette execution')}"
        )

        retrieved_chunks = result.get("retrieved_chunks", [])
        if retrieved_chunks:
            st.caption("Chunks recuperes avant generation")
            st.dataframe(
                [
                    {
                        "Document": chunk["source_name"],
                        "Langue": chunk.get("document_language", "unknown"),
                        "Page": chunk["page"],
                        "Extrait": chunk["chunk_id"],
                        "Pertinence": round(chunk.get("relevance_score", 0.0), 3),
                        "Score interne": round(chunk["score"], 3),
                        "Bonus lexical": round(chunk.get("lexical_score", 0.0), 3),
                    }
                    for chunk in retrieved_chunks
                ],
                use_container_width=True,
                hide_index=True,
            )

            for index, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(
                    f"**Chunk {index}** | {chunk['source_name']} | page {chunk['page']} | extrait {chunk['chunk_id']} | pertinence={chunk.get('relevance_score', 0.0):.3f} | score interne={chunk['score']:.3f}"
                )
                st.code(chunk["text"], language="text")
        else:
            st.info("Les details des chunks ne sont pas disponibles pour cette execution. Recharge le cache puis relance.")

        context = result.get("context")
        if context:
            st.caption("Contexte final envoye au modele")
            st.code(context, language="text")


knowledge_path = st.text_input("Chemin du dossier ou PDF", value=str(DEFAULT_KNOWLEDGE_PATH))
mode = st.selectbox("Mode DAWN", options=["medecin", "patient"], index=0)
provider = st.selectbox("Provider IA", options=["ollama", "anthropic"], index=0)
generation_model = st.text_input("Modele de generation", value="qwen2.5:14b")
ollama_base_url = st.text_input("URL Ollama", value="http://localhost:11434")
if st.button("Recharger la base documentaire"):
    load_assistant.clear()
    st.session_state.pop("last_analysis", None)
    st.success("Cache vide. La base sera rechargee au prochain test.")

question_options = [""] + [f"{item['theme']} | {item['question']}" for item in EVAL_QUESTIONS_FR]
selected_question_label = st.selectbox(
    "Question de test rapide",
    options=question_options,
    index=0,
)
selected_question = ""
if selected_question_label:
    selected_question = selected_question_label.split(" | ", maxsplit=1)[1]

question = st.text_area(
    "Question",
    value=selected_question,
    placeholder="Exemple : C'est quoi le paludisme ? ou Enfant avec fievre et vomissements depuis 2 jours.",
)

knowledge_path_obj = Path(knowledge_path)
if knowledge_path_obj.exists() and knowledge_path_obj.is_dir():
    with st.expander("Organisation recommandee du corpus"):
        st.write("Dossiers a utiliser pour ranger les PDF recents avant reindexation :")
        for directory in expected_corpus_directories(knowledge_path_obj):
            st.code(str(directory), language="text")

    pdf_files = sorted(knowledge_path_obj.rglob("*.pdf"))
    render_pdf_downloads("PDF disponibles au telechargement direct", pdf_files, knowledge_path_obj)
elif knowledge_path_obj.exists() and knowledge_path_obj.is_file() and knowledge_path_obj.suffix.lower() == ".pdf":
    render_pdf_downloads("PDF disponible au telechargement direct", [knowledge_path_obj], knowledge_path_obj)

if st.button("Analyser", type="primary"):
    if not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        try:
            corpus_signature = compute_corpus_signature(Path(knowledge_path))
            assistant = load_assistant(
                knowledge_path,
                corpus_signature,
                ASSISTANT_CACHE_VERSION,
                provider,
                generation_model,
                ollama_base_url,
            )
            result = assistant.answer(question, mode=mode)
            st.session_state["last_analysis"] = {
                "result": result,
                "loaded_documents": assistant.loaded_documents,
                "skipped_documents": assistant.skipped_documents,
            }
        except Exception as exc:
            st.error(
                "Le backend IA ne peut pas demarrer. Verifiez la configuration du provider, les dependances et la variable d'environnement de la cle API."
            )
            st.exception(exc)

last_analysis = st.session_state.get("last_analysis")
if last_analysis:
    render_analysis(
        last_analysis["result"],
        last_analysis["loaded_documents"],
        last_analysis["skipped_documents"],
    )
