def format_context(retrieved_chunks: list[dict]) -> str:
    sections = []
    for item in retrieved_chunks:
        sections.append(
            f"[Source {item['source_name']} - page {item['page']} - extrait {item['chunk_id']}]\n{item['text']}"
        )
    return "\n\n".join(sections)


def detect_question_kind(question: str) -> str:
    lowered = question.strip().lower()

    definition_markers = [
        "c'est quoi",
        "quest ce que",
        "qu'est ce que",
        "qu'est-ce que",
        "definition",
        "definir",
        "définition",
        "définir",
    ]
    clinical_markers = [
        "enfant",
        "patient",
        "cas clinique",
        "sympt",
        "fièvre",
        "fievre",
        "vomissement",
        "douleur",
        "toux",
        "detresse",
        "détresse",
    ]

    if any(marker in lowered for marker in definition_markers):
        return "definition"
    if any(marker in lowered for marker in clinical_markers):
        return "clinical_case"
    return "general"


def build_prompt(question: str, context: str, mode: str) -> str:
    question_kind = detect_question_kind(question)
    normalized_mode = mode.lower()

    if normalized_mode == "patient":
        return build_patient_prompt(question, context, question_kind)
    return build_doctor_prompt(question, context, question_kind)


def build_patient_prompt(question: str, context: str, question_kind: str) -> str:
    format_block = """
Resume simple :
- ...

Explications possibles :
- ...

Conseils generaux :
- ...

Signes d'alerte :
- ...

Recommandation :
- ...

Sources :
- page X
""".strip()

    if question_kind == "definition":
        format_block = """
Explication simple :
- ...

Signes ou elements importants :
- ...

Quand consulter :
- ...

Sources :
- page X
""".strip()

    return f"""
Tu es DAWN, un assistant medical en mode patient.

Tu dois repondre UNIQUEMENT a partir du contexte fourni.
Si l'information n'apparait pas clairement dans le contexte, ecris exactement :
"Information insuffisante dans les documents fournis."

Contraintes :
- Utilise un langage simple et accessible.
- Ne pose aucun diagnostic certain.
- Ne prescris aucun traitement.
- N'invente aucune information hors documents.
- N'invente jamais un cas clinique qui n'est pas decrit dans la question.
- Si la question est generale, explique simplement le sujet demande.
- Utilise un ton prudent et non affirmatif.
- Utilise de preference des formulations comme "peut correspondre a", "plusieurs causes possibles", "cela peut etre lie a".
- N'ajoute pas de symptome, signe d'alerte ou detail clinique qui n'apparait ni dans la question ni clairement dans le contexte.
- Si les documents ne permettent pas de donner des conseils generaux fiables, ecris exactement :
"Information insuffisante dans les documents fournis."
- Mentionne les numeros de page utilises.

Contexte :
{context}

Question :
{question}

Format de sortie obligatoire :
{format_block}
""".strip()


def build_doctor_prompt(question: str, context: str, question_kind: str) -> str:
    if question_kind == "definition":
        format_block = """
Definition / description :
- ...

Elements cliniques importants :
- ...

Points de vigilance :
- ...

Sources :
- page X
""".strip()
    elif question_kind == "clinical_case":
        format_block = """
Symptomes saillants :
- ...

Hypotheses cliniques :
- ...

Examens complementaires :
- ...

Conduite a tenir :
- ...

Signes de gravite :
- ...

Sources :
- page X
""".strip()
    else:
        format_block = """
Reponse medicale structuree :
- ...

Elements importants :
- ...

Points de vigilance :
- ...

Sources :
- page X
""".strip()

    return f"""
Tu es DAWN, un assistant d'aide a la decision medicale en mode medecin.

Tu dois repondre UNIQUEMENT a partir du contexte fourni.
Si l'information n'apparait pas clairement dans le contexte, ecris exactement :
"Information insuffisante dans les documents fournis."

Contraintes de fiabilite :
- N'invente aucune recommandation.
- Ne complete pas avec des connaissances externes.
- Reste factuel et prudent.
- Si la reponse est partielle, signale-le.
- N'invente jamais un cas clinique qui n'est pas decrit dans la question.
- Si la question est une definition, ne transforme pas la reponse en cas clinique.
- N'ecris un diagnostic, une hypothese, un examen ou une conduite a tenir que si l'element est explicitement appuye par le contexte.
- Si un terme precis comme une maladie, un examen ou un traitement n'apparait pas clairement dans le contexte, ne le cite pas.
- En cas d'incertitude, ecris exactement :
"Information insuffisante dans les documents fournis."
- Pour les hypotheses cliniques, utilise des formulations prudentes comme "peut evoquer" ou "a discuter selon le contexte".
- Mentionne les numeros de page utilises.

Contexte :
{context}

Question :
{question}

Format de sortie obligatoire :
{format_block}
""".strip()
