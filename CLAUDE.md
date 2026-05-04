# DAWN - Assistant Médical IA

## Projet
Assistant d'aide à la décision médicale basé sur un RAG multimodal pour l'Afrique francophone. Développé dans le cadre du M2 DSIA.

## Architecture

```
code/
├── app.py                    # Interface Streamlit
├── dawn/
│   ├── config.py             # Configuration (modèles, chunking, provider)
│   ├── corpus.py             # Gestion des dossiers du corpus
│   ├── generator.py          # Génération de réponse (Ollama/Anthropic)
│   ├── prompts.py            # Construction des prompts par mode
│   ├── retriever.py          # RAG avec FAISS + scoring hybride
│   ├── chunking.py           # Découpage des documents en chunks
│   ├── pdf_loader.py         # Chargement PDF avec OCR
│   ├── eval_questions.py     # Questions de test standardisées
│   └── __pycache__/
└── data/                     # Base documentaire PDF
    ├── pediatrie/
    │   ├── paludisme/        # Documents paludisme (B09145-fre.pdf = OMS)
    │   ├── anemie/
    │   ├── detresse_respiratoire/
    │   ├── diarrhee_deshydratation/
    │   ├── nutrition_malnutrition/
    │   └── urgences/
    ├── gyneco_obstetrique/
    │   ├── accouchement_travail/
    │   ├── consultation_prenatale/
    │   ├── hemorragies_obstetricales/
    │   └── hta_preeclampsie_eclampsie/
    └── communs/
        ├── protocoles_hospitaliers/
        ├── recommandations_generales/
        └── triage_signes_gravite/
```

## Modes de réponse

### Mode Médecin
- Structure standardisée : Symptômes saillants → Hypothèses cliniques → Examens → Conduite à tenir → Signes de gravité
- Hypothèses ordonnées du plus fréquent au plus rare
- "Données insuffisantes" si contexte insuffisant
- Jamais d'invention d'information

### Mode Patient
- Langage simple et accessible
- Ton prudent et non affirmatif
- Résumé + Explications + Conseils + Signes d'alerte

## Détection de question (prompts.py)

- `detect_question_kind()` → `definition` | `clinical_case` | `general`
- `detect_question_severity()` → `urgent` | `non_urgent` (marqueurs : urgence, grave, convulsion, choc...)
- `detect_question_focus()` → `severity_signs` | `general`

## Scoring RAG (retriever.py)

Score hybride = embedding.cosine + lexical_overlap + bonus/pénalités.

### Bonus
- `path_specialty_bonus` : +0.45 si document correspond au sujet queried
- `topic_alignment_bonus` : +1.1 si chunk matche le topic, -0.55 si chunk mentionne topics competitors
- `focus_alignment_bonus` : +1.25 si chunk contient marqueurs de focus (gravité)
- `language_alignment_bonus` : +0.2 si document français

### Pénalités
- `general_penalty` : -0.45 si document "communs" queried pour specialty
- `negative_markers` : -2.4 si chunk contient moustiquaires, prévention (pour questions gravité)

## Configuration (config.py)

```python
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
provider = "ollama"  # ou "anthropic"
generation_model = "qwen2.5:3b"  # qwen2.5:7b, llama3, etc.
chunk_size = 220
chunk_overlap = 40
top_k = 5
max_new_tokens = 280
temperature = 0.1
```

## ProblèmesKnown

- [ ] chunk_size=220 trop petit → perte de contexte clinique
- [ ] Modèle qwen2.5:3b trop petit pour raisonnement médical complexe
- [ ] documents PDF parfois illisibles (OCR qualité variable)
- [ ] paludisme_grave_local_transcription.md est basique, manque protocoles détaillés
- [ ] pas de validation des réponses générées contre le contexte

## Améliorations recommandées

1. **Chunking** : passer à 400-600 tokens pour préserver le contexte clinique
2. **Modèle** : utiliser qwen2.5:7b ou llama3:8b pour des réponses médicales satisfaisantes
3. **Documents** : intégrer les directives OMS 2024-2025 sur le paludisme
4. **Post-processing** : valider que la réponse générée utilise bien les données du contexte