# Fiches DAWN locales

## Pourquoi ajouter des fiches `.md`

Les PDF longs sont utiles comme references, mais ils ne donnent pas toujours de bons passages au RAG. Une fiche `.md` courte et structuree aide DAWN a recuperer directement les informations attendues : definition, signes, examens, conduite initiale, signes de gravite et sources.

## Regle de redaction

- Une fiche par maladie ou situation clinique.
- Phrases courtes.
- Titres explicites : `Definition`, `Signes de gravite`, `Examens de premiere intention`, `Conduite initiale`.
- Pas de longs paragraphes.
- Pas de traitements detailles ou posologies si le protocole local n'est pas valide.
- Toujours ajouter une section `Sources de reference`.

## Nom conseille

Dans chaque dossier :

```text
fiche_dawn_nom_du_theme.md
```

Exemples :

```text
pediatrie/paludisme/fiche_dawn_paludisme.md
pediatrie/anemie/fiche_dawn_anemie.md
pediatrie/detresse_respiratoire/fiche_dawn_detresse_respiratoire.md
```

## Apres modification

Dans l'application Streamlit, cliquer sur `Recharger la base documentaire`, puis relancer la question.
