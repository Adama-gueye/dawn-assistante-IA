## Corpus DAWN

Structure documentaire effectivement indexee par l'application.

- `pediatrie/anemie/`
- `pediatrie/paludisme/`
- `pediatrie/detresse_respiratoire/`
- `pediatrie/diarrhee_deshydratation/`
- `pediatrie/nutrition_malnutrition/`
- `gyneco_obstetrique/accouchement_travail/`
- `gyneco_obstetrique/consultation_prenatale/`
- `gyneco_obstetrique/hta_preeclampsie_eclampsie/`
- `gyneco_obstetrique/hemorragies_obstetricales/`
- `communs/protocoles_hospitaliers/`
- `communs/recommandations_generales/`
- `communs/triage_signes_gravite/`

Conseil de classement :

- PDF recent sur une maladie -> sous-dossier pathologie correspondant
- protocole local ou hospitalier -> `communs/protocoles_hospitaliers/`
- recommandation generale ou guide officiel -> `communs/recommandations_generales/`
- document centré signes de gravite, triage ou urgence -> `communs/triage_signes_gravite/`

Scans et photos

- si le document est un scan image, il vaut mieux le convertir en PDF OCR avant de l'ajouter
- DAWN peut tenter un OCR de secours sur les PDF scannes si `Tesseract`, `pytesseract` et `PyMuPDF` sont installes
- pour un protocole local sur le paludisme grave, ranger le PDF dans `pediatrie/paludisme/`
