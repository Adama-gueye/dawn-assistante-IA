# CLAUDE.COMSA - Analyse et Plan d'Amélioration du Projet DAWN

## Contexte : Problème des réponses médicales non satisfaisantes

Le modèle génère des réponses médicalement incomplètes ou incorrectes pour des questions comme :

> "Chez un enfant avec fievre en zone d'endemie, quelle conduite a tenir initiale est recommandee pour suspecter un paludisme ?"

**Exemple de réponse actuelle (insuffisante) :**
- Liste générique d'hypothèses sans rapport paludisme
- Ne priorise pas correctement le paludisme en zone d'endémie
- Omet l'évaluation des signes de gravité en première intention
- Conduite à tenir trop vague

---

## Diagnostic : Pourquoi les réponses ne sont pas satisfaisantes

### 1. DOCUMENTS - Problèmes identified

| Problème | Impact | Priorité |
|----------|--------|----------|
| `chunk_size=220` trop petit | Contexte clinique fragmenté, perte d'information | HAUTE |
| PDF paludisme (B09145-fre.pdf) de l'OMS mal-indexé ou OCR de mauvaise qualité | Document source peu accessible au RAG | HAUTE |
| `paludisme_grave_local_transcription.md` trop basique | Source locale insuffisante pour couvrir toutes les situations | MOYENNE |
| Manque directives OMS 2024-2025 sur le paludisme | Données potentiellement obsolètes | MOYENNE |

### 2. MODÈLE - Configuration sous-optimale

| Paramètre | Actuel | Recommandé | Raison |
|-----------|--------|------------|--------|
| `generation_model` | qwen2.5:3b | qwen2.5:7b ou llama3:8b | 3b trop petit pour raisonnement médical complexe |
| `max_new_tokens` | 280 | 800-1200 | Limite la longueur des réponses détaillées |
| `chunk_size` | 220 | 400-600 | Préserve le contexte clinique |
| `chunk_overlap` | 40 | 80-120 | Meilleure continuité entre chunks |

### 3. PROMPTS - Règles trop strictes

Les prompts actuels interdisent d'ajouter des informations même quand elles sont médicalement essentielles. Le modèle ne peut pas "dépasser" le contexte même quand le bon sens médical l'exige.

**Exemple de problème :** Le modèle refuse de mentionner l'artésunate (traitement de référence OMS) car il n'apparaît pas explicitement dans le document chunké.

---

## PLAN D'ACTION

### Étape 1 : Améliorer les documents sources

**Documents WHO à intégrer (téléchargeables) :**

```
Zone endémique palustre = fièvre chez l'enfant
→ https://iris.who.int/handle/10665/377239 (Directives OMS 2024)
→ https://www.who.int/publications/i/item/9789240062694 (TDR paludisme)
→ https://iris.who.int/handle/10665/375704 (Traitement paludisme simple)
```

**Documents complémentaires utiles :**
- Plan de gestion intégrée du paludisme (CVD) pour l'Afrique
- Protocoles nationaux des pays francophones (Côte d'Ivoire, Mali, Burkina, etc.)

### Étape 2 : Améliorer le chunking

```python
# config.py - Modifier chunking
chunk_size: int = 450      # 220 → 450
chunk_overlap: int = 90    # 40 → 90
```

**Pourquoi :** Un contexte clinique typique (symptômes + examen + conduite) nécessite au moins 300-400 tokens pour être préservé.

### Étape 3 : Améliorer les prompts

**Ajuster `build_doctor_prompt()` pour le cas "fièvre en zone d'endémie" :**

```python
# NOUVELLE RÈGLE À AJOUTER dans build_doctor_prompt()
# Lorsque la question mentionne fièvre + zone endémique + paludisme :
# - Le paludisme DOIT être en première hypothèse
# - Les signes de gravité DOIVENT être évalués
# - Le test diagnostique (TDR/goutte épaisse) EST obligatoire
```

### Étape 4 : Améliorer le modèle de génération

**Option A (locale, gratuit) :**
```python
generation_model: str = "qwen2.5:7b"  # si Ollama supporte
```

**Option B (API, payant mais plus puissant) :**
```python
provider: str = "anthropic"
generation_model: str = "claude-sonnet-4-6"  # ou opus-4-7
```

---

## Réponse type attendue après améliorations

Pour la question : "Chez un enfant avec fievre en zone d'endemie, quelle conduite a tenir initiale est recommandee pour suspecter un paludisme ?"

**Réponse DAWN v2 (attendue) :**

```
Symptômes saillants :
- Fièvre chez un enfant en zone d'endémie palustre

Hypothèses cliniques (du plus fréquent au plus rare) :
- Paludisme (à suspecter en priorité en zone d'endémie)
- Infection bactérienne (septicémie, pneumonie)
- Infection virale
- Autres parasitoses

Examens complémentaires (première intention uniquement) :
- Test de diagnostic rapide (TDR) du paludisme
- Goutte épaisse / frottis sanguin (si disponible)

Conduite à tenir :
1. Évaluer immédiatement les signes de gravité (troubles de conscience, convulsions, prostration, détresse respiratoire, vomissements incoercibles, ictère, collapsus)
2. Réaliser un test parasitologique en urgence
3. Si TDR positif → traitement antipalustre selon protocole
4. Si signes de gravité → hospitalisation urgente
5. Si TDR négatif → rechercher autre cause de fièvre

Signes de gravité :
- Coma / troubles de conscience
- Convulsions
- Prostration / incapacité à boire
- Détresse respiratoire
- Collapsus cardiovasculaire
- Ictère / hemoglobinurie
- Anémie sévère

Sources :
- B09145-fre.pdf (OMS) | pages 12-15
- paludisme_grave_local_transcription.md
```

---

## Fichiers à modifier

1. `code/dawn/config.py` - chunk_size, chunk_overlap, max_new_tokens
2. `code/dawn/prompts.py` - améliorer les règles pour fièvre+paludisme
3. `code/dawn/retriever.py` - améliorer le scoring pour documents français
4. `code/data/pediatrie/paludisme/` - ajouter nouveaux PDF OMS

## Commandes pour tester

```bash
cd code
streamlit run app.py

# OU en ligne de commande
python -c "
from dawn.config import DawnConfig
from dawn.generator import DawnAssistant
config = DawnConfig()
assistant = DawnAssistant(config)
result = assistant.answer('Chez un enfant avec fievre en zone d endemie, quelle conduite a tenir initiale est recommandee pour suspecter un paludisme ?', mode='medecin')
print(result['answer'])
"
```

---

## Note pour l'utilisateur

Le projet DAWN est bien conçu sur le plan architectural. Le problème principal est que les réponses médicales dépendent directement de la **qualité des documents sources** et de la **capacité du modèle de génération**.

**Recommandation immédiate :**
1. Passer à `qwen2.5:7b` au lieu de `qwen2.5:3b`
2. Augmenter `max_new_tokens` à 600 minimum
3. Augmenter `chunk_size` à 400-450

Ces 3 changements alone devraient améliorer significativement la qualité des réponses.