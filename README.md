# Datathon 2025 - Analyse IA de l'Impact Réglementaire

## 🎯 Objectif du Projet

Ce projet a été développé dans le cadre du **Datathon POLYFINANCES 2025**. Il vise à créer un outil d'analyse basé sur l'IA générative pour évaluer l'impact des réglementations financières sur les portefeuilles d'actions, spécifiquement le S&P 500.

## 📊 Contexte

Les marchés financiers sont de plus en plus influencés par :
- Un cadre réglementaire complexe et en constante évolution
- Des lois protectionnistes
- Des sanctions économiques internationales

Ces facteurs redefinissent la gestion d'actifs et nécessitent des outils d'aide à la décision plus agiles et plus intelligents.

## ✨ Fonctionnalités Principales

### 1. Analyse Automatique de Textes Réglementaires
- Extraction automatique des éléments clés (entités, secteurs, dates, mesures)
- Utilisation de techniques NLP et IA générative
- Adaptabilité à différents formats (lois, rapports, documents 10-K)

### 2. Évaluation d'Impact
- Calcul de scores de risque par entreprise
- Analyse des expositions sectorielles et géographiques
- Résultats chiffrés (perte estimée, % d'exposition)
- Explications transparentes du raisonnement

### 3. Recommandations Stratégiques
- Simulation de scénarios multiples
- Identification des zones de concentration du risque
- Suggestions d'ajustements concrets :
  - Réallocation sectorielle
  - Rotation sectorielle
  - Remplacement de titres
  - Réallocation géographique

### 4. Interface Web Interactive
- Visualisation claire de l'exposition du portefeuille
- Présentation intuitive des ajustements proposés
- Expérience utilisateur optimisée

## 🛠️ Technologies Utilisées

- **IA Générative** : Pour l'analyse et l'extraction d'informations
- **NLP (Natural Language Processing)** : Pour le traitement des textes réglementaires
- **Python** : Langage principal de développement
- **AWS Services** : Pour le traitement et l'hébergement

## 📊 Données

### Données Fournies
- `sp500_composition_2025-08-15.csv` : Composition du S&P 500 (tickers, poids, prix)
- `stocks-performance_2025-09-26.csv` : Performances des actions (capitalisation, EPS, FCF, etc.)

### Sources Externes Autorisées
- [SEC EDGAR](https://www.sec.gov/edgar/search/) : Rapports 10-K et 10-Q
- Yahoo Finance : Données de marché
- Morningstar : Analyses financières

## 📁 Structure du Projet

```
datathon-2025-regulatory-ai/
│
├── data/              # Données brutes et traitées
├── notebooks/         # Jupyter notebooks pour l'analyse
├── src/               # Code source de l'application
│   ├── extraction/    # Modules d'extraction de données
│   ├── analysis/      # Modules d'analyse et scoring
│   ├── recommendations/ # Génération de recommandations
│   └── web/           # Interface web
├── tests/             # Tests unitaires
├── docs/              # Documentation
├── requirements.txt   # Dépendances Python
└── README.md          # Ce fichier
```

## 🚀 Installation

```bash
# Cloner le repository
git clone https://github.com/Omar-Zed/datathon-2025-regulatory-ai.git
cd datathon-2025-regulatory-ai

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 📝 Utilisation

```python
# Exemple d'utilisation basique
from src.analysis import RegulatoryAnalyzer

# Initialiser l'analyseur
analyzer = RegulatoryAnalyzer()

# Analyser un document réglementaire
results = analyzer.analyze_document("path/to/regulatory_document.pdf")

# Générer des recommandations
recommendations = analyzer.generate_recommendations(results)
```

## 🏆 Critères d'Évaluation

Le projet sera évalué selon plusieurs critères, notamment :
- Précision de l'extraction d'informations
- Pertinence des scores de risque
- Qualité des recommandations stratégiques
- Interface utilisateur et visualisations
- Storytelling et présentation (25%)
- Originalité et valeur ajoutée

## 📅 Chronologie

- **Vendredi/Samedi** : Exploration des données et mise en place de l'infrastructure
- **Dimanche matin** : Réception du document supplémentaire pour évaluation
- **Dimanche après-midi** : Finalisation et préparation de la présentation

## ⚠️ Points d'Attention

- **Optimisation AWS** : Tester sur un échantillon restreint d'abord
- **Conservation des résultats** : Limiter les appels API répétés
- **Flexibilité** : L'outil doit s'adapter à différents types de documents
- **Transparence** : Expliquer le raisonnement derrière chaque recommandation

## 👥 Équipe

*[Ajouter les membres de votre équipe ici]*

## 📝 Licence

Ce projet a été créé dans le cadre du Datathon POLYFINANCES 2025.

## 🔗 Liens Utiles

- [Documentation POLYFINANCES](https://polyfinances.ca)
- [SEC EDGAR Database](https://www.sec.gov/edgar/search/)
- [S&P 500 Information](https://www.spglobal.com/spdji/en/indices/equity/sp-500/)

---

**Datathon POLYFINANCES 2025** | Transformer la complexité réglementaire en opportunités d'aide à la décision
