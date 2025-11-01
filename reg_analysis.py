# reg_analysis.py - Extraction enrichie avec NLP pour HTML / textes réglementaires
import re
import pandas as pd
import xml.etree.ElementTree as ET

try:
    from bs4 import BeautifulSoup
except ImportError:  # BeautifulSoup est facultatif mais recommandé
    BeautifulSoup = None

try:
    import spacy
    from spacy.matcher import PhraseMatcher
except ImportError:  # Le pipeline NLP reste optionnel
    spacy = None  # type: ignore
    PhraseMatcher = None  # type: ignore


_NLP_MODEL = None
_MEASURE_MATCHER = None


MEASURE_TERMS = [
    "tax", "minimum tax", "rebate", "tariff", "tariffs", "measure", "directive",
    "regulation", "law", "promotion", "advancement", "research", "development",
    "exploitation", "energy", "renewable energy", "carbon neutrality",
    "artificial intelligence", "ai", "consumer rights", "penalty", "penalties",
    "prohibition", "safety", "risk", "classification", "transparency",
    "cybersecurity", "environmental impact", "assessment", "emissions",
    "renewable sources", "subsidy", "subsidies", "fine", "fines", "sanction",
    "sanctions", "compliance", "reporting obligation", "reporting obligations",
    "drought relief", "mitigation", "financial assistance", "price cap",
    "price ceiling", "price control", "export ban", "import duty", "grant",
    "quota", "licencing requirement", "licensing requirement"
]

ENTITY_RULER_PATTERNS = [
    {"label": "LAW", "pattern": "Inflation Reduction Act"},
    {"label": "LAW", "pattern": "EU AI Act"},
    {"label": "LAW", "pattern": "Green Deal"},
    {"label": "LAW", "pattern": "General Data Protection Regulation"},
    {"label": "LAW", "pattern": "GDPR"},
    {"label": "ORG", "pattern": "European Commission"},
    {"label": "ORG", "pattern": "European Parliament"},
    {"label": "ORG", "pattern": "United States Congress"},
    {"label": "ORG", "pattern": "Securities and Exchange Commission"},
    {"label": "ORG", "pattern": "SEC"},
    {"label": "ORG", "pattern": "Food and Drug Administration"},
    {"label": "ORG", "pattern": "FDA"},
    {"label": "MISC", "pattern": "pharmaceutical industry"},
    {"label": "MISC", "pattern": "tech sector"},
    {"label": "MISC", "pattern": "energy sector"},
]

DATE_REGEX = r'\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|after\s+\w+\s+\d{1,2},\s+\d{4}|\d{4})\b'

FALLBACK_ENTITY_REGEX = r'\b(US|USA|United States|China|EU|European Union|Europe|Japan|pharma|pharmaceutical|tech|technology|energy|corporations|sectors|countries|AI|artificial intelligence|consumer|protection|inflation|reduction|act|directive|regulation|law|promotion|advancement|research|development|exploitation|carbon|neutrality|drought|relief|environmental|impact|assessment|renewable|sources|人工知能|能源法|parlement|conseil|penalties|risks|classification|transparency|cybersecurity)\b'


COMPANY_THEME_KEYWORDS = {
    "Energy": ["energy", "oil", "gas", "petro", "power", "utility", "renewable"],
    "Healthcare": ["health", "pharma", "drug", "bio", "med", "clinical"],
    "Technology": ["tech", "digital", "soft", "cloud", "data", "ai", "chip", "semi", "cyber"],
    "Consumer": ["consumer", "retail", "brand", "food", "beverage", "market", "store"],
    "Financials": ["bank", "financ", "capital", "insurance", "asset", "credit", "lending"],
    "Industrial": ["industrial", "manufact", "aero", "defense", "logistic", "transport"],
}

TEXT_THEME_KEYWORDS = {
    "pharma": "Healthcare",
    "drug": "Healthcare",
    "medical": "Healthcare",
    "energy": "Energy",
    "emission": "Energy",
    "carbon": "Energy",
    "renewable": "Energy",
    "oil": "Energy",
    "gas": "Energy",
    "ai": "Technology",
    "artificial intelligence": "Technology",
    "cyber": "Technology",
    "data": "Technology",
    "semiconductor": "Technology",
    "consumer": "Consumer",
    "retail": "Consumer",
    "agriculture": "Consumer",
    "bank": "Financials",
    "fiscal": "Financials",
    "tax": "Financials",
    "inflation": "Financials",
    "logistic": "Industrial",
    "automotive": "Industrial",
}

REG_TYPE_TARGETS = {
    "Taxation / Economic": {"Financials", "Consumer", "Technology"},
    "Consumer Protection": {"Consumer", "Technology"},
    "Energy / Environmental": {"Energy", "Industrial"},
    "AI Promotion / Regulation": {"Technology", "Financials"},
    "Other": {"Other"},
}


def _infer_company_theme(company):
    if not isinstance(company, str):
        return "Other"
    clower = company.lower()
    for theme, keywords in COMPANY_THEME_KEYWORDS.items():
        if any(keyword in clower for keyword in keywords):
            return theme
    return "Other"


def _themes_from_text(text):
    detected = set()
    lower = text.lower()
    for keyword, theme in TEXT_THEME_KEYWORDS.items():
        if keyword in lower:
            detected.add(theme)
    return detected


def _merge_theme_sets(*theme_sets):
    merged = set()
    for theme_set in theme_sets:
        merged.update(theme_set or set())
    return merged or {"Other"}


def _load_nlp_model():
    """Charge paresseusement un pipeline spaCy, avec retombée gracieuse."""
    global _NLP_MODEL
    if _NLP_MODEL is not None:
        return _NLP_MODEL

    if spacy is None:
        return None

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Modèle non disponible : on crée un pipeline minimal avec sentencizer
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

    # Ajout d'un EntityRuler pour saisir les concepts clés spécifiques
    if "entity_ruler" in nlp.pipe_names:
        ruler = nlp.get_pipe("entity_ruler")
    else:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(ENTITY_RULER_PATTERNS)

    _NLP_MODEL = nlp
    return _NLP_MODEL


def _get_measure_matcher(nlp):
    """Initialise un PhraseMatcher pour capter des mesures réglementaires."""
    global _MEASURE_MATCHER
    if _MEASURE_MATCHER is not None or nlp is None or PhraseMatcher is None:
        return _MEASURE_MATCHER

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("MEASURE", [nlp.make_doc(term) for term in MEASURE_TERMS])
    _MEASURE_MATCHER = matcher
    return _MEASURE_MATCHER


def _iter_spacy_docs(text, nlp, chunk_size=500000):
    """Découpe les textes trop longs pour spaCy en segments exploitables."""
    if nlp is None:
        return

    chunk_size = max(200000, chunk_size)
    length = len(text)
    if length == 0:
        return

    if length <= chunk_size:
        nlp.max_length = max(getattr(nlp, "max_length", 1000000), length + 1000)
        yield nlp(text)
        return

    nlp.max_length = max(getattr(nlp, "max_length", 1000000), chunk_size + 1000)
    start = 0
    while start < length:
        end = min(length, start + chunk_size)
        if end < length:
            overflow = 0
            while end < length and not text[end].isspace() and overflow < 1000:
                end += 1
                overflow += 1
        chunk = text[start:end]
        yield nlp(chunk)
        start = end


def clean_text(reg_text, file_extension):
    """Nettoie les textes réglementaires selon leur nature."""
    file_extension = file_extension.lower()

    if file_extension in ['html', 'htm'] and BeautifulSoup is not None:
        soup = BeautifulSoup(reg_text, "html.parser")
        for element in soup(["script", "style", "noscript", "header", "footer"]):
            element.decompose()
        reg_text = ' '.join(soup.stripped_strings)
    elif file_extension in ['html', 'htm']:
        # Fallback regex moins précis
        reg_text = re.sub(r'<[^>]+>', ' ', reg_text)
        reg_text = re.sub(r'&[^;]+;', ' ', reg_text)
    elif file_extension == 'xml':
        try:
            root = ET.fromstring(reg_text)
            reg_text = ' '.join(
                elem.text.strip()
                for elem in root.iter()
                if elem.text and elem.text.strip()
            )
        except ET.ParseError:
            reg_text = re.sub(r'<[^>]+>', ' ', reg_text)

    reg_text = re.sub(r'\s+', ' ', reg_text)
    return reg_text.strip()


def extract_reg_info(reg_text, file_extension='txt'):
    """Extrait entités, dates et mesures en combinant NLP et heuristiques."""
    reg_text = clean_text(reg_text, file_extension)
    nlp = _load_nlp_model()

    entities = set()
    dates = set(re.findall(DATE_REGEX, reg_text, re.I))
    measures = set()

    if nlp is not None:
        measure_matcher = _get_measure_matcher(nlp)
        docs = _iter_spacy_docs(reg_text, nlp)

        for doc in docs:
            for ent in doc.ents:
                if ent.label_ in {"ORG", "GPE", "NORP", "LAW", "FAC", "LOC", "MISC"}:
                    entities.add(ent.text.strip())
                if ent.label_ == "DATE":
                    dates.add(ent.text.strip())

            if measure_matcher is not None:
                matches = measure_matcher(doc)
                for _, start, end in matches:
                    span = doc[start:end]
                    measures.add(span.text.strip())
    else:
        # Fallback sur les regex existantes si spaCy indisponible
        entities = set(re.findall(FALLBACK_ENTITY_REGEX, reg_text, re.I))

    # Complète les entités avec les regex de secours pour plus de rappel
    entities.update(re.findall(FALLBACK_ENTITY_REGEX, reg_text, re.I))

    if not measures:
        fallback_measures = r'(tax|rebates|negotiations|tariffs|measure|directive|regulation|act|law|promotion|advancement|research|development|exploitation|energy|AI|artificial intelligence|consumer rights|penalties|prohibitions|safety|risks|classification|transparency|cybersecurity|environmental|impact|assessment|carbon|emissions|renewable|sources|subsidies|fines|sanctions|compliance|reporting|obligations|drought|relief|mitigation|financial assistance)'
        measures = set(re.findall(fallback_measures, reg_text, re.I))

    # Type inference (conserve la logique initiale)
    lower_text = reg_text.lower()
    if any(keyword in lower_text for keyword in ['tax', 'inflation', 'reduction act']):
        type_reg = 'Taxation / Economic'
    elif any(keyword in lower_text for keyword in ['price', 'consumer', 'directive']):
        type_reg = 'Consumer Protection'
    elif any(keyword in lower_text for keyword in ['energy', '碳', 'emission', 'renewable']):
        type_reg = 'Energy / Environmental'
    elif any(keyword in lower_text for keyword in ['ai', 'artificial intelligence', '人工知能']):
        type_reg = 'AI Promotion / Regulation'
    else:
        type_reg = 'Other'

    return {
        'entities': sorted(entities),
        'dates': sorted(dates),
        'measures': sorted(measures),
        'type_reg': type_reg,
    }


# analyze_reg_impact reste à implémenter selon vos besoins
def analyze_reg_impact(portfolio_df, reg_text, file_extension='txt'):
    if portfolio_df is None or (hasattr(portfolio_df, "empty") and portfolio_df.empty):
        empty_info = {'entities': [], 'dates': [], 'measures': [], 'type_reg': 'Other'}
        return portfolio_df, empty_info, 0.0, pd.DataFrame(), []

    reg_text = reg_text or ""
    extracted = extract_reg_info(reg_text, file_extension)
    df = portfolio_df.copy()

    required_cols = ['Risk Score', 'Impact Est. Loss %', 'Impact Est. Loss']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    if 'Weight' not in df.columns:
        df['Weight'] = 0.0

    df['Risk Theme'] = df.get('Risk Theme', df['Company'].apply(_infer_company_theme))
    lower_text = reg_text.lower()

    targeted_themes = _merge_theme_sets(
        REG_TYPE_TARGETS.get(extracted['type_reg'], set()),
        _themes_from_text(lower_text),
        _themes_from_text(' '.join(extracted.get('entities', []))),
        _themes_from_text(' '.join(extracted.get('measures', []))),
    )

    mention_boost = 0.4
    theme_boost = 0.25
    measure_intensity = min(0.05 * len(extracted.get('measures', [])), 0.35)
    penalty_flag = 0.1 if 'penalt' in lower_text or 'sanction' in lower_text else 0.0

    risk_drivers = []
    for idx, row in df.iterrows():
        risk = 0.05 + measure_intensity  # baseline risk tied to breadth of measures
        drivers = []
        company_lower = str(row.get('Company', '')).lower()
        symbol_lower = str(row.get('Symbol', '')).lower()

        if symbol_lower and symbol_lower in lower_text or company_lower and company_lower in lower_text:
            risk += mention_boost
            drivers.append("Mention explicite")

        theme = row['Risk Theme']
        if theme in targeted_themes:
            risk += theme_boost
            drivers.append(f"Thématique ciblée ({theme})")

        if penalty_flag:
            risk += penalty_flag
            drivers.append("Pénalités / sanctions mentionnées")

        if 'compliance' in lower_text or 'reporting' in lower_text:
            risk += 0.05
            drivers.append("Exigence de conformité")

        risk = min(risk, 1.0)
        loss_pct = min(risk * 5, 20.0)  # pourcentage exprimé en points
        market_cap = row.get('Market Cap', 0) or 0
        impact_loss = market_cap * (loss_pct / 100.0)

        df.at[idx, 'Risk Score'] = risk
        df.at[idx, 'Impact Est. Loss %'] = loss_pct
        df.at[idx, 'Impact Est. Loss'] = impact_loss
        risk_drivers.append(', '.join(drivers) if drivers else "Impact généralisé")

    df['Risk Drivers'] = risk_drivers

    weight_series = df.get('Weight')
    if weight_series is not None and weight_series.sum() > 0:
        portfolio_risk = float((df['Risk Score'] * weight_series).sum() / weight_series.sum())
    else:
        portfolio_risk = float(df['Risk Score'].mean())

    concentration = (
        df.groupby('Risk Theme')
        .agg(
            Weight_Exposure=('Weight', 'sum'),
            Avg_Risk=('Risk Score', 'mean'),
            Max_Risk=('Risk Score', 'max'),
            Est_Loss=('Impact Est. Loss', 'sum'),
        )
        .reset_index()
        .sort_values('Weight_Exposure', ascending=False)
    )

    recommendations = []
    highest_theme = concentration.iloc[0]['Risk Theme'] if not concentration.empty else 'Other'

    if portfolio_risk > 0.5:
        recommendations.append("Rééquilibrer immédiatement les positions les plus risquées identifiées.")
    if 'Technology' in targeted_themes:
        recommendations.append("Mettre en place un suivi renforcé des titres technologiques et de la conformité IA.")
    if 'Energy' in targeted_themes:
        recommendations.append("Évaluer l'impact sur les producteurs d'énergie et modéliser des scénarios carbone plus sévères.")
    if extracted.get('dates'):
        recommendations.append(f"Planifier les actions avant la prochaine échéance réglementaire détectée ({extracted['dates'][0]}).")
    if not recommendations:
        recommendations.append(f"Surveiller l'évolution du cadre pour la thématique dominante ({highest_theme}).")

    return df, extracted, portfolio_risk, concentration, recommendations
