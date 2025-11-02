# reg_analysis.py - Extraction enrichie avec NLP pour HTML / textes réglementaires
import json
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

CONFIG_START = "<<<REGAI_CONFIG_START>>>"
CONFIG_END = "<<<REGAI_CONFIG_END>>>"

INLINE_CONFIG_EXAMPLE = """<<<REGAI_CONFIG_START>>>
{
  "params": {
    "mention_boost": 0.25,
    "theme_boost": 0.10,
    "measure_intensity_scale": 0.02,
    "measure_intensity_cap": 0.20,
    "penalty_boost": 0.05,
    "loss_pct_multiplier": 20,
    "max_loss_pct": 20,
    "ticker_min_len": 3
  },
  "company_theme_keywords": {
    "Technology": ["software","cloud","saas","ai","chip","semiconductor","cyber","it services","analytics","datacenter"]
  },
  "reg_type_targets": {
    "AI Promotion / Regulation": ["Technology","Financials","Healthcare"],
    "Energy / Environmental": ["Energy","Industrial"]
  }
}
<<<REGAI_CONFIG_END>>>"""

DATE_MONTHS = "(January|February|March|April|May|June|July|August|September|October|November|December|Jan\\.?|Feb\\.?|Mar\\.?|Apr\\.?|Jun\\.?|Jul\\.?|Aug\\.?|Sep\\.?|Oct\\.?|Nov\\.?|Dec\\.?)"

# motifs "forts" (peu ambigus)
DATE_PATTERNS = [
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b",                                # 2025-09-30
    rf"\b\d{{1,2}}/\d{{1,2}}/\d{{4}}\b",                            # 09/30/2025
    rf"\b{DATE_MONTHS}\s+\d{{1,2}},\s+\d{{4}}\b",                   # September 30, 2025
    rf"\bfiscal\s+year\s+\d{{4}}\b",                                # fiscal year 2025
    rf"\bFY\s?20\d{{2}}\b",                                         # FY2025 / FY 2025
    rf"\bthrough\s+{DATE_MONTHS}\s+\d{{1,2}},\s+\d{{4}}\b",         # through Sept. 30, 2028
]

# motifs de citations à **masquer** avant l’extraction
LEGAL_CITATION_PATTERNS = [
    r"\b\d+\s*U\.?S\.?C\.?\b", r"\bC\.?F\.?R\.?\b", r"\bSTAT\.?\b",
    r"\bPub\.?\s*L\.?\b", r"\bH\.?R\.?\b", r"\bS\.?[\- ]?\d+\b",
    r"\bSec\.?\b", r"\bSection\b", r"§", r"\bTitle\s+\d+\b", r"\bChapter\s+\d+\b",
]

YEAR_CONTEXT = ("year", "fy", "fiscal", "by", "until", "through", "before", "after", "on", "effective", "expires", "sunset")

def _mask_legal_citations(text: str) -> str:
    out = text
    for pat in LEGAL_CITATION_PATTERNS:
        out = re.sub(pat, " ", out, flags=re.I)
    return out

def _extract_dates(text: str):
    """Retourne des dates plausibles, en évitant les numéros d’articles."""
    txt = _mask_legal_citations(text)
    results = set()

    # 1) Motifs forts
    for pat in DATE_PATTERNS:
        for m in re.findall(pat, txt, flags=re.I):
            results.add(m if isinstance(m, str) else " ".join(m))

    # 2) Années seules 1900–2100, **avec** contexte temporel proche
    # Fenêtre: 3 mots avant
    tokens = re.split(r"(\W+)", txt)
    for i, tok in enumerate(tokens):
        if re.fullmatch(r"\d{4}", tok or ""):
            y = int(tok)
            if 1900 <= y <= 2100:
                window = " ".join(t for t in tokens[max(0, i-6):i]).lower()
                if any(k in window for k in YEAR_CONTEXT):
                    results.add(tok)

    return sorted(results)


def _parse_inline_config(raw_text: str):
    if not raw_text:
        return {}, raw_text

    m = re.search(rf"{re.escape(CONFIG_START)}(.*?){re.escape(CONFIG_END)}", raw_text, re.S)
    if not m:
        return {}, raw_text

    block = m.group(1)
    text_wo_block = raw_text[:m.start()] + raw_text[m.end():]
    try:
        cfg = json.loads(block)
    except Exception:
        cfg = {}
    return (cfg or {}), text_wo_block


DEFAULT_PARAMS = {
    "mention_boost": 0.4,
    "theme_boost": 0.25,
    "measure_intensity_scale": 0.05,
    "measure_intensity_cap": 0.35,
    "penalty_boost": 0.1,
    "loss_pct_multiplier": 5,
    "max_loss_pct": 20,
    "ticker_min_len": 1,
}


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

#DATE_REGEX = r'\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|after\s+\w+\s+\d{1,2},\s+\d{4}|\d{4})\b'

FALLBACK_ENTITY_REGEX = r'\b(US|USA|United States|China|EU|European Union|Europe|Japan|pharma|pharmaceutical|tech|technology|energy|corporations|sectors|countries|AI|artificial intelligence|consumer|protection|inflation|reduction|act|directive|regulation|law|promotion|advancement|research|development|exploitation|carbon|neutrality|drought|relief|environmental|impact|assessment|renewable|sources|人工知能|能源法|parlement|conseil|penalties|risks|classification|transparency|cybersecurity)\b'


COMPANY_THEME_KEYWORDS = {
    "Energy": ["energy", "oil", "gas", "petro", "power", "utility", "renewable"],
    "Healthcare": ["health", "pharma", "drug", "bio", "med", "clinical"],
    "Technology": ["software", "cloud", "saas", "ai", "chip", "semiconductor", "cyber", "it services", "analytics", "datacenter"],
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


def _infer_company_theme(company, theme_keywords=None):
    if not isinstance(company, str):
        return "Other"
    clower = company.lower()
    theme_keywords = theme_keywords or COMPANY_THEME_KEYWORDS
    # Match par mots entiers (évite "Technologies" → "Technology" via "tech")
    for theme, keywords in theme_keywords.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r's?\b', clower):
                return theme
    return "Other"

def _themes_from_text(text, theme_keywords=None):
    detected = set()
    lower = text.lower()
    theme_keywords = theme_keywords or TEXT_THEME_KEYWORDS
    for keyword, theme in theme_keywords.items():
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
    cfg_overrides, reg_text = _parse_inline_config(reg_text)
    reg_text = clean_text(reg_text, file_extension)
    nlp = _load_nlp_model()

    entities = set()
    dates = set(_extract_dates(reg_text))
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
        'config': cfg_overrides,
    }


# analyze_reg_impact reste à implémenter selon vos besoins
def analyze_reg_impact(portfolio_df, reg_text, file_extension='txt'):
    if portfolio_df is None or (hasattr(portfolio_df, "empty") and portfolio_df.empty):
        empty_info = {'entities': [], 'dates': [], 'measures': [], 'type_reg': 'Other', 'config': {}}
        return portfolio_df, empty_info, 0.0, pd.DataFrame(), []

    reg_text = reg_text or ""
    extracted = extract_reg_info(reg_text, file_extension)
    cfg_overrides = extracted.get('config') or {}

    params = DEFAULT_PARAMS.copy()
    params.update(cfg_overrides.get('params', {}))

    company_theme_keywords = {k: list(v) for k, v in COMPANY_THEME_KEYWORDS.items()}
    for theme, keywords in cfg_overrides.get('company_theme_keywords', {}).items():
        if isinstance(keywords, (list, tuple, set)):
            company_theme_keywords[theme] = list(keywords)
        else:
            company_theme_keywords[theme] = [str(keywords)]

    text_theme_keywords = TEXT_THEME_KEYWORDS.copy()
    for theme, keywords in cfg_overrides.get('text_theme_keywords', {}).items():
        if isinstance(keywords, (list, tuple, set)):
            for kw in keywords:
                text_theme_keywords[str(kw).lower()] = theme
        else:
            text_theme_keywords[str(keywords).lower()] = theme

    reg_type_targets = {k: set(v) if isinstance(v, (set, list, tuple)) else {v} for k, v in REG_TYPE_TARGETS.items()}
    for reg_type, themes in cfg_overrides.get('reg_type_targets', {}).items():
        if isinstance(themes, (list, tuple, set)):
            reg_type_targets[reg_type] = set(themes)
        else:
            reg_type_targets[reg_type] = {themes}

    df = portfolio_df.copy()

    required_cols = ['Risk Score', 'Impact Est. Loss %', 'Impact Est. Loss']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    if 'Weight' not in df.columns:
        df['Weight'] = 0.0

    df['Risk Theme'] = df.get('Risk Theme', df['Company'].apply(lambda company: _infer_company_theme(company, company_theme_keywords)))
    lower_text = reg_text.lower()

    targeted_themes = _merge_theme_sets(
        reg_type_targets.get(extracted['type_reg'], set()),
        _themes_from_text(lower_text, text_theme_keywords),
        _themes_from_text(' '.join(extracted.get('entities', [])), text_theme_keywords),
        _themes_from_text(' '.join(extracted.get('measures', [])), text_theme_keywords),
    )

    mention_boost = float(params.get('mention_boost', DEFAULT_PARAMS['mention_boost']))
    theme_boost = float(params.get('theme_boost', DEFAULT_PARAMS['theme_boost']))
    measure_intensity_scale = float(params.get('measure_intensity_scale', DEFAULT_PARAMS['measure_intensity_scale']))
    measure_intensity_cap = float(params.get('measure_intensity_cap', DEFAULT_PARAMS['measure_intensity_cap']))
    measure_intensity = min(measure_intensity_scale * len(extracted.get('measures', [])), measure_intensity_cap)
    penalty_boost = float(params.get('penalty_boost', DEFAULT_PARAMS['penalty_boost']))
    penalty_flag = penalty_boost if 'penalt' in lower_text or 'sanction' in lower_text else 0.0
    ticker_min_len = max(1, int(params.get('ticker_min_len', DEFAULT_PARAMS['ticker_min_len'])))
    loss_multiplier = float(params.get('loss_pct_multiplier', DEFAULT_PARAMS['loss_pct_multiplier']))
    max_loss_pct = float(params.get('max_loss_pct', DEFAULT_PARAMS['max_loss_pct']))

    risk_drivers = []
    for idx, row in df.iterrows():
        risk = 0.05 + measure_intensity  # baseline risk tied to breadth of measures
        drivers = []
        company_lower = str(row.get('Company', '')).lower()
        symbol_lower = str(row.get('Symbol', '')).lower()

        # Détection robuste (évite les faux positifs, ex. ticker 'A')
        symbol_mentioned  = len(symbol_lower) >= ticker_min_len and re.search(r'\b' + re.escape(symbol_lower) + r'\b', lower_text)
        company_mentioned = company_lower and re.search(r'\b' + re.escape(company_lower) + r'\b', lower_text)

        if symbol_mentioned or company_mentioned:
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
        loss_pct = min(risk * loss_multiplier, max_loss_pct)  # pourcentage exprimé en points
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
