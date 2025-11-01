# reg_analysis_enhanced.py - Version améliorée avec 10-K et Bedrock
"""
Module d'analyse réglementaire enrichi combinant:
- NLP classique (spaCy)
- Analyse des rapports 10-K (supply chain, géographie)
- IA générative (AWS Bedrock)
"""

import re
import pandas as pd
from typing import Dict, List, Tuple
from reg_analysis import (
    extract_reg_info,
    analyze_reg_impact,
    DEFAULT_PARAMS,
    COMPANY_THEME_KEYWORDS,
    TEXT_THEME_KEYWORDS,
    REG_TYPE_TARGETS,
    _infer_company_theme,
    _themes_from_text,
    _merge_theme_sets
)


def analyze_reg_impact_enhanced(
    portfolio_df: pd.DataFrame,
    reg_text: str,
    fillings_dir: str,
    file_extension: str = 'txt',
    use_bedrock: bool = True
) -> Tuple[pd.DataFrame, Dict, float, pd.DataFrame, List[Dict]]:
    """
    Version améliorée de analyze_reg_impact qui intègre:
    1. Analyse des rapports 10-K
    2. IA générative via Bedrock
    3. Scoring multi-niveaux (direct, supply chain, géographique)
    
    Args:
        portfolio_df: DataFrame du portefeuille S&P 500
        reg_text: Texte réglementaire à analyser
        fillings_dir: Chemin vers dossier des rapports 10-K
        file_extension: Extension du fichier réglementaire
        use_bedrock: Si True, utilise Bedrock pour enrichissement
        
    Returns:
        Tuple de (df_analysé, extraction, risque_portfolio, concentration, recommandations)
    """
    
    # Validation
    if portfolio_df is None or portfolio_df.empty:
        empty_info = {'entities': [], 'dates': [], 'measures': [], 'type_reg': 'Other'}
        return portfolio_df, empty_info, 0.0, pd.DataFrame(), []
    
    # 1. EXTRACTION RÉGLEMENTAIRE (NLP + optionnel Bedrock)
    print("📄 Extraction des informations réglementaires...")
    extracted = extract_reg_info(reg_text, file_extension)
    
    bedrock = None
    bedrock_analysis = None
    bedrock_error = None
    use_bedrock_runtime = use_bedrock
    if use_bedrock:
        try:
            from aws_bedrock_integration import BedrockAnalyzer
            bedrock = BedrockAnalyzer()
            if bedrock.available:
                bedrock_analysis = bedrock.analyze_regulatory_text(reg_text)
                if not bedrock.available:
                    use_bedrock_runtime = False
                    bedrock_error = bedrock.last_error
                    print(f"⚠️ Analyse Bedrock désactivée: {bedrock_error}")
                else:
                    # Enrichir l'extraction avec résultats Bedrock
                    extracted = _merge_bedrock_extraction(extracted, bedrock_analysis)
                    print("✅ Analyse Bedrock complétée")
            else:
                use_bedrock_runtime = False
                bedrock_error = bedrock.last_error or "Client Bedrock non disponible"
                print(f"⚠️ Bedrock indisponible: {bedrock_error}")
        except Exception as e:
            use_bedrock_runtime = False
            bedrock_error = str(e)
            print(f"⚠️ Bedrock non disponible: {e}")
            bedrock = None

    extracted['bedrock_status'] = {
        "enabled": bool(use_bedrock_runtime and bedrock and bedrock.available),
        "error": bedrock_error
    }

    # 2. CHARGEMENT ANALYSEUR 10-K
    print("📊 Chargement des données 10-K...")
    try:
        from filing_analyzer import FilingAnalyzer
        filing_analyzer = FilingAnalyzer(fillings_dir)
    except Exception as e:
        print(f"⚠️ Erreur chargement FilingAnalyzer: {e}")
        filing_analyzer = None
    
    # 3. PRÉPARATION DU DATAFRAME
    df = portfolio_df.copy()
    
    # Assurer colonnes nécessaires
    required_cols = {
        'Risk Score': 0.0,
        'Impact Est. Loss %': 0.0,
        'Impact Est. Loss': 0.0,
        'Supply Chain Risk': 0.0,
        'Geographic Risk': 0.0,
        'Direct Risk': 0.0,
        'Weight': 0.0
    }
    
    for col, default_val in required_cols.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Inférer thèmes si absent
    if 'Risk Theme' not in df.columns:
        df['Risk Theme'] = df['Company'].apply(
            lambda c: _infer_company_theme(c, COMPANY_THEME_KEYWORDS)
        )
    
    # 4. ANALYSE DÉTAILLÉE PAR ENTREPRISE
    print("🔍 Analyse détaillée par entreprise...")
    
    lower_text = reg_text.lower()
    cfg_overrides = extracted.get('config', {})
    params = DEFAULT_PARAMS.copy()
    params.update(cfg_overrides.get('params', {}))
    
    # Identifier thèmes ciblés par la réglementation
    targeted_themes = _merge_theme_sets(
        REG_TYPE_TARGETS.get(extracted['type_reg'], set()),
        _themes_from_text(lower_text, TEXT_THEME_KEYWORDS),
        _themes_from_text(' '.join(extracted.get('entities', [])), TEXT_THEME_KEYWORDS),
    )
    
    # Extraction paramètres
    mention_boost = float(params.get('mention_boost', 0.4))
    theme_boost = float(params.get('theme_boost', 0.25))
    measure_intensity = min(
        float(params.get('measure_intensity_scale', 0.05)) * len(extracted.get('measures', [])),
        float(params.get('measure_intensity_cap', 0.35))
    )
    penalty_boost = float(params.get('penalty_boost', 0.1))
    penalty_flag = penalty_boost if 'penalt' in lower_text or 'sanction' in lower_text else 0.0
    
    # Données enrichies pour chaque entreprise
    supply_chain_impacts = []
    
    # Itération sur chaque action
    for idx, row in df.iterrows():
        ticker = row['Symbol']
        company = row['Company']
        
        # === RISQUE DIRECT ===
        direct_risk = calculate_direct_risk(
            row, lower_text, extracted, targeted_themes,
            mention_boost, theme_boost, measure_intensity, penalty_flag
        )
        
        # === ANALYSE 10-K ===
        supply_chain_risk = 0.0
        geographic_risk = 0.0
        filing_info = {}
        
        if filing_analyzer:
            filing_info = filing_analyzer.extract_key_info(ticker)
            
            # Risque chaîne d'approvisionnement
            supply_chain_risk = calculate_supply_chain_risk(
                filing_info, extracted, lower_text
            )
            
            # Risque géographique
            geographic_risk = calculate_geographic_risk(
                filing_info.get('geographic_exposure', {}),
                extracted,
                lower_text
            )
        
        # === ANALYSE BEDROCK SPÉCIFIQUE (optionnel) ===
        bedrock_company_risk = 0.0
        if use_bedrock_runtime and bedrock and bedrock_analysis and filing_analyzer and bedrock.available and filing_info:
            try:
                company_analysis = bedrock.compare_regulation_with_filing(
                    reg_text[:8000],  # Limiter pour coûts
                    ticker,
                    filing_info
                )
                bedrock_company_risk = (
                    company_analysis.get('impact_direct', 0) +
                    company_analysis.get('impact_fournisseurs', 0) +
                    company_analysis.get('impact_geographique', 0)
                ) / 300.0  # Normaliser 0-1
            except Exception as e:
                bedrock_error = bedrock_error or str(e)
                bedrock.available = False
                use_bedrock_runtime = False
                print(f"⚠️ Analyse Bedrock désactivée pour {ticker}: {e}")
        
        # === SCORING GLOBAL PONDÉRÉ ===
        # Poids ajustables
        w_direct = 0.40
        w_supply = 0.25
        w_geo = 0.20
        w_bedrock = 0.15
        
        if not use_bedrock or bedrock_company_risk == 0:
            # Sans Bedrock, redistribuer le poids
            w_direct = 0.45
            w_supply = 0.30
            w_geo = 0.25
            w_bedrock = 0.0
        
        enhanced_risk = (
            direct_risk * w_direct +
            supply_chain_risk * w_supply +
            geographic_risk * w_geo +
            bedrock_company_risk * w_bedrock
        )
        
        enhanced_risk = min(enhanced_risk, 1.0)  # Cap à 1.0
        
        # === CALCUL PERTE ESTIMÉE ===
        loss_multiplier = float(params.get('loss_pct_multiplier', 15))
        max_loss_pct = float(params.get('max_loss_pct', 20))
        
        loss_pct = min(enhanced_risk * loss_multiplier, max_loss_pct)
        market_cap = row.get('Market Cap', 0) or 0
        impact_loss = market_cap * (loss_pct / 100.0)
        
        # === MISE À JOUR DATAFRAME ===
        df.at[idx, 'Direct Risk'] = direct_risk
        df.at[idx, 'Supply Chain Risk'] = supply_chain_risk
        df.at[idx, 'Geographic Risk'] = geographic_risk
        df.at[idx, 'Risk Score'] = enhanced_risk
        df.at[idx, 'Impact Est. Loss %'] = loss_pct
        df.at[idx, 'Impact Est. Loss'] = impact_loss
        
        # Drivers de risque
        drivers = []
        if direct_risk > 0.3:
            drivers.append("Impact direct élevé")
        if supply_chain_risk > 0.2:
            drivers.append(f"Exposition supply chain ({len(filing_info.get('suppliers', []))} fournisseurs)")
        if geographic_risk > 0.2:
            regions = list(filing_info.get('geographic_exposure', {}).keys())
            drivers.append(f"Exposition géographique ({', '.join(regions[:3])})")
        if not drivers:
            drivers.append("Impact généralisé")
        
        df.at[idx, 'Risk Drivers'] = ', '.join(drivers)
        
        # Stocker pour recommandations
        supply_chain_impacts.append({
            'ticker': ticker,
            'company': company,
            'risk_score': enhanced_risk,
            'filing_info': filing_info,
            'direct_risk': direct_risk,
            'supply_chain_risk': supply_chain_risk,
            'geographic_risk': geographic_risk
        })
    
    # 5. AGRÉGATION PORTFOLIO
    print("📈 Calcul des métriques de portefeuille...")
    
    weight_series = df['Weight']
    if weight_series.sum() > 0:
        portfolio_risk = float((df['Risk Score'] * weight_series).sum() / weight_series.sum())
    else:
        portfolio_risk = float(df['Risk Score'].mean())
    
    # Concentration par thème
    concentration = (
        df.groupby('Risk Theme')
        .agg(
            Weight_Exposure=('Weight', 'sum'),
            Avg_Risk=('Risk Score', 'mean'),
            Max_Risk=('Risk Score', 'max'),
            Est_Loss=('Impact Est. Loss', 'sum'),
            Count=('Symbol', 'count')
        )
        .reset_index()
        .sort_values('Weight_Exposure', ascending=False)
    )
    
    # 6. GÉNÉRATION RECOMMANDATIONS
    print("💡 Génération des recommandations...")
    
    portfolio_summary = {
        'portfolio_risk': portfolio_risk,
        'total_market_cap': df['Market Cap'].sum(),
        'total_estimated_loss': df['Impact Est. Loss'].sum(),
        'sector_concentration': concentration.to_dict('records')
    }
    
    top_risks_data = df.nlargest(15, 'Risk Score').to_dict('records')
    
    recommendations = generate_enhanced_recommendations(
        df,
        extracted,
        supply_chain_impacts,
        portfolio_summary,
        top_risks_data,
        bedrock_analysis,
        use_bedrock_runtime
    )
    
    print("✅ Analyse complète terminée")
    
    return df, extracted, portfolio_risk, concentration, recommendations


def calculate_direct_risk(
    row: pd.Series,
    lower_text: str,
    extracted: Dict,
    targeted_themes: set,
    mention_boost: float,
    theme_boost: float,
    measure_intensity: float,
    penalty_flag: float
) -> float:
    """Calcule le risque direct d'une entreprise"""
    
    risk = 0.05 + measure_intensity  # Baseline
    
    company_lower = str(row.get('Company', '')).lower()
    symbol_lower = str(row.get('Symbol', '')).lower()
    
    # Mention explicite (robuste)
    if len(symbol_lower) >= 2:
        if re.search(r'\b' + re.escape(symbol_lower) + r'\b', lower_text):
            risk += mention_boost
    
    if company_lower and re.search(r'\b' + re.escape(company_lower) + r'\b', lower_text):
        risk += mention_boost * 1.2  # Mention du nom complet = plus fort
    
    # Thème ciblé
    theme = row.get('Risk Theme', 'Other')
    if theme in targeted_themes:
        risk += theme_boost
    
    # Pénalités mentionnées
    if penalty_flag:
        risk += penalty_flag
    
    # Compliance
    if 'compliance' in lower_text or 'reporting' in lower_text:
        risk += 0.05
    
    return min(risk, 1.0)


def calculate_supply_chain_risk(
    filing_info: Dict,
    extracted: Dict,
    reg_text: str
) -> float:
    """Calcule le risque lié à la chaîne d'approvisionnement"""
    
    risk = 0.0
    
    suppliers = filing_info.get('suppliers', [])
    entities = extracted.get('entities', [])
    reg_lower = reg_text.lower()
    
    if not suppliers:
        return risk
    
    # Fournisseurs directement mentionnés
    for supplier in suppliers:
        if supplier.lower() in reg_lower:
            risk += 0.20  # Fournisseur impacté = risque significatif
    
    # Régions de fournisseurs
    geographic_entities = [
        e for e in entities 
        if any(country in e for country in ['China', 'Taiwan', 'Europe', 'US', 'Japan', 'India'])
    ]
    
    # Croiser avec localisation des fournisseurs (heuristique)
    for geo_entity in geographic_entities:
        for supplier in suppliers:
            if geo_entity.lower() in supplier.lower():
                risk += 0.10
    
    # Cap
    return min(risk, 0.60)


def calculate_geographic_risk(
    geo_exposure: Dict[str, float],
    extracted: Dict,
    reg_text: str
) -> float:
    """Calcule le risque basé sur l'exposition géographique"""
    
    risk = 0.0
    
    if not geo_exposure:
        return risk
    
    entities = extracted.get('entities', [])
    geo_entities = [
        e for e in entities 
        if any(country in e for country in ['China', 'Taiwan', 'Europe', 'US', 'Japan', 'India', 'Korea'])
    ]
    
    # Pour chaque région d'exposition
    for region, percentage in geo_exposure.items():
        # Vérifier si région mentionnée dans réglementation
        region_mentioned = any(
            region.lower() in entity.lower() 
            for entity in geo_entities
        )
        
        if region_mentioned:
            # Risque proportionnel à l'exposition
            risk += (percentage / 100.0) * 0.50
    
    # Cap
    return min(risk, 0.60)


def _merge_bedrock_extraction(nlp_extracted: Dict, bedrock_analysis: Dict) -> Dict:
    """Fusionne les résultats NLP et Bedrock"""
    
    merged = nlp_extracted.copy()
    
    # Enrichir avec données Bedrock
    if bedrock_analysis:
        merged['type_reg'] = bedrock_analysis.get('type_regulation', merged['type_reg'])
        
        # Ajouter entités Bedrock
        merged['entities'] = list(set(
            merged.get('entities', []) +
            bedrock_analysis.get('entreprises_a_risque', [])
        ))
        
        # Ajouter dates Bedrock
        merged['dates'] = list(set(
            merged.get('dates', []) +
            bedrock_analysis.get('dates_cles', [])
        ))
        
        # Ajouter mesures Bedrock
        merged['measures'] = list(set(
            merged.get('measures', []) +
            bedrock_analysis.get('mesures', [])
        ))
        
        # Métadonnées Bedrock
        merged['bedrock_analysis'] = bedrock_analysis
        merged['severite'] = bedrock_analysis.get('severite', 5)
        merged['resume_ia'] = bedrock_analysis.get('resume', '')
    
    return merged


def generate_enhanced_recommendations(
    df: pd.DataFrame,
    extracted: Dict,
    supply_chain_impacts: List[Dict],
    portfolio_summary: Dict,
    top_risks_data: List[Dict],
    bedrock_analysis: Dict,
    use_bedrock: bool
) -> List[Dict]:
    """Génère des recommandations actionnables et détaillées"""
    
    recommendations = []
    
    # === RECOMMANDATIONS BEDROCK (si disponible) ===
    if use_bedrock and bedrock_analysis:
        try:
            from aws_bedrock_integration import BedrockAnalyzer
            bedrock = BedrockAnalyzer()
            if bedrock.available:
                bedrock_recs = bedrock.generate_portfolio_recommendations(
                    portfolio_summary,
                    top_risks_data,
                    extracted
                )
                if bedrock_recs:
                    return bedrock_recs  # Utiliser recs IA générative si disponibles
        except Exception as e:
            print(f"⚠️ Erreur génération recs Bedrock: {e}")
    
    # === RECOMMANDATIONS RULE-BASED (fallback) ===
    
    # 1. Recommandations SELL/REDUCE pour top risques
    high_risk = df[df['Risk Score'] > 0.60].nlargest(10, 'Risk Score')
    
    for _, row in high_risk.iterrows():
        supply_info = next(
            (s for s in supply_chain_impacts if s['ticker'] == row['Symbol']),
            None
        )
        
        rec = {
            'action': 'REDUCE' if row['Risk Score'] < 0.75 else 'SELL',
            'ticker': row['Symbol'],
            'company': row['Company'],
            'current_weight': f"{row['Weight']:.2f}%",
            'risk_score': f"{row['Risk Score']:.1%}",
            'estimated_loss': f"${row['Impact Est. Loss']/1e9:.2f}B",
            'reason': row.get('Risk Drivers', 'Risque élevé identifié'),
            'urgency': 'HIGH' if row['Risk Score'] > 0.75 else 'MEDIUM',
            'timeline': extracted.get('dates', ['Court terme'])[0] if extracted.get('dates') else 'Court terme'
        }
        
        if supply_info:
            rec['suppliers_at_risk'] = supply_info['filing_info'].get('suppliers', [])[:3]
            rec['geographic_exposure'] = supply_info['filing_info'].get('geographic_exposure', {})
        
        recommendations.append(rec)
    
    # 2. Recommandations HEDGE pour risque modéré
    medium_risk = df[(df['Risk Score'] > 0.35) & (df['Risk Score'] <= 0.60)]
    
    for _, row in medium_risk.head(5).iterrows():
        rec = {
            'action': 'HEDGE',
            'ticker': row['Symbol'],
            'company': row['Company'],
            'hedge_strategy': 'Options put 6-12 mois',
            'strike': f"-{row['Impact Est. Loss %']:.0f}%",
            'reason': 'Risque modéré, protection recommandée',
            'cost_estimate': f"~{row['Impact Est. Loss %'] * 0.15:.1f}% du nominal",
            'urgence': 'MEDIUM'
        }
        recommendations.append(rec)
    
    # 3. Rotation sectorielle
    concentration = df.groupby('Risk Theme').agg({
        'Risk Score': 'mean',
        'Weight': 'sum'
    }).reset_index()
    
    high_risk_sectors = concentration[concentration['Risk Score'] > 0.40]
    
    for _, sector_row in high_risk_sectors.iterrows():
        safe_sectors = concentration[
            (concentration['Risk Score'] < 0.30) &
            (concentration['Risk Theme'] != sector_row['Risk Theme'])
        ].nsmallest(3, 'Risk Score')
        
        rec = {
            'action': 'SECTOR_ROTATION',
            'from_sector': sector_row['Risk Theme'],
            'sector_risk': f"{sector_row['Risk Score']:.1%}",
            'sector_weight': f"{sector_row['Weight']:.2f}%",
            'recommendation': f"Réduire exposition {sector_row['Risk Theme']} de {sector_row['Weight'] * 0.25:.2f}%",
            'target_sectors': safe_sectors['Risk Theme'].tolist(),
            'urgence': 'MEDIUM'
        }
        recommendations.append(rec)
    
    # 4. Monitoring pour faible risque
    if df['Risk Score'].mean() < 0.25:
        recommendations.append({
            'action': 'MONITOR',
            'recommendation': 'Risque global faible. Maintenir surveillance régulière.',
            'next_review': 'Dans 3 mois',
            'urgence': 'LOW'
        })
    
    return recommendations


if __name__ == "__main__":
    print("Module d'analyse réglementaire enrichi chargé")
    print("Utilisation: from reg_analysis_enhanced import analyze_reg_impact_enhanced")
