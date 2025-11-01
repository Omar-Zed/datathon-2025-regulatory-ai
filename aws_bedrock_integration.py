# aws_bedrock_integration.py - Intégration AWS Bedrock pour analyse IA générative
"""
Module d'intégration avec AWS Bedrock (Claude) pour analyse avancée 
des textes réglementaires et génération de recommandations stratégiques.
"""

import boto3
import json
from typing import Dict, List, Optional
import os
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError


class BedrockAnalyzer:
    """Interface avec AWS Bedrock pour analyse réglementaire via IA générative"""
    
    def __init__(self, region: str = 'us-east-1', model: str = 'claude-3-sonnet'):
        """
        Initialise le client Bedrock.
        
        Args:
            region: Région AWS (défaut: us-east-1)
            model: Modèle à utiliser ('claude-3-sonnet' ou 'claude-3-haiku' pour plus rapide)
        """
        self.region = region
        self.last_error: Optional[str] = None
        
        # Mapping des modèles
        model_ids = {
            'claude-3-opus': 'anthropic.claude-3-opus-20240229-v1:0',
            'claude-3-sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
            'claude-3-haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
        }
        
        self.model_id = model_ids.get(model, model_ids['claude-3-sonnet'])
        
        try:
            self.client = boto3.client('bedrock-runtime', region_name=region)
            self.available = True
        except Exception as e:
            self.last_error = str(e)
            print(f"⚠️ Bedrock non disponible: {e}")
            print("Mode dégradé: utilisation NLP basique seulement")
            self.available = False
    
    def analyze_regulatory_text(self, reg_text: str, filing_text: Optional[str] = None) -> Dict:
        """
        Analyse approfondie d'un texte réglementaire avec IA générative.
        
        Args:
            reg_text: Texte de la réglementation à analyser
            filing_text: Optionnel - Contexte additionnel (extrait de 10-K)
            
        Returns:
            Dictionnaire avec analyse structurée:
            - type_regulation: Classification du type de règlement
            - secteurs_impactes: Liste des secteurs économiques touchés
            - entreprises_a_risque: Types d'entreprises spécifiquement visées
            - dates_cles: Timeline d'application
            - mesures: Liste des obligations/restrictions
            - implications_indirectes: Impacts chaîne approvisionnement/géo
            - severite: Score 1-10 de l'impact
            - resume: Résumé exécutif
        """
        if not self.available:
            return self._empty_analysis()
        
        # Construire le prompt pour Claude
        prompt = self._build_regulatory_analysis_prompt(reg_text, filing_text)
        
        try:
            response = self._invoke_claude(prompt, max_tokens=4000, temperature=0.3)
            analysis = json.loads(response)
            return analysis
        
        except json.JSONDecodeError as e:
            print(f"⚠️ Erreur parsing JSON Bedrock: {e}")
            print(f"Réponse brute: {response[:200]}...")
            return self._empty_analysis()
        
        except Exception as e:
            self.last_error = str(e)
            print(f"⚠️ Erreur Bedrock analyze_regulatory_text: {e}")
            return self._empty_analysis()
    
    def compare_regulation_with_filing(
        self, 
        reg_text: str, 
        ticker: str, 
        filing_info: Dict
    ) -> Dict:
        """
        Compare une réglementation avec les données spécifiques d'une entreprise.
        
        Args:
            reg_text: Texte réglementaire
            ticker: Symbole boursier
            filing_info: Informations extraites du 10-K
            
        Returns:
            Dictionnaire avec:
            - impact_direct: Score 0-100 de l'impact direct
            - impact_fournisseurs: Score 0-100 via chaîne approvisionnement
            - impact_geographique: Score 0-100 exposition géographique
            - fournisseurs_a_risque: Liste des fournisseurs impactés
            - regions_a_risque: Liste des régions impactées
            - recommandation: Action recommandée
            - justification: Explication détaillée
        """
        if not self.available:
            return self._empty_company_analysis()
        
        prompt = self._build_company_comparison_prompt(reg_text, ticker, filing_info)
        
        try:
            response = self._invoke_claude(prompt, max_tokens=2500, temperature=0.3)
            analysis = json.loads(response)
            return analysis
        
        except Exception as e:
            self.last_error = str(e)
            print(f"⚠️ Erreur compare_regulation_with_filing pour {ticker}: {e}")
            return self._empty_company_analysis()
    
    def generate_portfolio_recommendations(
        self, 
        portfolio_summary: Dict,
        top_risks: List[Dict],
        regulatory_context: Dict
    ) -> List[Dict]:
        """
        Génère des recommandations stratégiques pour le portefeuille.
        
        Args:
            portfolio_summary: Résumé du portefeuille (risque global, concentration, etc.)
            top_risks: Liste des actions les plus à risque
            regulatory_context: Contexte réglementaire analysé
            
        Returns:
            Liste de recommandations actionnables avec:
            - action: Type d'action (SELL/REDUCE/HEDGE/ROTATE)
            - tickers_concernes: Liste des tickers
            - justification: Raison quantifiée
            - timeline: Échéance suggérée
            - impact_estime: Impact financier estimé
            - urgence: HIGH/MEDIUM/LOW
        """
        if not self.available:
            return self._fallback_recommendations()
        
        prompt = self._build_recommendations_prompt(
            portfolio_summary, top_risks, regulatory_context
        )
        
        try:
            response = self._invoke_claude(prompt, max_tokens=3500, temperature=0.5)
            recommendations = json.loads(response)
            
            # S'assurer que c'est une liste
            if isinstance(recommendations, dict):
                recommendations = [recommendations]
            
            return recommendations
        
        except Exception as e:
            self.last_error = str(e)
            print(f"⚠️ Erreur generate_portfolio_recommendations: {e}")
            return self._fallback_recommendations()
    
    def extract_supply_chain_insights(
        self, 
        reg_text: str, 
        suppliers_list: List[str],
        geographic_data: Dict[str, float]
    ) -> Dict:
        """
        Analyse les implications réglementaires sur une chaîne d'approvisionnement.
        
        Args:
            reg_text: Texte réglementaire
            suppliers_list: Liste des fournisseurs
            geographic_data: Exposition géographique {région: %}
            
        Returns:
            Analyse des risques supply chain
        """
        if not self.available or not suppliers_list:
            return {"risk_level": "unknown", "affected_suppliers": []}
        
        prompt = f"""Analysez l'impact de cette réglementation sur la chaîne d'approvisionnement:

<regulation>
{reg_text[:10000]}
</regulation>

<supply_chain>
Fournisseurs: {', '.join(suppliers_list[:15])}
Exposition géographique: {json.dumps(geographic_data)}
</supply_chain>

Identifiez:
1. Fournisseurs directement impactés
2. Risques géographiques
3. Score de risque global (0-100)
4. Recommandations de mitigation

Format JSON:
{{
  "risk_level": "HIGH|MEDIUM|LOW",
  "risk_score": int,
  "affected_suppliers": ["list"],
  "geographic_risks": {{"region": "risk description"}},
  "mitigation_strategies": ["list"]
}}
"""
        
        try:
            response = self._invoke_claude(prompt, max_tokens=2000)
            return json.loads(response)
        except:
            return {"risk_level": "unknown", "affected_suppliers": []}
    
    def _invoke_claude(
        self, 
        prompt: str, 
        max_tokens: int = 2000, 
        temperature: float = 0.3
    ) -> str:
        """
        Invoque le modèle Claude via Bedrock.
        
        Args:
            prompt: Texte du prompt
            max_tokens: Nombre maximum de tokens en réponse
            temperature: Température (créativité) de la génération
            
        Returns:
            Réponse textuelle du modèle
        """
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "top_p": 0.9
        })
        
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body
            )
        except (ClientError, BotoCoreError, NoCredentialsError) as e:
            self.last_error = str(e)
            self.available = False
            raise
        
        response_body = json.loads(response['body'].read())
        content = response_body['content'][0]['text']
        
        return content
    
    def _build_regulatory_analysis_prompt(
        self, 
        reg_text: str, 
        filing_text: Optional[str]
    ) -> str:
        """Construit le prompt pour l'analyse réglementaire"""
        
        filing_context = ""
        if filing_text:
            filing_context = f"""
<company_context>
{filing_text[:15000]}
</company_context>
"""
        
        prompt = f"""Vous êtes un analyste financier expert spécialisé dans l'évaluation d'impact réglementaire sur les marchés financiers.

<regulatory_document>
{reg_text[:20000]}
</regulatory_document>
{filing_context}

Analysez ce document réglementaire et fournissez une évaluation structurée:

1. **Type de réglementation**: Quelle catégorie (fiscale, environnementale, technologique, etc.)?
2. **Secteurs économiques impactés**: Quels secteurs du S&P 500 sont concernés?
3. **Entreprises à risque**: Types ou noms d'entreprises spécifiquement visées
4. **Timeline**: Dates d'application, d'entrée en vigueur, échéances
5. **Mesures contraignantes**: Obligations, taxes, restrictions, pénalités
6. **Implications indirectes**: 
   - Impact sur chaînes d'approvisionnement
   - Implications géographiques (pays/régions affectés)
   - Effets en cascade
7. **Sévérité globale**: Score de 1-10 de l'impact sur les marchés
8. **Résumé exécutif**: En 2-3 phrases, l'essentiel pour un investisseur

⚠️ IMPORTANT: Répondez UNIQUEMENT avec un objet JSON valide, sans texte avant ou après.

Format JSON attendu:
{{
  "type_regulation": "string - catégorie principale",
  "secteurs_impactes": ["liste des secteurs"],
  "entreprises_a_risque": ["types ou noms d'entreprises"],
  "dates_cles": ["dates importantes au format YYYY-MM-DD ou description"],
  "mesures": ["liste des obligations/restrictions/taxes"],
  "implications_indirectes": {{
    "chaine_approvisionnement": "description des impacts supply chain",
    "geographique": "description des impacts par région",
    "autres": "autres effets indirects"
  }},
  "severite": 7,
  "resume": "résumé en 2-3 phrases"
}}
"""
        return prompt
    
    def _build_company_comparison_prompt(
        self, 
        reg_text: str, 
        ticker: str, 
        filing_info: Dict
    ) -> str:
        """Construit le prompt pour la comparaison entreprise vs réglementation"""
        
        prompt = f"""Évaluez l'impact spécifique de cette réglementation sur l'entreprise {ticker}.

<regulation>
{reg_text[:12000]}
</regulation>

<company_data>
Ticker: {ticker}

Fournisseurs principaux:
{chr(10).join([f"- {s}" for s in filing_info.get('suppliers', [])[:10]])}

Exposition géographique:
{json.dumps(filing_info.get('geographic_exposure', {}), indent=2)}

Produits principaux:
{chr(10).join([f"- {p}" for p in filing_info.get('main_products', [])[:5]])}

Dépendances critiques:
{chr(10).join([f"- {d}" for d in filing_info.get('dependencies', [])[:5]])}
</company_data>

Analysez:
1. **Impact direct** (0-100): L'entreprise est-elle directement visée?
2. **Impact via fournisseurs** (0-100): Ses fournisseurs sont-ils affectés?
3. **Impact géographique** (0-100): Son exposition géographique l'expose-t-elle?
4. **Fournisseurs à risque**: Lesquels sont impactés?
5. **Régions à risque**: Quelles zones d'opération sont touchées?
6. **Recommandation**: Quelle action (HOLD/REDUCE/SELL/HEDGE)?
7. **Justification**: Pourquoi cette recommandation? Avec données quantifiées.

⚠️ Répondez UNIQUEMENT en JSON valide:

{{
  "impact_direct": 65,
  "impact_fournisseurs": 80,
  "impact_geographique": 40,
  "fournisseurs_a_risque": ["Nom1", "Nom2"],
  "regions_a_risque": ["Région1", "Région2"],
  "recommandation": "REDUCE",
  "justification": "Justification détaillée avec chiffres",
  "score_risque_global": 75
}}
"""
        return prompt
    
    def _build_recommendations_prompt(
        self,
        portfolio_summary: Dict,
        top_risks: List[Dict],
        regulatory_context: Dict
    ) -> str:
        """Construit le prompt pour générer les recommandations stratégiques"""
        
        prompt = f"""Vous êtes un gestionnaire de portefeuille senior. Générez des recommandations stratégiques actionnables.

<portfolio_analysis>
Risque global du portefeuille: {portfolio_summary.get('portfolio_risk', 'N/A')}
Capitalisation totale: ${portfolio_summary.get('total_market_cap', 0)/1e12:.2f}T
Perte estimée potentielle: ${portfolio_summary.get('total_estimated_loss', 0)/1e9:.2f}B

Top 10 positions à risque:
{json.dumps(top_risks[:10], indent=2)}

Concentration sectorielle:
{json.dumps(portfolio_summary.get('sector_concentration', {}), indent=2)}
</portfolio_analysis>

<regulatory_context>
Type: {regulatory_context.get('type_reg', 'N/A')}
Sévérité: {regulatory_context.get('severite', 'N/A')}/10
Secteurs visés: {', '.join(regulatory_context.get('secteurs_impactes', []))}
Timeline: {', '.join(regulatory_context.get('dates_cles', []))}
</regulatory_context>

Générez 6-8 recommandations stratégiques avec:
- Actions concrètes (SELL/REDUCE/HEDGE/ROTATE/MONITOR)
- Tickers spécifiques concernés
- Justification quantifiée (montants, pourcentages, timeline)
- Impact financier estimé
- Niveau d'urgence (HIGH si >60 jours, MEDIUM si >6 mois, LOW sinon)
- Alternatives d'investissement si pertinent

⚠️ Répondez UNIQUEMENT avec un array JSON valide:

[
  {{
    "action": "SELL",
    "tickers_concernes": ["NVDA", "AMD"],
    "secteur": "Technology - Semiconductors",
    "justification": "Exposition directe aux tarifs semi-conducteurs chinois. Impact estimé -15% sur 12 mois.",
    "montant_concerne": "$450B en capitalisation",
    "timeline": "Avant le 15/01/2026",
    "impact_estime": "-$67.5B en valeur de portefeuille",
    "urgence": "HIGH",
    "alternatives": ["ASML (Europe)", "TSM (Taiwan)"]
  }},
  {{
    "action": "HEDGE",
    "tickers_concernes": ["AAPL"],
    "secteur": "Technology - Consumer Electronics",
    "justification": "Impact indirect via chaîne approvisionnement asiatique",
    "strategie_hedge": "Options put 6 mois, strike -10%",
    "cout_estime": "2.3% du nominal",
    "protection": "Protège contre baisse >10%",
    "urgence": "MEDIUM"
  }}
]
"""
        return prompt
    
    def _empty_analysis(self) -> Dict:
        """Analyse vide en cas d'erreur"""
        return {
            "type_regulation": "Unknown",
            "secteurs_impactes": [],
            "entreprises_a_risque": [],
            "dates_cles": [],
            "mesures": [],
            "implications_indirectes": {
                "chaine_approvisionnement": "Analyse non disponible",
                "geographique": "Analyse non disponible"
            },
            "severite": 0,
            "resume": "Analyse IA générative non disponible. Utilisation du fallback NLP."
        }
    
    def _empty_company_analysis(self) -> Dict:
        """Analyse entreprise vide en cas d'erreur"""
        return {
            "impact_direct": 0,
            "impact_fournisseurs": 0,
            "impact_geographique": 0,
            "fournisseurs_a_risque": [],
            "regions_a_risque": [],
            "recommandation": "HOLD",
            "justification": "Analyse non disponible",
            "score_risque_global": 0
        }
    
    def _fallback_recommendations(self) -> List[Dict]:
        """Recommandations génériques en cas d'erreur"""
        return [
            {
                "action": "MONITOR",
                "tickers_concernes": [],
                "justification": "Analyse IA générative non disponible. Surveillez les positions à risque identifiées.",
                "timeline": "Continu",
                "urgence": "MEDIUM"
            }
        ]


def test_bedrock_integration():
    """Test basique de l'intégration Bedrock"""
    
    analyzer = BedrockAnalyzer(model='claude-3-haiku')  # Plus rapide pour test
    
    # Test avec un texte réglementaire exemple
    test_reg_text = """
    The Inflation Reduction Act of 2022 imposes a 15% minimum tax on corporations 
    with average annual adjusted financial statement income exceeding $1 billion. 
    This affects primarily large technology companies and pharmaceutical firms.
    The act also includes provisions for renewable energy tax credits.
    Implementation date: January 1, 2023.
    """
    
    if analyzer.available:
        print("✅ Bedrock disponible - Test d'analyse...")
        
        result = analyzer.analyze_regulatory_text(test_reg_text)
        
        print("\nRésultat de l'analyse:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("❌ Bedrock non disponible - Mode dégradé activé")


if __name__ == "__main__":
    test_bedrock_integration()
