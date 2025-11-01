# filing_analyzer.py - Module d'analyse des rapports 10-K
"""
Module pour extraire et analyser les informations clés des rapports 10-K.
Utilisé pour identifier les chaînes d'approvisionnement, exposition géographique,
et facteurs de risque des entreprises du S&P 500.
"""

import os
import re
from typing import Dict, List, Optional
import json
from pathlib import Path


class FilingAnalyzer:
    """Analyseur de rapports 10-K pour extraction d'informations stratégiques"""
    
    def __init__(self, fillings_dir: str):
        """
        Initialise l'analyseur avec le répertoire contenant les rapports 10-K.
        
        Args:
            fillings_dir: Chemin vers le dossier 'fillings' contenant les sous-dossiers par ticker
        """
        self.fillings_dir = fillings_dir
        self.filing_cache = {}  # Cache pour éviter de relire les mêmes fichiers
        
        # Patterns de recherche optimisés
        self.supplier_patterns = [
            r'suppliers?\s+(?:include|are|such as|primarily)\s+([^.]{10,200})',
            r'manufacturing.{0,50}(?:by|from|with)\s+([A-Z][a-zA-Z\s,&]{5,50})',
            r'sourced?\s+(?:from|through)\s+([A-Z][a-zA-Z\s,&]{5,50})',
            r'key\s+suppliers?\s+(?:include|are)\s+([^.]{10,200})',
            r'depend(?:s|ent)\s+on\s+([A-Z][a-zA-Z\s,&]{5,50})',
        ]
        
        self.country_patterns = {
            'United States': r'\b(?:United States|U\.S\.|USA|America)\b',
            'China': r'\b(?:China|Chinese|PRC)\b',
            'Europe': r'\b(?:Europe|European Union|EU)\b',
            'Japan': r'\b(?:Japan|Japanese)\b',
            'Taiwan': r'\b(?:Taiwan|Taiwanese)\b',
            'India': r'\bIndia\b',
            'Canada': r'\bCanada\b',
            'Mexico': r'\bMexico\b',
            'Korea': r'\b(?:Korea|Korean|South Korea)\b',
            'Singapore': r'\bSingapore\b',
        }
    
    def get_filing_path(self, ticker: str) -> Optional[str]:
        """
        Trouve le chemin du rapport 10-K pour un ticker donné.
        
        Args:
            ticker: Symbole boursier (ex: 'AAPL', 'MSFT')
            
        Returns:
            Chemin complet vers le fichier 10-K, ou None si non trouvé
        """
        ticker_dir = os.path.join(self.fillings_dir, ticker)
        
        if not os.path.exists(ticker_dir):
            return None
        
        # Chercher un fichier contenant '10-k' ou '10k' (insensible à la casse)
        for file in os.listdir(ticker_dir):
            if '10-k' in file.lower() or '10k' in file.lower():
                return os.path.join(ticker_dir, file)
        
        # Si aucun fichier explicitement nommé 10-K, prendre le premier fichier
        files = [f for f in os.listdir(ticker_dir) if os.path.isfile(os.path.join(ticker_dir, f))]
        if files:
            return os.path.join(ticker_dir, files[0])
        
        return None
    
    def extract_key_info(self, ticker: str) -> Dict:
        """
        Extrait toutes les informations clés d'un rapport 10-K.
        
        Args:
            ticker: Symbole boursier
            
        Returns:
            Dictionnaire contenant:
            - suppliers: Liste des fournisseurs identifiés
            - geographic_exposure: Dictionnaire {région: pourcentage}
            - risk_factors: Liste des facteurs de risque
            - revenue_by_region: Répartition du CA par région
            - main_products: Liste des principaux produits
            - dependencies: Liste des dépendances critiques
        """
        # Vérifier le cache
        if ticker in self.filing_cache:
            return self.filing_cache[ticker]
        
        # Trouver et lire le fichier
        filing_path = self.get_filing_path(ticker)
        if not filing_path:
            return self._empty_info(ticker)
        
        try:
            with open(filing_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Erreur lecture {ticker}: {e}")
            return self._empty_info(ticker)
        
        # Extraction des différentes informations
        info = {
            'ticker': ticker,
            'suppliers': self._extract_suppliers(content),
            'geographic_exposure': self._extract_geography(content),
            'risk_factors': self._extract_risks(content),
            'revenue_by_region': self._extract_revenue_geography(content),
            'main_products': self._extract_products(content),
            'dependencies': self._extract_dependencies(content),
            'key_customers': self._extract_customers(content),
            'manufacturing_locations': self._extract_manufacturing(content),
        }
        
        # Mettre en cache
        self.filing_cache[ticker] = info
        return info
    
    def _extract_suppliers(self, content: str) -> List[str]:
        """Extrait les noms de fournisseurs du rapport"""
        suppliers = set()
        
        # Recherche dans sections pertinentes
        sections = [
            self._extract_section(content, 'supply chain'),
            self._extract_section(content, 'suppliers'),
            self._extract_section(content, 'manufacturing'),
            self._extract_section(content, 'sourcing'),
            self._extract_section(content, 'vendor'),
        ]
        
        for section in sections:
            if not section:
                continue
            
            # Appliquer les patterns de recherche
            for pattern in self.supplier_patterns:
                matches = re.findall(pattern, section, re.I)
                for match in matches:
                    # Extraire noms propres (majuscules)
                    companies = re.findall(
                        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Co|Corporation|Limited))?)\b',
                        match
                    )
                    suppliers.update(companies)
        
        # Filtrer les faux positifs
        filtered = [
            s for s in suppliers 
            if len(s) > 3 and s not in ['The', 'Our', 'Company', 'Item', 'Part', 'Form', 'Table', 'Note']
        ]
        
        return sorted(list(set(filtered)))[:30]  # Top 30 uniques
    
    def _extract_geography(self, content: str) -> Dict[str, float]:
        """Extrait l'exposition géographique avec pourcentages si disponibles"""
        geography = {}
        
        # Chercher section géographique
        geo_sections = [
            self._extract_section(content, 'geographic', chars=8000),
            self._extract_section(content, 'segment', chars=8000),
            self._extract_section(content, 'revenue by', chars=5000),
        ]
        
        for section in geo_sections:
            if not section:
                continue
            
            # Pour chaque pays/région, chercher des pourcentages
            for region, pattern in self.country_patterns.items():
                if re.search(pattern, section, re.I):
                    # Chercher des pourcentages dans les 200 caractères suivant la mention
                    context_pattern = pattern + r'.{0,200}?(\d{1,3})%'
                    matches = re.findall(context_pattern, section, re.I)
                    
                    if matches:
                        # Prendre le pourcentage le plus élevé (probablement le plus pertinent)
                        geography[region] = float(max(matches, key=float))
                    else:
                        # Si pas de pourcentage, juste noter la présence
                        if region not in geography:
                            geography[region] = 0.0
        
        return geography
    
    def _extract_risks(self, content: str) -> List[str]:
        """Extrait les facteurs de risque principaux"""
        risks = []
        
        # Section Risk Factors (obligatoire dans les 10-K)
        risk_section = self._extract_section(content, 'risk factors', chars=15000)
        
        if not risk_section:
            return risks
        
        # Thématiques de risque à rechercher
        risk_themes = {
            'Réglementaire': r'regulat(?:ion|ory)|compliance|law|legislation|government',
            'Chaîne approvisionnement': r'supply chain|supplier|manufacturing|shortage|disruption',
            'Géopolitique': r'geopolit|tariff|trade|sanction|war|tension',
            'Technologie': r'technolog|innovation|cyber|data breach|ai|digital',
            'Concurrence': r'compet(?:ition|itor)|market share|pricing pressure',
            'Financier': r'interest rate|currency|foreign exchange|liquidity|debt',
            'Environnemental': r'climate|environmental|sustainability|emissions',
        }
        
        # Extraire phrases de risque par thème
        for theme, pattern in risk_themes.items():
            # Chercher des phrases contenant le pattern
            sentences = re.findall(
                rf'[^.!?]*{pattern}[^.!?]*[.!?]',
                risk_section,
                re.I
            )
            
            if sentences:
                # Prendre la phrase la plus substantielle (la plus longue)
                best_sentence = max(sentences, key=len)
                if len(best_sentence) > 50:  # Ignorer phrases trop courtes
                    risks.append(f"{theme}: {best_sentence[:200]}...")
        
        return risks[:15]  # Top 15 risques
    
    def _extract_revenue_geography(self, content: str) -> Dict[str, float]:
        """Extrait la répartition du CA par région (similaire à geography mais plus spécifique)"""
        return self._extract_geography(content)
    
    def _extract_products(self, content: str) -> List[str]:
        """Extrait les principaux produits et services"""
        products = set()
        
        product_sections = [
            self._extract_section(content, 'products and services', chars=5000),
            self._extract_section(content, 'business segment', chars=5000),
            self._extract_section(content, 'principal products', chars=5000),
        ]
        
        product_patterns = [
            r'products?\s+(?:include|are|consist of)\s+([^.]{20,300})',
            r'offerings?\s+(?:include|are)\s+([^.]{20,300})',
            r'portfolio\s+(?:includes|consists of)\s+([^.]{20,300})',
        ]
        
        for section in product_sections:
            if not section:
                continue
            
            for pattern in product_patterns:
                matches = re.findall(pattern, section, re.I)
                for match in matches:
                    # Split sur virgules et "and"
                    items = re.split(r',\s*(?:and\s+)?|and\s+', match)
                    products.update([
                        item.strip() 
                        for item in items 
                        if 5 < len(item.strip()) < 100
                    ])
        
        return sorted(list(products))[:20]
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extrait les dépendances critiques mentionnées"""
        dependencies = set()
        
        dependency_patterns = [
            r'depend(?:s|ent|ence)\s+on\s+([^.]{15,150})',
            r'relian(?:ce|t)\s+on\s+([^.]{15,150})',
            r'critical\s+to.{0,30}([^.]{15,150})',
            r'essential\s+(?:to|for).{0,20}([^.]{15,150})',
        ]
        
        for pattern in dependency_patterns:
            matches = re.findall(pattern, content, re.I)
            dependencies.update([m.strip() for m in matches if len(m.strip()) > 15])
        
        return sorted(list(dependencies))[:15]
    
    def _extract_customers(self, content: str) -> List[str]:
        """Extrait les principaux clients si mentionnés"""
        customers = set()
        
        customer_section = self._extract_section(content, 'customers', chars=3000)
        
        if customer_section:
            # Chercher noms de sociétés
            companies = re.findall(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Co))?)\b',
                customer_section
            )
            customers.update([c for c in companies if len(c) > 3])
        
        return sorted(list(customers))[:10]
    
    def _extract_manufacturing(self, content: str) -> List[str]:
        """Extrait les sites de fabrication"""
        locations = set()
        
        manuf_section = self._extract_section(content, 'manufacturing', chars=5000)
        
        if manuf_section:
            # Chercher pays et villes
            for region in self.country_patterns.keys():
                if re.search(self.country_patterns[region], manuf_section, re.I):
                    locations.add(region)
        
        return sorted(list(locations))
    
    def _extract_section(self, content: str, section_name: str, chars: int = 3000) -> str:
        """
        Extrait une section spécifique du rapport.
        
        Args:
            content: Contenu complet du rapport
            section_name: Nom de la section à extraire
            chars: Nombre de caractères à extraire après le début de section
            
        Returns:
            Texte de la section, ou chaîne vide si non trouvée
        """
        # Normaliser pour la recherche
        pattern = rf'(?:^|\n).*{re.escape(section_name)}.*?(?:\n|$)'
        match = re.search(pattern, content, re.I | re.M)
        
        if not match:
            return ""
        
        start = match.start()
        return content[start:start + chars]
    
    def _empty_info(self, ticker: str) -> Dict:
        """Retourne un dictionnaire vide pour un ticker sans données"""
        return {
            'ticker': ticker,
            'suppliers': [],
            'geographic_exposure': {},
            'risk_factors': [],
            'revenue_by_region': {},
            'main_products': [],
            'dependencies': [],
            'key_customers': [],
            'manufacturing_locations': [],
        }
    
    def export_cache_to_json(self, output_path: str):
        """Exporte le cache vers un fichier JSON pour réutilisation"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.filing_cache, f, indent=2, ensure_ascii=False)
    
    def load_cache_from_json(self, input_path: str):
        """Charge le cache depuis un fichier JSON"""
        if os.path.exists(input_path):
            with open(input_path, 'r', encoding='utf-8') as f:
                self.filing_cache = json.load(f)


def test_filing_analyzer():
    """Fonction de test du module"""
    # Test sur quelques tickers
    analyzer = FilingAnalyzer('./jeu_de_donnees/fillings')
    
    test_tickers = ['AAPL', 'MSFT', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"Analyse de {ticker}")
        print(f"{'='*60}")
        
        info = analyzer.extract_key_info(ticker)
        
        print(f"\nFournisseurs ({len(info['suppliers'])}):")
        for supplier in info['suppliers'][:5]:
            print(f"  - {supplier}")
        
        print(f"\nExposition géographique:")
        for region, pct in sorted(info['geographic_exposure'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {region}: {pct:.1f}%")
        
        print(f"\nFacteurs de risque ({len(info['risk_factors'])}):")
        for risk in info['risk_factors'][:3]:
            print(f"  - {risk}")


if __name__ == "__main__":
    test_filing_analyzer()
