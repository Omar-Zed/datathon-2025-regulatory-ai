# visualization_utils.py - Graphiques et visualisations avancées
"""
Module de visualisations interactives pour l'analyse de portefeuille.
Utilise Plotly pour des graphiques interactifs de qualité professionnelle.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def create_risk_weight_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Crée un scatter plot interactif risque vs poids dans le portefeuille.
    
    Args:
        df: DataFrame avec colonnes 'Weight', 'Risk Score', 'Market Cap', 'Risk Theme', etc.
        
    Returns:
        Figure Plotly
    """
    fig = px.scatter(
        df,
        x='Weight',
        y='Risk Score',
        size='Market Cap',
        color='Risk Theme',
        hover_data={
            'Symbol': True,
            'Company': True,
            'Impact Est. Loss %': ':.2f',
            'Weight': ':.2f',
            'Risk Score': ':.2%',
            'Market Cap': ':,.0f'
        },
        title='📊 Matrice Risque-Poids du Portefeuille S&P 500',
        labels={
            'Weight': 'Poids dans le S&P 500 (%)',
            'Risk Score': 'Score de Risque',
            'Risk Theme': 'Secteur'
        },
        size_max=60,
        template='plotly_white'
    )
    
    # Lignes de référence
    avg_risk = df['Risk Score'].mean()
    fig.add_hline(
        y=avg_risk,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Risque moyen: {avg_risk:.2%}",
        annotation_position="right"
    )
    
    # Zone de haute attention (haut à droite)
    fig.add_shape(
        type="rect",
        x0=df['Weight'].quantile(0.75),
        x1=df['Weight'].max(),
        y0=df['Risk Score'].quantile(0.75),
        y1=1.0,
        fillcolor="red",
        opacity=0.1,
        line_width=0,
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_sector_concentration(concentration: pd.DataFrame) -> go.Figure:
    """
    Crée un graphique de concentration sectorielle avec risque.
    
    Args:
        concentration: DataFrame groupé par secteur avec métriques
        
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    
    # Barres principales (poids)
    fig.add_trace(go.Bar(
        x=concentration['Risk Theme'],
        y=concentration['Weight_Exposure'],
        name='Poids dans portefeuille (%)',
        marker_color='lightblue',
        text=concentration['Weight_Exposure'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Poids: %{y:.2f}%<extra></extra>'
    ))
    
    # Ligne de risque moyen
    fig.add_trace(go.Scatter(
        x=concentration['Risk Theme'],
        y=concentration['Avg_Risk'] * 100,  # Convertir en %
        name='Risque Moyen (%)',
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=10),
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Risque: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='🎯 Concentration Sectorielle et Niveau de Risque',
        xaxis_title='Secteur',
        yaxis_title='Poids Total (%)',
        yaxis2=dict(
            title='Risque Moyen (%)',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        height=500,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig


def create_top_risks_bar(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Crée un graphique en barres des actions les plus à risque.
    
    Args:
        df: DataFrame du portefeuille
        top_n: Nombre d'actions à afficher
        
    Returns:
        Figure Plotly
    """
    top_risks = df.nlargest(top_n, 'Risk Score')
    
    # Colormap selon niveau de risque
    colors = top_risks['Risk Score'].apply(
        lambda x: 'darkred' if x > 0.7 else ('orangered' if x > 0.5 else 'orange')
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_risks['Symbol'],
        y=top_risks['Risk Score'],
        marker_color=colors,
        text=top_risks['Risk Score'].apply(lambda x: f'{x:.1%}'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                      'Company: ' + top_risks['Company'] + '<br>' +
                      'Risk Score: %{y:.2%}<br>' +
                      'Drivers: ' + top_risks['Risk Drivers'] + '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'⚠️ Top {top_n} Actions à Risque Élevé',
        xaxis_title='Ticker',
        yaxis_title='Score de Risque',
        yaxis=dict(tickformat='.0%'),
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    # Ligne de seuil critique
    fig.add_hline(
        y=0.60,
        line_dash="dash",
        line_color="red",
        annotation_text="Seuil critique (60%)",
        annotation_position="right"
    )
    
    return fig


def create_loss_estimation(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Graphique des pertes estimées en capitalisation.
    
    Args:
        df: DataFrame du portefeuille
        top_n: Nombre d'actions à afficher
        
    Returns:
        Figure Plotly
    """
    top_losses = df.nlargest(top_n, 'Impact Est. Loss')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_losses['Symbol'],
        y=top_losses['Impact Est. Loss'] / 1e9,  # Convertir en milliards
        marker_color='crimson',
        text=top_losses['Impact Est. Loss'].apply(lambda x: f'${x/1e9:.1f}B'),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                      'Company: ' + top_losses['Company'] + '<br>' +
                      'Estimated Loss: $%{y:.2f}B<br>' +
                      'Loss %: ' + top_losses['Impact Est. Loss %'].apply(lambda x: f'{x:.1f}%') +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'💸 Top {top_n} - Pertes Estimées en Capitalisation',
        xaxis_title='Ticker',
        yaxis_title='Perte Estimée (Milliards $)',
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_risk_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogramme de la distribution des scores de risque.
    
    Args:
        df: DataFrame du portefeuille
        
    Returns:
        Figure Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['Risk Score'],
        nbinsx=25,
        marker_color='steelblue',
        marker_line_color='white',
        marker_line_width=1.5,
        hovertemplate='Score: %{x:.2%}<br>Count: %{y}<extra></extra>'
    ))
    
    # Ligne de moyenne
    mean_risk = df['Risk Score'].mean()
    fig.add_vline(
        x=mean_risk,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Moyenne: {mean_risk:.2%}",
        annotation_position="top right"
    )
    
    # Ligne de médiane
    median_risk = df['Risk Score'].median()
    fig.add_vline(
        x=median_risk,
        line_dash="dot",
        line_color="orange",
        line_width=2,
        annotation_text=f"Médiane: {median_risk:.2%}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title='📈 Distribution des Scores de Risque',
        xaxis_title='Score de Risque',
        yaxis_title='Nombre d\'Actions',
        xaxis=dict(tickformat='.0%'),
        height=450,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def create_risk_components_comparison(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Compare les composantes de risque (direct, supply chain, géographique).
    
    Args:
        df: DataFrame avec colonnes Direct Risk, Supply Chain Risk, Geographic Risk
        top_n: Nombre d'actions à afficher
        
    Returns:
        Figure Plotly
    """
    top_risks = df.nlargest(top_n, 'Risk Score')
    
    fig = go.Figure()
    
    # Risque direct
    fig.add_trace(go.Bar(
        name='Risque Direct',
        x=top_risks['Symbol'],
        y=top_risks['Direct Risk'],
        marker_color='crimson',
        hovertemplate='Direct: %{y:.2%}<extra></extra>'
    ))
    
    # Risque supply chain
    fig.add_trace(go.Bar(
        name='Risque Supply Chain',
        x=top_risks['Symbol'],
        y=top_risks['Supply Chain Risk'],
        marker_color='orange',
        hovertemplate='Supply Chain: %{y:.2%}<extra></extra>'
    ))
    
    # Risque géographique
    fig.add_trace(go.Bar(
        name='Risque Géographique',
        x=top_risks['Symbol'],
        y=top_risks['Geographic Risk'],
        marker_color='gold',
        hovertemplate='Géographique: %{y:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'🔍 Décomposition du Risque - Top {top_n}',
        xaxis_title='Ticker',
        yaxis_title='Niveau de Risque',
        yaxis=dict(tickformat='.0%'),
        barmode='stack',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Matrice de corrélation entre différentes métriques de risque.
    
    Args:
        df: DataFrame du portefeuille
        
    Returns:
        Figure Plotly
    """
    # Sélectionner colonnes numériques pertinentes
    cols = ['Risk Score', 'Direct Risk', 'Supply Chain Risk', 
            'Geographic Risk', 'Weight', 'Impact Est. Loss %']
    
    # Filtrer colonnes existantes
    available_cols = [col for col in cols if col in df.columns]
    
    if len(available_cols) < 2:
        # Pas assez de données pour corrélation
        fig = go.Figure()
        fig.add_annotation(
            text="Données insuffisantes pour matrice de corrélation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    corr_matrix = df[available_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='🔗 Matrice de Corrélation des Métriques de Risque',
        labels=dict(color="Corrélation")
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_geographic_exposure_chart(
    filing_analyzer,
    top_tickers: List[str]
) -> go.Figure:
    """
    Graphique d'exposition géographique pour les top tickers.
    
    Args:
        filing_analyzer: Instance de FilingAnalyzer
        top_tickers: Liste des tickers à analyser
        
    Returns:
        Figure Plotly
    """
    geo_data = []
    
    for ticker in top_tickers[:15]:  # Limiter à 15 pour lisibilité
        filing_info = filing_analyzer.extract_key_info(ticker)
        geo_exposure = filing_info.get('geographic_exposure', {})
        
        for region, percentage in geo_exposure.items():
            geo_data.append({
                'Ticker': ticker,
                'Region': region,
                'Exposure': percentage
            })
    
    if not geo_data:
        fig = go.Figure()
        fig.add_annotation(
            text="Données géographiques non disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    geo_df = pd.DataFrame(geo_data)
    
    fig = px.bar(
        geo_df,
        x='Ticker',
        y='Exposure',
        color='Region',
        title='🌍 Exposition Géographique des Actions à Risque',
        labels={'Exposure': 'Exposition (%)'},
        barmode='stack',
        height=500
    )
    
    fig.update_layout(
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_timeline_chart(extracted: Dict) -> go.Figure:
    """
    Timeline des dates clés réglementaires.
    
    Args:
        extracted: Dictionnaire d'extraction réglementaire
        
    Returns:
        Figure Plotly
    """
    dates = extracted.get('dates', [])
    
    if not dates:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucune date clé identifiée dans la réglementation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Parser dates (simplifié)
    timeline_data = []
    for i, date_str in enumerate(dates[:10]):  # Max 10 dates
        timeline_data.append({
            'Date': date_str,
            'Event': f'Échéance {i+1}',
            'Y': 1
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = px.scatter(
        timeline_df,
        x='Date',
        y='Y',
        text='Event',
        title='📅 Timeline des Dates Clés Réglementaires',
        height=300
    )
    
    fig.update_traces(
        marker=dict(size=20, color='red'),
        textposition='top center'
    )
    
    fig.update_layout(
        showlegend=False,
        yaxis=dict(visible=False),
        template='plotly_white'
    )
    
    return fig


def create_all_visualizations(
    df: pd.DataFrame,
    concentration: pd.DataFrame,
    extracted: Dict,
    filing_analyzer = None
) -> Dict[str, go.Figure]:
    """
    Génère toutes les visualisations en une seule fois.
    
    Args:
        df: DataFrame analysé du portefeuille
        concentration: DataFrame de concentration sectorielle
        extracted: Informations extraites de la réglementation
        filing_analyzer: Optionnel, instance de FilingAnalyzer
        
    Returns:
        Dictionnaire {nom: figure} de toutes les visualisations
    """
    figs = {}
    
    figs['risk_weight_scatter'] = create_risk_weight_scatter(df)
    figs['sector_concentration'] = create_sector_concentration(concentration)
    figs['top_risks_bar'] = create_top_risks_bar(df)
    figs['loss_estimation'] = create_loss_estimation(df)
    figs['risk_distribution'] = create_risk_distribution(df)
    
    # Composantes de risque (si colonnes disponibles)
    if all(col in df.columns for col in ['Direct Risk', 'Supply Chain Risk', 'Geographic Risk']):
        figs['risk_components'] = create_risk_components_comparison(df)
    
    figs['correlation_heatmap'] = create_correlation_heatmap(df)
    
    # Exposition géographique (si filing_analyzer disponible)
    if filing_analyzer:
        top_tickers = df.nlargest(10, 'Risk Score')['Symbol'].tolist()
        figs['geographic_exposure'] = create_geographic_exposure_chart(
            filing_analyzer, top_tickers
        )
    
    figs['timeline'] = create_timeline_chart(extracted)
    
    return figs


if __name__ == "__main__":
    print("Module de visualisations chargé")
    print("Utilisation: from visualization_utils import create_all_visualizations")
