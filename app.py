# app.py — RegAI Portfolio Analyzer (avec AWS Lambda)
import os
import json
import base64
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
from botocore.exceptions import ClientError, NoRegionError, NoCredentialsError

from data_loader import (
    load_sp500_composition,
    load_stocks_performance,
    merge_sp500_data,
)
from reg_analysis import extract_reg_info
from analyze_reg_impact_enhanced import analyze_reg_impact_enhanced
from visualization_utils import create_all_visualizations

# -----------------------------------------------------------
# Configuration Streamlit
# -----------------------------------------------------------
st.set_page_config(page_title="RegAI Portfolio Analyzer", layout="wide")
st.title("Team 13")

# -----------------------------------------------------------
# AWS • Contrôles (sidebar)
# -----------------------------------------------------------
st.sidebar.title("⚙️ AWS / Exécution distante")
use_lambda = st.sidebar.toggle("Utiliser AWS Lambda pour l’analyse", value=True)
aws_region = st.sidebar.text_input("Région AWS", value=os.getenv("AWS_REGION", "us-west-2"))
lambda_name = st.sidebar.text_input("Nom de la fonction Lambda", value=os.getenv("REGAI_LAMBDA_NAME", "daily-laws-fr"))
aws_profile = st.sidebar.text_input("Profil AWS (optionnel)", value=os.getenv("AWS_PROFILE", ""))

with st.sidebar.expander("🩺 Diagnostics AWS"):
    def _session_and_lambda(region: str, profile: str | None):
        if profile:
            sess = boto3.Session(profile_name=profile, region_name=region)
        else:
            sess = boto3.Session(region_name=region)
        return sess, sess.client("lambda", region_name=region)

    colA, colB = st.columns(2)
    if colA.button("DryRun Lambda"):
        try:
            _, lc = _session_and_lambda(aws_region, aws_profile or None)
            r = lc.invoke(FunctionName=lambda_name, InvocationType="DryRun")
            st.success(f"DryRun → StatusCode={r.get('StatusCode')}")
        except Exception as e:
            st.error(f"DryRun error: {e}")

    if colB.button("Qui suis-je ? (STS)"):
        try:
            sess, _ = _session_and_lambda(aws_region, aws_profile or None)
            sts = sess.client("sts")
            st.json(sts.get_caller_identity())
        except Exception as e:
            st.error(f"STS error: {e}")

def _invoke_regai_lambda(text: str) -> tuple[dict, str]:
    """Appelle la Lambda RegAI et renvoie (payload_json, log_tail)."""
    sess = boto3.Session(
        profile_name=(aws_profile or None),
        region_name=aws_region
    )
    lc = sess.client("lambda", region_name=aws_region)
    resp = lc.invoke(
        FunctionName=lambda_name,
        InvocationType="RequestResponse",
        LogType="Tail",
        Payload=json.dumps({"regulation_text": text}).encode("utf-8"),
    )
    # Logs
    logs = base64.b64decode(resp.get("LogResult", "")).decode("utf-8", errors="ignore")
    # Payload
    raw = resp.get("Payload").read()
    outer = json.loads(raw) if raw else {}
    final = {}
    if isinstance(outer, dict) and "body" in outer:
        try:
            final = json.loads(outer["body"])
        except Exception:
            final = outer
    elif isinstance(outer, dict):
        final = outer
    return final, logs

# -----------------------------------------------------------
# Étapes UI
# -----------------------------------------------------------
st.sidebar.title("Étapes de l'Application")
steps = [
    "1. Chargement des Données S&P 500",
    "2. Upload et Extraction du Texte Réglementaire",
    "3. Modélisation de l'Impact",
    "4. Évaluation Globale du Portefeuille",
    "5. Visualisations et Recommandations"
]
selected_step = st.sidebar.selectbox("Sélectionnez une étape", steps)

# -----------------------------------------------------------
# Données S&P 500 (CSV / défaut)
# -----------------------------------------------------------
DEFAULT_DATA_DIR = Path(__file__).parent / "jeu_de_donnees"
if 'data_dir' not in st.session_state:
    st.session_state['data_dir'] = str(DEFAULT_DATA_DIR)

st.sidebar.header("Jeu de Données")
data_dir_input = st.sidebar.text_input("Répertoire jeu_de_donnees", st.session_state['data_dir'])
st.session_state['data_dir'] = data_dir_input.strip() or st.session_state['data_dir']
data_dir_path = Path(st.session_state['data_dir'])

if st.sidebar.button("Charger les fichiers par défaut"):
    comp_path = data_dir_path / "2025-08-15_composition_sp500.csv"
    perf_path = data_dir_path / "2025-09-26_stocks-performance.csv"
    if comp_path.exists() and perf_path.exists():
        df_comp = load_sp500_composition(comp_path)
        df_perf = load_stocks_performance(perf_path)
        portfolio_df = merge_sp500_data(df_comp, df_perf)
        if portfolio_df is not None:
            st.session_state['portfolio_df'] = portfolio_df
            st.sidebar.success("Données chargées depuis le jeu de données.")
        else:
            st.sidebar.error("Fusion impossible : vérifiez les fichiers.")
    else:
        st.sidebar.error("Impossible de trouver les CSV dans le répertoire indiqué.")

st.sidebar.header("Chargement des Fichiers CSV (upload manuel)")
comp_file = st.sidebar.file_uploader("composition_sp500.csv", type=['csv'], key="comp_uploader")
perf_file = st.sidebar.file_uploader("stocks-performance.csv", type=['csv'], key="perf_uploader")

if comp_file and perf_file and 'portfolio_df' not in st.session_state:
    df_comp = load_sp500_composition(comp_file)
    df_perf = load_stocks_performance(perf_file)
    portfolio_df = merge_sp500_data(df_comp, df_perf)
    if portfolio_df is not None:
        st.session_state['portfolio_df'] = portfolio_df
        st.sidebar.success("Données chargées avec succès !")

# Récup session
portfolio_df = st.session_state.get('portfolio_df', None)

if st.sidebar.button("Recharger les Données CSV"):
    if comp_file and perf_file:
        df_comp = load_sp500_composition(comp_file)
        df_perf = load_stocks_performance(perf_file)
        portfolio_df = merge_sp500_data(df_comp, df_perf)
        st.session_state['portfolio_df'] = portfolio_df
        st.sidebar.success("Données rechargées !")
    else:
        st.sidebar.error("Veuillez uploader les fichiers d'abord.")

# -----------------------------------------------------------
# Étape 1
# -----------------------------------------------------------
if selected_step == "1. Chargement des Données S&P 500":
    st.header("Étape 1: Chargement et Aperçu des Données S&P 500")
    st.write("Utilisez les uploaders dans le sidebar pour charger les fichiers. Les données persistent via session_state.")
    if portfolio_df is not None:
        st.dataframe(portfolio_df.head(10))
        st.write(f"Nombre total d'actions: {len(portfolio_df)}")
        st.write("Exemple de métriques dérivées:")
        cols_preview = [c for c in ['Symbol', 'Op. Margin', 'Net Margin'] if c in portfolio_df.columns]
        if cols_preview:
            st.dataframe(portfolio_df[cols_preview].head())
    else:
        st.warning("Veuillez uploader les fichiers dans le sidebar.")

# -----------------------------------------------------------
# Étape 2 — Upload & Extraction (multi-fichiers)
# -----------------------------------------------------------
if selected_step == "2. Upload et Extraction du Texte Réglementaire":
    st.header("Étape 2: Upload du Document Réglementaire et Extraction des Informations")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    else:
        uploaded_files = st.file_uploader(
            "Téléchargez des fichiers texte réglementaires (TXT, HTML, XML)", 
            type=['txt', 'html', 'xml', 'htm'], 
            accept_multiple_files=True
        )
        reg_texts = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1]
                reg_text = uploaded_file.read().decode('utf-8', errors='ignore')
                reg_texts.append((reg_text, file_extension, uploaded_file.name))

        manual_text = st.text_area("Ou collez le texte réglementaire ici", value="")
        if manual_text.strip():
            reg_texts.append((manual_text, 'txt', 'Manual Input'))

        if st.button("Extraire les Informations"):
            all_extracted = []
            combined_text = ''
            for reg_text, ext, name in reg_texts:
                extracted = extract_reg_info(reg_text, ext)
                all_extracted.append((name, extracted))
                combined_text += reg_text + ' '

            st.session_state['all_extracted'] = all_extracted
            st.session_state['reg_texts'] = reg_texts

            st.markdown("## 📑 Résumé des Textes Réglementaires Analysés")
            for name, extracted in all_extracted:
                with st.expander(f"📘 {name}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Entités détectées", len(extracted.get('entities', [])))
                    col2.metric("Périodes mentionnées", len(extracted.get('dates', [])))
                    col3.metric("Thèmes / Mesures", len(extracted.get('measures', [])))

                    st.markdown(f"**Type de Réglementation :** `{extracted.get('type_reg', 'N/A')}`")
                    st.markdown("### 🏛️ Entités Principales")
                    ents = sorted(list(extracted.get('entities', [])))[:15]
                    st.write(", ".join(ents) + (" ..." if len(extracted.get('entities', [])) > 15 else ""))

                    st.markdown("### 🧩 Thèmes / Mots-clés Dominants")
                    common_measures = Counter(extracted.get("measures", [])).most_common(10)
                    if common_measures:
                        df_common = pd.DataFrame(common_measures, columns=["Mot-clé", "Occurrences"])
                        fig = px.bar(df_common, x="Mot-clé", y="Occurrences", title="Top 10 Thèmes Détectés", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Aucun mot-clé dominant détecté.")

                    st.markdown("### 📅 Périodes Mentionnées")
                    dprev = sorted(list(extracted.get('dates', [])))[:10]
                    st.write(", ".join(dprev) + (" ..." if len(extracted.get('dates', [])) > 10 else ""))

                    st.markdown("### 🧠 Synthèse")
                    st.info(
                        f"La directive contient **{len(extracted.get('entities', []))} entités**, "
                        f"**{len(extracted.get('dates', []))} références temporelles** "
                        f"et **{len(extracted.get('measures', []))} thèmes économiques**."
                    )

# -----------------------------------------------------------
# Étape 3 — Modélisation (Lambda by default)
# -----------------------------------------------------------
if selected_step == "3. Modélisation de l'Impact":
    st.header("Étape 3: Modélisation de l'Impact Réglementaire sur les Actions")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    elif 'reg_texts' not in st.session_state:
        st.warning("Veuillez d'abord extraire les informations à l'étape 2.")
    else:
        default_fillings = st.session_state.get('fillings_dir', str((data_dir_path / "fillings").resolve()))
        fillings_dir = st.text_input("Dossier des rapports 10-K (fillings)", value=default_fillings)

        st.caption("ℹ️ Pour éviter les erreurs d'identifiants, l'appel Bedrock local est désactivé. Toute l'analyse se fait via Lambda si le toggle est activé.")

        if st.button("Modéliser l'Impact"):
            combined_text = ' '.join([text for text, _, _ in st.session_state['reg_texts']])

            analyzed_df = pd.DataFrame()
            extracted = {}
            portfolio_risk = None
            concentration = None
            recommendations = []

            if use_lambda:
                with st.spinner("Analyse distante (Lambda)…"):
                    try:
                        final, logs = _invoke_regai_lambda(combined_text)
                        if logs:
                            with st.expander("CloudWatch Logs (tail)"):
                                st.code(logs)

                        # Adapter ici si ta Lambda renvoie un autre schéma
                        analyzed_df = pd.DataFrame(final.get("analyzed_df", []))
                        extracted = final.get("extracted", {})
                        portfolio_risk = final.get("portfolio_risk", 0.0)
                        concentration = pd.DataFrame(final.get("concentration", [])) if final.get("concentration") else None
                        recommendations = final.get("recommendations", [])

                        # Fallback si la Lambda ne renvoie pas ces objets
                        if analyzed_df.empty:
                            st.warning("La Lambda n'a pas renvoyé de DataFrame d'analyse. Bascule sur l'analyse locale (heuristique).")
                            raise ValueError("payload_incomplet")

                    except Exception as e:
                        st.info(f"⤵️ Bascule locale (raison : {e})")
                        with st.spinner("Analyse locale (heuristique)…"):
                            analyzed_df, extracted, portfolio_risk, concentration, recommendations = analyze_reg_impact_enhanced(
                                portfolio_df.copy(),
                                combined_text,
                                fillings_dir=fillings_dir,
                                file_extension='txt',
                                use_bedrock=False
                            )
            else:
                with st.spinner("Analyse locale (heuristique)…"):
                    analyzed_df, extracted, portfolio_risk, concentration, recommendations = analyze_reg_impact_enhanced(
                        portfolio_df.copy(),
                        combined_text,
                        fillings_dir=fillings_dir,
                        file_extension='txt',
                        use_bedrock=False
                    )

            # Persist
            st.session_state['analyzed_df'] = analyzed_df
            st.session_state['extracted_combined'] = extracted
            st.session_state['portfolio_risk'] = portfolio_risk
            st.session_state['concentration'] = concentration
            st.session_state['recommendations'] = recommendations

            st.subheader("Scores de Risque par Action (Top 10 impactées):")
            if not analyzed_df.empty and "Risk Score" in analyzed_df.columns:
                high_risk = analyzed_df[analyzed_df['Risk Score'] > 0].sort_values('Risk Score', ascending=False).head(10)
                cols = ['Symbol', 'Company', 'Risk Score', 'Direct Risk', 'Supply Chain Risk', 'Geographic Risk', 'Impact Est. Loss %', 'Impact Est. Loss']
                existing_cols = [c for c in cols if c in high_risk.columns]
                st.dataframe(high_risk[existing_cols])

# -----------------------------------------------------------
# Étape 4 — Évaluation Globale
# -----------------------------------------------------------
if selected_step == "4. Évaluation Globale du Portefeuille":
    st.header("Étape 4: Évaluation de l'Effet Global sur le Portefeuille S&P 500")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    elif 'analyzed_df' not in st.session_state:
        st.warning("Veuillez d'abord modéliser l'impact à l'étape 3.")
    else:
        analyzed_df = st.session_state['analyzed_df']
        portfolio_risk = st.session_state.get('portfolio_risk')
        concentration = st.session_state.get('concentration')
        if portfolio_risk is None or concentration is None:
            st.warning("Aucun résultat stocké. Relancez la modélisation à l'étape 3.")
        else:
            st.subheader("Risque Global du Portefeuille:")
            st.write(f"**Score de Risque Agrégé:** {portfolio_risk:.4f}")
            st.subheader("Concentrations de Risque par Secteur:")
            st.dataframe(concentration)

# -----------------------------------------------------------
# Étape 5 — Visualisations & Recos (charts triés/pertinents)
# -----------------------------------------------------------
if selected_step == "5. Visualisations et Recommandations":
    st.header("Étape 5 : Visualisations interactives et recommandations")

    if portfolio_df is None:
        st.warning("⚠️ Veuillez d'abord charger les données S&P 500 à l'étape 1.")
    elif 'analyzed_df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord modéliser l'impact réglementaire à l'étape 3.")
    else:
        analyzed_df = st.session_state['analyzed_df']
        concentration = st.session_state.get('concentration', None)
        extracted = st.session_state.get('extracted_combined', {})
        recommendations = st.session_state.get('recommendations', [])

        # Figures (on ne conserve que les pertinentes)
        figs = create_all_visualizations(analyzed_df, concentration, extracted)

        # 1) Résumé global
        st.markdown("### 📊 Indicateurs clés du portefeuille")
        col1, col2, col3 = st.columns(3)
        if "Risk Score" in analyzed_df.columns and "Impact Est. Loss" in analyzed_df.columns:
            col1.metric("Score de risque moyen", f"{analyzed_df['Risk Score'].mean():.2%}")
            col2.metric("Perte estimée totale", f"{analyzed_df['Impact Est. Loss'].sum()/1e9:.1f} B $")
            col3.metric("Nb d'actions à risque >", len(analyzed_df[analyzed_df['Risk Score'] > 0.6]))
        st.markdown("---")

        # 2) Visualisations pertinentes seulement
        tabs = st.tabs([
            "⚠️ Top Risques",
            "🎯 Concentration Macro-Secteurs",
            "💸 Pertes Estimées",
            "🧩 Décomposition du Risque",
            "🔗 Corrélations"
        ])

        with tabs[0]:
            st.plotly_chart(figs["top_risks_bar"], use_container_width=True)

        with tabs[1]:
            st.plotly_chart(figs["sector_concentration"], use_container_width=True)

        with tabs[2]:
            st.plotly_chart(figs["loss_estimation"], use_container_width=True)

        with tabs[3]:
            st.plotly_chart(figs["risk_components"], use_container_width=True)
            st.caption("Formule de perte estimée (exemple) : "
                       "**Impact Est. Loss = MarketCap × Impact Est. Loss %**. "
                       "Le score de risque agrège les composantes (Direct, Supply, Géographique) pondérées.")

        with tabs[4]:
            st.plotly_chart(figs["correlation_heatmap"], use_container_width=True)

        st.markdown("---")

        # 3) Scénarios rapides
        st.markdown("### 🔮 Simulation de scénarios")
        scenario = st.radio(
            "Choisissez un scénario :",
            ["Base", "High Impact (+20 %)", "Low Impact (−20 %)"],
            horizontal=True
        )
        if "Impact Est. Loss" in analyzed_df.columns:
            impact_loss = analyzed_df['Impact Est. Loss'].sum() / 1e12
            if scenario == "High Impact (+20 %)":
                impact_loss *= 1.2
            elif scenario == "Low Impact (−20 %)":
                impact_loss *= 0.8
            st.success(f"💰 Perte estimée totale ({scenario}) : **{impact_loss:.2f} T $**")

        # 4) Recommandations
        st.markdown("### 🧠 Recommandations & Insights")
        if not recommendations:
            st.info("Aucune recommandation générée. Relancez l'analyse.")
        else:
            st.markdown("---")
            for rec in recommendations:
                with st.container():
                    st.markdown(
                        f"#### 🏢 {rec.get('company', 'Entreprise N/A')} "
                        f"({rec.get('ticker', 'Ticker N/A')})"
                    )
                    risk_score = rec.get('risk_score', 'N/A')
                    if isinstance(risk_score, (float, int)):
                        st.write(f"**Score Risque :** {risk_score:.4f}")
                    else:
                        st.write(f"**Score Risque :** {risk_score}")
                    st.write(f"**Action suggérée :** {rec.get('action', 'N/A')}")
                    st.write(f"**Commentaire :** {rec.get('recommendation', rec.get('reason', 'Aucun'))}")
                    if rec.get('estimated_loss'):
                        st.caption(f"💸 Perte estimée : {rec['estimated_loss']}")
                    st.markdown("---")
