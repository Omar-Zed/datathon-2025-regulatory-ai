# app.py - Fichier principal de l'application Streamlit
import streamlit as st
from data_loader import load_sp500_composition, load_stocks_performance, merge_sp500_data
from reg_analysis import extract_reg_info, analyze_reg_impact
import plotly.express as px

st.set_page_config(page_title="RegAI Portfolio Analyzer", layout="wide")

# Menu latéral pour les étapes
st.sidebar.title("Étapes de l'Application")
steps = [
    "1. Chargement des Données S&P 500",
    "2. Upload et Extraction du Texte Réglementaire",
    "3. Modélisation de l'Impact",
    "4. Évaluation Globale du Portefeuille",
    "5. Visualisations et Recommandations"
]
selected_step = st.sidebar.selectbox("Sélectionnez une étape", steps)

# Uploaders dans le sidebar pour persistance
st.sidebar.header("Chargement des Fichiers CSV (disponible partout)")
comp_file = st.sidebar.file_uploader("composition_sp500.csv", type=['csv'], key="comp_uploader")
perf_file = st.sidebar.file_uploader("stocks-performance.csv", type=['csv'], key="perf_uploader")

# Chargement automatique si fichiers uploadés et DF pas en session
if comp_file and perf_file and 'portfolio_df' not in st.session_state:
    df_comp = load_sp500_composition(comp_file)
    df_perf = load_stocks_performance(perf_file)
    portfolio_df = merge_sp500_data(df_comp, df_perf)
    if portfolio_df is not None:
        st.session_state['portfolio_df'] = portfolio_df
        st.sidebar.success("Données chargées avec succès !")

# Récupération du DF depuis session
portfolio_df = st.session_state.get('portfolio_df', None)

# Bouton pour recharger si besoin
if st.sidebar.button("Recharger les Données CSV"):
    if comp_file and perf_file:
        df_comp = load_sp500_composition(comp_file)
        df_perf = load_stocks_performance(perf_file)
        portfolio_df = merge_sp500_data(df_comp, df_perf)
        st.session_state['portfolio_df'] = portfolio_df
        st.sidebar.success("Données rechargées !")
    else:
        st.sidebar.error("Veuillez uploader les fichiers d'abord.")

# Étape 1: Chargement des Données S&P 500
if selected_step == "1. Chargement des Données S&P 500":
    st.header("Étape 1: Chargement et Aperçu des Données S&P 500")
    st.write("Utilisez les uploaders dans le sidebar pour charger les fichiers. Les données persistent via session_state.")
    if portfolio_df is not None:
        st.dataframe(portfolio_df.head(10))  # Afficher top 10
        st.write(f"Nombre total d'actions: {len(portfolio_df)}")
        st.write("Exemple de métriques dérivées:")
        st.dataframe(portfolio_df[['Symbol', 'Op. Margin', 'Net Margin']].head())
    else:
        st.warning("Veuillez uploader les fichiers dans le sidebar.")

# Étape 2: Upload et Extraction du Texte Réglementaire (updated for multiple files)
if selected_step == "2. Upload et Extraction du Texte Réglementaire":
    st.header("Étape 2: Upload du Document Réglementaire et Extraction des Informations")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    else:
        uploaded_files = st.file_uploader("Téléchargez des fichiers texte réglementaires (TXT, HTML, XML)", type=['txt', 'html', 'xml', 'htm'], accept_multiple_files=True)
        
        reg_texts = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1]
                reg_text = uploaded_file.read().decode('utf-8', errors='ignore')
                reg_texts.append((reg_text, file_extension, uploaded_file.name))
        
        # Optional text area for manual input
        manual_text = st.text_area("Ou collez le texte réglementaire ici (exemple par défaut fourni)", 
                                   value="The Inflation Reduction Act imposes 15% minimum tax on corporations with over $1B income. Application: after Dec 31, 2022. Measures: excise tax on stock repurchases, drug price negotiations for pharma. Entities: US corporations, pharma manufacturers, foreign-parented groups.")
        if manual_text:
            reg_texts.append((manual_text, 'txt', 'Manual Input'))
        
        if st.button("Extraire les Informations"):
            all_extracted = []
            combined_text = ''
            for reg_text, ext, name in reg_texts:
                extracted = extract_reg_info(reg_text, ext)
                all_extracted.append((name, extracted))
                combined_text += reg_text + ' '  # For combined analysis
            
            st.session_state['all_extracted'] = all_extracted
            st.session_state['reg_texts'] = reg_texts  # Store list
            
            # Display per file
            for name, extracted in all_extracted:
                st.subheader(f"Éléments Extraits de {name}:")
                st.write(f"**Entités:** {', '.join(sorted(extracted['entities']))}")
                st.write(f"**Dates:** {', '.join(sorted(extracted['dates']))}")
                st.write(f"**Mesures:** {', '.join(sorted(extracted['measures']))}")
                st.write(f"**Type de Réglementation:** {extracted['type_reg']}")

# Étape 3: Modélisation de l'Impact
if selected_step == "3. Modélisation de l'Impact":
    st.header("Étape 3: Modélisation de l'Impact Réglementaire sur les Actions")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    elif 'reg_texts' not in st.session_state:
        st.warning("Veuillez d'abord extraire les informations à l'étape 2.")
    else:
        if st.button("Modéliser l'Impact"):
            # Combined text for analysis
            combined_text = ' '.join([text for text, _, _ in st.session_state['reg_texts']])
            analyzed_df, extracted, portfolio_risk, concentration, recommendations = analyze_reg_impact(portfolio_df.copy(), combined_text)
            st.session_state['analyzed_df'] = analyzed_df
            st.session_state['extracted_combined'] = extracted  # For later use
            st.subheader("Scores de Risque par Action (Top 10 impactées):")
            high_risk = analyzed_df[analyzed_df['Risk Score'] > 0].sort_values('Risk Score', ascending=False).head(10)
            st.dataframe(high_risk[['Symbol', 'Company', 'Risk Score', 'Impact Est. Loss %', 'Impact Est. Loss']])

# Étape 4: Évaluation Globale du Portefeuille
if selected_step == "4. Évaluation Globale du Portefeuille":
    st.header("Étape 4: Évaluation de l'Effet Global sur le Portefeuille S&P 500")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    elif 'analyzed_df' not in st.session_state:
        st.warning("Veuillez d'abord modéliser l'impact à l'étape 3.")
    else:
        analyzed_df = st.session_state['analyzed_df']
        _, _, portfolio_risk, concentration, _ = analyze_reg_impact(analyzed_df, ' '.join([text for text, _, _ in st.session_state['reg_texts']]))
        st.subheader("Risque Global du Portefeuille:")
        st.write(f"**Score de Risque Agrégé:** {portfolio_risk:.4f}")
        st.subheader("Concentrations de Risque par Secteur:")
        st.write(concentration)

# Étape 5: Visualisations et Recommandations
if selected_step == "5. Visualisations et Recommandations":
    st.header("Étape 5: Visualisations, Simulations et Recommandations")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    elif 'analyzed_df' not in st.session_state:
        st.warning("Veuillez d'abord modéliser l'impact à l'étape 3.")
    else:
        analyzed_df = st.session_state['analyzed_df']
        _, _, _, _, recommendations = analyze_reg_impact(analyzed_df, ' '.join([text for text, _, _ in st.session_state['reg_texts']]))
        
        # Visualisation
        st.subheader("Visualisation des Risques")
        fig = px.bar(analyzed_df[analyzed_df['Risk Score'] > 0].sort_values('Risk Score', ascending=False).head(20),
                     x='Symbol', y='Risk Score', title="Top 20 Actions à Risque")
        st.plotly_chart(fig)
        
        # Simulations
        st.subheader("Simulations de Scénarios")
        scenario = st.selectbox("Choisissez un scénario", ["Base", "High Impact (+20%)", "Low Impact (-20%)"])
        impact_loss = analyzed_df['Impact Est. Loss'].sum() / 1e12
        if scenario == "High Impact (+20%)":
            impact_loss *= 1.2
        elif scenario == "Low Impact (-20%)":
            impact_loss *= 0.8
        st.write(f"Perte Estimée Totale ({scenario}): {impact_loss:.2f} T$")
        
        # Recommandations
        st.subheader("Recommandations Stratégiques")
        for rec in recommendations:
            st.write(f"- {rec}")