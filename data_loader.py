# data_loader.py - Module pour le chargement et la fusion des données
import pandas as pd

def load_sp500_composition(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = ['Rank', 'Company', 'Symbol', 'Weight', 'Price']
        df['Weight'] = df['Weight'].str.replace(',', '.').astype(float)
        df['Price'] = df['Price'].str.replace(',', '.').astype(float)
        df['Company'] = df['Company'].str.strip('"')
        df = df.sort_values(by='Weight', ascending=False)
        return df
    return None

def load_stocks_performance(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        numeric_cols = ['Market Cap', 'Revenue', 'Op. Income', 'Net Income', 'EPS', 'FCF']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].str.replace(',', '') if df[col].dtype == 'object' else df[col], errors='coerce')
        df[numeric_cols] = df[numeric_cols].fillna(0)
        df['Company Name'] = df['Company Name'].str.strip()
        return df
    return None

def merge_sp500_data(df_comp, df_perf):
    if df_comp is None or df_perf is None:
        return None
    df_merged = pd.merge(df_comp, df_perf, left_on='Symbol', right_on='Symbol', how='left')
    df_merged = df_merged[['Symbol', 'Company', 'Weight', 'Price', 'Market Cap', 'Revenue', 'Op. Income', 'Net Income', 'EPS', 'FCF']]
    df_merged['Op. Margin'] = df_merged['Op. Income'] / df_merged['Revenue'].replace(0, float('nan'))
    df_merged['Net Margin'] = df_merged['Net Income'] / df_merged['Revenue'].replace(0, float('nan'))
    df_merged.fillna(0, inplace=True)
    df_merged['Risk Score'] = 0.0
    df_merged['Impact Est. Loss %'] = 0.0
    df_merged['Impact Est. Loss'] = 0.0
    return df_merged