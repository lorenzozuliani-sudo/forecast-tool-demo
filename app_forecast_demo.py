import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import logging

# Disabilita log pesanti di Prophet
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# 1. CONFIGURAZIONE PAGINA
st.set_page_config(page_title="Forecasting Strategico Pro - DEMO", layout="wide")

# STILE CSS PROFESSIONALE
st.markdown("""
    <style>
    /* 1. STILI BASE (LIGHT MODE) */
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e6e6e6; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { background-color: #ffffff; border-radius: 10px; border: 1px solid #e1e4e8; }
    h1, h2, h3 { color: #2c3e50; }
    .ai-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e0e0e0; color: #1e1e1e; }
    .ai-score-high { background-color: #f0fff4; border-left: 5px solid #48bb78; }
    .ai-score-med { background-color: #fffaf0; border-left: 5px solid #ed8936; }
    .ai-score-low { background-color: #fff5f5; border-left: 5px solid #f56565; }
    .ai-alert { background-color: #fff5f5; color: #c53030; padding: 8px; border-radius: 5px; margin-top: 10px; font-size: 0.85em; border: 1px solid #feb2b2; }
    .ai-tag { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; margin-right: 5px; background-color: #e2e8f0; color: #4a5568; }
{ background-color: #0e1117; }
        .stMetric, div[data-testid="stExpander"] { background-color: #262730; border: 1px solid #464855; color: #ffffff; }
        h1, h2, h3 { color: #ffffff !important; }
        .ai-box { background-color: #1e1e1e; border-color: #464855; color: #ffffff; }
        /* Varianti colori AI per Dark Mode per mantenere leggibilità */
        .ai-score-high { background-color: #1a2e23; border-left: 5px solid #48bb78; }
        .ai-score-med { background-color: #2d261e; border-left: 5px solid #ed8936; }
        .ai-score-low { background-color: #2d1e1e; border-left: 5px solid #f56565; }
        .ai-alert { background-color: #442a2a
    /* 2. ADATTAMENTO PER DARK MODE (MAC/SYSTEM) */
    @media (prefers-color-scheme: dark) {
        .main ; color: #ff8080; border-color: #663333; }
        .ai-tag { background-color: #313d4f; color: #e2e8f0; }
    }
    </style>
    """, unsafe_allow_html=True)

# --- INIZIALIZZAZIONE SESSION STATE ---
if 'trend_val' not in st.session_state: st.session_state.trend_val = 0.0
if 'google_scale' not in st.session_state: st.session_state.google_scale = 1.0
if 'meta_scale' not in st.session_state: st.session_state.meta_scale = 1.0
if 'sat_val' not in st.session_state: st.session_state.sat_val = 0.85
if 'is_demo_loaded' not in st.session_state: st.session_state.is_demo_loaded = False
if 'last_uploaded_file' not in st.session_state: st.session_state.last_uploaded_file = None

# --- FUNZIONI DI UTILITÀ ---

def clean_currency_us(column):
    if column is None: return 0
    s = column.astype(str)
    s = s.str.replace('€', '', regex=False).str.strip()
    s = s.str.replace(',', '', regex=False) 
    return pd.to_numeric(s, errors='coerce').fillna(0)

def parse_iso_week(week_str):
    try:
        week_str = str(week_str).strip()
        if len(week_str) < 6: return pd.NaT
        year = int(week_str[:4])
        week = int(week_str[4:])
        return datetime.fromisocalendar(year, week, 1) 
    except:
        return pd.NaT

def get_week_range_label_with_year(date):
    if pd.isna(date): return ""
    start = date
    end = date + timedelta(days=6)
    return f"{start.strftime('%d %b')} - {end.strftime('%d %b %Y')}"

def clean_percentage(val):
    if pd.isna(val): return 0.0
    s = str(val).replace('%', '').strip()
    try: return float(s)
    except: return 0.0

def generate_demo_data():
    """Genera dati casuali ma realistici per la demo (2020-2026)."""
    # Impostiamo date fisse dal 2020 al 2026
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2026, 5, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')
    
    data = []
    
    # 1. Stagionalità Settimanale (Week 1-53) - Clonato dal CSV reale
    # Notiamo: Q1 basso, Picco estivo (Week 26-28), Picco enorme Q4 (Black Friday Week 47-48)
    seasonal_profile = {
        1: 0.8, 2: 0.7, 3: 0.6, 4: 0.6, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.55, 9: 0.6, 10: 0.6,
        11: 0.65, 12: 0.7, 13: 0.75, 14: 0.8, 15: 0.8, 16: 0.8, 17: 0.85, 18: 0.9, 19: 0.9, 20: 0.95,
        21: 1.0, 22: 1.05, 23: 1.1, 24: 1.15, 25: 1.2, 26: 1.3, 27: 1.4, 28: 1.3, 29: 1.1, 30: 1.0,
        31: 0.9, 32: 0.8, 33: 0.7, 34: 0.7, 35: 0.8, 36: 0.9, 37: 0.95, 38: 1.0, 39: 1.0, 40: 1.05,
        41: 1.1, 42: 1.15, 43: 1.2, 44: 1.4, 45: 1.8, 46: 2.5, 47: 4.5, 48: 3.8, 49: 3.2, 50: 2.5,
        51: 1.5, 52: 1.0, 53: 0.9
    }

    # 2. Trend Annuale Non-Lineare (Fattore moltiplicativo base)
    yearly_trend = {
        2020: 1.0,
        2021: 1.4,  # Boom post-2020
        2022: 1.3,  # Assestamento/Calo
        2023: 1.5,  # Ripresa
        2024: 1.7,  # Crescita solida
        2025: 1.9,  # Crescita continua
        2026: 2.1   # Proiezione
    }

    base_sales = 5000.0 # Valore base settimanale
    
    for d in dates:
        year = d.year
        week = d.isocalendar().week
        
        s_fact = seasonal_profile.get(week, 1.0)
        y_fact = yearly_trend.get(year, 1.0)
        
        # Randomicità controllata
        noise = np.random.uniform(0.9, 1.1)
        
        # Calcolo Vendite Totali
        total_sales = base_sales * s_fact * y_fact * noise
        
        # Spesa Ads (Segue le vendite ma con efficienza variabile)
        # Quando il fatturato esplode (Black Friday), il ROAS sale ma il CPM costa di più
        marketing_pressure = 0.20 # 20% del fatturato va in ads di media
        if s_fact > 2.0: marketing_pressure = 0.15 # Efficienza sale nei picchi
        
        total_spend = total_sales * marketing_pressure * np.random.uniform(0.95, 1.05)
        
        # Split Google/Meta (Google prende più brand search nei picchi)
        google_share = 0.40
        if s_fact > 1.5: google_share = 0.50
        
        g_cost = total_spend * google_share
        m_cost = total_spend * (1 - google_share)
        
        # KPI Derivati
        aov = 120.0 + np.random.uniform(-10, 10)
        orders = int(total_sales / aov)
        
        # Resi (più alti dopo i picchi)
        return_rate = 0.12
        if week in [1, 2, 3, 4, 5]: return_rate = 0.25 # Gennaio resi alti
        returns = - (total_sales * return_rate * np.random.uniform(0.8, 1.2))
        
        discounts = - (total_sales * 0.05) if s_fact < 2 else - (total_sales * 0.15) # Più sconti nei picchi
        
        # ROAS Simulato
        roas_g = (total_sales * 0.6) / g_cost if g_cost > 0 else 0
        roas_m = (total_sales * 0.5) / m_cost if m_cost > 0 else 0
        
        data.append({
            'Year Week': f"{year}{week:02d}",
            'Cost': g_cost,
            'Amount Spent': m_cost,
            'Total sales': total_sales,
            'Returns': returns,
            'Discounts': discounts,
            'Average order value': aov,
            'Orders': orders,
            'Returning customer rate': f"{np.random.randint(12, 28)}%",
            'Conversions Value': g_cost * roas_g,
            'Website Purchases Conversion Value': m_cost * roas_m,
            'Avg. CPC': 0.85,
            'CPC (All)': 0.65,
            'CPM (Cost per 1,000 Impressions)': 12.50,
            'Impressions': int(m_cost / 12.50 * 1000),
            'Frequency': 1.2,
            'Items': int(orders * 1.5),
            'Gross sales': total_sales - discounts
        })
        
    return pd.DataFrame(data)

# --- HEADER ---
st.title("📈 Simulatore Business & Forecasting")

with st.expander("ℹ️ Guida Rapida: Cosa fa questo strumento?"):
    st.markdown("""
    Questo strumento è un **CFO Virtuale e Simulatore Strategico** progettato per e-commerce. Non si limita a visualizzare i dati passati, ma ti aiuta a pianificare il futuro economico.
    
    **A cosa serve:**
    1.  **💰 Controllo Profittabilità (Business Economics):** Inserendo i tuoi margini nella colonna di sinistra, il tool calcola il *Break-Even ROAS* (il punto di pareggio) e ti dice se stai realmente guadagnando o se stai solo "muovendo soldi".
    2.  **🔮 Forecasting (Previsione):** Ti permette di simulare scenari futuri ("Cosa succede se raddoppio il budget su Meta?"). Usa i dati storici e la stagionalità per proiettare fatturato e costi.
    3.  **⚖️ Analisi Efficienza (Saturazione):** Ti aiuta a capire se aumentando la spesa pubblicitaria il fatturato cresce di pari passo (alta elasticità) o se stai saturando il pubblico (bassa efficienza).
    4.  **🧠 Intelligence Automatica (AI):** Un algoritmo analizza la qualità del business (Retention, Sconti, Canali) e ti dà un punteggio di salute mensile.
    """)

# --- SIDEBAR: BUSINESS ECONOMICS ---
st.sidebar.header("⚙️ Business Economics")
with st.sidebar.expander("1. Input Metriche", expanded=True):
    # INPUT UTENTE
    be_aov = st.number_input(
        "Average Order Value (€)", 
        value=145.0, step=1.0,
        help="Il valore medio del carrello (Lordo). Corrisponde al prezzo pagato dal cliente alla cassa."
    )
    
    be_vat = st.number_input(
        "Tax/VAT (%)", 
        value=22.0, step=1.0,
        help="Aliquota IVA media. Il calcolo scorpora l'IVA dal fatturato lordo. (Es. 22% in Italia)."
    )
    
    be_returns = st.number_input(
        "Return Rate (%)", 
        value=13.0, step=0.5,
        help="Percentuale di fatturato persa a causa dei resi. Questi soldi vengono sottratti prima di calcolare i margini."
    )
    
    be_margin_prod = st.number_input(
        "Gross Margin (%)", 
        value=45.0, step=5.0,
        help="Margine lordo sul prodotto dopo il costo del venduto (COGS). Esempio: Vendi a 100€, ti costa 70€ produrlo -> Margine 30%."
    )
    
    be_fulfillment = st.number_input(
        "Fulfillment Cost (€)", 
        value=5.0, step=0.5,
        help="Costo fisso per ordine per logistica, imballaggio e spedizione (Pick & Pack + Shipping)."
    )
    
    st.markdown("**Metriche Retention**")
    be_returning_perc = st.number_input(
        "Returning Customers (%)", 
        value=13.0, step=1.0,
        help="Percentuale di clienti che effettuano un secondo acquisto."
    )
    
    be_repeat_rate = st.number_input(
        "Repeat Order Rate (Freq)", 
        value=1.0, step=0.1,
        help="Frequenza media di riacquisto per i clienti che ritornano (es. 1.0 = comprano 1 volta in più)."
    )

    # --- CALCOLI (BACKEND - INSERITO PER EVITARE NAME ERROR) ---
    # 1. AOV Netto
    aov_post_tax_returns = (be_aov * (1 - be_returns/100)) / (1 + be_vat/100)

    # 2. Profit per Order
    profit_order = (aov_post_tax_returns * (be_margin_prod/100)) - be_fulfillment

    # 3. Profit per Customer
    profit_per_customer = profit_order + (profit_order * (be_returning_perc/100) * be_repeat_rate)

    # 4. Break Even CPA
    be_cpa = profit_per_customer

    # 5. Break Even ROAS
    be_roas_val = be_aov / be_cpa if be_cpa > 0 else 99.9

with st.sidebar.expander("2. Output Calcolati (Live)", expanded=True):
    st.markdown("---")
    
    st.markdown(f"**AOV (Netto)**: € {aov_post_tax_returns:.2f}")
    st.caption("Formula: `(AOV * (1-Resi)) / (1+IVA)`")
    
    st.markdown(f"**Profitto/Ordine**: € {profit_order:.2f}")
    st.caption("Formula: `(AOV Netto * Margine%) - Spedizioni`")
    
    st.markdown(f"**Profitto/Cliente**: € {profit_per_customer:.2f}")
    st.caption("Formula: `Profitto Ordine + Valore Ricorsivo`")
    
    st.markdown("---")
    
    st.metric(
        "🎯 Break-Even CPA", 
        f"€ {be_cpa:.2f}",
        help="Costo per Acquisizione massimo sostenibile. Se spendi più di così per acquisire un cliente, sei in perdita."
    )
    
    st.metric(
        "🎯 Break-Even ROAS", 
        f"{be_roas_val*100:.0f}% ({be_roas_val:.2f})",
        help="Ritorno sulla spesa pubblicitaria minimo necessario. Formula: `AOV / Break Even CPA`."
    )

st.sidebar.divider()

# --- SIDEBAR: CONTROLLI ---
st.sidebar.header("🕹️ Dati & Pannello")
demo_mode = st.sidebar.toggle("🚀 Usa Modalità DEMO (Dati Casuali)", value=False)

uploaded_file = None
if not demo_mode:
    uploaded_file = st.sidebar.file_uploader("Carica il file .csv", type="csv")

# --- GUIDA FORMATO CSV ---
with st.expander("📋 Guida: Come formattare il CSV per la versione completa"):
    st.markdown("""
    Per utilizzare la versione completa, il file CSV deve contenere le seguenti colonne:
    | Colonna | Descrizione |
    | :--- | :--- |
    | `Year Week` | Data settimanale (es. 202501) |
    | `Cost` | Spesa Google Ads |
    | `Amount Spent` | Spesa Meta Ads |
    | `Total sales` | Fatturato netto Shopify |
    | `Returns` | Valore dei resi |
    | `Orders` | Numero totale ordini |
    """)

# --- LOGICA CARICAMENTO E PULIZIA (UNIFICATA) ---
df = None

# 1. Recupero DataFrame (Demo o File)
if demo_mode:
    df = generate_demo_data()
    st.success("✅ Dati DEMO generati (Pattern Non-Lineare)!")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df = df.dropna(how='all')
    except Exception as e:
        st.error(f"Errore: {e}")

# 2. Elaborazione Completa (Se df esiste)
if df is not None:
    try:
        df.columns = df.columns.str.strip()

        col_date = next((c for c in df.columns if 'Year Week' in c or 'Settimana' in c), None)
        col_google = next((c for c in df.columns if 'Cost' in c), 'Cost')
        col_meta = next((c for c in df.columns if 'Amount Spent' in c), 'Amount Spent')
        col_sales = next((c for c in df.columns if 'Total sales' in c), 'Total sales')
        col_returns = next((c for c in df.columns if 'Returns' in c), 'Returns')
        col_orders = next((c for c in df.columns if 'Orders' in c), 'Orders')
        col_aov = next((c for c in df.columns if 'Average order value' in c), 'Average order value')
        
        col_g_val = 'Conversions Value'
        col_m_val = 'Website Purchases Conversion Value'
        col_g_cpc = 'Avg. CPC'
        col_m_cpc = 'CPC (All)'
        col_m_cpm = 'CPM (Cost per 1,000 Impressions)'
        col_g_imps = 'Impressions'
        col_m_freq = 'Frequency'
        
        col_items = 'Items'
        col_ret_rate = 'Returning customer rate'
        col_discounts = 'Discounts'

        if not col_date:
            st.error("Errore: Manca colonna data.")
            st.stop()

        df['Data_Interna'] = df[col_date].apply(parse_iso_week)
        df = df.dropna(subset=['Data_Interna']).sort_values('Data_Interna')
        df['Periodo'] = df['Data_Interna'].apply(get_week_range_label_with_year)

        # === CREAZIONE COLONNE GLOBALI PER TAB 4 ===
        df['Year'] = df['Data_Interna'].dt.year
        df['Week'] = df['Data_Interna'].dt.isocalendar().week
        # ==========================================================

        money_cols = [col_google, col_meta, col_sales, col_returns, col_aov, 'Gross sales', col_discounts, 
                      col_g_val, col_m_val, col_g_cpc, col_m_cpc, col_m_cpm]
        for c in money_cols:
            if c in df.columns: df[c] = clean_currency_us(df[c])

        if col_g_imps in df.columns: df[col_g_imps] = pd.to_numeric(df[col_g_imps], errors='coerce').fillna(0)
        if col_m_freq in df.columns: df[col_m_freq] = pd.to_numeric(df[col_m_freq], errors='coerce').fillna(0)
        
        # Pulizia specifica per l'AI
        if col_ret_rate in df.columns: df[col_ret_rate] = df[col_ret_rate].apply(clean_percentage)
        if col_items in df.columns: df[col_items] = pd.to_numeric(df[col_items], errors='coerce').fillna(0)

        df = df.fillna(0)

        df['Fatturato_Netto'] = df[col_sales].clip(lower=0)
        df['Spesa_Ads_Totale'] = df[col_google] + df[col_meta]
        df['Spesa_Ads_Totale'] = df['Spesa_Ads_Totale'].replace(0, np.nan)
        df['MER'] = (df[col_sales] / df['Spesa_Ads_Totale']).fillna(0)
        df['Tasso_Resi'] = (df[col_returns].abs() / df[col_sales].replace(0, np.nan)) * 100
        df['Tasso_Resi'] = df['Tasso_Resi'].fillna(0)
        
        # Calcolo CoS Storico
        df['CoS'] = (df['Spesa_Ads_Totale'] / df['Fatturato_Netto'].replace(0, np.nan)) * 100
        df['CoS'] = df['CoS'].fillna(0)
        
        # --- CALCOLO PROFITTO NETTO STIMATO NEL DF ---
        num_orders = df[col_orders] if col_orders in df.columns else (df['Fatturato_Netto'] / be_aov)
        
        # Profitto Operativo = (Numero Ordini * Profitto per Ordine) - Spesa Ads
        # FIX: Uso la variabile corretta profit_order definita nella sidebar
        df['Profitto_Operativo'] = (num_orders * profit_order) - df['Spesa_Ads_Totale']

        # Inizializzazione sicura ROAS
        df['ROAS_Google'] = 0.0
        df['ROAS_Meta'] = 0.0
        if col_g_val in df.columns: df['ROAS_Google'] = df[col_g_val] / df[col_google].replace(0, np.nan).fillna(0)
        if col_m_val in df.columns: df['ROAS_Meta'] = df[col_m_val] / df[col_meta].replace(0, np.nan).fillna(0)

        # --- AUTO-CALCOLO ELASTICITÀ ---
        df_annual = df.groupby('Year').agg({'Spesa_Ads_Totale': 'sum', 'Fatturato_Netto': 'sum'}).sort_index()
        suggested_saturation = 0.85 
        if len(df_annual) >= 2:
            last_year = df_annual.index[-1]
            prev_year = df_annual.index[-2]
            d_spend = (df_annual.loc[last_year, 'Spesa_Ads_Totale'] - df_annual.loc[prev_year, 'Spesa_Ads_Totale']) / df_annual.loc[prev_year, 'Spesa_Ads_Totale']
            d_rev = (df_annual.loc[last_year, 'Fatturato_Netto'] - df_annual.loc[prev_year, 'Fatturato_Netto']) / df_annual.loc[prev_year, 'Fatturato_Netto']
            if d_spend > 0.05:
                raw_elasticity = d_rev / d_spend
                suggested_saturation = np.clip(raw_elasticity, 0.60, 1.0)

        # --- CALCOLO TREND YoY ---
        last_date = df['Data_Interna'].max()
        start_last_year = last_date - pd.Timedelta(weeks=52)
        start_prev_year = start_last_year - pd.Timedelta(weeks=52)
        sales_ly = df[(df['Data_Interna'] > start_last_year) & (df['Data_Interna'] <= last_date)]['Fatturato_Netto'].sum()
        sales_py = df[(df['Data_Interna'] > start_prev_year) & (df['Data_Interna'] <= start_last_year)]['Fatturato_Netto'].sum()
        growth_rate = (sales_ly - sales_py) / sales_py if sales_py > 0 else 0.0

        # Storico Annuale
        historical_growth_data = []
        years_avail = sorted(df['Year'].unique(), reverse=True)
        for i in range(len(years_avail) - 1):
            curr_y = years_avail[i]
            prev_y = years_avail[i+1]
            val_curr = df_annual.loc[curr_y, 'Fatturato_Netto']
            val_prev = df_annual.loc[prev_y, 'Fatturato_Netto']
            g_y = (val_curr - val_prev) / val_prev if val_prev > 0 else 0
            historical_growth_data.append(f"📅 {curr_y} vs {prev_y}: **{g_y:+.1%}**")

        # === 🚀 AUTO-SETTING AL PRIMO CARICAMENTO (O AVVIO DEMO) ===
        current_source_name = "DEMO" if demo_mode else (uploaded_file.name if uploaded_file else None)
        
        if st.session_state.last_uploaded_file != current_source_name:
            st.session_state.trend_val = 0.0
            st.session_state.google_scale = 1.0
            st.session_state.meta_scale = 1.0
            st.session_state.sat_val = float(suggested_saturation)
            st.session_state.last_uploaded_file = current_source_name
            st.rerun()
        # =============================================

        # --- SIDEBAR: AZIONI RAPIDE ---
        st.sidebar.subheader("⚡ Azioni Rapide")
        
        with st.sidebar.expander("ℹ️ Info Scenari"):
            st.markdown("""
            * **🛡️ Prudente:** Mantiene budget attuali, stima efficienza bassa (0.80).
            * **🚀 Aggressivo:** Aumenta budget del 50%, assume trend positivo (+15%) e ottima efficienza.
            * **🎯 Auto-Calibra:** Usa i tuoi dati storici per settare la saturazione reale e propone una crescita sostenibile (+20% budget).
            """)

        col_b1, col_b2 = st.sidebar.columns(2)
        if col_b1.button("🛡️ Prudente"):
            st.session_state.trend_val = 0.0; st.session_state.google_scale = 1.0; st.session_state.meta_scale = 1.0; st.session_state.sat_val = 0.80; st.rerun()
        if col_b2.button("🚀 Aggressivo"):
            st.session_state.trend_val = 0.15; st.session_state.google_scale = 1.5; st.session_state.meta_scale = 1.5; st.session_state.sat_val = 0.90; st.rerun()
        if st.sidebar.button(f"🎯 Auto-Calibra (Sat: {suggested_saturation:.2f})"):
            st.session_state.trend_val = 0.0; st.session_state.google_scale = 1.0; st.session_state.meta_scale = 1.0; st.session_state.sat_val = float(suggested_saturation); st.rerun()

        st.sidebar.divider()

        # --- SIDEBAR: SLIDER E LEGENDE ---
        st.sidebar.subheader("🚀 Trend & Crescita")
        
        with st.sidebar.expander("ℹ️ Come viene calcolato?"):
            st.markdown(f"**Formula:** `(Fatturato Ultimi 12 Mesi - Fatturato 12 Mesi Precedenti) / Precedenti`")
            st.markdown(f"**Dato Rilevato:** `{growth_rate:+.1%}`")
            if historical_growth_data:
                st.markdown("---")
                for hist_row in historical_growth_data[:4]: st.markdown(hist_row)

        st.sidebar.markdown(f"**Crescita Rilevata (YoY):** `{growth_rate:+.1%}`.")
        manual_trend = st.sidebar.slider("Aggiusta Trend Futuro", -0.5, 2.0, value=st.session_state.trend_val, key="trend_val", help="...")
        st.sidebar.divider()
        st.sidebar.subheader("Scenari Budget")
        st.sidebar.markdown("""
        **Scala Budget:** Moltiplica la spesa storica media per questo fattore.
        * **1.0**: Spesa standard (uguale agli anni passati).
        * **2.0**: Simula cosa succede se raddoppi l'investimento.
        """)        
        # Usiamo il valore dello stato per rendere lo slider "intelligente"
        m_google = st.sidebar.slider("🚀 Google Ads Budget Scale", 0.5, 3.0, value=st.session_state.google_scale, key="google_scale")
        m_meta = st.sidebar.slider("🚀 Meta Ads Budget Scale", 0.5, 3.0, value=st.session_state.meta_scale, key="meta_scale")
        
        st.sidebar.divider()
        # Titolo della sezione
        st.sidebar.subheader("🧪 Calibrazione")
        
        # Lo slider deve apparire UNA sola volta
        sat_factor = st.sidebar.slider("Saturazione", 0.5, 1.0, key="sat_val")

        # L'expander contiene solo la documentazione
        with st.sidebar.expander("ℹ️ Come viene calcolato?"):
            st.markdown("""
            Modella i **rendimenti decrescenti**: spiega al simulatore quanto cala l'efficacia delle Ads all'aumentare del budget.
            
            * **1.0 (Lineare):** Caso teorico. Raddoppi il budget = raddoppi le vendite. Poco realistico per scale elevate.
            * **0.85 (Realistico):** Valore consigliato. Tiene conto che scalando il pubblico, il costo per acquisizione (CPA) tende a salire leggermente.
            * **0.70 (Competitivo):** Mercato saturo o molto competitivo. Grandi incrementi di budget portano a piccoli incrementi di fatturato.
            
            > **Suggerimento:** Se hai un ROAS molto alto ora, puoi stare verso 0.90. Se sei già al limite della profittabilità, usa 0.75 per sicurezza.
            """)
        # Grafico Saturazione con Assi
        x_sat = np.linspace(1, 4, 20)
        y_sat = x_sat ** sat_factor
        fig_sat, ax_sat = plt.subplots(figsize=(4, 2))
        ax_sat.plot(x_sat, x_sat, linestyle='--', color='gray', alpha=0.5, label='Ideale')
        ax_sat.plot(x_sat, y_sat, color='#e74c3c', linewidth=2, label=f'Reale')
        ax_sat.set_title("Curva Rendimenti", fontsize=9)
        ax_sat.set_xlabel("Moltiplicatore Spesa", fontsize=7)
        ax_sat.set_ylabel("Moltiplicatore Ricavi", fontsize=7)
        ax_sat.tick_params(labelsize=6)
        ax_sat.legend(fontsize=6, frameon=False)
        ax_sat.spines['top'].set_visible(False)
        ax_sat.spines['right'].set_visible(False)
        st.sidebar.pyplot(fig_sat)

        mesi_prev = st.sidebar.number_input("Mesi di Previsione", 1, 24, 6)

        # --- 4. DASHBOARD KPI (Ultime 4 Settimane) ---
        st.divider()
        last_4 = df.tail(4)
        tot_sales = last_4['Fatturato_Netto'].sum()
        tot_ads = last_4['Spesa_Ads_Totale'].sum()
        mer = tot_sales / tot_ads if tot_ads > 0 else 0
        cos = (tot_ads / tot_sales * 100) if tot_sales > 0 else 0
        profit = last_4['Profitto_Operativo'].sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Fatturato (4w)", f"€ {tot_sales:,.0f}")
        c2.metric("Spesa Ads (4w)", f"€ {tot_ads:,.0f}")
        c3.metric("MER / BE", f"{mer:.2f} / {be_roas_val:.2f}", delta=f"{mer-be_roas_val:.2f}")
        c4.metric("CoS (Spesa/Fatt.)", f"{cos:.1f}%")
        c5.metric("Profitto Stimato", f"€ {profit:,.0f}", help="Profitto Operativo dopo Merce, Tasse, Logistica e Ads.")

        # --- 5. CALCOLO PREVISIONALE ---
        # Pre-calcolo delle feature per tutti i modelli (ML e Backtest)
        df['Week'] = df['Data_Interna'].dt.isocalendar().week
        df['Week_Sin'] = np.sin(2 * np.pi * df['Week'] / 53)
        df['Week_Cos'] = np.cos(2 * np.pi * df['Week'] / 53)
        df['Lag_Sales_1'] = df['Fatturato_Netto'].shift(1)
        df['Lag_Sales_4'] = df['Fatturato_Netto'].shift(4)

        df['Week_Num'] = df['Week'] # Per retrocompatibilità stagionale
        seasonal = df.groupby('Week_Num').agg({
            'Fatturato_Netto': 'mean', col_google: 'mean', col_meta: 'mean', col_orders: 'mean'
        }).reset_index()

        avg_hist_sales = df['Fatturato_Netto'].mean()
        
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=int(mesi_prev*4.34), freq='W-MON')

        rows = []
        for d in future_dates:
            w = d.isocalendar().week
            base = seasonal[seasonal['Week_Num'] == w]
            if base.empty: base = seasonal.mean().to_frame().T
            
            # Trend applicato (Base storica + Slider)
            base_trend = (1 + growth_rate) * (1 + manual_trend)
            
            proj_sales_base = base['Fatturato_Netto'].values[0] * base_trend
            proj_google_base = base[col_google].values[0] * base_trend
            proj_meta_base = base[col_meta].values[0] * base_trend
            
            new_g, new_m = proj_google_base * m_google, proj_meta_base * m_meta
            total_new_spend = new_g + new_m
            total_base_spend = proj_google_base + proj_meta_base
            
            # Modello Esponenziale: raddoppiare la spesa non raddoppia i risultati
            spend_ratio = total_new_spend / total_base_spend if total_base_spend > 0 else 1.0
            saturation_effect = 1 - np.exp(-sat_factor * spend_ratio)
            # Normalizzazione per mantenere la coerenza con la baseline
            efficiency_boost = saturation_effect / (1 - np.exp(-sat_factor)) if sat_factor > 0 else 1.0
            
            f_sales = proj_sales_base * efficiency_boost
            
            f_orders = f_sales / be_aov
            
            rows.append({
                'Data': d, 
                'Periodo': get_week_range_label_with_year(d),
                'Google Previsto': new_g, 
                'Meta Previsto': new_m, 
                'Fatturato Previsto': f_sales, 
                'Ordini Previsti': f_orders
            })
        
        df_prev = pd.DataFrame(rows)
        df_prev['Spesa Totale'] = df_prev['Google Previsto'] + df_prev['Meta Previsto']
        df_prev['MER Previsto'] = df_prev['Fatturato Previsto'] / df_prev['Spesa Totale']
        # Calcolo CoS Previsto
        df_prev['CoS Previsto'] = (df_prev['Spesa Totale'] / df_prev['Fatturato Previsto'].replace(0, np.nan)) * 100
        df_prev['CoS Previsto'] = df_prev['CoS Previsto'].fillna(0)

        # --- 5.2 CALCOLO ML AVANZATO ---
        def run_ml_forecast(df_hist, periods, g_scale, m_scale, sat):
            # Prepariamo le feature
            df_train = df_hist.copy()
            
            # Feature critiche: Solo i "DRIVER" (quello che conosciamo o possiamo stimare)
            features_list = ['Week_Sin', 'Week_Cos', col_google, col_meta, 'Lag_Sales_1', 'Lag_Sales_4']
            
            # DRIVER DI EFFICIENZA (Influenzano il risultato ma non lo contengono)
            driver_metrics = [
                'Average order value', 'Returning customer rate', 'Discounts',
                'Avg. CPC', 'CPC (All)', 'CPM (Cost per 1,000 Impressions)', 'sessions', 'Frequency'
            ]
            valid_drivers = [m for m in driver_metrics if m in df_train.columns]
            features_list += valid_drivers
            
            # METRICHE DI RISULTATO (Sola diagnostica, rimosse dal forecast futuro per evitare pacchi)
            result_metrics = ['Conversions Value', 'Website Purchases Conversion Value', 'Orders', 'Items', 'Returns']
            
            # Gestione NaNs
            for f in features_list:
                if f in df_train.columns:
                    df_train[f] = df_train[f].fillna(df_train[f].median())
            
            df_train = df_train.dropna(subset=['Fatturato_Netto'])
            X = df_train[features_list]
            y = df_train['Fatturato_Netto']
            
            # Modello più conservativo (meno profondità, più alberi)
            model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            # Per il futuro, usiamo le medie delle ultime 4 settimane (più reattive al post-festività)
            avg_metrics = {m: df_hist[m].tail(4).mean() for m in valid_drivers}

            # Proiezione
            curr_sales_lag1 = df_hist['Fatturato_Netto'].iloc[-1]
            curr_sales_lag4 = df_hist['Fatturato_Netto'].iloc[-4]
            ml_rows = []
            
            for i in range(len(df_prev)):
                d = df_prev.iloc[i]['Data']
                w = d.isocalendar().week
                w_sin, w_cos = np.sin(2 * np.pi * w / 53), np.cos(2 * np.pi * w / 53)
                
                row_pred = {
                    'Week_Sin': w_sin, 'Week_Cos': w_cos,
                    col_google: df_prev.iloc[i]['Google Previsto'],
                    col_meta: df_prev.iloc[i]['Meta Previsto'],
                    'Lag_Sales_1': curr_sales_lag1, 'Lag_Sales_4': curr_sales_lag4
                }
                # Aggiungiamo le medie storiche per i parametri extra
                row_pred.update(avg_metrics)
                
                X_pred = pd.DataFrame([row_pred])[features_list] # Assicura ordine colonne
                pred_sales = model.predict(X_pred)[0]
                
                curr_sales_lag4 = curr_sales_lag1
                curr_sales_lag1 = pred_sales
                
                ml_rows.append({'Data': d, 'Fatturato_ML': pred_sales, 'Spesa_ML': row_pred[col_google] + row_pred[col_meta]})
            
            return pd.DataFrame(ml_rows), model, valid_drivers

        def run_prophet_forecast(df_hist, periods, g_scale, m_scale, seasonal_ref, extra_cols):
            # Filtriamo extra_cols per usare solo i driver (evitiamo overfitting su Conversion Value)
            drivers_only = [c for c in extra_cols if c not in ['Conversions Value', 'Orders', 'Items', 'Returns', 'Website Purchases Conversion Value']]
            
            # Prophet con Regressori + Configurazione per serie storiche lunghe
            df_p = df_hist[['Data_Interna', 'Fatturato_Netto', col_google, col_meta] + drivers_only].copy()
            mapping = {'Data_Interna': 'ds', 'Fatturato_Netto': 'y', col_google: 'google', col_meta: 'meta'}
            df_p = df_p.rename(columns=mapping)
            
            # Changepoint prior scale: 0.05 è bilanciato. Se troppo alto segue troppo i picchi, se troppo basso è troppo rigido.
            m = Prophet(
                yearly_seasonality=True, 
                weekly_seasonality=True, 
                daily_seasonality=False,
                changepoint_prior_scale=0.08, # Leggermente più flessibile per catturare trend pluriennali
                seasonality_prior_scale=10.0
            )
            m.add_regressor('google')
            m.add_regressor('meta')
            for exc in drivers_only:
                m.add_regressor(exc)
            
            m.add_country_holidays(country_name='IT')
            m.fit(df_p)
            
            future = m.make_future_dataframe(periods=int(periods*4.34), freq='W-MON')
            # Usiamo una media pesata (esponenziale) per i parametri business, 
            # dando più peso alle ultime 4-8 settimane rispetto a 6 anni fa.
            avg_metrics = {c: df_p[c].tail(8).mean() for c in drivers_only}

            def fill_future(ds):
                w = ds.isocalendar().week
                base = seasonal_ref[seasonal_ref['Week_Num'] == w]
                if base.empty: base = seasonal_ref.mean().to_frame().T
                return base[col_google].values[0] * g_scale, base[col_meta].values[0] * m_scale

            hist_len = len(df_p)
            future_budget = future.tail(len(future) - hist_len)['ds'].apply(fill_future)
            
            future['google'] = df_p['google'].tolist() + [x[0] for x in future_budget]
            future['meta'] = df_p['meta'].tolist() + [x[1] for x in future_budget]
            for exc in drivers_only:
                future[exc] = df_p[exc].tolist() + [avg_metrics[exc]] * (len(future) - hist_len)
            
            forecast = m.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(future) - hist_len), m

        def run_historical_backtest(df_hist, drivers, target_col):
            # Eseguiamo un backtest sugli ultimi 12 mesi per misurare l'affidabilità reale
            # Per ogni mese, alleniamo il modello sul passato e compariamo la previsione col reale
            backtest_results = []
            
            # Troviamo le date degli ultimi 12 mesi
            last_date = df_hist['Data_Interna'].max()
            start_backtest = last_date - pd.DateOffset(months=12)
            test_dates = df_hist[df_hist['Data_Interna'] > start_backtest]['Data_Interna'].unique()
            
            # Filtriamo i test_dates per prenderne uno per settimana (per velocità)
            for d in test_dates:
                d = pd.to_datetime(d)
                # Alleniamo solo con i dati precedenti a questa data
                train_data = df_hist[df_hist['Data_Interna'] < d].copy()
                if len(train_data) < 20: continue # Serve un minimo di storico
                
                # Allenamento veloce Random Forest
                feat_bt = ['Week_Sin', 'Week_Cos', col_google, col_meta, 'Lag_Sales_1', 'Lag_Sales_4'] + drivers
                X_train = train_data[feat_bt]
                y_train = train_data['Fatturato_Netto']
                
                # Imputazione NaNs nel training
                X_train = X_train.fillna(X_train.median())
                
                bm = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                bm.fit(X_train, y_train)
                
                # Previsione per la riga reale di quella settimana
                actual_row = df_hist[df_hist['Data_Interna'] == d].iloc[0]
                X_test = pd.DataFrame([{
                    'Week_Sin': np.sin(2 * np.pi * d.isocalendar().week / 53),
                    'Week_Cos': np.cos(2 * np.pi * d.isocalendar().week / 53),
                    col_google: actual_row[col_google],
                    col_meta: actual_row[col_meta],
                    'Lag_Sales_1': train_data['Fatturato_Netto'].iloc[-1],
                    'Lag_Sales_4': train_data['Fatturato_Netto'].iloc[-4] if len(train_data) > 4 else train_data['Fatturato_Netto'].iloc[-1]
                }])
                for drv in drivers: X_test[drv] = actual_row[drv]
                
                pred = bm.predict(X_test)[0]
                actual = actual_row['Fatturato_Netto']
                
                error = abs(pred - actual) / actual if actual > 0 else 0
                accuracy = max(0, 1 - error)
                
                backtest_results.append({
                    'Anno': d.year,
                    'Mese_Num': d.month,
                    'Mese': d.strftime('%b'),
                    'Accuratezza': accuracy
                })
            
            if not backtest_results: return None
            
            bt_df = pd.DataFrame(backtest_results)
            # Pivot table: Righe = Mese, Colonne = Anno
            pivot_bt = bt_df.groupby(['Mese', 'Anno', 'Mese_Num'])['Accuratezza'].mean().reset_index()
            pivot_bt = pivot_bt.pivot(index=['Mese_Num', 'Mese'], columns='Anno', values='Accuratezza').sort_index()
            return pivot_bt

        df_ml, ml_model, businesses_found = run_ml_forecast(df, mesi_prev, m_google, m_meta, sat_factor)
        df_prophet, p_model = run_prophet_forecast(df, mesi_prev, m_google, m_meta, seasonal, businesses_found)
        df_backtest = run_historical_backtest(df, businesses_found, 'Fatturato_Netto')
        
        # --- 6. VISUALIZZAZIONE TABS ---
        tabs = st.tabs([
            "🔮 ML Forecasting", "🔵 Analisi Google Ads", 
            "🔵 Analisi Meta Ads", "🧪 Analisi Saturazione Storica", "📊 Analisi Resi", 
            "🗂️ Dati CSV", "🧠 Insight AI", "🎯 Ottimizzazione", "🏥 Health Check"
        ])
        
        # COLORI
        DARKEST_BLUE = '#000080'  
        META_COLOR = '#3b5998'    
        GREEN_COLOR = '#2ecc71'   
        ORANGE_COLOR = '#e67e22'  
        

        with tabs[0]:
            st.header("🔮 Advanced AI Forecasting: Battle of Models")
            
            with st.expander("📖 Guida ai Modelli: Cosa sto leggendo?"):
                st.markdown("""
                In questa sezione, tre diverse "intelligenze" analizzano i tuoi dati per prevedere il futuro. Ognuna ha un punto di vista differente:
                
                1.  **🔸 Heuristic (Stagionalità Media):** 
                    *   **Cos'è:** Un modello basato sulla media storica. 
                    *   **Cosa guarda:** Ripete semplicemente l'andamento degli anni passati. 
                    *   **Limiti:** Non capisce se aumenti il budget o se il mercato è cambiato. È la tua "linea di base".
                
                2.  **💜 ML: Random Forest (Budget Focus):** 
                    *   **Cos'è:** Un algoritmo di Machine Learning puro. 
                    *   **Cosa guarda:** È molto sensibile alla **spesa pubblicitaria**. Cerca di capire: *"Se spendo 1€ in più su Meta, quanto fatturato extra ottengo?"*.
                    *   **Punto di forza:** È il migliore per simulare scenari di scalabilità del budget.
                
                3.  **🔹 AI: Facebook Prophet (Season Focus):** 
                    *   **Cos'è:** Un modello avanzato creato da Meta per i dati di business. 
                    *   **Cosa guarda:** Eccelle nel trovare **pattern ciclici** (es. ogni lunedì vendi di più) e l'effetto delle **festività** (Black Friday, Natale).
                    *   **Punto di forza:** Include la *"nuvola di incertezza"*, mostrandoti il rischio della previsione.
                
                4.  **📉 Media Ensemble:** 
                    *   **La verità sta nel mezzo:** Spesso la previsione più accurata è la media tra il focus sul budget (ML) e il focus sulla stagionalità (Prophet).
                """)

            st.caption("Confronto tra Random Forest (più sensibile al budget) e Facebook Prophet (più sensibile a stagionalità e festività).")
            
            col_ml1, col_ml2 = st.columns([2, 1])
            
            with col_ml1:
                fig_ml, ax_ml = plt.subplots(figsize=(10, 5))
                ax_ml.plot(df['Data_Interna'], df['Fatturato_Netto'], label='Storico Reale', color='gray', alpha=0.3)
                
                # Modello Heuristic
                ax_ml.plot(df_prev['Data'], df_prev['Fatturato Previsto'], label='Heuristic (Stag. Media)', color=ORANGE_COLOR, linestyle=':')
                
                # Modello Random Forest
                ax_ml.plot(df_ml['Data'], df_ml['Fatturato_ML'], label='ML: Random Forest (Budget Focus)', color='#9b59b6', linewidth=2)
                
                # Modello Prophet
                ax_ml.plot(df_prophet['ds'], df_prophet['yhat'], label='AI: Facebook Prophet (Season Focus)', color='#3498db', linewidth=2)
                ax_ml.fill_between(df_prophet['ds'], df_prophet['yhat_lower'], df_prophet['yhat_upper'], color='#3498db', alpha=0.15, label='Incertezza Prophet')
                
                ax_ml.set_title("Proiezione Multimodale")
                ax_ml.legend(fontsize=8)
                st.pyplot(fig_ml)
            
            with col_ml2:
                st.subheader("💡 Analisi Strategica AI")
                
                # Calcola quale modello prevede più fatturato
                rf_total = df_ml['Fatturato_ML'].sum()
                p_total = df_prophet['yhat'].sum()
                
                st.info(f"""
                **Previsione Totale Periodo:**
                * **Random Forest:** € {rf_total:,.0f}
                * **Prophet:** € {p_total:,.0f}
                """)
                
                diff = abs(rf_total - p_total) / min(rf_total, p_total)
                if diff < 0.1:
                    st.success("🟢 **Consenso Elevato:** I due modelli concordano. La previsione è molto solida.")
                else:
                    st.warning("🟡 **Divergenza:** I modelli hanno visioni diverse. Prophet vede più/meno stagionalità rispetto all'impatto del budget stimato dal RF.")
                
                st.subheader("Feature Importance (RF)")
                importances = ml_model.feature_importances_
                
                # Creiamo una lista leggibile delle feature presenti
                base_features = ['Stag. (S)', 'Stag. (C)', 'Budget G.', 'Budget M.', 'Lag 1w', 'Lag 4w']
                all_feature_names = base_features + [b.replace('Website Purchases ', 'Meta ').replace('Conversion Value', 'Val. Conv.') for b in businesses_found]
                
                feat_df = pd.DataFrame({'Feature': all_feature_names, 'Importanza': importances}).sort_values('Importanza', ascending=True)
                fig_feat, ax_feat = plt.subplots(figsize=(5, 8))
                ax_feat.barh(feat_df['Feature'], feat_df['Importanza'], color='#9b59b6')
                ax_feat.set_title("Cosa guida veramente il tuo business?", fontsize=10)
                ax_feat.tick_params(axis='both', which='major', labelsize=8)
                st.pyplot(fig_feat)

            st.divider()
            st.subheader("📈 Analisi Affidabilità: Quanto ha indovinato l'AI nel passato?")
            st.caption("Questa tabella mostra la percentuale di accuratezza che il modello avrebbe ottenuto se fosse stato usato negli anni scorsi (Backtesting).")
            
            if df_backtest is not None:
                # Applichiamo una colorazione per rendere la tabella leggibile
                def color_accuracy(val):
                    if pd.isna(val): return ''
                    color = 'green' if val > 0.9 else 'orange' if val > 0.8 else 'red'
                    return f'color: {color}'

                st.dataframe(df_backtest.style.format("{:.1%}") \
                           .applymap(color_accuracy))
                
                st.info("""
                **Cosa indicano i colori?**
                *   🟢 **Sopra 90%**: Il modello è estremamente solido in questo periodo.
                *   🟠 **80% - 90%**: Buona precisione, ma ci sono fattori esterni che influenzano il risultato.
                *   🔴 **Sotto 80%**: Periodo di forte instabilità (es. inizio saldi, eventi spot), l'AI qui va presa con cautela.
                """)
            else:
                st.info("Carica uno storico più lungo (almeno 6 mesi) per vedere l'analisi di affidabilità mensile.")

            st.divider()
            st.subheader("📅 Tabella Comparativa")
            # Uniamo le previsioni per una tabella chiara
            df_comp_final = df_ml.copy()
            df_prophet_clean = df_prophet[['ds', 'yhat']].copy()
            df_prophet_clean.columns = ['Data', 'Fatturato_Prophet']
            
            # NORMALIZZAZIONE DATE PER IL MERGE (Togliamo ore/minuti e allineiamo)
            df_comp_final['Data'] = pd.to_datetime(df_comp_final['Data']).dt.normalize()
            df_prophet_clean['Data'] = pd.to_datetime(df_prophet_clean['Data']).dt.normalize()
            
            # Usiamo un merge 'outer' o 'left' per sicurezza e debug
            df_final_tab = pd.merge(df_comp_final, df_prophet_clean, on='Data', how='inner')
            
            if df_final_tab.empty:
                st.warning("⚠️ Nota: I dati di Prophet e ML non sono allineati temporalmente. Prova a ricaricare i dati.")
            else:
                df_final_tab['Media_Ensemble'] = (df_final_tab['Fatturato_ML'] + df_final_tab['Fatturato_Prophet']) / 2
                
                st.dataframe(df_final_tab.style.format({
                    'Fatturato_ML': '€ {:,.0f}', 
                    'Fatturato_Prophet': '€ {:,.0f}', 
                    'Media_Ensemble': '€ {:,.0f}',
                    'Spesa_ML': '€ {:,.0f}'
                }))

            st.divider()
            st.subheader("🧪 Simulatore di Precisione (Actual vs Forecast)")
            st.write("Scegli una settimana dal tuo storico e confronta quello che è successo realmente con quello che l'AI avrebbe previsto.")
            
            # Selettore della settimana
            date_options = df['Data_Interna'].dt.date.unique()
            selected_test_date = st.selectbox(
                "📅 Seleziona la settimana da verificare", 
                options=date_options,
                index=len(date_options)-1,
                help="I dati di spesa e fatturato verranno pre-compilati automaticamente."
            )
            
            # Recupera dati reali per quella data
            actual_row = df[df['Data_Interna'].dt.date == selected_test_date].iloc[0]
            
            with st.container():
                c_test1, c_test2, c_test3 = st.columns(3)
                test_g = c_test1.number_input("Spesa Google (€)", value=float(actual_row[col_google]), step=100.0)
                test_m = c_test2.number_input("Spesa Meta (€)", value=float(actual_row[col_meta]), step=100.0)
                test_actual = c_test3.number_input("Fatturato Reale (€)", value=float(actual_row['Fatturato_Netto']), step=500.0)
                
                if st.button("🚀 Avvia Test di Validazione AI"):
                    # Trova l'indice per calcolare i LAG precedenti a quella data
                    idx_list = df[df['Data_Interna'].dt.date == selected_test_date].index
                    if not idx_list.empty:
                        idx = idx_list[0]
                        
                        # 1. Prediction con Random Forest
                        test_dt = pd.to_datetime(selected_test_date)
                        curr_w = test_dt.isocalendar().week
                        w_sin, w_cos = np.sin(2 * np.pi * curr_w / 53), np.cos(2 * np.pi * curr_w / 53)
                        
                        # Prendi i Lag basandoti sulla riga selezionata (usiamo dati disponibili PRIMA)
                        l1 = df['Fatturato_Netto'].iloc[idx-1] if idx > 0 else actual_row['Fatturato_Netto']
                        l4 = df['Fatturato_Netto'].iloc[idx-4] if idx > 3 else actual_row['Fatturato_Netto']
                        
                        row_val = {
                            'Week_Sin': w_sin, 'Week_Cos': w_cos, 
                            col_google: test_g, col_meta: test_m, 
                            'Lag_Sales_1': l1, 'Lag_Sales_4': l4
                        }
                        # Aggiungiamo i valori business reali di quella riga per il test
                        for b in businesses_found:
                            row_val[b] = actual_row[b]

                        X_val = pd.DataFrame([row_val])
                        rf_pred = ml_model.predict(X_val)[0]
                        
                        # 2. Prediction con Prophet
                        test_dict_p = {'ds': test_dt, 'google': test_g, 'meta': test_m}
                        for b in businesses_found:
                            test_dict_p[b] = actual_row[b]
                        
                        test_df_p = pd.DataFrame([test_dict_p])
                        p_pred = p_model.predict(test_df_p)['yhat'].values[0]
                        
                        avg_pred = (rf_pred + p_pred) / 2

                        st.markdown("---")
                        res_c1, res_c2, res_c3 = st.columns(3)
                        res_c1.metric("Previsione RF", f"€ {rf_pred:,.2f}")
                        res_c2.metric("Previsione Prophet", f"€ {p_pred:,.2f}")
                        res_c3.metric("Media Ensemble", f"€ {avg_pred:,.2f}", delta="Target AI")
                        
                        if test_actual > 0:
                            error = abs(avg_pred - test_actual) / test_actual
                            accuracy = (1 - error) * 100
                            st.subheader(f"🎯 Accuratezza Riscontrata: {accuracy:.1f}%")
                            
                            if accuracy > 92: st.success("🏆 Eccellente! I modelli hanno catturato perfettamente il trend di questa settimana.")
                            elif accuracy > 85: st.info("📉 Buona precisione. Lo scostamento rientra nei margini statistici.")
                            else: 
                                st.warning(f"⚠️ Scostamento del {error*100:.1f}%.")
                                # Analisi anomalie
                                st.write("**Possibili cause individuate dall'AI:**")
                                historical_cpc = df['Avg. CPC'].mean() if 'Avg. CPC' in df.columns else 0
                                if test_g/test_actual < df[col_google].mean()/df['Fatturato_Netto'].mean() * 0.8:
                                    st.write("- 🚩 **Efficienza Ads anomala**: Hai speso molto meno del solito per generare questo fatturato. C'era un evento organico?")
                                if 'Avg. CPC' in actual_row and actual_row['Avg. CPC'] > historical_cpc * 1.3:
                                    st.write("- 🚩 **CPC Alert**: Il costo per click di questa settimana era il 30% più alto della media, distorcendo la previsione.")
                                st.write("- 🚩 **Dato mancante**: L'AI non vede sconti o stock-out che potrebbero aver influenzato il risultato.")

        with tabs[1]:
            st.caption("Focus sulle performance storiche di Google Ads.")
            st.subheader("🔵 Performance Google Ads")
            if col_g_val in df.columns:
                g_metrics = df.tail(4)[[col_google, col_g_val, 'ROAS_Google', col_g_cpc, col_g_imps]].sum()
                st.columns(5)[0].metric("Spesa (4w)", f"€ {g_metrics[col_google]:,.0f}")
                
                fig_g, ax_g1 = plt.subplots(figsize=(12, 5))
                ax_g1.bar(df['Data_Interna'], df[col_google], color=DARKEST_BLUE, alpha=0.7, label='Spesa Google')
                ax_g2 = ax_g1.twinx()
                ax_g2.plot(df['Data_Interna'], df[col_g_val], color=GREEN_COLOR, linewidth=2, label='Valore Conversione')
                st.pyplot(fig_g)
                st.dataframe(df[['Periodo', col_google, col_g_val, 'ROAS_Google', col_g_cpc]].iloc[::-1].style.format({col_google: '€ {:,.2f}', col_g_val: '€ {:,.2f}', 'ROAS_Google': '{:.2f}', col_g_cpc: '€ {:,.2f}'}))

        with tabs[2]:
            st.caption("Focus sulle performance storiche di Meta Ads.")
            st.subheader("🔵 Performance Meta Ads")
            if col_m_val in df.columns:
                m_metrics = df.tail(4)[[col_meta, col_m_val, 'ROAS_Meta', col_m_cpc, col_m_cpm, col_m_freq]].sum()
                st.columns(6)[0].metric("Spesa (4w)", f"€ {m_metrics[col_meta]:,.0f}")
                
                fig_m, ax_m1 = plt.subplots(figsize=(12, 5))
                ax_m1.bar(df['Data_Interna'], df[col_meta], color=DARKEST_BLUE, alpha=0.7, label='Spesa Meta')
                ax_m2 = ax_m1.twinx()
                ax_m2.plot(df['Data_Interna'], df[col_m_val], color=GREEN_COLOR, linewidth=2, label='Website Purch. Value')
                st.pyplot(fig_m)
                st.dataframe(df[['Periodo', col_meta, col_m_val, 'ROAS_Meta', col_m_cpc, col_m_cpm, col_m_freq]].iloc[::-1].style.format({col_meta: '€ {:,.2f}', col_m_val: '€ {:,.2f}', 'ROAS_Meta': '{:.2f}', col_m_cpc: '€ {:,.2f}', col_m_cpm: '€ {:,.2f}', col_m_freq: '{:.2f}'}))

        with tabs[3]:
            st.caption("Analisi dell'elasticità: misura quanto il fatturato reagisce alle variazioni di spesa pubblicitaria.")
            st.header("🧪 Analisi Saturazione e Scalabilità")
            st.subheader("1. Riepilogo Annuale Completo")
            
            annual_agg = df.groupby('Year').agg({'Spesa_Ads_Totale': 'sum', 'Fatturato_Netto': 'sum'}).sort_index(ascending=False)
            annual_rows = []
            years_list = annual_agg.index.tolist()
            
            for i in range(len(years_list) - 1):
                curr_y, prev_y = years_list[i], years_list[i+1]
                s_curr, s_prev = annual_agg.loc[curr_y, 'Spesa_Ads_Totale'], annual_agg.loc[prev_y, 'Spesa_Ads_Totale']
                r_curr, r_prev = annual_agg.loc[curr_y, 'Fatturato_Netto'], annual_agg.loc[prev_y, 'Fatturato_Netto']
                
                d_spend = ((s_curr - s_prev) / s_prev * 100) if s_prev > 0 else 0
                d_rev = ((r_curr - r_prev) / r_prev * 100) if r_prev > 0 else 0
                elasticity = d_rev / d_spend if d_spend != 0 else 0
                
                annual_rows.append({'Confronto': f"{curr_y} vs {prev_y}", 'Delta Spesa %': d_spend, 'Delta Fatturato %': d_rev, 'Elasticità': elasticity})
            
            if annual_rows:
                st.dataframe(pd.DataFrame(annual_rows).style.format({'Delta Spesa %': '{:+.1f}%', 'Delta Fatturato %': '{:+.1f}%', 'Elasticità': '{:.2f}'}).background_gradient(subset=['Elasticità'], cmap='RdYlGn', vmin=0.5, vmax=1.5))

            st.divider()
            st.subheader("2. Dettaglio Settimanale")
            
            if not annual_rows:
                st.warning("Dati insufficienti.")
            else:
                comp_options = [row['Confronto'] for row in annual_rows]
                selected_comp = st.selectbox("Seleziona Anno da Confrontare", comp_options)
                curr_year_sel = int(selected_comp.split(" vs ")[0])
                prev_year_sel = int(selected_comp.split(" vs ")[1])
                
                all_weeks = pd.DataFrame({'Week': range(1, 54)})
                df_curr = df[df['Year'] == curr_year_sel][['Week', 'Spesa_Ads_Totale', 'Fatturato_Netto', 'Periodo']]
                df_hist_prev = df[df['Year'] == prev_year_sel][['Week', 'Spesa_Ads_Totale', 'Fatturato_Netto']]
                
                df_comp = pd.merge(all_weeks, df_curr, on='Week', how='left')
                df_comp = pd.merge(df_comp, df_hist_prev, on='Week', suffixes=('_Curr', '_Prev'), how='left').fillna(0)
                
                df_comp['Delta Spesa %'] = np.where(df_comp['Spesa_Ads_Totale_Prev'] > 0, ((df_comp['Spesa_Ads_Totale_Curr'] - df_comp['Spesa_Ads_Totale_Prev']) / df_comp['Spesa_Ads_Totale_Prev']) * 100, 0)
                df_comp['Delta Ricavi %'] = np.where(df_comp['Fatturato_Netto_Prev'] > 0, ((df_comp['Fatturato_Netto_Curr'] - df_comp['Fatturato_Netto_Prev']) / df_comp['Fatturato_Netto_Prev']) * 100, 0)
                df_comp['Elasticità'] = np.where(df_comp['Delta Spesa %'] != 0, df_comp['Delta Ricavi %'] / df_comp['Delta Spesa %'], 0)
                
                df_view = df_comp[(df_comp['Spesa_Ads_Totale_Curr'] > 0) | (df_comp['Spesa_Ads_Totale_Prev'] > 0)].sort_values('Week', ascending=False)
                
                st.dataframe(df_view[['Week', 'Periodo', 'Spesa_Ads_Totale_Curr', 'Spesa_Ads_Totale_Prev', 'Delta Spesa %', 'Delta Ricavi %', 'Elasticità']].style.format({'Spesa_Ads_Totale_Curr': '€ {:,.0f}', 'Spesa_Ads_Totale_Prev': '€ {:,.0f}', 'Delta Spesa %': '{:+.1f}%', 'Delta Ricavi %': '{:+.1f}%', 'Elasticità': '{:.2f}'}).background_gradient(subset=['Elasticità'], cmap='RdYlGn', vmin=0.5, vmax=1.5))
                
                fig_sat, ax_sat = plt.subplots(figsize=(10, 5))
                ax_sat.plot([-100, 500], [-100, 500], ls='--', color='gray', alpha=0.5)
                scatter = ax_sat.scatter(df_view['Delta Spesa %'], df_view['Delta Ricavi %'], c=df_view['Elasticità'], cmap='RdYlGn', s=80, edgecolor='black', vmin=0.6, vmax=1.4)
                ax_sat.set_xlabel("Variazione Spesa (%)")
                ax_sat.set_ylabel("Variazione Fatturato (%)")
                plt.colorbar(scatter, label='Elasticità')
                st.pyplot(fig_sat)

        with tabs[4]:
            st.caption("Confronto tra spesa e resi.")
            st.subheader("🔍 Spesa Ads vs Tasso Resi")
            fig2, ax1_2 = plt.subplots(figsize=(12, 6))
            ax1_2.bar(df['Data_Interna'], df['Spesa_Ads_Totale'], color=DARKEST_BLUE, alpha=0.5)
            ax2_2 = ax1_2.twinx()
            ax2_2.plot(df['Data_Interna'], df['Tasso_Resi'].rolling(4).mean(), color='#e74c3c', linewidth=2)
            st.pyplot(fig2)

        with tabs[5]:
            st.caption("Il database grezzo importato.")
            st.subheader("🗂️ Database Storico")
            display_cols = [col_date, 'Periodo', 'Total sales', col_google, col_g_val, col_g_cpc, col_g_imps, 
                            col_meta, col_m_val, col_m_cpc, col_m_cpm, col_m_freq, 'CoS', 'Profitto_Operativo']
            valid_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[valid_cols].iloc[::-1].style.format({'CoS': '{:.1f}%', 'Profitto_Operativo': '€ {:,.0f}'}, precision=2))

        # --- 8. TAB AI AVANZATA ---
        with tabs[6]:
            st.caption("Analisi automatica che incrocia Profitto, Retention e Performance Canali.")
            st.header("🧠 Insight AI: Analisi Strategica Completa")
            
            df['Month_Date'] = df['Data_Interna'].dt.to_period('M')
            ai_agg = {
                'Spesa_Ads_Totale': 'sum', 'Fatturato_Netto': 'sum', 'Orders': 'sum', 
                col_returns: 'sum', col_discounts: 'sum', col_google: 'sum', col_meta: 'sum', 
                col_g_val: 'sum', col_m_val: 'sum', 'Gross sales': 'sum', 'Profitto_Operativo': 'sum'
            }
            if col_ret_rate in df.columns: ai_agg[col_ret_rate] = 'mean'
            if col_m_freq in df.columns: ai_agg[col_m_freq] = 'mean'
            if col_m_cpm in df.columns: ai_agg[col_m_cpm] = 'mean'
            if col_g_cpc in df.columns: ai_agg[col_g_cpc] = 'mean'
            if col_m_cpc in df.columns: ai_agg[col_m_cpc] = 'mean'

            ai_df = df.groupby('Month_Date').agg(ai_agg).sort_index(ascending=False)
            
            ai_df['MER'] = ai_df['Fatturato_Netto'] / ai_df['Spesa_Ads_Totale'].replace(0, np.nan)
            ai_df['Discount_Rate'] = (ai_df[col_discounts].abs() / ai_df['Gross sales'].replace(0, np.nan)) * 100
            ai_df['ROAS_Google'] = ai_df[col_g_val] / ai_df[col_google].replace(0, np.nan)
            ai_df['ROAS_Meta'] = ai_df[col_m_val] / ai_df[col_meta].replace(0, np.nan)
            
            avg_sales = ai_df['Fatturato_Netto'].mean()
            ai_df['Seasonality'] = ai_df['Fatturato_Netto'] / avg_sales
            
            bench_mer = ai_df['MER'].mean()
            bench_ret = ai_df[col_ret_rate].mean() if col_ret_rate in df.columns else 0
            
            for m in ai_df.index[:12]:
                row = ai_df.loc[m]
                m_str = str(m)
                
                score = 50
                tags = []
                alerts = []
                
                if row['MER'] >= be_roas_val: 
                    score += 20
                    tags.append(f"Profittevole (> {be_roas_val:.2f})")
                else: 
                    score -= 20
                    alerts.append(f"Sotto Break-Even (MER {row['MER']:.2f})")
                
                if col_ret_rate in df.columns:
                    if row[col_ret_rate] > bench_ret * 1.1: score += 15; tags.append(f"Retention {row[col_ret_rate]:.1f}%")
                    elif row[col_ret_rate] < bench_ret * 0.8: score -= 10; alerts.append("Crollo Retention")
                
                if row['ROAS_Google'] > row['ROAS_Meta']: tags.append("Win: Google")
                else: tags.append("Win: Meta")
                
                seas_txt = "Media"
                if row['Seasonality'] > 1.2: seas_txt = "Alta Stagionalità 🔥"
                elif row['Seasonality'] < 0.8: seas_txt = "Bassa Stagionalità ❄️"
                
                color_class = "ai-score-high" if score >= 70 else "ai-score-med" if score >= 50 else "ai-score-low"
                
                with st.container():
                    st.markdown(f"""
                    <div class="ai-box {color_class}">
                        <div class="ai-title" style="display:flex; justify-content:space-between;">
                            <span>📅 {m_str} | Score: {score}/100</span>
                            <span style="font-size:0.8em; color:#666;">{seas_txt}</span>
                        </div>
                        <div style="margin:8px 0;">
                            {' '.join([f'<span class="ai-tag tag-blue">{t}</span>' for t in tags])}
                        </div>
                        <p style="margin:0; font-size:0.95em;">
                            Generati <b>€ {row['Fatturato_Netto']:,.0f}</b> con MER <b>{row['MER']:.2f}</b>. 
                            Profitto Operativo: <b>€ {row['Profitto_Operativo']:,.0f}</b>.
                            Incidenza sconti: <b>{row['Discount_Rate']:.1f}%</b>.
                        </p>
                        {''.join([f'<div class="ai-alert">⚠️ {a}</div>' for a in alerts])}
                    </div>
                    """, unsafe_allow_html=True)

        with tabs[7]:
            st.header("🎯 Strategia di Scalabilità Safe")
            st.subheader("Pianificazione Budget Ads per il Prossimo Mese")
            
            # Calcolo dei dati per il prossimo mese (4 settimane)
            # Usiamo df_prev che ora contiene correttamente le previsioni future
            next_4_weeks = df_prev.head(4)
            next_month_sales = df_prev.head(4)['Fatturato Previsto'].sum()
            
            # Budget Massimo = Fatturato stimato / ROAS di Pareggio
            max_safe_budget = next_month_sales / be_roas_val
            
            # Calcolo MER medio dell'ultimo mese storico per il semaforo
            mer_attuale = df['MER'].tail(4).mean()

            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                st.markdown(f"### ⚖️ Punto di Pareggio Reale\nIl tuo **Break-Even ROAS è {be_roas_val:.2f}**.")
                st.metric(
                    "💰 Budget Totale Max Sostenibile", 
                    f"€ {max_safe_budget:,.0f}", 
                    help="Questa è la spesa Ads totale (Google + Meta) massima per le prossime 4 settimane per non chiudere il mese in perdita operativa."
                )

            with col_opt2:
                st.markdown("### 🚀 Roadmap Strategica")
                # Il semaforo si basa sulle performance attuali rispetto al limite di pareggio
                if mer_attuale > be_roas_val * 1.2:
                    st.success(f"✅ **Semaforo Verde:** Hai un'ottima marginalità (MER {mer_attuale:.2f}). Puoi aumentare il budget Ads totale del 15-20% per le prossime settimane.")
                elif mer_attuale >= be_roas_val:
                    st.warning(f"⚠️ **Semaforo Giallo:** Sei vicino al break-even (MER {mer_attuale:.2f}). Scala il budget solo se riesci ad aumentare l'AOV o migliorare il Conversion Rate.")
                else:
                    st.error(f"🚨 **Semaforo Rosso:** Attualmente sei sotto soglia (MER {mer_attuale:.2f}). Non aumentare il budget! Rischieresti di scalare le perdite.")

            st.divider()
            
            # Simulatore di Scenario Target
            st.subheader("🔮 Simulatore: Obiettivo Profitto Netto")
            st.write("Imposta quanto vuoi guadagnare 'pulito' e ti dirò quanto puoi permetterti di spendere in Ads.")
            
            target_profit = st.slider(
                "Profitto desiderato per il prossimo mese (€)", 
                0, int(next_month_sales/2), 5000,
                help="Profitto operativo dopo aver pagato Tasse, Merce, Logistica e Pubblicità."
            )
            
            # Budget Target = Budget di Pareggio - Profitto Desiderato
            target_budget = max_safe_budget - target_profit
            needed_mer = next_month_sales / target_budget if target_budget > 0 else 0
            
            if target_budget > 0:
                st.info(f"""
                Per incassare **€ {target_profit:,.0f}** di profitto netto il prossimo mese:
                1. La tua spesa Ads totale (Google+Meta) deve essere di circa **€ {target_budget:,.0f}**.
                2. Devi mantenere un **MER minimo di {needed_mer:.2f}**.
                """)
            else:
                st.error("Il profitto richiesto è troppo alto rispetto al fatturato previsto. Riduci l'obiettivo o aumenta i margini.")      
     
        with tabs[8]:
            st.header("🏥 Stato di Salute del Progetto (YoY)")
            
            years_avail = sorted(df['Year'].unique(), reverse=True)
            if len(years_avail) < 2:
                st.warning("Dati insufficienti per un confronto YoY.")
            else:
                col_f1, col_f2 = st.columns(2)
                year_target = col_f1.selectbox("Anno", years_avail, index=0, key="year_t_9")
                year_comp = col_f2.selectbox("Confronta con", years_avail, index=min(1, len(years_avail)-1), key="year_c_9")

                def get_y_metrics(year):
                    d = df[df['Year'] == year].copy()
                    sales = d['Fatturato_Netto'].sum()
                    spend = d['Spesa_Ads_Totale'].sum()
                    orders = d['Orders'].sum() if 'Orders' in d.columns else (sales / be_aov)
                    
                    # Pulizia Returning Customer Rate (dal tuo CSV Euterpe)
                    if 'Returning customer rate' in d.columns:
                        ret_values = d['Returning customer rate'].astype(str).str.replace('%', '').str.strip()
                        ret_mean = pd.to_numeric(ret_values, errors='coerce').mean()
                    else:
                        ret_mean = 0

                    return {
                        'sales': sales,
                        'mer': sales / spend if spend > 0 else 0,
                        'cpa': spend / orders if orders > 0 else 0,
                        'cpc': d['CPC_Medio'].mean() if 'CPC_Medio' in d.columns else 0,
                        'aov': sales / orders if orders > 0 else 0,
                        'returns': (d['Returns'].abs().sum() / sales * 100) if sales > 0 else 0,
                        'ret': ret_mean
                    }

                mt = get_y_metrics(year_target)
                mc = get_y_metrics(year_comp)

                # PRIMA RIGA KPI
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    delta_sales = ((mt['sales'] - mc['sales']) / mc['sales'] * 100) if mc['sales'] > 0 else 0
                    st.metric("Fatturato", f"€ {mt['sales']:,.0f}", f"{delta_sales:+.1f}%")
                with c2:
                    delta_mer = ((mt['mer'] - mc['mer']) / mc['mer'] * 100) if mc['mer'] > 0 else 0
                    st.metric("ROAS (MER)", f"{mt['mer']:.2f}", f"{delta_mer:+.1f}%")
                with c3:
                    delta_cpa = ((mt['cpa'] - mc['cpa']) / mc['cpa'] * 100) if mc['cpa'] > 0 else 0
                    st.metric("CPA Medio", f"€ {mt['cpa']:.2f}", f"{delta_cpa:+.1f}%", delta_color="inverse")
                with c4:
                    delta_cpc = ((mt['cpc'] - mc['cpc']) / mc['cpc'] * 100) if mc['cpc'] > 0 else 0
                    st.metric("CPC Medio", f"€ {mt['cpc']:.2f}", f"{delta_cpc:+.1f}%", delta_color="inverse")

                st.divider()

                # SECONDA RIGA KPI (Qui c'era l'errore: ora definiamo c5, c6, c7, c8)
                c5, c6, c7, c8 = st.columns(4)
                with c5:
                    delta_aov = ((mt['aov'] - mc['aov']) / mc['aov'] * 100) if mc['aov'] > 0 else 0
                    st.metric("Carrello Medio (AOV)", f"€ {mt['aov']:.2f}", f"{delta_aov:+.1f}%")
                with c6:
                    delta_res = mt['returns'] - mc['returns']
                    st.metric("Tasso Resi (%)", f"{mt['returns']:.1f}%", f"{delta_res:+.1f}% (pt)", delta_color="inverse")
                with c7:
                    delta_ret = mt['ret'] - mc['ret']
                    st.metric("Returning Customer %", f"{mt['ret']:.2f}%", f"{delta_ret:+.2f}% (pt)")
                with c8:
                    status = "🟢 SANO" if mt['mer'] > be_roas_val else "🔴 CRITICO"
                    st.metric("Status Progetto", status)

                # Grafico
                st.subheader(f"Confronto Mensile: {year_target} vs {year_comp}")
                df_yoy = df[df['Year'].isin([year_target, year_comp])].copy()
                df_yoy['Mese'] = df_yoy['Data_Interna'].dt.month
                yoy_chart = df_yoy.groupby(['Year', 'Mese'])['Fatturato_Netto'].sum().unstack(level=0)
                st.bar_chart(yoy_chart)

    except Exception as e:
        st.error(f"⚠️ Errore: {e}")
else:
    st.info("👋 Carica il file CSV per iniziare.")
