import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import requests
import logging
import yfinance as yf

# Disabilita log pesanti di Prophet
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# 1. CONFIGURAZIONE PAGINA
st.set_page_config(page_title="Forecasting Strategico Pro - DEMO", layout="wide")

# STILE CSS PROFESSIONALE
st.markdown("""
    <style>
    /* ===== PREMIUM UNIVERSAL STYLING ===== */
    :root {
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Metric Cards Premium Look */
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        padding: 20px !important;
        border-radius: 15px !important;
        border: 1px solid var(--glass-border) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: var(--primary-color) !important;
    }

    /* Expander Fix & Design */
    div[data-testid="stExpander"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="stExpanderSummary"] {
        padding: 1rem !important;
    }

    /* AI Insight Boxes - Adaptive Colors */
    .ai-box {
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid var(--glass-border);
        color: var(--text-color) !important;
        backdrop-filter: blur(10px);
        background-color: var(--secondary-background-color);
        box-shadow: inset 0 0 20px rgba(0,0,0,0.02);
    }
    
    .ai-score-high { border-left: 6px solid #48bb78 !important; background-color: rgba(72, 187, 120, 0.1) !important; }
    .ai-score-med  { border-left: 6px solid #ed8936 !important; background-color: rgba(237, 137, 54, 0.1) !important; }
    .ai-score-low  { border-left: 6px solid #f56565 !important; background-color: rgba(245, 101, 101, 0.1) !important; }

    .ai-alert {
        background-color: rgba(197, 48, 48, 0.1);
        color: #f56565 !important;
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(197, 48, 48, 0.2);
        font-size: 0.9em;
        font-weight: 500;
    }

    .ai-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
        margin-right: 8px;
        background-color: var(--secondary-background-color);
        border: 1px solid var(--glass-border);
        color: var(--text-color);
    }

    /* Professional Sidebar */
    .stSidebar {
        background-color: var(--secondary-background-color) !important;
    }

    /* Typography Upgrades */
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    /* Improve form elements visual consistency */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        border-radius: 8px !important;
        border: 1px solid var(--glass-border) !important;
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
    """Pulisce valute in formato US (1,200.50) o EU (1.200,50) -> float."""
    if column is None: return 0
    s = column.astype(str).str.replace('\u20ac', '', regex=False).str.replace('$', '', regex=False).str.strip()
    # Rilevamento automatico del formato: se la virgola appare DOPO il punto è formato US
    # se il punto appare DOPO la virgola è formato EU
    def parse_val(v):
        if not isinstance(v, str): return 0.0
        v = v.strip()
        comma_pos = v.rfind(',')
        dot_pos = v.rfind('.')
        if comma_pos > dot_pos:
            # Formato EU: 1.234,56 -> rimuove punti, sostituisce virgola con punto
            v = v.replace('.', '').replace(',', '.')
        else:
            # Formato US/standard: 1,234.56 -> rimuove virgole
            v = v.replace(',', '')
        try:
            return float(v)
        except:
            return 0.0
    return pd.to_numeric(s.apply(parse_val), errors='coerce').fillna(0)

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

def format_euro(val, precision=0):
    if pd.isna(val) or val is None: return "€ 0"
    s = f"{val:,.{precision}f}"
    return "€ " + s.replace(",", "X").replace(".", ",").replace("X", ".")

def format_num_eu(val, precision=2, delta=False):
    if pd.isna(val) or val is None: return "0"
    prefix = "+" if delta and val > 0 else ""
    s = f"{val:,.{precision}f}"
    return prefix + s.replace(",", "X").replace(".", ",").replace("X", ".")

@st.cache_data(ttl=3600)
def fetch_global_signals():
    """Recupera segnali geopolitici ed economici reali via Yahoo Finance."""
    symbols = {
        'Oil': 'BZ=F',    # Brent Crude Oil (Logistica/Energia)
        'VIX': '^VIX',    # Volatility Index (Paura/Sentiment)
        'Gold': 'GC=F',   # Gold (Bene Rifugio/Guerra)
        'DXY': 'DX-Y.NYB' # US Dollar Index (Forza Valuta)
    }
    data = {}
    for name, sym in symbols.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="1y") # Recuperiamo 1 anno per il ledger
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev_7d = hist['Close'].iloc[-7] if len(hist) >= 7 else current
                prev_30d = hist['Close'].iloc[0]
                trend_7d = (current - prev_7d) / prev_7d
                trend_30d = (current - prev_30d) / prev_30d
                data[name] = {
                    'current': current,
                    'trend_7d': trend_7d,
                    'trend_30d': trend_30d,
                    'history': hist['Close']
                }
        except:
            pass
    return data

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
st.title("📈 E-commerce Strategic Decision Engine")

with st.expander("ℹ️ Guida Rapida: Cosa fa questo strumento?"):
    st.markdown("""
    Questo ecosistema rappresenta un **Advanced Strategic Decision Engine** progettato per trasformare i dati storici in una roadmap di crescita resiliente. Non è un semplice calcolatore, ma un framework di **Prescriptive Analytics** che guida l'imprenditore e il media buyer nella pianificazione finanziaria e nell'allocazione del budget pubblicitario.

    ### Architettura e Funzioni Core:

    1.  **📊 Financial Modeling & LTV Intelligence:**
        Oltre al calcolo del *Break-Even ROAS*, il sistema integra ora modelli di **Lifetime Value (LTV)**. Analizzando la *Retention Rate* e l'inerzia del fatturato (*Lag Sales*), il motore distingue tra acquisizioni "one-shot" e crescita strutturale, calcolando un **Health Score** che premia il valore nel tempo e non solo il ritorno immediato.

    2.  **🔮 Multi-Model Ensemble Forecasting:**
        Le proiezioni non sono lineari. Il motore combina **Facebook Prophet** (specializzato in stagionalità e trend macro) con **Random Forest Machine Learning** (sensibile alla spesa Ads). Questa combinazione permette di isolare l'impatto reale dei canali Paid dai flussi organici, fornendo intervalli di confidenza per decisioni basate sul rischio calcolato.

    3.  **🧪 Stress Test & Scenario Analysis (Novità):**
        Il sistema permette di simulare scenari di mercato ostili. Attraverso gli *Stress Slider*, è possibile prevedere l'impatto sul profitto netto in caso di aumento dei costi pubblicitari (**CPM/CPC**) o cali del tasso di conversione (**CR**). È lo strumento definitivo per valutare la **resilienza** del tuo business model.

    4.  **⚖️ Attribution Synergy (Market Gen vs Direct Intent):**
        Lo strumento abbandona la logica dell'attribuzione "last-click" semplificata. Riconosce a **Meta** il ruolo di *Market Generation* (creazione della domanda) e a **Google** (Shopping/PMax/Search) il ruolo di *Direct Intent* (conversione della domanda). L'algoritmo di allocazione suggerisce il mix di budget ideale per non "soffocare" il funnel di acquisizione.

    5.  **🏥 Like-for-Like (LFL) Health Check:**
        L'analisi della salute del progetto (YoY) utilizza ora una logica **LFL**, confrontando solo settimane omogenee tra anni diversi. Questo garantisce che la crescita rilevata sia reale e non distorta da dati parziali o anni incompleti.
    """)

# --- SIDEBAR: CONTROLLI ---
st.sidebar.header("🕹️ Dati & Pannello")
demo_mode = st.sidebar.toggle("🚀 Usa Modalità DEMO (Dati Casuali)", value=False)

uploaded_file = None
if not demo_mode:
    uploaded_file = st.sidebar.file_uploader("Carica il file .csv", type="csv")

st.sidebar.divider()

# --- SIDEBAR: BUSINESS ECONOMICS ---
st.sidebar.header("⚙️ Business Economics")
st.sidebar.info("**A cosa serve?** Questa sezione serve a calcolare il tuo **Break-Even point** (punto di pareggio). Inserendo i tuoi costi reali, il simulatore capisce qual è il ROAS minimo necessario per non andare in perdita.")

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
    st.caption("Questi sono i risultati derivati dai tuoi input. Calcolano quanto valore genera ogni ordine/cliente e definiscono i tuoi limiti di spesa (Break-Even) per il simulatore.")
    st.markdown("---")
    
    st.markdown(f"**AOV (Netto)**: {format_euro(aov_post_tax_returns, 2)}")
    st.caption("Formula: `(AOV * (1-Resi)) / (1+IVA)`")
    
    st.markdown(f"**Profitto/Ordine**: {format_euro(profit_order, 2)}")
    st.caption("Formula: `(AOV Netto * Margine%) - Spedizioni`")
    
    st.markdown(f"**Profitto/Cliente**: {format_euro(profit_per_customer, 2)}")
    st.caption("Formula: `Profitto Ordine + Valore Ricorsivo`")
    
    st.markdown("---")
    
    st.metric(
        "🎯 Break-Even CPA", 
        format_euro(be_cpa, 2),
        help="Costo per Acquisizione massimo sostenibile. Se spendi più di così per acquisire un cliente, sei in perdita."
    )
    
    st.metric(
        "🎯 Break-Even ROAS", 
        f"{format_num_eu(be_roas_val*100, 0)}% ({format_num_eu(be_roas_val, 2)})",
        help="Ritorno sulla spesa pubblicitaria minimo necessario. Formula: `AOV / Break Even CPA`."
    )




# --- GUIDA FORMATO CSV ---
with st.expander("📋 Guida: Come formattare il CSV per la versione completa"):
    st.markdown("""
    Per far funzionare correttamente gli algoritmi di AI e le analisi finanziarie, il file CSV deve seguire questa struttura:

    ### 🟢 Colonne Obbligatorie (Core)
    | Colonna | Descrizione | Formato Esempio |
    | :--- | :--- | :--- |
    | **`Year Week`** | Identificativo univoco Anno + Settimana | `202501` (Settimana 1 del 2025) |
    | **`Cost`** | Spesa totale su Google Ads | `1200.50` o `€ 1.200,50` |
    | **`Amount Spent`** | Spesa totale su Meta Ads | `850.00` o `€ 850,00` |
    | **`Total sales`** | Fatturato netto totale (Shopify) | `15000.00` o `€ 15.000,00` |

    ### 🟡 Colonne Opzionali (Analisi Avanzata)
    *Consigliate per abilitare i tab "Insight AI", "Resi" e "Ottimizzazione".*

    | Colonna | Utilizzo | Formato Esempio |
    | :--- | :--- | :--- |
    | `Returns` | Calcolo marginalità reale e Tab Resi | `-120.00` (Valore possibilmente negativo) |
    | `Discounts` | Analisi pressione promozionale in AI Insight | `-450.00` |
    | `Orders` | Calcolo Carrello Medio (AOV) e CPA | `120` (Numero intero) |
    | `Average order value` | Backup per Business Economics | `125.00` |
    | `Returning customer rate` | Analisi Retention in AI Insight | `15%` o `0.15` |
    | `Conversions Value` | ROAS specifico Google Ads | `5000.00` |
    | `Website Purchases Conversion Value` | ROAS specifico Meta Ads | `4000.00` |
    | `Avg. CPC` / `CPC (All)` | Qualità del traffico e Costi Click | `0.85` |
    | `CPM` / `Frequency` | Analisi saturazione Meta | `12.50` / `1.2` |
    """)

# --- LOGICA CARICAMENTO E PULIZIA (UNIFICATA) ---
df = None

# 1. Recupero DataFrame (Demo o File)
with st.spinner("📂 Caricamento e preparazione dati..."):
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

        # Rilevazione robusta colonne: priorita' a match esatto, fallback su contiene
        def find_col(df, keywords_exact, keywords_contains, fallback):
            for k in keywords_exact:
                if k in df.columns: return k
            for k in keywords_contains:
                matches = [c for c in df.columns if k.lower() in c.lower()]
                if matches: return matches[0]
            return fallback

        col_date = find_col(df, ['Year Week', 'Settimana'], ['year week', 'settimana'], None)
        col_google = find_col(df, ['Cost'], [], 'Cost')  # Match esatto 'Cost' per evitare 'Cost per Click' ecc.
        col_meta = find_col(df, ['Amount Spent'], ['amount spent'], 'Amount Spent')
        col_sales = find_col(df, ['Total sales'], ['total sales'], 'Total sales')
        col_returns = find_col(df, ['Returns'], ['returns'], 'Returns')
        col_orders = find_col(df, ['Orders'], ['orders'], 'Orders')
        col_aov = find_col(df, ['Average order value'], ['average order value'], 'Average order value')
        
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
        df['Year'] = df['Data_Interna'].dt.isocalendar().year
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
        # Spesa_Ads_Totale conserva il valore 0 (non NaN) per KPI e somme corrette
        df['Spesa_Ads_Totale'] = df[col_google] + df[col_meta]
        # Versione sicura per divisioni (evita NaN cascade sui KPI)
        spesa_safe = df['Spesa_Ads_Totale'].replace(0, np.nan)
        df['MER'] = (df[col_sales] / spesa_safe).fillna(0)
        df['Tasso_Resi'] = (df[col_returns].abs() / df[col_sales].replace(0, np.nan) * 100).fillna(0)
        
        # Calcolo CoS Storico
        df['CoS'] = (df['Spesa_Ads_Totale'] / df['Fatturato_Netto'].replace(0, np.nan) * 100).fillna(0)
        
        # --- CALCOLO PROFITTO NETTO STIMATO NEL DF ---
        num_orders = df[col_orders] if col_orders in df.columns else (df['Fatturato_Netto'] / be_aov)
        # Profitto Operativo: usa la spesa_safe per il calcolo corretto (0 se nessuna spesa)
        df['Profitto_Operativo'] = ((num_orders * profit_order) - df['Spesa_Ads_Totale']).fillna(0)

        # Inizializzazione sicura ROAS
        df['ROAS_Google'] = 0.0
        df['ROAS_Meta'] = 0.0
        if col_g_val in df.columns: df['ROAS_Google'] = df[col_g_val] / df[col_google].replace(0, np.nan).fillna(0)
        if col_m_val in df.columns: df['ROAS_Meta'] = df[col_m_val] / df[col_meta].replace(0, np.nan).fillna(0)

        # --- AUTO-CALCOLO ELASTICITÀ (solo su anni completi >= 48 settimane) ---
        df_annual = df.groupby('Year').agg(
            spesa=('Spesa_Ads_Totale', 'sum'),
            fatturato=('Fatturato_Netto', 'sum'),
            n_weeks=('Fatturato_Netto', 'count')
        ).sort_index()
        suggested_saturation = 0.85
        anni_completi = df_annual[df_annual['n_weeks'] >= 48]
        if len(anni_completi) >= 2:
            last_year = anni_completi.index[-1]
            prev_year = anni_completi.index[-2]
            d_spend = (anni_completi.loc[last_year, 'spesa'] - anni_completi.loc[prev_year, 'spesa']) / anni_completi.loc[prev_year, 'spesa']
            d_rev = (anni_completi.loc[last_year, 'fatturato'] - anni_completi.loc[prev_year, 'fatturato']) / anni_completi.loc[prev_year, 'fatturato']
            if d_spend > 0.05 and d_rev != 0:
                raw_elasticity = d_rev / d_spend
                suggested_saturation = float(np.clip(raw_elasticity, 0.60, 1.0))

        # --- CALCOLO TREND YoY ---
        last_date = df['Data_Interna'].max()
        start_last_year = last_date - pd.Timedelta(weeks=52)
        start_prev_year = start_last_year - pd.Timedelta(weeks=52)
        sales_ly = df[(df['Data_Interna'] > start_last_year) & (df['Data_Interna'] <= last_date)]['Fatturato_Netto'].sum()
        sales_py = df[(df['Data_Interna'] > start_prev_year) & (df['Data_Interna'] <= start_last_year)]['Fatturato_Netto'].sum()
        growth_rate = (sales_ly - sales_py) / sales_py if sales_py > 0 else 0.0

        # Storico Annuale Like-for-Like (LFL) - Logica Unificata con Health Check
        historical_growth_data = []
        years_avail = sorted(df['Year'].unique(), reverse=True)
        for i in range(len(years_avail) - 1):
            curr_y = years_avail[i]
            prev_y = years_avail[i+1]
            
            # 1. Identifichiamo le settimane disponibili per entrambi gli anni (Confronto Omogeneo)
            weeks_curr = set(df[df['Year'] == curr_y]['Week'].unique())
            weeks_prev = set(df[df['Year'] == prev_y]['Week'].unique())
            common_weeks = sorted(list(weeks_curr.intersection(weeks_prev)))
            
            if common_weeks:
                # 2. Sommiamo il fatturato solo per le settimane comuni
                val_curr_lfl = df[(df['Year'] == curr_y) & (df['Week'].isin(common_weeks))]['Fatturato_Netto'].sum()
                val_prev_lfl = df[(df['Year'] == prev_y) & (df['Week'].isin(common_weeks))]['Fatturato_Netto'].sum()
                
                if val_prev_lfl > 0:
                    g_y = (val_curr_lfl - val_prev_lfl) / val_prev_lfl
                    # Se stiamo confrontando un anno parziale (es. l'attuale), aggiungiamo il flag LFL
                    is_partial = len(common_weeks) < 48
                    tag = " (LFL)" if is_partial else ""
                    historical_growth_data.append(f"📅 {curr_y} vs {prev_y}: **{g_y:+.1%}{tag}**")
                else:
                    historical_growth_data.append(f"📅 {curr_y} vs {prev_y}: **N/A**")
            else:
                # Fallback: Se non c'è nessuna settimana in comune, mostriamo il dato totale (es. primo anno)
                val_curr = df[df['Year'] == curr_y]['Fatturato_Netto'].sum()
                val_prev = df[df['Year'] == prev_y]['Fatturato_Netto'].sum()
                if val_prev > 0:
                    g_y = (val_curr - val_prev) / val_prev
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


        st.sidebar.divider()
        
        # --- SIDEBAR: SIMULATORE AVANZATO (Raggruppato) ---
        with st.sidebar.expander("🎮 Simulatore Avanzato", expanded=False):
            st.markdown("### ⚡ Azioni Rapide (Presets)")
            col_b1, col_b2 = st.columns(2)
            if col_b1.button("🛡️ Prudente"):
                st.session_state.trend_val = 0.0; st.session_state.google_scale = 1.0; st.session_state.meta_scale = 1.0; st.session_state.sat_val = 0.80; st.rerun()
            if col_b2.button("🚀 Aggressivo"):
                st.session_state.trend_val = 0.15; st.session_state.google_scale = 1.5; st.session_state.meta_scale = 1.5; st.session_state.sat_val = 0.90; st.rerun()
            if st.button(f"🎯 Auto-Calibra (Saturazione: {suggested_saturation:.2f})"):
                st.session_state.trend_val = 0.0; st.session_state.google_scale = 1.0; st.session_state.meta_scale = 1.0; st.session_state.sat_val = float(suggested_saturation); st.rerun()
            
            st.divider()
            st.markdown("### 🚀 Configurazione Manuale")
            
            # Sezione Trend
            st.markdown("**1. Trend & Crescita**", help="Aggiusta la proiezione futura rispetto alla crescita storica rilevata. 0% significa seguire il trend attuale, valori positivi simulano un'accelerazione del brand.")
            st.sidebar.markdown(f"*(Crescita Rilevata YoY: {growth_rate:+.1%})*")
            manual_trend = st.slider("Aggiusta Trend Futuro", -0.5, 2.0, key="trend_val")
            
            st.divider()
            
            # Sezione Budget
            st.markdown("**2. Scenari Budget**", help="Moltiplica la spesa storica media per testare scenari di investimento. 1.0x = spesa attuale, 2.0x = raddoppio del budget.")
            m_google = st.slider("🚀 Google Ads Budget Scale", 0.5, 3.0, key="google_scale")
            m_meta = st.slider("🚀 Meta Ads Budget Scale", 0.5, 3.0, key="meta_scale")
            
            st.divider()
            
            # Sezione Saturazione
            st.markdown("**3. Calibrazione Saturazione**", help="Definisce quanto l'efficacia delle Ads cala all'aumentare della spesa (Rendimenti Decrescenti). 0.85 è il valore standard, 0.70 indica un mercato molto saturo.")
            sat_factor = st.slider("Coefficiente Saturazione", 0.5, 1.0, key="sat_val")
            
            # Grafico Saturazione ridotto
            x_sat = np.linspace(1, 4, 20)
            y_sat = x_sat ** sat_factor
            fig_sat, ax_sat = plt.subplots(figsize=(4, 2))
            ax_sat.plot(x_sat, x_sat, ls='--', color='gray', alpha=0.3)
            ax_sat.plot(x_sat, y_sat, color='#e74c3c', linewidth=2)
            ax_sat.set_title("Curva Rendimenti Marginali", fontsize=8)
            ax_sat.tick_params(labelsize=6)
            st.pyplot(fig_sat)
            
            st.divider()
            
            # Durata
            mesi_prev = st.number_input("Mesi di Previsione", 1, 24, 6)

        st.sidebar.divider()
        st.sidebar.subheader("📉 Stress Test (Scenario Analysis)")
        with st.sidebar.expander("🧪 Simula un peggioramento del mercato", expanded=False):
            st.markdown("""
            Usa questi slider per capire quanto è **resiliente** il tuo business. 
            Cosa succede se i costi salgono o le persone convertono meno?
            """)
            
            # --- INTEGRAZIONE SEGNALI REALI NELLO STRESS TEST ---
            risk_signals = fetch_global_signals()
            if risk_signals:
                oil_risk = risk_signals.get('Oil', {}).get('trend_7d', 0)
                vix_risk = risk_signals.get('VIX', {}).get('trend_7d', 0)
                
                if oil_risk > 0.05 or vix_risk > 0.10:
                    st.error(f"⚡ **Segnale di Rischio Rilevato!**")
                    if oil_risk > 0.05: st.caption(f"Petrolio: +{oil_risk:.1%} (7gg)")
                    if vix_risk > 0.10: st.caption(f"Volatilità: +{vix_risk:.1%} (7gg)")
                    if st.button("🚀 Auto-Applica Stress"):
                        cpm_val = int(min(100, (oil_risk * 100) * 2))
                        cr_val = int(min(50, (vix_risk * 100) * 1.5))
                        # Nota: Questi slider sono definiti sotto, quindi usiamo session_state
                        st.session_state['cpm_stress_slider'] = cpm_val
                        st.session_state['cr_stress_slider'] = cr_val
                        st.rerun()

            cpm_stress = st.slider("Aumento Costi Pubblicitari (CPM/CPC %)", 0, 100, 0, key="cpm_stress_slider", help="Simula un aumento dei costi di acquisizione a parità di budget.")
            cr_stress = st.slider("Calo Tasso di Conversione (%)", 0, 50, 0, key="cr_stress_slider", help="Simula un calo dell'efficacia del sito o della propensione all'acquisto.")
        
        # Trasformiamo in moltiplicatori (es. +20% CPM -> 0.83x efficacia spesa; -10% CR -> 0.90x fatturato)
        cpm_mult = 1 / (1 + (cpm_stress / 100))
        cr_mult = 1 - (cr_stress / 100)
        stress_total_mult = cpm_mult * cr_mult

        # --- 4. DASHBOARD KPI (Status Attuale vs Precedente) ---
        st.divider()
        
        # Ultime 4 settimane (ordinate per data per sicurezza)
        df_sorted = df.sort_values('Data_Interna')
        last_4 = df_sorted.tail(4)
        tot_sales = last_4['Fatturato_Netto'].sum()
        tot_ads = last_4['Spesa_Ads_Totale'].fillna(0).sum()
        mer_attuale = tot_sales / tot_ads if tot_ads > 0 else 0
        cos = (tot_ads / tot_sales * 100) if tot_sales > 0 else 0
        profit = last_4['Profitto_Operativo'].fillna(0).sum()

        # 4 settimane ancora precedenti (per Delta)
        prev_4 = df_sorted.iloc[-8:-4] if len(df_sorted) >= 8 else None
        d_sales = d_ads = d_profit = d_mer = d_cos = d_aov = None
        
        if prev_4 is not None:
            p_sales = prev_4['Fatturato_Netto'].sum()
            p_ads = prev_4['Spesa_Ads_Totale'].sum()
            p_profit = prev_4['Profitto_Operativo'].sum()
            p_mer = p_sales / p_ads if p_ads > 0 else 0
            p_cos = (p_ads / p_sales * 100) if p_sales > 0 else 0
            
            # Calcolo AOV per delta
            c_orders = last_4[col_orders].sum() if col_orders in last_4.columns else (tot_sales / be_aov)
            p_orders = prev_4[col_orders].sum() if col_orders in prev_4.columns else (p_sales / be_aov)
            c_aov = tot_sales / c_orders if c_orders > 0 else 0
            p_aov = p_sales / p_orders if p_orders > 0 else 0
            
            d_sales = f"{(tot_sales - p_sales)/p_sales:+.1%}" if p_sales > 0 else None
            d_ads = f"{(tot_ads - p_ads)/p_ads:+.1%}" if p_ads > 0 else None
            d_profit = f"{(profit - p_profit)/abs(p_profit):+.1%}" if abs(p_profit) > 0 else None
            d_mer = f"{mer_attuale - p_mer:+.2f}"
            d_cos = f"{cos - p_cos:+.1f}%"
            d_aov = f"{(c_aov - p_aov)/p_aov:+.1%}" if p_aov > 0 else None

        st.subheader("📊 Analisi Performance Recente (Last 4w)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Fatturato (4w)", format_euro(tot_sales, 0), delta=d_sales)
        c2.metric("Spesa Ads (4w)", format_euro(tot_ads, 0), delta=d_ads, delta_color="inverse")
        c3.metric("MER / BE", f"{format_num_eu(mer_attuale, 2)} / {format_num_eu(be_roas_val, 2)}", delta=f"{format_num_eu(mer_attuale-be_roas_val, 2)} (vs BE)")
        c4.metric("CoS (Spesa/Ricavi)", f"{format_num_eu(cos, 1)}%", delta=d_cos, delta_color="inverse")
        c5.metric("Profitto Stimato", format_euro(profit, 0), delta=d_profit, help="Profitto Operativo netto stimato.")

        # --- 🌐 ANALISI DEL CONTESTO (INTELLIGENT INSIGHTS) ---
        with st.expander("🌐 Analisi del Contesto: Perché questo andamento?"):
            st.markdown(f"**Periodo Corrente:** {last_4['Periodo'].iloc[0]} ➔ {last_4['Periodo'].iloc[-1]}")
            if prev_4 is not None:
                st.markdown(f"**Confronto con:** {prev_4['Periodo'].iloc[0]} ➔ {prev_4['Periodo'].iloc[-1]} (Periodo Precedente)")
            
            st.divider()
            
            # --- 1. PROMOZIONI (Basate su Discounts) ---
            disc_curr = abs(last_4[col_discounts].sum()) / last_4['Fatturato_Netto'].sum() if last_4['Fatturato_Netto'].sum() > 0 else 0
            disc_prev = abs(prev_4[col_discounts].sum()) / prev_4['Fatturato_Netto'].sum() if prev_4 is not None and prev_4['Fatturato_Netto'].sum() > 0 else 0
            
            col_ctx1, col_ctx2 = st.columns(2)
            with col_ctx1:
                st.markdown("##### 🏷️ Pressione Promozionale")
                if disc_curr > 0.15:
                    st.warning(f"**Alta Promozionalità:** Sconti al {disc_curr:.1%}. Questo spiega il volume ma occhio al margine!")
                elif disc_curr > disc_prev + 0.05:
                    st.info(f"**Promo Attiva:** Pressione sconti del {disc_curr:.1%} (era {disc_prev:.1%}).")
                else:
                    st.success(f"**Prezzi Stabili:** Incidenza sconti contenuta ({disc_curr:.1%}).")

            # --- 2. EVENTI ESTERNI (Meteo/Stagione) ---
            with col_ctx2:
                st.markdown("##### 📅 Eventi & Stagionalità")
                curr_months = last_4['Data_Interna'].dt.month.unique()
                if 1 in curr_months:
                    st.info("❄️ **Periodo Saldi Invernali:** Alta propensione all'acquisto ma traffico comparativo più caro.")
                elif 7 in curr_months:
                    st.info("☀️ **Periodo Saldi Estivi:** Possibile calo del conversion rate fuori dai giorni di picco.")
                elif 11 in curr_months:
                    st.warning("🖤 **Black Friday Approaching:** I CPM di Meta tendono a rimpiazzare l'efficienza con la saturazione.")
                else:
                    st.write("Dinamiche di mercato standard per il periodo selezionato.")

            st.caption("Nota: Il confronto è Month-over-Month (MoM). Se vedi cali drastici, verifica se il periodo precedente includeva festività (Natale, BF, etc.)")

        # --- MINI ADVISER STRATEGICO ---
        with st.container():
            st.markdown("""<style>.adviser-box { padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; margin-top: 10px; }</style>""", unsafe_allow_html=True)
            
            if mer_attuale < be_roas_val:
                st.error(f"🔴 **CRITICO:** Il tuo MER ({mer_attuale:.2f}) è inferiore al punto di pareggio ({be_roas_val:.2f}). Stai perdendo € {abs(profit):,.0f} al mese. Intervieni subito sui costi Ads o sul magazzino.")
            elif mer_attuale < be_roas_val * 1.25:
                st.warning(f"🟡 **ATTENZIONE:** Sei in profitto ma con margini ridotti. Il tuo cuscinetto di sicurezza è del {(mer_attuale/be_roas_val-1):.1%}. Un aumento del CAC su Meta potrebbe mandarti in perdita.")
            else:
                st.success(f"🟢 **SCALABILITÀ:** Efficienza eccellente ({mer_attuale:.2f} vs {be_roas_val:.2f}). Hai margine operativo per aumentare i budget Ads del 15-20% senza compromettere la stabilità.")

        # --- 5. CALCOLO PREVISIONALE ---
        # Pre-calcolo delle feature per tutti i modelli (ML e Backtest)
        df['Week'] = df['Data_Interna'].dt.isocalendar().week
        df['Week_Sin'] = np.sin(2 * np.pi * df['Week'] / 53)
        df['Week_Cos'] = np.cos(2 * np.pi * df['Week'] / 53)
        df['Lag_Sales_1'] = df['Fatturato_Netto'].shift(1)
        df['Lag_Sales_4'] = df['Fatturato_Netto'].shift(4)

        df['Week_Num'] = df['Week']  # Per retrocompatibilità stagionale
        seasonal = df.groupby('Week_Num').agg({
            'Fatturato_Netto': 'mean', col_google: 'mean', col_meta: 'mean', col_orders: 'mean'
        }).reset_index()

        # Calcolo future_dates qui per passarlo come parametro esplicito al modello ML
        future_dates_list = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=int(mesi_prev * 4.34),
            freq='W-MON'
        )

        rows = []
        for d in future_dates_list:
            w = d.isocalendar().week
            base = seasonal[seasonal['Week_Num'] == w]
            if base.empty: base = seasonal.mean(numeric_only=True).to_frame().T
            
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
        def run_ml_forecast(df_hist, future_df, g_scale, m_scale, sat, stress_mult):
            """Proiezione con Random Forest. future_df è il DataFrame Heuristic già calcolato."""
            df_train = df_hist.copy()
            
            # Feature critiche
            features_list = ['Week_Sin', 'Week_Cos', col_google, col_meta, 'Lag_Sales_1', 'Lag_Sales_4']
            driver_metrics = [
                'Average order value', 'Returning customer rate', 'Discounts',
                'Avg. CPC', 'CPC (All)', 'CPM (Cost per 1,000 Impressions)', 'Frequency'
            ]
            valid_drivers = [m for m in driver_metrics if m in df_train.columns]
            features_list += valid_drivers
            
            for f in features_list:
                if f in df_train.columns:
                    df_train[f] = df_train[f].fillna(df_train[f].median())
            
            df_train = df_train.dropna(subset=['Fatturato_Netto'])
            if len(df_train) < 10:
                return pd.DataFrame(), None, valid_drivers

            X = df_train[features_list]
            y = df_train['Fatturato_Netto']
            
            model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            avg_metrics = {m: df_hist[m].tail(4).mean() for m in valid_drivers}
            curr_sales_lag1 = df_hist['Fatturato_Netto'].iloc[-1]
            curr_sales_lag4 = df_hist['Fatturato_Netto'].iloc[-4] if len(df_hist) >= 4 else curr_sales_lag1
            ml_rows = []
            
            for i in range(len(future_df)):
                d = future_df.iloc[i]['Data']
                w = d.isocalendar().week
                w_sin, w_cos = np.sin(2 * np.pi * w / 53), np.cos(2 * np.pi * w / 53)
                row_pred = {
                    'Week_Sin': w_sin, 'Week_Cos': w_cos,
                    col_google: future_df.iloc[i]['Google Previsto'],
                    col_meta: future_df.iloc[i]['Meta Previsto'],
                    'Lag_Sales_1': curr_sales_lag1, 'Lag_Sales_4': curr_sales_lag4
                }
                row_pred.update(avg_metrics)
                X_pred = pd.DataFrame([row_pred])[features_list]
                pred_sales = model.predict(X_pred)[0] * stress_mult
                curr_sales_lag4 = curr_sales_lag1
                curr_sales_lag1 = pred_sales
                ml_rows.append({'Data': d, 'Fatturato_ML': pred_sales,
                                'Spesa_ML': row_pred[col_google] + row_pred[col_meta]})
            
            return pd.DataFrame(ml_rows), model, valid_drivers

        def run_prophet_forecast(df_hist, periods, g_scale, m_scale, seasonal_ref, extra_cols, stress_mult):
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
            # Applicazione Stress Multiplier alle colonne di Prophet
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                forecast[col] = forecast[col] * stress_mult
            
            # Estrazione sicura: solo date > storico reale
            is_future = forecast['ds'] > df_p['ds'].max()
            df_forecast_future = forecast[is_future][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            df_forecast_hist = forecast[~is_future][['ds', 'yhat']].copy()
            
            return df_forecast_future, m, df_forecast_hist

        def run_historical_backtest(df_hist, drivers, target_col, seasonal_df, prophet_hist_df=None):
            backtest_results = []
            last_date = df_hist['Data_Interna'].max()
            start_backtest = last_date - pd.DateOffset(months=12)
            test_dates = df_hist[df_hist['Data_Interna'] > start_backtest]['Data_Interna'].unique()
            
            for d in test_dates:
                d = pd.to_datetime(d)
                train_data = df_hist[df_hist['Data_Interna'] < d].copy()
                if len(train_data) < 20: continue 
                
                feat_bt = ['Week_Sin', 'Week_Cos', col_google, col_meta, 'Lag_Sales_1', 'Lag_Sales_4'] + drivers
                X_train = train_data[feat_bt].fillna(train_data[feat_bt].median(numeric_only=True))
                y_train = train_data['Fatturato_Netto']
                
                # Modello potenziato per riflettere le performance reali dell'app
                bm = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
                bm.fit(X_train, y_train)
                
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
                
                pred_ml = bm.predict(X_test)[0]
                
                # 1. Heuristic (Stagionalità Media)
                w_num = d.isocalendar().week
                h_row = seasonal_df[seasonal_df['Week_Num'] == w_num]
                pred_h = h_row['Fatturato_Netto'].values[0] if not h_row.empty else seasonal_df['Fatturato_Netto'].mean()
                
                # 3. Prophet
                pred_p = pred_ml # Fallback
                if prophet_hist_df is not None:
                    p_val = prophet_hist_df[pd.to_datetime(prophet_hist_df['ds']).dt.date == d.date()]['yhat'].values
                    pred_p = p_val[0] if len(p_val) > 0 else pred_ml
                
                # 4. Ensemble (Media)
                pred_ens = (pred_ml + pred_p) / 2

                actual = actual_row['Fatturato_Netto']
                
                # Calcolo Errori per trovare il Vincente
                errs = {
                    'Heuristic': abs(pred_h - actual),
                    'Random Forest': abs(pred_ml - actual),
                    'Prophet': abs(pred_p - actual),
                    'Ensemble': abs(pred_ens - actual)
                }
                winner = min(errs, key=errs.get)
                
                # Calcolo Accuratezze Individuali
                def get_acc(p, a): return max(0, 1 - (abs(p - a) / a)) if a > 0 else 0
                
                backtest_results.append({
                    'Anno': d.year,
                    'Mese': d.strftime('%b'),
                    'Mese_Num': d.month,
                    'Vincente': winner,
                    # Ensemble (Principale)
                    'Accuratezza': get_acc(pred_ens, actual),
                    'Errore_Euro': abs(pred_ens - actual),
                    # Heuristic
                    'Acc_Heuristic': get_acc(pred_h, actual),
                    'Err_Heuristic': abs(pred_h - actual),
                    # ML
                    'Acc_ML': get_acc(pred_ml, actual),
                    'Err_ML': abs(pred_ml - actual),
                    # Prophet
                    'Acc_Prophet': get_acc(pred_p, actual),
                    'Err_Prophet': abs(pred_p - actual)
                })
            
            return pd.DataFrame(backtest_results) if backtest_results else None

        with st.spinner("🧠 Calcolo algoritmi predittivi e analisi in corso..."):
            df_ml, ml_model, businesses_found = run_ml_forecast(df, df_prev, m_google, m_meta, sat_factor, stress_total_mult)
            df_prophet, p_model, df_prophet_hist = run_prophet_forecast(df, mesi_prev, m_google, m_meta, seasonal, businesses_found, stress_total_mult)
            df_backtest = run_historical_backtest(df, businesses_found, 'Fatturato_Netto', seasonal, df_prophet_hist)
        
        # --- 6. VISUALIZZAZIONE TABS ---
        tabs = st.tabs([
            "🔮 ML Forecasting", "🔵 Analisi Google Ads", 
            "🔵 Analisi Meta Ads", "🧪 Market Elasticity Hub", "📊 Analisi Resi", 
            "🗂️ Dati CSV", "🧠 Insight AI", "🎯 Ottimizzazione", "🏥 Health Check",
            "🌍 Market Intelligence"
        ])
        
        # COLORI
        DARKEST_BLUE = '#000080'  
        META_COLOR = '#3b5998'    
        GREEN_COLOR = '#2ecc71'   
        ORANGE_COLOR = '#e67e22'  
        

        with tabs[0]:
            st.info("**Cosa fa:** Confronta tre metodologie (Heuristic, Machine Learning, Facebook Prophet) per prevedere il fatturato futuro basato sui piani di budget.  \n**Logica:** Allena gli algoritmi sui dati storici per capire l'impatto della spesa pubblicitaria e della stagionalità.")
            st.header("🔮 Advanced AI Forecasting: Battle of Models")
            
            # --- ALERT STRESS TEST ATTIVO ---
            if stress_total_mult < 1.0:
                st.error(f"""
                ### 🚨 SCENARIO DI CRISI ATTIVO
                **Stai simulando un mercato ostile:**
                *   Hai ipotizzato un calo dei ricavi del **{(1-stress_total_mult)*100:.0f}%** dovuto a costi pubblicitari più alti o conversioni più basse.
                *   I numeri che vedi qui sotto sono **estremamente conservativi** per prepararti al peggiore dei casi.
                """)
            
            # --- NOTA DINAMICA SULL'ACCURATEZZA ---
            if df_backtest is not None:
                # Calcoliamo la media dell'accuratezza (escludendo i NaN)
                avg_accuracy = df_backtest['Accuratezza'].mean()
                
                if avg_accuracy >= 0.85:
                    status_icon, status_label, status_color = "✅", "ALTA", "success"
                    status_text = "Il modello ha trovato pattern solidi e ricorrenti. Previsioni molto affidabili per decisioni di budget a lungo termine."
                    st_func = st.success
                elif avg_accuracy >= 0.70:
                    status_icon, status_label, status_color = "🟠", "MEDIA", "warning"
                    status_text = "Il business ha una buona stabilità, ma ci sono rumori statistici o fattori esterni (es. promozioni variabili) che l'IA non può prevedere con certezza totale."
                    st_func = st.warning
                else:
                    status_icon, status_label, status_color = "🚨", "BASSA", "error"
                    status_text = "Alta volatilità rilevata. Il business potrebbe essere influenzato pesantemente da sconti spot, flash sales o mancare di variabili chiave (es. database email/SMS). Usa i dati con cautela."
                    st_func = st.error

                st_func(f"**{status_icon} Affidabilità IA: {status_label} ({avg_accuracy:.1%})**  \n{status_text}")
            else:
                st.info("📊 **Affidabilità in calcolo:** Carica uno storico più lungo (almeno 6-8 mesi) per generare il bollino di qualità dell'IA.")

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
                # Storico Reale con markers annuali alla base per "timeline" e linea grigia
                ax_ml.plot(df['Data_Interna'], df['Fatturato_Netto'], label='Storico Reale', color='gray', alpha=0.4)
                
                # Markers e Label Fatturato Annuo alla base per Timeline
                annual_summary = df.groupby('Year').agg({'Data_Interna': 'min', 'Fatturato_Netto': 'sum'})
                for yr, row in annual_summary.iterrows():
                    # Posizione alla base
                    base_y = df['Fatturato_Netto'].min() * 0.1
                    ax_ml.scatter(row['Data_Interna'], base_y, s=60, color='gray', alpha=0.4, marker='|')
                    
                    # Formattazione K/M compatta
                    val = row['Fatturato_Netto']
                    label_val = f"€ {val/1e6:.1f}M" if val >= 1e6 else f"€ {val/1e3:.0f}k"
                    
                    # Testo sopra la tacca
                    ax_ml.text(row['Data_Interna'], base_y * 1.8, label_val, 
                              color='gray', fontsize=7, ha='center', va='bottom', fontweight='bold', alpha=0.7)
                
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
                
                # Calcolo della Previsione Combinata (Ensemble)
                rf_total = df_ml['Fatturato_ML'].sum()
                p_total = df_prophet['yhat'].sum()
                ensemble_total = (rf_total + p_total) / 2
                
                st.metric("Fatturato Totale Previsto (Ensemble)", format_euro(ensemble_total, 0), 
                          help="Questa è la media ponderata tra l'impatto del budget e la stagionalità. È il dato più affidabile per pianificare il cashflow.")
                
                st.divider()
                st.info(f"🔮 **Inizio Previsioni:** Dalla settimana del **{(last_date + pd.Timedelta(weeks=1)).strftime('%d %b %Y')}** ({mesi_prev} mesi)")
                
                viz_mode = st.radio("Seleziona dettaglio tabella:", ["Mensile", "Settimanale"], horizontal=True, key="forecast_viz")
                
                # Sincronizzazione e normalizzazione date per il merge
                df_p_m = df_prev[['Data', 'Fatturato Previsto']].copy()
                df_p_m['Data'] = pd.to_datetime(df_p_m['Data']).dt.normalize()
                
                df_m_m = df_ml[['Data', 'Fatturato_ML']].copy()
                df_m_m['Data'] = pd.to_datetime(df_m_m['Data']).dt.normalize()
                
                df_pr_m = df_prophet[['ds', 'yhat']].copy()
                df_pr_m['Data'] = pd.to_datetime(df_pr_m['ds']).dt.normalize()
                
                # Unificazione dati settimanali con merge robusto
                df_all_w = df_p_m.rename(columns={'Fatturato Previsto': 'Heuristic'})
                df_all_w = df_all_w.merge(df_m_m.rename(columns={'Fatturato_ML': 'Random Forest (Budget)'}), on='Data', how='left')
                df_all_w = df_all_w.merge(df_pr_m[['Data', 'yhat']].rename(columns={'yhat': 'Prophet (Season)'}), on='Data', how='left')
                
                # Calcolo Ensemble (Media che ignora i NaN per sicurezza operativa)
                df_all_w['Ensemble (Consigliato)'] = df_all_w[['Random Forest (Budget)', 'Prophet (Season)']].mean(axis=1)
                
                if viz_mode == "Settimanale":
                    st.markdown("##### 📅 Tabella Comparativa Previsionale (Settimanale)")
                    df_display = df_all_w.copy()
                    df_display['Data'] = df_display['Data'].dt.strftime('%d-%m-%Y')
                    st.dataframe(df_display.style.format({
                        'Heuristic': '€ {:,.0f}',
                        'Random Forest (Budget)': '€ {:,.0f}',
                        'Prophet (Season)': '€ {:,.0f}',
                        'Ensemble (Consigliato)': '€ {:,.0f}'
                    }), use_container_width=True)
                else:
                    st.markdown("##### 📅 Tabella Comparativa Previsionale (Mensile)")
                    # Aggregazione mensile intelligente
                    df_all_w['Mese'] = df_all_w['Data'].dt.to_period('M').astype(str)
                    df_summary_m = df_all_w.groupby('Mese').agg({
                        'Heuristic': 'sum',
                        'Random Forest (Budget)': 'sum',
                        'Prophet (Season)': 'sum',
                        'Ensemble (Consigliato)': 'sum'
                    }).reset_index()
                    
                    st.dataframe(df_summary_m.style.format({
                        'Heuristic': '€ {:,.0f}',
                        'Random Forest (Budget)': '€ {:,.0f}',
                        'Prophet (Season)': '€ {:,.0f}',
                        'Ensemble (Consigliato)': '€ {:,.0f}'
                    }), use_container_width=True)

                st.divider()
                
                diff_pct = abs(rf_total - p_total) / min(rf_total, p_total)
                optimistic_model = "Machine Learning (Budget Focus)" if rf_total > p_total else "Prophet (Stagionalità)"
                cautious_model = "Prophet (Stagionalità)" if rf_total > p_total else "Machine Learning (Budget Focus)"
                
                # Integriamo l'accuratezza storica nel verdetto di consenso
                is_unstable = avg_accuracy < 0.70 if 'avg_accuracy' in locals() else False

                if diff_pct < 0.15:
                    if not is_unstable:
                        st.success("🟢 **Consenso Elevato:** I modelli concordano. La previsione è molto solida.")
                    else:
                        st.warning("🟡 **Consenso Fragile:** I modelli concordano sulla cifra, ma l'alta volatilità storica (vedasi a sinistra) suggerisce comunque cautela nell'esecuzione.")
                elif diff_pct < 0.35:
                    st.warning(f"🟡 **Divergenza Moderata ({diff_pct:.1%}):** Il modello **{optimistic_model}** è più ottimista rispetto a **{cautious_model}**. Questa è una classica forchetta di mercato.")
                else:
                    st.error(f"🚨 **Alta Discrepanza ({diff_pct:.1%}):** C'è una forte tensione tra l'andamento recente (Momentum) e lo storico degli anni passati.")
                
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

                with st.expander("📖 Legenda: Cosa significano queste voci?"):
                    st.markdown("""
                    Il grafico mostra quali variabili pesano di più nel calcolo delle previsioni dell'IA:
                    
                    *   **🌊 Stag. (S) / (C):** Componenti stagionali (Seno/Coseno). Se sono alte, il tuo business è influenzato ciclicamente dal periodo dell'anno.
                    *   **💰 Budget G. / M.:** Impatto della spesa pubblicitaria su Google e Meta. Indicano quanto il fatturato "risponde" ai tuoi investimenti.
                    *   **⏳ Lag 1w / 4w:** Inerzia del fatturato (1 sett. e 4 sett. fa). Se sono alte, il tuo business è molto stabile e basato sul brand o sulla retention.
                    *   **🎯 Val. Conv. / Altro:** Parametri qualitativi (es. valore di conversione storico) rilevati nel tuo CSV.
                    """)

            st.divider()
            st.subheader("🛡️ Audit di Affidabilità AI: Backtesting Report")
            st.caption("Analisi storica delle performance: l'AI ha sfidato i dati reali degli ultimi 12 mesi per misurare la sua precisione.")
            
            if df_backtest is not None:
                # Tab per ogni modello
                audit_tabs = st.tabs(["🧩 Ensemble (Media)", "💜 Random Forest (ML)", "🔹 Facebook Prophet (AI)", "🔸 Heuristic (Stagionalità)"])
                
                # Setup per il loop dei tab
                model_configs = [
                    {"tab": audit_tabs[0], "acc_col": "Accuratezza", "err_col": "Errore_Euro", "name": "Ensemble"},
                    {"tab": audit_tabs[1], "acc_col": "Acc_ML", "err_col": "Err_ML", "name": "Random Forest"},
                    {"tab": audit_tabs[2], "acc_col": "Acc_Prophet", "err_col": "Err_Prophet", "name": "Prophet"},
                    {"tab": audit_tabs[3], "acc_col": "Acc_Heuristic", "err_col": "Err_Heuristic", "name": "Heuristic"}
                ]

                def style_acc(val):
                    if pd.isna(val): return ''
                    color = '#27ae60' if val > 0.9 else '#f39c12' if val > 0.8 else '#e74c3c'
                    return f'color: {color}; font-weight: bold'

                for config in model_configs:
                    with config["tab"]:
                        m_acc = df_backtest[config["acc_col"]].mean()
                        m_err = df_backtest[config["err_col"]].mean()
                        
                        ca, cb = st.columns(2)
                        ca.metric(f"Accuratezza {config['name']}", f"{m_acc:.1%}")
                        cb.metric(f"Errore Medio (€)", f"€ {m_err:,.0f}")
                        
                        # Heatmap Accuratezza
                        st.markdown(f"**📈 Accuratezza {config['name']} (%)**")
                        p_acc = df_backtest.groupby(['Mese', 'Anno', 'Mese_Num'])[[config["acc_col"]]].mean().reset_index()
                        p_acc = p_acc.pivot(index=['Mese_Num', 'Mese'], columns='Anno', values=config["acc_col"]).sort_index()
                        st.dataframe(p_acc.style.format("{:.1%}") \
                                   .map(style_acc), use_container_width=True)
                        
                        # Tabella Errore
                        st.markdown(f"**💶 Scostamento {config['name']} (€)**")
                        p_err = df_backtest.groupby(['Mese', 'Anno', 'Mese_Num'])[[config["err_col"]]].mean().reset_index()
                        p_err = p_err.pivot(index=['Mese_Num', 'Mese'], columns='Anno', values=config["err_col"]).sort_index()
                        st.dataframe(p_err.style.format("€ {:,.0f}") \
                                   .background_gradient(cmap='YlOrRd'), use_container_width=True)
                
                # --- SPIEGAZIONE METODOLOGICA (Richiesta Utente) ---
                with st.expander("🧠 Scienza e Metodologia: Come arriviamo a questo numero?"):
                    ens_acc = df_backtest['Accuratezza'].mean()
                    ens_mae = df_backtest['Errore_Euro'].mean()
                    st.markdown(f"""
                    L'accuratezza del **{ens_acc:.1%}** non è una stima teorica, ma il risultato di un rigoroso processo di **Backtesting (Walk-forward Validation)**.
                    
                    ### 1. Il Processo di "Esame"
                    Per ogni settimana dell'ultimo anno, l'IA ha "sfidato" se stessa:
                    *   **Simulazione del Passato:** Si è posizionata in una data passata (es. Ottobre 2024), facendo finta di non conoscere il futuro.
                    *   **Training Dinamico:** Si è allenata solo sui dati disponibili *prima* di quella data.
                    *   **Previsione vs Realtà:** Ha previsto il fatturato usando i budget Ads reali di quel periodo e lo ha confrontato con l'incasso effettivo registrato nel tuo CSV.
                    
                    ### 2. I Due "Cervelli" (Modelli Ensemble)
                    Il numero che vedi è la media di due diverse intelligenze che lavorano insieme:
                    *   **💜 Random Forest (Il Muscolo):** Un algoritmo di Machine Learning avanzato che analizza l'impatto diretto del **Budget**. Capisce quanto ogni Euro investito su Google/Meta muove l'ago della bilancia.
                    *   **🔹 Facebook Prophet (L'Orologio):** Un'IA statistica di Meta specializzata nella **Stagionalità**. Individua i cicli ricorrenti (giorni festivi, weekend, stagioni) che si ripetono ogni anno nel tuo business.
                    
                    ### 3. Cosa significa il Verdetto?
                    *   **ALTA (>85%):** Business estremamente prevedibile. L'efficienza Ads è costante.
                    *   **MEDIA (70-85%):** Caso tipico dell'E-commerce. L'IA cattura i trend principali, ma esistono "rumori" esterni (flash sales, promo email, stockout) che creano variazioni non tracciate.
                    *   **BASSA (<70%):** Alta volatilità. Il business è guidato da fattori che non sono nel CSV.
                    
                    **Conclusione:** Un errore medio di **€ {ens_mae:,.0f}** indica la "forchetta" di rischio da tenere in conto quando pianifichi le tue prossime scalate di budget.
                    """)
                
                # Tabella 3: Modello Vincente
                st.markdown("#### 🏆 Modello Vincente (Battle of Models)")
                pivot_win = df_backtest.groupby(['Mese', 'Anno', 'Mese_Num'])['Vincente'].first().reset_index()
                pivot_win = pivot_win.pivot(index=['Mese_Num', 'Mese'], columns='Anno', values='Vincente').sort_index()
                
                def color_winner(val):
                    if val == 'Ensemble': color = '#2ecc71'
                    elif val == 'Random Forest': color = '#9b59b6'
                    elif val == 'Prophet': color = '#3498db'
                    else: color = '#e67e22' # Heuristic
                    return f'background-color: {color}; color: white; font-weight: bold'

                st.dataframe(pivot_win.style.map(color_winner), use_container_width=True)
                
                # Summary Vincitori
                top_winner = df_backtest['Vincente'].mode()[0]
                st.info(f"🏅 **Analisi Storica:** Il modello più preciso per il tuo business è stato **{top_winner}**. Questo significa che storicamente {'la media dei modelli' if top_winner=='Ensemble' else 'l impatto del budget' if top_winner=='Random Forest' else 'la stagionalità pura'} ha fornito i risultati più vicini alla realtà.")
            else:
                st.info("Carica uno storico più lungo (almeno 6 mesi) per generare il report di affidabilità.")

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
                        res_c1.metric("Previsione RF", format_euro(rf_pred, 2))
                        res_c2.metric("Previsione Prophet", format_euro(p_pred, 2))
                        res_c3.metric("Media Ensemble", format_euro(avg_pred, 2), delta="Target AI")
                        
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
            st.info("**Cosa fa:** Analisi verticale delle performance di Google Ads (Spesa vs Valore Conversione).  \n**Logica:** Calcola metriche dinamiche come ROAS e CPC medio degli ultimi 30 giorni per misurare l'efficienza diretta del canale.")
            st.caption("Focus sulle performance storiche di Google Ads.")
            st.subheader("🔵 Performance Google Ads")
            if col_g_val in df.columns:
                g_metrics = df.tail(4)[[col_google, col_g_val, 'ROAS_Google', col_g_cpc, col_g_imps]].sum()
                st.columns(5)[0].metric("Spesa (4w)", format_euro(g_metrics[col_google], 0))
                
                fig_g, ax_g1 = plt.subplots(figsize=(12, 5))
                ax_g1.bar(df['Data_Interna'], df[col_google], color=DARKEST_BLUE, alpha=0.7, label='Spesa Google')
                ax_g2 = ax_g1.twinx()
                ax_g2.plot(df['Data_Interna'], df[col_g_val], color=GREEN_COLOR, linewidth=2, label='Valore Conversione')
                st.pyplot(fig_g)
                st.dataframe(df[['Periodo', col_google, col_g_val, 'ROAS_Google', col_g_cpc]].iloc[::-1].style.format({col_google: '€ {:,.2f}', col_g_val: '€ {:,.2f}', 'ROAS_Google': '{:.2f}', col_g_cpc: '€ {:,.2f}'}))

        with tabs[2]:
            st.info("**Cosa fa:** Analisi verticale delle performance di Meta Ads.  \n**Logica:** Monitora ROAS, CPM e frequenza per valutare la salute delle campagne social e l'impatto del pixel di tracciamento.")
            st.caption("Focus sulle performance storiche di Meta Ads.")
            st.subheader("🔵 Performance Meta Ads")
            if col_m_val in df.columns:
                m_metrics = df.tail(4)[[col_meta, col_m_val, 'ROAS_Meta', col_m_cpc, col_m_cpm, col_m_freq]].sum()
                st.columns(6)[0].metric("Spesa (4w)", format_euro(m_metrics[col_meta], 0))
                
                fig_m, ax_m1 = plt.subplots(figsize=(12, 5))
                ax_m1.bar(df['Data_Interna'], df[col_meta], color=DARKEST_BLUE, alpha=0.7, label='Spesa Meta')
                ax_m2 = ax_m1.twinx()
                ax_m2.plot(df['Data_Interna'], df[col_m_val], color=GREEN_COLOR, linewidth=2, label='Website Purch. Value')
                st.pyplot(fig_m)
                st.dataframe(df[['Periodo', col_meta, col_m_val, 'ROAS_Meta', col_m_cpc, col_m_cpm, col_m_freq]].iloc[::-1].style.format({col_meta: '€ {:,.2f}', col_m_val: '€ {:,.2f}', 'ROAS_Meta': '{:.2f}', col_m_cpc: '€ {:,.2f}', col_m_cpm: '€ {:,.2f}', col_m_freq: '{:.2f}'}))

        with tabs[3]:
            st.info("**Cosa fa:** Analizza la reattività del tuo business agli aumenti di budget. Misura l'elasticità storica per aiutarti a calibrare correttamente il simulatore.  \n**Logica:** Un coefficiente di 1.0 indica una crescita lineare (spendi 2x, incassi 2x). Valori inferiori indicano che il mercato sta saturando.")
            st.header("🧪 Market Elasticity Hub")
            
            # --- 1. IL VERDETTO DELL'IA ---
            # --- 1. IL VERDETTO DELL'IA (Logica LFL) ---
            years_avail = sorted(df['Year'].unique(), reverse=True)
            annual_rows = []
            
            for i in range(len(years_avail) - 1):
                y_curr, y_prev = years_avail[i], years_avail[i+1]
                
                # Trova settimane comuni per un confronto LFL reale
                weeks_curr = set(df[df['Year'] == y_curr]['Week'].unique())
                weeks_prev = set(df[df['Year'] == y_prev]['Week'].unique())
                common_w = weeks_curr.intersection(weeks_prev)
                
                if common_w:
                    d_curr = df[(df['Year'] == y_curr) & (df['Week'].isin(common_w))]
                    d_prev = df[(df['Year'] == y_prev) & (df['Week'].isin(common_w))]
                    
                    s_curr, s_prev = d_curr['Spesa_Ads_Totale'].sum(), d_prev['Spesa_Ads_Totale'].sum()
                    r_curr, r_prev = d_curr['Fatturato_Netto'].sum(), d_prev['Fatturato_Netto'].sum()
                    
                    d_spend = ((s_curr - s_prev) / s_prev) if s_prev > 0 else 0
                    d_rev = ((r_curr - r_prev) / r_prev) if r_prev > 0 else 0
                    
                    # Elasticità = % Delta Rev / % Delta Spend
                    elasticity = d_rev / d_spend if abs(d_spend) > 0.01 else 0
                    
                    annual_rows.append({
                        'Confronto': f"{y_curr} vs {y_prev} (LFL)", 
                        'Delta Spesa %': d_spend * 100, 
                        'Delta Fatturato %': d_rev * 100, 
                        'Elasticità': elasticity,
                        'Settimane': len(common_w)
                    })

            if annual_rows:
                avg_elasticity = np.mean([r['Elasticità'] for r in annual_rows if r['Elasticità'] > 0])
                
                col_v1, col_v2 = st.columns([1, 2])
                with col_v1:
                    st.metric("Elasticità Media Storica", f"{avg_elasticity:.2f}")
                with col_v2:
                    if avg_elasticity > 0.95:
                        status_msg = "🚀 **Business Altamente Scalabile:** Il mercato risponde quasi linearmente. Puoi aumentare il budget con fiducia."
                    elif avg_elasticity > 0.75:
                        status_msg = "⚖️ **Efficienza Standard:** Rilevati rendimenti decrescenti fisiologici. Scalare richiede attenzione ai margini."
                    else:
                        status_msg = "⚠️ **Segnali di Saturazione:** La crescita del fatturato è molto più lenta della spesa. Focus sull'efficienza prima di scalare."
                    st.success(status_msg)
                
                st.info(f"👉 **Consiglio Tecnico:** Imposta il 'Coefficiente Saturazione' nel **Simulatore Avanzato** (sidebar) su un valore vicino a **{avg_elasticity:.2f}** per proiezioni massimamente accurate.")

            st.divider()
            st.subheader("📊 Analisi Comparativa Annuale")
            
            if annual_rows:
                st.dataframe(pd.DataFrame(annual_rows).style.format({'Delta Spesa %': '{:+.1f}%', 'Delta Fatturato %': '{:+.1f}%', 'Elasticità': '{:.2f}'}) \
                           .background_gradient(subset=['Elasticità'], cmap='RdYlGn', vmin=0.5, vmax=1.5))

            st.divider()
            st.subheader("2. Dettaglio Settimanale")
            
            if not annual_rows:
                st.warning("Dati insufficienti.")
            else:
                comp_options = [row['Confronto'] for row in annual_rows]
                selected_comp = st.selectbox("Seleziona Anno da Confrontare", comp_options)
                
                # Parsing dei nomi degli anni (rimuovendo il suffisso LFL se presente)
                years_parts = selected_comp.replace(" (LFL)", "").split(" vs ")
                curr_year_sel = int(years_parts[0])
                prev_year_sel = int(years_parts[1])
                
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
                
                with st.expander("📖 Guida alla Lettura: Come interpretare questa Mappa di Scalabilità?"):
                    st.markdown("""
                    Questo grafico a dispersione (Scatter Plot) mette in relazione la tua spesa pubblicitaria con la risposta del mercato:
                    
                    *   **Asse X (Variazione Spesa %):** Quanto hai aumentato o diminuito il budget rispetto allo scorso anno.
                    *   **Asse Y (Variazione Fatturato %):** Come sono cambiate le vendite reali in risposta a quel budget.
                    *   **La Diagonale Tratteggiata:** Rappresenta l'equilibrio (Elasticità 1.0). 
                        - Se i punti sono **SOPRA** la linea: Sei in zona di **Iper-Efficienza**. Il mercato risponde meglio di quanto investi.
                        - Se i punti sono **SOTTO** la linea: Sei in zona di **Saturazione**. Stai pagando un "costo marginale" più alto per ogni nuova vendita.
                    
                    **Il Significato dei Colori:**
                    - 🟢 **Verde (Elasticità > 1.2):** Mercato reattivo. Ogni euro speso in più genera una crescita di fatturato più che proporzionale. **ZONA DI SCALA.**
                    - 🟡 **Giallo/Arancio (0.8 - 1.2):** Rendimenti costanti. La crescita è lineare. **ZONA DI MANTENIMENTO.**
                    - 🔴 **Rosso (Elasticità < 0.7):** Saturazione rilevata. Hai raggiunto il limite del pubblico attuale o l'offerta ha perso appeal. **ZONA DI OTTIMIZZAZIONE.**
                    """)

        with tabs[4]:
            st.info("**Cosa fa:** Monitora l'incidenza dei resi sul fatturato lordo per valutare la qualità delle vendite.  \n**Logica:** Applica una media mobile a 4 settimane per identificare trend strutturali e impatti sulla marginalità finale.")
            st.caption("Confronto tra spesa e resi.")
            st.subheader("🔍 Spesa Ads vs Tasso Resi")
            fig2, ax1_2 = plt.subplots(figsize=(12, 6))
            ax1_2.bar(df['Data_Interna'], df['Spesa_Ads_Totale'], color=DARKEST_BLUE, alpha=0.5)
            ax2_2 = ax1_2.twinx()
            ax2_2.plot(df['Data_Interna'], df['Tasso_Resi'].rolling(4).mean(), color='#e74c3c', linewidth=2)
            st.pyplot(fig2)

        with tabs[5]:
            st.info("**Cosa fa:** Visualizza il database normalizzato utilizzato per i calcoli.  \n**Logica:** È il risultato del processo di pulizia che trasforma i dati grezzi in numeri pronti per l'analisi statistica e il Machine Learning.")
            st.caption("Il database grezzo importato.")
            st.subheader("🗂️ Database Storico")
            display_cols = [col_date, 'Periodo', 'Total sales', col_google, col_g_val, col_g_cpc, col_g_imps, 
                            col_meta, col_m_val, col_m_cpc, col_m_cpm, col_m_freq, 'CoS', 'Profitto_Operativo']
            valid_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[valid_cols].iloc[::-1].style.format({'CoS': '{:.1f}%', 'Profitto_Operativo': '€ {:,.0f}'}, precision=2))

        # --- 8. TAB AI AVANZATA ---
        with tabs[6]:
            st.info("**Cosa fa:** Analisi automatica che assegna un punteggio di salute mensile al business.  \n**Logica:** Incrocia profitto, retention e performance dei canali utilizzando benchmark storici per generare alert e suggerimenti strategici.")
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
            
            # Calcolo Metriche Avanzate Mensili
            ai_df['MER'] = ai_df['Fatturato_Netto'] / ai_df['Spesa_Ads_Totale'].replace(0, np.nan)
            ai_df['AOV'] = ai_df['Fatturato_Netto'] / ai_df['Orders'].replace(0, np.nan)
            ai_df['CPA'] = ai_df['Spesa_Ads_Totale'] / ai_df['Orders'].replace(0, np.nan)
            
            # --- LOGICA LTV INTEGRATA ---
            ret_col = col_ret_rate if col_ret_rate in ai_df.columns else None
            if ret_col:
                ai_df['RET_RATE'] = pd.to_numeric(ai_df[ret_col], errors='coerce').fillna(0) / 100
                # LTV Proxy = AOV / (1 - Retention) - Cappato al 90% per evitare infiniti
                ai_df['LTV'] = ai_df['AOV'] / (1 - ai_df['RET_RATE'].clip(upper=0.90))
                ai_df['LTV_CPA'] = ai_df['LTV'] / ai_df['CPA'].replace(0, np.nan)
            
            ai_df['Discount_Rate'] = (ai_df[col_discounts].abs() / ai_df['Gross sales'].replace(0, np.nan)) * 100
            ai_df['ROAS_Google'] = ai_df[col_g_val] / ai_df[col_google].replace(0, np.nan)
            ai_df['ROAS_Meta'] = ai_df[col_m_val] / ai_df[col_meta].replace(0, np.nan)
            
            avg_sales = ai_df['Fatturato_Netto'].mean()
            ai_df['Seasonality'] = ai_df['Fatturato_Netto'] / avg_sales
            
            # Benchmark storici per lo scoring
            bench_mer = ai_df['MER'].mean()
            bench_ltv_cpa = ai_df['LTV_CPA'].mean() if 'LTV_CPA' in ai_df.columns else 0
            
            for m in ai_df.index[:12]:
                row = ai_df.loc[m]
                m_str = str(m)
                
                score = 50
                tags, alerts = [], []
                
                # 1. Salute Finanziaria Immediata (MER)
                if row['MER'] >= be_roas_val: 
                    score += 15
                    tags.append(f"MER Pro: {row['MER']:.2f}")
                else: 
                    score -= 15
                    alerts.append(f"Margine Basso (MER {row['MER']:.2f})")
                
                # 2. Salute Strategica a Lungo Termine (LTV)
                if 'LTV_CPA' in ai_df.columns:
                    if row['LTV_CPA'] > 3.5: 
                        score += 20
                        tags.append(f"High LTV: {row['LTV_CPA']:.1f}x")
                    elif row['LTV_CPA'] < 2.0:
                        score -= 10
                        alerts.append("Bassa Retention/Valore Cliente")
                
                # 3. Canale Dominante (Efficienza Diretta)
                if row['ROAS_Google'] > row['ROAS_Meta']: 
                    tags.append("Top: Google (Intent)")
                else: 
                    tags.append("Top: Meta (Visual)")
                
                # 4. Pressione Sconti
                if row['Discount_Rate'] > 15:
                    score -= 10
                    alerts.append(f"Eccesso Sconti ({row['Discount_Rate']:.1f}%)")
                
                seas_txt = "Standard"
                if row['Seasonality'] > 1.2: seas_txt = "Alta Stagionalità 🔥"
                elif row['Seasonality'] < 0.8: seas_txt = "Bassa Stagionalità ❄️"
                
                color_class = "ai-score-high" if score >= 70 else "ai-score-med" if score >= 50 else "ai-score-low"
                
                with st.container():
                    st.markdown(f"""
                    <div class="ai-box {color_class}">
                        <div class="ai-title" style="display:flex; justify-content:space-between;">
                            <span>📅 {m_str} | <b>Health Score: {min(100, score)}/100</b></span>
                            <span style="font-size:0.8em; color:#666;">{seas_txt}</span>
                        </div>
                        <div style="margin:8px 0;">
                            {' '.join([f'<span class="ai-tag">{t}</span>' for t in tags])}
                        </div>
                        <p style="margin:0; font-size:0.95em;">
                            Fatturato: <b>€ {row['Fatturato_Netto']:,.0f}</b>. 
                            CPA: <b>€ {row['CPA']:.2f}</b> | 
                            AOV: <b>€ {row['AOV']:.2f}</b> |
                            LTV Est.: <b>€ {row['LTV'] if 'LTV' in row else 0:,.0f}</b>
                        </p>
                        {''.join([f'<div class="ai-alert">⚠️ {a}</div>' for a in alerts])}
                    </div>
                    """, unsafe_allow_html=True)

        with tabs[7]:
            st.info("**Cosa fa:** Elabora una roadmap di investimento basata sulla tua efficienza reale e sui segnali tecnici dei canali.  \n**Logica:** Analizza il MER, le finestre di attribuzione e le metriche di saturazione (CPM, Frequency, CPC) per definire una scalabilità sostenibile.")
            st.header("🎯 Analisi Strategica di Scalabilità")
            
            # --- LOGICA DI CALCOLO STRATEGICO AVANZATA ---
            last_4 = df.tail(4)
            prev_4 = df.iloc[-8:-4] if len(df) >= 8 else last_4
            
            g_spend_last = last_4[col_google].sum()
            m_spend_last = last_4[col_meta].sum()
            g_roas_last = last_4[col_g_val].sum() / g_spend_last if g_spend_last > 0 else 0
            m_roas_last = last_4[col_m_val].sum() / m_spend_last if m_spend_last > 0 else 0
            
            # Segnali Tecnici (Ultimi 4w vs Precedenti 4w)
            m_cpm_last = last_4[col_m_cpm].mean()
            m_cpm_prev = prev_4[col_m_cpm].mean()
            m_freq_last = last_4[col_m_freq].mean()
            g_cpc_last = last_4[col_g_cpc].mean()
            g_cpc_prev = prev_4[col_g_cpc].mean()
            
            m_cpm_delta = (m_cpm_last - m_cpm_prev) / m_cpm_prev if m_cpm_prev > 0 else 0
            g_cpc_delta = (g_cpc_last - g_cpc_prev) / g_cpc_prev if g_cpc_prev > 0 else 0
            
            # Determinazione Canale Dominante e Bias di Attribuzione
            best_channel = "Google Ads" if g_roas_last > m_roas_last else "Meta Ads"
            performance_gap = abs(g_roas_last - m_roas_last) / max(g_roas_last, m_roas_last, 0.01)
            
            # Calcolo Budget Sostenibile basato su Forecasting
            next_month_sales = df_prev.head(4)['Fatturato Previsto'].sum()
            max_safe_budget = next_month_sales / be_roas_val
            
            # --- RENDERING UI PARTE 1: SEGNALI TECNICI ---
            st.subheader("📡 Segnali di Mercato & Saturazione")
            col_sig1, col_sig2, col_sig3 = st.columns(3)
            
            with col_sig1:
                st.metric("CPM Meta", f"€ {m_cpm_last:.2f}", delta=f"{m_cpm_delta:+.1%}", delta_color="inverse")
                if m_cpm_delta > 0.15: st.warning("⚠️ **Pressione Aste:** I costi pubblicitari su Meta sono in netto aumento.")
            
            with col_sig2:
                st.metric("Frequency Meta (4w)", f"{m_freq_last:.2f}")
                if m_freq_last > 1.4: st.error("🚨 **Saturazione:** Frequenza alta su Meta. Hai bisogno di nuovi creativi per scalare.")
                else: st.success("✅ **Audience Fresca:** Hai spazio per aumentare il budget su Meta.")
                
            with col_sig3:
                st.metric("CPC Google Ads", f"€ {g_cpc_last:.2f}", delta=f"{g_cpc_delta:+.1%}", delta_color="inverse")

            # --- PARTE 2: SUGGERIMENTO STRATEGICO ---
            st.divider()
            
            # Logica Proposta Strategica
            if mer_attuale > be_roas_val * 1.3:
                strat_title = "🚀 SCALABILITÀ AGGRESSIVA"
                strat_desc = "Le tue Unit Economics sono eccellenti. L'obiettivo è catturare quota di mercato dominando la scoperta (Meta) e scalando l'intenzione d'acquisto diretta (Google Shopping/PMax)."
                rec_scale, rec_trend, rec_sat = 1.25, growth_rate + 0.05, 0.92
            elif mer_attuale >= be_roas_val:
                strat_title = "⚖️ CRESCITA EQUILIBRATA"
                strat_desc = "Sei in profitto. Focus sull'ottimizzazione del mix tra canali per massimizzare il profitto netto senza sprecare budget."
                rec_scale, rec_trend, rec_sat = 1.10, growth_rate, 0.85
            else:
                strat_title = "🛡️ DIFESA E OTTIMIZZAZIONE"
                strat_desc = "Stai operando vicino o sotto il break-even. Priorità al recupero dell'efficienza prima di qualsiasi aumento di spesa."
                rec_scale, rec_trend, rec_sat = 0.85, growth_rate - 0.05, 0.80

            st.markdown(f"### 🤖 Suggerimento AI: **{strat_title}**")
            st.write(strat_desc)

            col_rec1, col_rec2 = st.columns([2, 1])
            with col_rec1:
                st.info(f"""
                **Analisi Multimetrica:**
                * **Attribuzione:** {best_channel} ha un ROAS più alto, ma ricorda che Meta genera la domanda che Google spesso converte in Shopping. Non tagliare Meta se alimenta il volume di traffico "fresco".
                * **Finestre di conversione:** Google (30gg) intercetta l'intenzione d'acquisto nata spesso da impulsi visivi su Meta (7gg). Valuta il MER complessivo.
                * **Efficienza:** Il tuo cuscinetto di sicurezza è del **{((mer_attuale/be_roas_val)-1):+.1%}**.
                """)
            
            with col_rec2:
                st.metric("Target Budget Prossime 4w", format_euro(max_safe_budget * (rec_scale/1.1), 0))
                st.caption("Budget totale stimato per mantenere il profitto operativo.")

            st.divider()

            col_p1, col_p2 = st.columns(2)
            
            # --- LOGICA DI ALLOCAZIONE INTEGRATA (ROAS + ML IMPORTANCE) ---
            # Estraiamo l'importanza delle feature dal modello Random Forest
            importances = ml_model.feature_importances_
            feature_names = ['Week_Sin', 'Week_Cos', col_google, col_meta, 'Lag_Sales_1', 'Lag_Sales_4'] # Da run_ml_forecast
            for drv in businesses_found: feature_names.append(drv)
            
            fi_dict = dict(zip(feature_names, importances))
            fi_g = fi_dict.get(col_google, 0)
            fi_m = fi_dict.get(col_meta, 0)
            
            # Determinazione del "Vero Motore" del Business (ML vs ROAS)
            ml_engine = "Google Ads" if fi_g > fi_m else "Meta Ads"
            roas_efficiency = "Google Ads" if g_roas_last > m_roas_last else "Meta Ads"
            
            # Spiegazione approfondita delle tre voci tramite Tabs informative
            st.divider()
            opt_tabs = st.tabs(["📍 Setup Simulatore", "📂 Allocazione Consigliata", "⚖️ Nota sull'Attribuzione", "💶 EBITDA & Profitto"])

            with opt_tabs[0]:
                col_s1, col_s2 = st.columns([1, 2])
                with col_s1:
                    st.code(f"""
Trend Futuro:  {rec_trend:+.1%}
Budget Scale:  {rec_scale:.2f}x
Saturazione:   {rec_sat:.2f}
                    """)
                    st.caption(f"Basato su MER attuale di **{mer_attuale:.2f}** vs Break-Even di **{be_roas_val:.2f}**.")
                with col_s2:
                    st.markdown(f"""
                    **Logica dei Suggerimenti:**  
                    Questi valori derivano dal tuo **Cuscinetto di Sicurezza** e non solo dal ROAS:
                    
                    *   **Trend Futuro ({rec_trend:+.1%}):** Unisce la crescita YoY (+{growth_rate:1%}) con la capacità di assorbimento del mercato rilevata dall'IA.
                    *   **Budget Scale ({rec_scale:.2f}x):** Poiché il tuo MER è {f'superiore del 30% al BE' if mer_attuale > be_roas_val * 1.3 else 'sopra il BE' if mer_attuale >= be_roas_val else 'sotto il BE'}, l'IA consiglia di {f'investire con un +{(rec_scale-1)*100:.0f}%' if rec_scale > 1 else 'ridurre la spesa'}. 
                    *   **Saturazione ({rec_sat:.2f}):** Indica la reattività delle tue campagne. Più è alto, più l'IA crede che tu possa scalare senza distruggere il ROAS.
                    """)

            with opt_tabs[1]:
                col_a1, col_a2 = st.columns([1, 2])
                
                # Split pesato con Bias Attribuzione: Diamo più peso al canale con più "Importanza" (Machine Learning)
                # se ROAS e ML concordano, spingiamo ignorando. Se discordano, bilanciamo.
                if ml_engine == roas_efficiency:
                    split_best = 0.65 # Convergenza totale
                    best_channel = roas_efficiency
                else:
                    split_best = 0.55 # Discordano: Prudenza, non togliere ossigeno al motore (ML)
                    best_channel = ml_engine # Diamo precedenza a chi muove davvero il business secondo l'ML
                
                val_best = max_safe_budget * rec_scale * split_best
                val_other = (max_safe_budget * rec_scale) - val_best
                
                with col_a1:
                    st.success(f"**{best_channel}:**  \n€ {val_best:,.0f}")
                    st.info(f"**Altro Canale:**  \n€ {val_other:,.0f}")
                
                with col_a2:
                    st.markdown(f"""
                    **Perché questa divisione?**  
                    Abbiamo incrociato l'Efficienza (ROAS) con l'Impatto (Machine Learning):
                    *   L'IA ha rilevato che **{ml_engine}** è il canale che **"muove maggiormente il fatturato"** (Feature Importance).
                    *   Sebbene l'altro canale possa sembrare più efficiente a parità di spesa, togliere budget a {ml_engine} potrebbe causare un crollo delle vendite organiche o di ricerca.
                    *   Consigliamo di mantenere il **{split_best*100:.0f}%** del budget su **{best_channel}** per stabilità.
                    """)

            with opt_tabs[2]:
                st.markdown(f"""
                ### ⚠️ Il Bias delle Finestre di Conversione
                È fondamentale distinguere tra la **Funzione** dei canali:
                
                1.  **Meta Ads (Market Generation):** Intercetta la domanda latente. È il tuo **Motore di Scoperta**. Senza Meta, il volume di ricerche specifiche per i tuoi prodotti su Google potrebbe calare nel lungo termine.
                2.  **Google Ads (Direct Intent):** Grazie a Shopping e PMax, cattura l'utente nel momento esatto del bisogno. È il tuo **Motore di Conversione**.
                
                **Analisi AntiGravity:**  
                Dall'analisi del Random Forest, risulta che la variabile **{col_google if fi_g > fi_m else col_meta}** correla meglio con le vendite finali. Anche se il ROAS sembra inferiore, non ridurlo: è il canale che garantisce la massa critica di ordini.
                """)

            with opt_tabs[3]:
                ebitda_val = (next_month_sales * rec_scale) - (max_safe_budget * rec_scale)
                col_e1, col_e2 = st.columns([1, 2])
                with col_e1:
                    st.metric("EBITDA Stimato", f"€ {ebitda_val:,.0f}")
                with col_e2:
                    st.markdown(f"""
                    **Il tuo guadagno reale (stimato)**  
                    L'EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) qui rappresenta il tuo **Margine Operativo Lordo** previsto per il prossimo mese.
                    
                    **Formula:** `Fatturato Previsto - Spesa Ads Prevista`
                    
                    *Nota: Questo valore non tiene conto di costi fissi, tasse o costi di prodotto (COGS) non inseriti nella sidebar.*
                    """)
      
     
        with tabs[8]:
            st.info("**Cosa fa:** Confronto Anno su Anno (YoY) per valutare la crescita strutturale del progetto.  \n**Logica:** Analizza metriche chiave come CPA, AOV e Retention Rate comparando periodi omogenei tra anni diversi.")
            st.header("🏥 Stato di Salute del Progetto (YoY)")
            
            years_avail = sorted(df['Year'].unique(), reverse=True)
            if len(years_avail) < 2:
                st.warning("Dati insufficienti per un confronto YoY.")
            else:
                with st.expander("⚙️ Opzioni di Confronto", expanded=False):
                    col_f1, col_f2 = st.columns(2)
                    year_target = col_f1.selectbox("Anno", years_avail, index=0, key="year_t_9")
                    year_comp = col_f2.selectbox("Confronta con", years_avail, index=min(1, len(years_avail)-1), key="year_c_9")

                # --- LOGICA DI ALLINEAMENTO SETTIMANALE (INTERSEZIONE) ---
                weeks_target = set(df[df['Year'] == year_target]['Week'].unique())
                weeks_comp = set(df[df['Year'] == year_comp]['Week'].unique())
                
                # Intersezione: prendiamo solo le settimane presenti in ENTRAMBI gli anni
                common_weeks = sorted(list(weeks_target.intersection(weeks_comp)))
                
                def get_y_metrics(year, week_filter=None):
                    d = df[df['Year'] == year].copy()
                    if week_filter:
                        d = d[d['Week'].isin(week_filter)]
                        
                    sales = d['Fatturato_Netto'].sum()
                    spend = d['Spesa_Ads_Totale'].sum()
                    orders = d['Orders'].sum() if 'Orders' in d.columns else (sales / be_aov)
                    
                    # Pulizia Returning Customer Rate
                    if 'Returning customer rate' in d.columns:
                        ret_values = d['Returning customer rate'].astype(str).str.replace('%', '').str.strip()
                        ret_mean = pd.to_numeric(ret_values, errors='coerce').mean()
                    else:
                        ret_mean = 0

                    aov = sales / orders if orders > 0 else 0
                    cpa = spend / orders if orders > 0 else 0
                    ret_rate = ret_mean / 100
                    # Formula LTV Proxy: AOV / (1 - Retention Rate)
                    ltv = aov / (1 - ret_rate) if ret_rate < 0.99 else aov
                    ltv_cpa_ratio = ltv / cpa if cpa > 0 else 0

                    return {
                        'sales': sales,
                        'mer': sales / spend if spend > 0 else 0,
                        'cpa': cpa,
                        'cpc_g': d[col_g_cpc].mean() if col_g_cpc in d.columns else 0,
                        'cpc_m': d[col_m_cpc].mean() if col_m_cpc in d.columns else 0,
                        'cpm_m': d[col_m_cpm].mean() if col_m_cpm in d.columns else 0,
                        'freq_m': d[col_m_freq].mean() if col_m_freq in d.columns else 0,
                        'imps_g': d[col_g_imps].sum() if col_g_imps in d.columns else 0,
                        'aov': aov,
                        'returns': (d['Returns'].abs().sum() / sales * 100) if sales > 0 else 0,
                        'ret': ret_mean,
                        'ltv': ltv,
                        'ltv_cpa': ltv_cpa_ratio,
                        'weeks_count': d['Week'].nunique(),
                        'week_list': sorted(d['Week'].unique().tolist())
                    }

                # Applichiamo l'intersezione per un confronto 1:1 perfetto
                mt = get_y_metrics(year_target, week_filter=common_weeks)
                mc = get_y_metrics(year_comp, week_filter=common_weeks)

                st.subheader(f"🏥 Statistiche Vitali: {year_target} vs {year_comp}")
                
                # Notifica di Allineamento Tecnico
                with st.expander("🔬 Nota Tecnica: Metodologia di Confronto"):
                    st.markdown(fr"""
                    **Allineamento Temporale:**
                    *   Il sistema ha individuato le settimane presenti in **entrambi** gli anni ({len(common_weeks)} settimane totali).
                    *   Settimane incluse: `{common_weeks}`.
                    
                    **Calcolo Lifetime Value (LTV):**
                    *   Utilizziamo la formula predittiva: $LTV = \frac{{AOV}}{{1 - Retention Rate}}$
                    *   Questo valore indica quanto fatturato genera mediamente un cliente nel tempo, considerando la sua propensione al riacquisto attuale.
                    """)

                # Caption sui periodi
                col_cap1, col_cap2 = st.columns(2)
                with col_cap1:
                    st.caption(f"📅 **{year_target}**: {mt['weeks_count']} settimane selezionate")
                with col_cap2:
                    st.caption(f"📅 **{year_comp}**: {mc['weeks_count']} settimane selezionate")
                
                st.info(f"💡 **Confronto 1:1 Attivo:** Analisi basata sulle {len(common_weeks)} settimane comuni rilevate.")
                st.divider()

                # --- RIGA 1: FINANCIALS ---
                st.subheader("💰 Performance Finanziaria (PTD)")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    delta_sales = ((mt['sales'] - mc['sales']) / mc['sales'] * 100) if mc['sales'] > 0 else 0
                    st.metric("Fatturato", format_euro(mt['sales'], 0), format_num_eu(delta_sales, 1, delta=True) + "%", help="Somma totale del Fatturato Netto nel periodo selezionato. Rappresenta il volume d'affari lordo al netto di IVA.")
                with c2:
                    st.metric("ROAS (MER)", format_num_eu(mt['mer'], 2), format_num_eu(mt['mer'] - mc['mer'], 2, delta=True), help="Marketing Efficiency Ratio. Calcolato come Fatturato Totale / Spesa Ads Totale. Indica l'efficienza globale di ogni euro investito in pubblicità.")
                with c3:
                    delta_cpa = ((mt['cpa'] - mc['cpa']) / mc['cpa'] * 100) if mc['cpa'] > 0 else 0
                    st.metric("CPA Medio", format_euro(mt['cpa'], 2), format_num_eu(delta_cpa, 1, delta=True) + "%", delta_color="inverse", help="Costo Per Acquisizione. Calcolato come Spesa Ads Totale / Numero Ordini. Indica quanto paghi in media per ottenere una vendita.")
                with c4:
                    delta_aov = ((mt['aov'] - mc['aov']) / mc['aov'] * 100) if mc['aov'] > 0 else 0
                    st.metric("Carrello Medio (AOV)", format_euro(mt['aov'], 2), format_num_eu(delta_aov, 1, delta=True) + "%", help="Average Order Value. Fatturato Totale / Numero Ordini. Indica la spesa media di un cliente per singolo acquisto.")

                st.divider()

                # --- RIGA 2: TECHNICAL HEALTH ---
                st.subheader("📡 Analisi Tecnica Canali (Deep Dive)")
                ct1, ct2, ct3, ct4 = st.columns(4)
                with ct1:
                    delta_cpm = ((mt['cpm_m'] - mc['cpm_m']) / mc['cpm_m'] * 100) if mc['cpm_m'] > 0 else 0
                    st.metric("CPM Meta", format_euro(mt['cpm_m'], 2), format_num_eu(delta_cpm, 1, delta=True) + "%", delta_color="inverse", help="Cost Per Mille. Costo medio pagato su Meta per 1.000 visualizzazioni dell'annuncio.")
                with ct2:
                    delta_freq = mt['freq_m'] - mc['freq_m']
                    st.metric("Frequency Meta", format_num_eu(mt['freq_m'], 2), format_num_eu(delta_freq, 2, delta=True), help="Frequenza media. Indica quante volte mediamente una persona ha visto i tuoi annunci su Meta. Un valore troppo alto (sopra 3-4) può indicare saturazione del pubblico.")
                with ct3:
                    delta_cpc_g = ((mt['cpc_g'] - mc['cpc_g']) / mc['cpc_g'] * 100) if mc['cpc_g'] > 0 else 0
                    st.metric("CPC Google", format_euro(mt['cpc_g'], 2), format_num_eu(delta_cpc_g, 1, delta=True) + "%", delta_color="inverse", help="Cost Per Click. Costo medio pagato per ogni click sui tuoi annunci Google.")
                with ct4:
                    delta_imps = ((mt['imps_g'] - mc['imps_g']) / mc['imps_g'] * 100) if mc['imps_g'] > 0 else 0
                    st.metric("Impression Google", format_num_eu(mt['imps_g'], 0), format_num_eu(delta_imps, 1, delta=True) + "%", help="Somma totale delle visualizzazioni ottenute sui posizionamenti Google Ads.")

                st.info(f"""
                **Diagnostica Rapida:**
                * {'🔴 **Meta Saturazione:** La frequenza è aumentata!' if mt['freq_m'] > mc['freq_m'] else '🟢 **Meta Audience:** La frequenza è stabile o in calo.'}
                * {'🔴 **Costi in ascesa:** Il CPM di Meta è aumentato del ' + format_num_eu(delta_cpm, 1) + '%' if delta_cpm > 5 else '🟢 **Efficienza Costi:** I costi delle aste sono sotto controllo.'}
                * {'🔵 **Visibilità Google:** Hai ottenuto il ' + format_num_eu(delta_imps, 1, delta=True) + '%' + ' di impression rispetto al passato.'}
                """)
                
                st.divider()

                # --- RIGA 3: BRAND SOLIDITY & LTV ---
                st.subheader("🛠️ Solidità del Brand & Lifetime Value")
                st.write("Formula: $LTV = AOV / (1 - Retention Rate)$")
                c_op1, c_op2, c_op3, c_op4 = st.columns(4)
                with c_op1:
                    delta_ret = mt['ret'] - mc['ret']
                    st.metric("Retention Rate", f"{format_num_eu(mt['ret'], 2)}%", format_num_eu(delta_ret, 2, delta=True) + "% (pt)", help="Percentuale di clienti che tornano ad acquistare. Estratto direttamente dalla colonna 'Returning customer rate' del tuo report Shopify.")
                with c_op2:
                    delta_res = mt['returns'] - mc['returns']
                    st.metric("Incidenza Resi", f"{format_num_eu(mt['returns'], 1)}%", format_num_eu(delta_res, 1, delta=True) + "% (pt)", delta_color="inverse", help="Rapporto tra il valore dei resi/rimborsi e il fatturato lordo. Indica la qualità della vendita e della logistica.")
                with c_op3:
                    delta_ltv = ((mt['ltv'] - mc['ltv']) / mc['ltv'] * 100) if mc['ltv'] > 0 else 0
                    st.metric("LTV Stimato (12m)", format_euro(mt['ltv'], 2), format_num_eu(delta_ltv, 1, delta=True) + "%", help="Lifetime Value Stimato. Calcolato matematicamente come AOV / (1 - Retention Rate). Indica quanto vale mediamente un cliente in un anno.")
                with c_op4:
                    delta_ratio = mt['ltv_cpa'] - mc['ltv_cpa']
                    st.metric("LTV / CPA Ratio", format_num_eu(mt['ltv_cpa'], 2), format_num_eu(delta_ratio, 2, delta=True), help="Indice di sostenibilità. Rapporto tra quanto il cliente vale nel tempo (LTV) e quanto costa acquisirlo (CPA). Un valore sopra 3.0 indica eccellente scalabilità.")

                st.info(f"""
                **Analisi Qualitativa & Valore Cliente:**
                * {'🟢 **LTV in crescita:** Ogni nuovo cliente acquisito oggi vale più che in passato.' if mt['ltv'] > mc['ltv'] else '🟡 **LTV Statico:** Il valore nel tempo del cliente non sta crescendo. Lavora su bundle e cross-selling.'}
                * {'  **Efficienza Scalabilità:**' if mt['ltv_cpa'] > 3 else '⚖️ **Efficienza Moderata:**'} Il tuo rapporto LTV/CPA è di **{mt['ltv_cpa']:.2f}**. {'Puoi permetterti di alzare il CAC per dominare il mercato.' if mt['ltv_cpa'] > 3 else 'Monitora attentamente i costi di acquisizione.'}
                """)

        with tabs[9]:
            st.header("🌍 Market Intelligence: Geopolitical Risk Watch")
            st.info("""
            **Sincronizzazione Eventi:** Questa sezione analizza come i fattori macroeconomici e geopolitici influenzano il mercato. 
            I dati sono recuperati in tempo reale dai mercati globali.
            """)
            
            risk_data = fetch_global_signals()
            if risk_data:
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                
                # Metrics
                if 'Oil' in risk_data:
                    col_r1.metric("Brent Oil", format_euro(risk_data['Oil']['current'], 2), f"{risk_data['Oil']['trend_7d']:+.1%}")
                if 'VIX' in risk_data:
                    col_r2.metric("Fear Index (VIX)", format_num_eu(risk_data['VIX']['current'], 2), f"{risk_data['VIX']['trend_7d']:+.1%}", delta_color="inverse")
                if 'Gold' in risk_data:
                    col_r3.metric("Gold", format_euro(risk_data['Gold']['current'], 1), f"{risk_data['Gold']['trend_7d']:+.1%}")
                if 'DXY' in risk_data:
                    col_r4.metric("Dollar Index", format_num_eu(risk_data['DXY']['current'], 2), f"{risk_data['DXY']['trend_7d']:+.1%}")
                
                st.divider()
                
                # Charts
                st.subheader("📈 Trend Mercati negli ultimi 30 giorni")
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.write("**Petrolio Brent**")
                    st.line_chart(risk_data['Oil']['history'] if 'Oil' in risk_data else None)
                    st.caption("📦 **Costi Logistica:** L'aumento del petrolio è un indicatore diretto di rincari nei trasporti e nelle tariffe di spedizione per il tuo e-commerce.")
                with chart_col2:
                    st.write("**VIX Index**")
                    st.line_chart(risk_data['VIX']['history'] if 'VIX' in risk_data else None)
                    st.caption("🧠 **Sentiment Consumatori:** Un VIX alto indica incertezza e paura; in questi periodi il tasso di conversione tende a calare a causa della minor fiducia degli acquirenti.")
                
                # AI Geopolitical Commentary
                st.subheader("🤖 Analisi AI del Rischio Geopolitico")
                
                recent_oil = risk_data['Oil']['trend_7d'] if 'Oil' in risk_data else 0
                recent_vix = risk_data['VIX']['trend_7d'] if 'VIX' in risk_data else 0
                
                if recent_oil > 0.08:
                    st.error(f"**ALERT GUERRA/LOGISTICA:** Il forte aumento del petrolio (+{recent_oil:.1%}) suggerisce possibili blocchi alle rotte commerciali o rincari dei trasporti. Proteggi i margini aumentando il prezzo medio o ottimizzando la logistica.")
                elif recent_vix > 0.15:
                    st.warning(f"**ALERT SENTIMENT:** L'indice della paura è balzato del {recent_vix:.1%}. Gli utenti tendono a ritardare gli acquisti non necessari. Considera di spingere prodotti 'essenziali' o offerte di valore.")
                else:
                    st.success("✅ **Stabilità Macro:** Non si rilevano scossoni geopolitici critici nelle ultime 72 ore. Scenario favorevole allo scale-up del budget.")
            
            st.divider()
            st.subheader("🌦️ Weather Intelligence (Dati Storici)")
            st.info("Sposta lo slider per vedere l'evoluzione del meteo sull'Italia in tempo reale.")

            # 1. Recupero date totali dal CSV
            full_start = df['Data_Interna'].min()
            full_end = df['Data_Interna'].max()

            # 2. Logica Weather Time-Series (Open-Meteo)
            @st.cache_data(ttl=86400) # Cache 24h visto che sono dati storici fissi
            def fetch_weather_timeseries(start_date, end_date):
                cities = {
                    'Milano': (45.46, 9.19), 'Torino': (45.07, 7.68), 'Venezia': (45.44, 12.31),
                    'Bologna': (44.49, 11.34), 'Genova': (44.40, 8.94), 'Firenze': (43.76, 11.25),
                    'Ancona': (43.61, 13.51), 'Perugia': (43.11, 12.38), 'Roma': (41.90, 12.49),
                    'Pescara': (42.46, 14.21), 'Napoli': (40.85, 14.26), 'Bari': (41.11, 16.87),
                    'Potenza': (40.64, 15.80), 'Catanzaro': (38.90, 16.58), 'Palermo': (38.11, 13.36),
                    'Catania': (37.50, 15.08), 'Cagliari': (39.22, 9.12), 'Sassari': (40.72, 8.56),
                    'Trento': (46.06, 11.12), 'Trieste': (45.64, 13.77)
                }
                
                lats = ",".join([str(c[0]) for c in cities.values()])
                lons = ",".join([str(c[1]) for c in cities.values()])
                s_str = start_date.strftime('%Y-%m-%d')
                e_str = end_date.strftime('%Y-%m-%d')
                
                # Limitiamo il range per non sovraccaricare l'API (max 6 mesi per test)
                is_recent = (datetime.now() - end_date).days < 5
                base_url = "https://api.open-meteo.com/v1/forecast" if is_recent else "https://archive-api.open-meteo.com/v1/archive"
                url = f"{base_url}?latitude={lats}&longitude={lons}&start_date={s_str}&end_date={e_str}&daily=temperature_2m_max,precipitation_sum&timezone=Europe%2FBerlin"
                
                all_days_data = []
                try:
                    resp = requests.get(url, timeout=16).json()
                    if not isinstance(resp, list): resp = [resp]
                    
                    for i, (city, coords) in enumerate(cities.items()):
                        data = resp[i]
                        if 'daily' in data:
                            for day_idx in range(len(data['daily']['time'])):
                                t = data['daily']['temperature_2m_max'][day_idx]
                                r = data['daily']['precipitation_sum'][day_idx]
                                d = data['daily']['time'][day_idx]
                                
                                # Heatmap color
                                if t < 5: c = '#0000FF'
                                elif t < 12: c = '#3498db'
                                elif t < 20: c = '#f1c40f'
                                elif t < 28: c = '#e67e22'
                                else: c = '#e74c3c'
                                
                                all_days_data.append({
                                    'Date': d, 'City': city, 'lat': coords[0], 'lon': coords[1],
                                    'temp': t, 'rain': r, 'color': c, 'size': 6000 + (r * 400)
                                })
                except Exception as ex:
                    st.error(f"Errore API: {ex}")
                return pd.DataFrame(all_days_data)

            # --- ESECUZIONE PLAYER ---
            st.write("---")
            weather_full_df = fetch_weather_timeseries(full_start, full_end)

            if not weather_full_df.empty:
                # Slider Temporale
                unique_days = sorted(weather_full_df['Date'].unique())
                sel_date_str = st.select_slider("📅 Seleziona il giorno da analizzare", options=unique_days, value=unique_days[-1])
                
                # Filtraggio dati per il giorno scelto
                day_weather = weather_full_df[weather_full_df['Date'] == sel_date_str]
                
                # Visualizzazione Mappa Dinamica
                st.subheader(f"🌦️ Stato Meteo Italia: {datetime.strptime(sel_date_str, '%Y-%m-%d').strftime('%d %B %Y')}")
                st.map(day_weather, color='color', size='size')
                
                col_w1, col_w2, col_w3 = st.columns(3)
                col_w1.metric("Temperatura Media", f"{day_weather['temp'].mean():.1f}°C")
                col_w2.metric("Precipitazioni Max", f"{day_weather['rain'].max():.1f} mm")
                col_w3.metric("Città più Calda", day_weather.loc[day_weather['temp'].idxmax()]['City'])

                # Correlazione con i dati del business
                target_dt = pd.to_datetime(sel_date_str)
                biz_day = df[df['Data_Interna'].dt.date == target_dt.date()]
                
                if not biz_day.empty:
                    st.write("---")
                    st.subheader("📊 Performance Business del Giorno")
                    cb1, cb2 = st.columns(2)
                    day_sales = biz_day['Fatturato_Netto'].values[0]
                    day_mer = biz_day['Fatturato_Netto'].values[0] / biz_day['Spesa_Ads_Totale'].values[0] if biz_day['Spesa_Ads_Totale'].values[0] > 0 else 0
                    
                    cb1.metric("Fatturato Giorno", format_euro(day_sales, 0))
                    cb2.metric("MER Reale Giorno", format_num_eu(day_mer, 2))
                    
                    st.info(f"💡 **Insight AI:** In questa giornata di {day_weather.loc[day_weather['temp'].idxmin()]['status'] if 'status' in day_weather else 'clima variabile'}, il business ha generato un'efficienza di {format_num_eu(day_mer, 2)}. {'Ottima correlazione con il maltempo!' if day_weather['rain'].sum() > 30 else 'Performance guidata da fattori interni (Ads/Promo).'}")
            else:
                st.warning("Caricamento dati meteo in corso o fallito. Verifica la connessione.")

            st.divider()
            
            # --- SEZIONE 3: STRATEGIC CORRELATION LEDGER ---
            st.subheader("📊 Strategic Correlation Ledger (LFL Monthly)")
            st.info("Questa tabella incrocia i tuoi KPI con i 'fatti del mondo' per identificare le cause esterne di successo o fallimento.")
            
            # Prepariamo i dati mensili degli ultimi 12 mesi
            df_last_12 = df[df['Data_Interna'] > (full_end - timedelta(days=365))].copy()
            df_last_12['Month_Year'] = df_last_12['Data_Interna'].dt.strftime('%Y-%m')
            
            monthly_biz = df_last_12.groupby('Month_Year').agg({
                'Fatturato_Netto': 'sum',
                'Spesa_Ads_Totale': 'sum',
                'Data_Interna': 'min'
            }).reset_index()
            
            monthly_biz['MER'] = monthly_biz['Fatturato_Netto'] / monthly_biz['Spesa_Ads_Totale']
            
            # Database Eventi Sincronizzato (Espanso)
            events_template = [
                {"month": 1, "day_start": 3, "day_end": 31, "title": "🛍️ Saldi Invernali", "type": "Commercial", "desc": "Sconti stagionali massicci."},
                {"month": 2, "day_start": 5, "day_end": 12, "title": "🎶 Festival di Sanremo", "type": "Cultural", "desc": "Picco di attenzione media su TV/Social."},
                {"month": 2, "day_start": 10, "day_end": 14, "title": "❤️ San Valentino", "type": "Commercial", "desc": "Focus su gifting."},
                {"month": 4, "day_start": 1, "day_end": 15, "title": "🐣 Pasqua", "type": "Cultural", "desc": "Festività nazionali."},
                {"month": 5, "day_start": 1, "day_end": 12, "title": "💐 Festa della Mamma", "type": "Commercial", "desc": "Lifestyle gifting."},
                {"month": 7, "day_start": 1, "day_end": 31, "title": "☀️ Saldi Estivi", "type": "Commercial", "desc": "Svuotamento inventory."},
                {"month": 11, "day_start": 20, "day_end": 30, "title": "🖤 Black Friday", "type": "Commercial", "desc": "Picco annuale Ads."},
                {"month": 12, "day_start": 1, "day_end": 24, "title": "🎄 Natale", "type": "Commercial", "desc": "Massimo volume ordini."},
            ]

            correlation_rows = []
            for idx, row in monthly_biz.iterrows():
                m_start = row['Data_Interna'].replace(day=1)
                m_end = (m_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                
                # Dati Meteo Sintetizzati
                if not weather_full_df.empty and 'Date' in weather_full_df.columns:
                    city_w = weather_full_df[pd.to_datetime(weather_full_df['Date']).dt.to_period('M') == m_start.strftime('%Y-%m')]
                else:
                    city_w = pd.DataFrame()
                
                if not city_w.empty:
                    avg_t, sum_r = city_w['temp'].mean(), city_w['rain'].mean() 
                else:
                    avg_t, sum_r = 15.0, 40.0

                # --- INTEGRAZIONE DATI MERCATO NEL LEDGER ---
                oil_m_avg = vix_m_avg = 0.0
                if risk_data:
                    if 'Oil' in risk_data:
                        h_oil = risk_data['Oil']['history']
                        m_oil = h_oil[h_oil.index.to_period('M') == m_start.strftime('%Y-%m')]
                        oil_m_avg = m_oil.mean() if not m_oil.empty else 0
                    if 'VIX' in risk_data:
                        h_vix = risk_data['VIX']['history']
                        m_vix = h_vix[h_vix.index.to_period('M') == m_start.strftime('%Y-%m')]
                        vix_m_avg = m_vix.mean() if not m_vix.empty else 0

                # --- ANALISI DINAMICA DEL CONTESTO (GEOPOLITICA & MERCATI) ---
                mkt_sentiment = "✅ Stabile"
                if vix_m_avg > 28: mkt_sentiment = "🚨 Panico/Crisi"
                elif vix_m_avg > 22: mkt_sentiment = "⚠️ Alta Incertezza"
                elif vix_m_avg > 18: mkt_sentiment = "🔸 Volatilità Moderata"
                
                logistics_status = "🟢 Standard"
                if oil_m_avg > 92: logistics_status = "🚨 Shock Prezzi Petrolio"
                elif oil_m_avg > 85: logistics_status = "⚠️ Costi Logistica ↑"
                
                # Eventi commerciali
                m_events = [ev['title'] for ev in events_template if ev['month'] == m_start.month]
                event_str = m_events[0] if m_events else "Nessun Evento"
                
                correlation_rows.append({
                    'Mese': row['Month_Year'],
                    'Fatturato': row['Fatturato_Netto'],
                    'Spesa Ads': row['Spesa_Ads_Totale'],
                    'MER': row['MER'],
                    'Temp Media': avg_t,
                    'Petrolio (Avg)': oil_m_avg,
                    'VIX (Avg)': vix_m_avg,
                    'Contesto Esterno': f"{event_str} | {mkt_sentiment} | {logistics_status}"
                })
            
            ledger_df = pd.DataFrame(correlation_rows)
            
            # Styling della tabella
            st.dataframe(
                ledger_df.style.format({
                    'Fatturato': lambda x: format_euro(x, 0),
                    'Spesa Ads': lambda x: format_euro(x, 0),
                    'MER': lambda x: format_num_eu(x, 2),
                    'Temp Media': lambda x: format_num_eu(x, 1) + "°C",
                    'Petrolio (Avg)': lambda x: format_euro(x, 2),
                    'VIX (Avg)': lambda x: format_num_eu(x, 2)
                })
                .background_gradient(subset=['MER'], cmap='RdYlGn', vmin=monthly_biz['MER'].min(), vmax=monthly_biz['MER'].max())
                .background_gradient(subset=['Petrolio (Avg)'], cmap='YlOrRd')
                .background_gradient(subset=['VIX (Avg)'], cmap='YlOrRd')
                .background_gradient(subset=['Temp Media'], cmap='coolwarm')
                , use_container_width=True
            )
            
            # --- CONCLUSIONI AUTOMATICHE ---
            st.subheader("💡 Verdetto di Correlazione")
            best_month = monthly_biz.loc[monthly_biz['MER'].idxmax()]
            worst_month = monthly_biz.loc[monthly_biz['MER'].idxmin()]
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.success(f"**Mese Top: {best_month['Month_Year']}**")
                # Cerchiamo se c'era pioggia o eventi
                best_info = ledger_df[ledger_df['Mese'] == best_month['Month_Year']].iloc[0]
                st.write(f"In questo periodo l'efficienza è stata massima ({best_month['MER']:.2f}).")
                st.caption(f"Fattori Esterni: {best_info['Contesto Esterno']} | Meteo: {best_info['Temp Media']}")
            
            with col_res2:
                st.error(f"**Mese Critico: {worst_month['Month_Year']}**")
                worst_info = ledger_df[ledger_df['Mese'] == worst_month['Month_Year']].iloc[0]
                st.write(f"Calo di efficienza rilevato ({worst_month['MER']:.2f}).")
                st.caption(f"Fattori Esterni: {worst_info['Contesto Esterno']} | Meteo: {worst_info['Temp Media']}")
            
            st.warning("⚠️ **Conclusione Strategica:** Se vedi un MER alto in mesi con 'Saldi' o 'Pioggia > 60mm', la tua crescita è drogata da fattori esterni. Non scalare il budget nel mese successivo se le condizioni meteo/commerciali cambiano.")

    except Exception as e:
        st.error(f"⚠️ Errore: {e}")
else:
    st.info("👋 Carica il file CSV nel menu a sinistra per iniziare, oppure attiva la modalità DEMO.")
