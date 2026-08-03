# -*- coding: utf-8 -*-
"""
SIMULADOR DE CRÉDITOS DE CARBONO PARA COMPOSTAGEM
COM ENTRADA POR BOMBONAS DE 50 LITROS
BASELINE ALINHADA À UNFCCC A6.4‑AMT‑003 (2025) – APENAS CH₄, OX=0.383, SEM N₂O
"""

import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed
import warnings
from matplotlib.ticker import FuncFormatter
from SALib.sample.sobol import sample
from SALib.analyze.sobol import analyze
import yfinance as yf

# CSS customizado (apenas justificação e centralização)
st.markdown("""
<style>
    p, .stMarkdown, .stInfo, .stSuccess, .stWarning, .stException, .stText, .stCaption, .stMetric, .stDataFrame {
        text-align: justify !important;
    }
    .stMetric label, .stMetric .metric-value, .stMetric .metric-delta {
        text-align: center !important;
    }
    .test-stats {
        font-size: 0.9rem;
    }
    @media (max-width: 768px) {
        .stMetric label, .stMetric .metric-value {
            font-size: 0.8rem;
        }
        .test-stats {
            font-size: 0.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)

np.random.seed(50)

st.set_page_config(
    page_title="Simulador de Emissões de GEE e Créditos de Carbono",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# PARÂMETROS GLOBAIS – BASELINE CALIBRADO PARA RIBEIRÃO PRETO (ATERRO GUATAPARÁ)
# ALINHADO À UNFCCC A6.4‑AMT‑003 (2025) – APENAS CH₄, SEM N₂O
# =============================================================================
MCF_BASELINE = 1.0
OX_BASELINE = 0.383                # Application B – tropical wet (UNFCCC A6.4‑AMT‑003)
PHI_BASELINE = 0.85                # Fator φ para clima úmido (UNFCCC 2024)

# Fatores de emissão padrão da metodologia UNFCCC (AMS‑III.F / TOOL13)
EF_CH4_STD = 0.002      # t CH₄ / t resíduo úmido  → 0.002 kg CH₄ / kg resíduo
EF_N2O_STD = 0.0002     # t N₂O / t resíduo úmido  → 0.0002 kg N₂O / kg resíduo

# GWP FIXO – IPCC AR5 (CONFORME UNFCCC A6.4‑AMT‑003)
GWP_CH4 = 28.0
GWP_N2O = 265.0

# Parâmetros fixos baseados na literatura (Yang et al. 2017) – usados apenas para comparação
TOC = 0.436
TN = 0.0142
F_CH4_VERMI = 0.0013
F_N2O_VERMI = 0.0092
F_CH4_THERMO = 0.0060
F_N2O_THERMO = 0.0196
COMPOSTING_DAYS = 50

# Perfis de emissão diários (mantidos para compostagem)
profile_ch4_vermi = np.array([
    0.02,0.02,0.02,0.03,0.03,0.04,0.04,0.05,0.05,0.06,
    0.07,0.08,0.09,0.10,0.09,0.08,0.07,0.06,0.05,0.04,
    0.03,0.02,0.02,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
    0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
    0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001
])
profile_ch4_vermi /= profile_ch4_vermi.sum()

profile_n2o_vermi = np.array([
    0.15,0.10,0.20,0.05,0.03,0.03,0.03,0.04,0.05,0.06,
    0.08,0.09,0.10,0.08,0.07,0.06,0.05,0.04,0.03,0.02,
    0.01,0.01,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,
    0.002,0.002,0.002,0.002,0.002,0.001,0.001,0.001,0.001,0.001,
    0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001
])
profile_n2o_vermi /= profile_n2o_vermi.sum()

profile_ch4_thermo = profile_ch4_vermi.copy()
profile_n2o_thermo = profile_n2o_vermi.copy()

# =============================================================================
# CLASSE DE CÁLCULO (CORRIGIDA – BASELINE APENAS CH₄, OX=0.383, SEM N₂O)
# =============================================================================
class GHGEmissionCalculator:
    def __init__(self):
        self.TOC = TOC
        self.TN = TN
        self.f_CH4_vermi = F_CH4_VERMI
        self.f_N2O_vermi = F_N2O_VERMI
        self.f_CH4_thermo = F_CH4_THERMO
        self.f_N2O_thermo = F_N2O_THERMO
        self.EF_CH4_std = EF_CH4_STD
        self.EF_N2O_std = EF_N2O_STD
        self.COMPOSTING_DAYS = COMPOSTING_DAYS
        self.GWP_CH4_20 = GWP_CH4
        self.GWP_N2O_20 = GWP_N2O
        self.MCF = MCF_BASELINE
        self.F = 0.5
        self.OX = OX_BASELINE          # 0.383
        self.Ri = 0.0
        self.profile_ch4_vermi = profile_ch4_vermi
        self.profile_n2o_vermi = profile_n2o_vermi
        self.profile_ch4_thermo = profile_ch4_thermo
        self.profile_n2o_thermo = profile_n2o_thermo

    def calculate_landfill_emissions(self, w, k, T, doc, docf, umid, years=20, phi=PHI_BASELINE, fy=0.0):
        """
        Baseline UNFCCC A6.4‑AMT‑003 (2025): apenas CH₄, com φ, OX e captura fy.
        N₂O do aterro e pré‑descarte NÃO são considerados.
        """
        days = years * 365
        # Potencial de geração de metano (kg CH₄ / dia)
        ch4_pot = (doc * docf * self.MCF * self.F * (16/12) * (1 - self.Ri) * (1 - self.OX)) * w
        t = np.arange(1, days + 1)
        kernel = np.exp(-k * (t - 1) / 365) - np.exp(-k * t / 365)
        ch4 = np.convolve(np.ones(days), kernel, mode='full')[:days] * ch4_pot
        ch4 *= phi * (1 - fy)   # correção climática e captura
        # N₂O zerado (norma não inclui)
        n2o = np.zeros(days)
        return ch4, n2o

    def calculate_vermicomposting_emissions(self, w, umid, years=20):
        days = years * 365
        dry = 1 - umid
        ch4_batch = w * self.TOC * self.f_CH4_vermi * (16/12) * dry
        n2o_batch = w * self.TN * self.f_N2O_vermi * (44/28) * dry
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e + d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4_vermi[d]
                    n2o[ed] += n2o_batch * self.profile_n2o_vermi[d]
        return ch4, n2o

    def calculate_thermophilic_emissions(self, w, umid, years=20):
        days = years * 365
        dry = 1 - umid
        ch4_batch = w * self.TOC * self.f_CH4_thermo * (16/12) * dry
        n2o_batch = w * self.TN * self.f_N2O_thermo * (44/28) * dry
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e + d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4_thermo[d]
                    n2o[ed] += n2o_batch * self.profile_n2o_thermo[d]
        return ch4, n2o

    def calculate_standard_emissions(self, w, umid, years=20):
        """Emissões com fatores padrão UNFCCC (AMS‑III.F / TOOL13)."""
        days = years * 365
        ch4_batch = w * self.EF_CH4_std
        n2o_batch = w * self.EF_N2O_std
        ch4 = np.zeros(days)
        n2o = np.zeros(days)
        for e in range(days):
            for d in range(self.COMPOSTING_DAYS):
                ed = e + d
                if ed < days:
                    ch4[ed] += ch4_batch * self.profile_ch4_vermi[d]
                    n2o[ed] += n2o_batch * self.profile_n2o_vermi[d]
        return ch4, n2o

    def calculate_avoided_emissions(self, w, k, T, doc, docf, umid, years, fy=0.0):
        ch4_l, _ = self.calculate_landfill_emissions(w, k, T, doc, docf, umid, years, fy=fy)
        ch4_v, n2o_v = self.calculate_vermicomposting_emissions(w, umid, years)
        ch4_t, n2o_t = self.calculate_thermophilic_emissions(w, umid, years)
        ch4_s, n2o_s = self.calculate_standard_emissions(w, umid, years)

        base = (ch4_l * self.GWP_CH4_20) / 1000   # apenas CH₄, convertido para tCO₂eq
        vermi = (ch4_v * self.GWP_CH4_20 + n2o_v * self.GWP_N2O_20) / 1000
        thermo = (ch4_t * self.GWP_CH4_20 + n2o_t * self.GWP_N2O_20) / 1000
        std = (ch4_s * self.GWP_CH4_20 + n2o_s * self.GWP_N2O_20) / 1000

        return {
            'baseline': base.sum(),
            'vermi_avoided': base.sum() - vermi.sum(),
            'thermo_avoided': base.sum() - thermo.sum(),
            'std_avoided': base.sum() - std.sum(),
            'base_series': base, 'vermi_series': vermi, 'thermo_series': thermo, 'std_series': std
        }

    def calculate_avoided_emissions_fast(self, w, k, T, doc, docf, umid, years, fy=0.0):
        ch4_l, _ = self.calculate_landfill_emissions(w, k, T, doc, docf, umid, years, fy=fy)
        ch4_v, n2o_v = self.calculate_vermicomposting_emissions(w, umid, years)
        ch4_t, n2o_t = self.calculate_thermophilic_emissions(w, umid, years)
        ch4_s, n2o_s = self.calculate_standard_emissions(w, umid, years)

        base = (ch4_l * self.GWP_CH4_20) / 1000
        vermi = (ch4_v * self.GWP_CH4_20 + n2o_v * self.GWP_N2O_20) / 1000
        thermo = (ch4_t * self.GWP_CH4_20 + n2o_t * self.GWP_N2O_20) / 1000
        std = (ch4_s * self.GWP_CH4_20 + n2o_s * self.GWP_N2O_20) / 1000

        return (base.sum() - vermi.sum()), (base.sum() - thermo.sum()), (base.sum() - std.sum())

# =============================================================================
# FUNÇÕES DE COTAÇÃO, FORMATAÇÃO E ESTADO (mantidas idênticas)
# =============================================================================
def obter_cotacao_carbono():
    try:
        ticker = yf.Ticker("CO2.L")
        data = ticker.history(period="1d")
        if not data.empty:
            preco = data['Close'].iloc[-1]
            if 10 < preco < 200:
                return preco, "€", "Carbon Futures (CO2.L)", True, "Yahoo Finance (CO2.L)"
        return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"
    except:
        return 85.50, "€", "Carbon Emissions (Referência)", False, "Referência"

def obter_cotacao_euro_real():
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return float(data['EURBRL']['bid']), "R$", True, "AwesomeAPI"
    except:
        pass
    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['rates']['BRL'], "R$", True, "ExchangeRate-API"
    except:
        pass
    return 5.50, "R$", False, "Referência"

def calcular_valor_creditos(e, preco, moeda, taxa=1):
    return e * preco * taxa

def formatar_br(num):
    if pd.isna(num) or not np.isfinite(num):
        return "N/A"
    num = round(num, 2)
    return f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def exibir_cotacao_carbono():
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    if not st.session_state.get('cotacao_carregada', False):
        st.session_state.mostrar_atualizacao = True
        st.session_state.cotacao_carregada = True
    col1, col2 = st.sidebar.columns([3,1])
    with col1:
        if st.button("🔄 Atualizar Cotações"):
            st.session_state.cotacao_atualizada = True
            st.session_state.mostrar_atualizacao = True
    if st.session_state.get('mostrar_atualizacao', False):
        st.sidebar.info("🔄 Atualizando cotações...")
        preco_carbono, moeda, _, _, fonte_carbono = obter_cotacao_carbono()
        preco_euro, moeda_real, _, _ = obter_cotacao_euro_real()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        st.session_state.fonte_cotacao = fonte_carbono
        st.session_state.mostrar_atualizacao = False
        st.session_state.cotacao_atualizada = False
        st.rerun()
    st.sidebar.metric("Preço do Carbono (tCO₂eq)", f"{st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}", help=f"Fonte: {st.session_state.fonte_cotacao}")
    st.sidebar.metric("Euro (EUR/BRL)", f"{st.session_state.moeda_real} {formatar_br(st.session_state.taxa_cambio)}")
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    st.sidebar.metric("Carbono em Reais (tCO₂eq)", f"R$ {formatar_br(preco_carbono_reais)}")
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**  
        - Preço: {st.session_state.moeda_carbono} {formatar_br(st.session_state.preco_carbono)}/tCO₂eq  
        - Câmbio: 1 Euro = R$ {formatar_br(st.session_state.taxa_cambio)}  
        - Carbono em Reais: R$ {formatar_br(preco_carbono_reais)}/tCO₂eq  
        **🌍 Fonte:** {st.session_state.fonte_cotacao} (ICE CO2.L)  
        """)

def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        p, m, _, _, f = obter_cotacao_carbono()
        st.session_state.preco_carbono = p
        st.session_state.moeda_carbono = m
        st.session_state.fonte_cotacao = f
    if 'taxa_cambio' not in st.session_state:
        euro, real, _, _ = obter_cotacao_euro_real()
        st.session_state.taxa_cambio = euro
        st.session_state.moeda_real = real
    if 'moeda_real' not in st.session_state:
        st.session_state.moeda_real = "R$"
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'k_ano' not in st.session_state:
        st.session_state.k_ano = 0.06

inicializar_session_state()

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================
st.title("🌍 Simulador de Emissões de GEE e Créditos de Carbono")
st.caption("Comparação: Vermicompostagem (Yang et al. 2017) vs Compostagem Termofílica (Yang et al. 2017) vs Fatores Padrão UNFCCC (AMS‑III.F / TOOL13). Baseline = Aterro em Guatapará, destino da maior parte dos RSU de Ribeirão Preto")

with st.container():
    st.markdown("""
    **📘 Nota metodológica:**  
    - **Baseline de aterro** agora **totalmente alinhada à UNFCCC A6.4‑AMT‑003 (2025)**: apenas **CH₄**, **OX = 0,383** (Application B), **sem N₂O** e **sem pré‑descarte**.  
    - O fator de captura de metano (`fy`) continua sendo parametrizável (padrão 0,0).  
    - As comparações com fatores experimentais (Yang et al.) são mantidas para fins científicos, mas os créditos elegíveis segundo a norma são aqueles calculados com a baseline aqui implementada e os fatores padrão da TOOL13.
    """)
    st.divider()

exibir_cotacao_carbono()

# =============================================================================
# SIDEBAR COM PARÂMETROS (mantida idêntica, exceto pela remoção do aviso sobre DOC_f)
# =============================================================================
with st.sidebar:
    st.header("⚙️ Parâmetros")
    
    unidade = st.radio("Unidade de entrada:", ["kg/dia", "Bombonas de 50L"])
    if unidade == "kg/dia":
        residuos_kg_dia = st.slider("Resíduos (kg/dia)", 10, 1000, 100, 10)
    else:
        col1, col2 = st.columns(2)
        with col1:
            num_bombonas = st.number_input("Bombonas/dia", min_value=1, max_value=100, value=10, step=1)
        with col2:
            densidade = st.selectbox("Densidade (kg/L)", [0.50, 0.60, 0.70, 0.80], index=1)
            densidade = st.slider("ou ajuste manual", 0.30, 0.90, 0.60, 0.01) if densidade == 0.60 else densidade
        residuos_kg_dia = num_bombonas * 50 * densidade
        st.caption(f"→ Estimativa: **{residuos_kg_dia:.1f} kg/dia**")
    
    opcao_k = st.selectbox("k (ano⁻¹)", ["0,06 (lento)", "0,40 (rápido)"], index=0)
    k_ano = 0.40 if "0,40" in opcao_k else 0.06
    st.session_state.k_ano = k_ano
    T = st.slider("Temperatura média (°C)", 20, 40, 25, 1)
    DOC = st.slider("DOC (fração)", 0.10, 0.25, 0.15, 0.01)
    DOC_f = 0.7
    st.info("📌 **DOC_f fixo em 0,7** – resíduos altamente decomponíveis (Tabela 7 UNFCCC).")
    umidade_valor = st.slider("Umidade (%)", 50, 95, 85, 1)
    umidade = umidade_valor/100.0

    st.subheader("🏭 Captura de Metano no Aterro (fy)")
    fy = st.slider(
        "Fração capturada e destruída (fy)",
        min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.2f",
        help="Conforme Data/Parameter table 10 da A6.4-AMT-003."
    )
    st.caption(f"Valor atual: **{fy:.2f}** ({fy*100:.0f}% de captura)")

    anos_simulacao = st.slider("Anos de simulação", 5, 50, 20, 5)
    n_simulations = st.slider("Monte Carlo (n)", 50, 1000, 100, 50)
    n_samples = st.slider("Sobol (amostras)", 32, 256, 64, 16)
    
    if st.button("🚀 Executar Simulação", type="primary"):
        st.session_state.run_simulation = True

# =============================================================================
# CACHE DAS SIMULAÇÕES (mantida idêntica, apenas adaptada aos novos outputs)
# =============================================================================
@st.cache_data(show_spinner=False)
def cached_sobol(n_samples, w, k, T, doc, docf, umid, years, fy_bound=[0.0, 0.8]):
    problem = {'num_vars':4, 'names':['k','T','DOC','fy'], 'bounds':[[0.06,0.40],[20,40],[0.10,0.25], fy_bound]}
    param_values = sample(problem, n_samples, seed=50)
    calc = GHGEmissionCalculator()
    def f(p):
        return calc.calculate_avoided_emissions_fast(w, p[0], p[1], p[2], docf, umid, years, fy=p[3])
    res = Parallel(n_jobs=-1)(delayed(f)(p) for p in param_values)
    arr_v = np.array([r[0] for r in res])
    arr_t = np.array([r[1] for r in res])
    arr_s = np.array([r[2] for r in res])
    Si_v = analyze(problem, arr_v, print_to_console=False)
    Si_t = analyze(problem, arr_t, print_to_console=False)
    Si_s = analyze(problem, arr_s, print_to_console=False)
    return Si_v, Si_t, Si_s

@st.cache_data(show_spinner=False)
def cached_montecarlo(n, w, k, T, doc, docf, umid, years):
    np.random.seed(50)
    u = np.random.uniform(0.75, 0.90, n)
    t = np.random.normal(25, 3, n)
    d = np.random.triangular(0.12, 0.15, 0.18, n)
    fy_samples = np.random.uniform(0.0, 0.8, n)
    calc = GHGEmissionCalculator()
    def run(i):
        np.random.seed(50+i)
        return calc.calculate_avoided_emissions_fast(w, k, t[i], d[i], docf, u[i], years, fy=fy_samples[i])
    res = Parallel(n_jobs=-1)(delayed(run)(i) for i in range(n))
    arr_v = np.array([r[0] for r in res])
    arr_t = np.array([r[1] for r in res])
    arr_s = np.array([r[2] for r in res])
    return arr_v, arr_t, arr_s

# =============================================================================
# EXECUÇÃO PRINCIPAL (mantida, apenas atualizadas as mensagens)
# =============================================================================
if st.session_state.get('run_simulation', False):
    with st.spinner("Executando simulação..."):
        calc = GHGEmissionCalculator()
        res = calc.calculate_avoided_emissions(residuos_kg_dia, k_ano, T, DOC, DOC_f, umidade, anos_simulacao, fy=fy)
        
        base_series = res['base_series']
        vermi_series = res['vermi_series']
        termo_series = res['thermo_series']
        std_series = res['std_series']

        dias = len(base_series)
        datas = pd.date_range(start=datetime.now(), periods=dias, freq='D')
        df_dia = pd.DataFrame({'Data':datas, 'Base':base_series, 'Vermi':vermi_series, 'Termo':termo_series, 'Std':std_series})
        df_dia['Year'] = df_dia['Data'].dt.year
        df_anual = df_dia.groupby('Year').agg({'Base':'sum','Vermi':'sum','Termo':'sum','Std':'sum'}).reset_index()
        df_anual['Evitado_Vermi'] = df_anual['Base'] - df_anual['Vermi']
        df_anual['Evitado_Termo'] = df_anual['Base'] - df_anual['Termo']
        df_anual['Evitado_Std'] = df_anual['Base'] - df_anual['Std']

        base_acum = np.cumsum(base_series)
        vermi_acum = np.cumsum(vermi_series)
        termo_acum = np.cumsum(termo_series)
        std_acum = np.cumsum(std_series)

        st.header(f"📈 Resultados da Simulação (GWP AR5 – CH₄=28 | N₂O=265)")
        st.info(f"""
        **Parâmetros – Ribeirão Preto (Aterro Guatapará):**  
        - k = {formatar_br(k_ano)} ano⁻¹  
        - Temperatura = {formatar_br(T)} °C  
        - DOC = {formatar_br(DOC)}  
        - **DOC_f (Tabela 7 UNFCCC) = 0,7** (fixo para resíduos altamente decomponíveis)  
        - Umidade = {formatar_br(umidade_valor)}%  
        - **f_y (captura de metano) = {formatar_br(fy)}**  
        - Resíduos totais = {formatar_br(residuos_kg_dia*365*anos_simulacao/1000)} t  
        - **Baseline UNFCCC: apenas CH₄, φ = 0,85, OX = 0,383 (sem N₂O)**  
        """)

        st.subheader(f"💰 Valor Financeiro (GWP Fixo – AR5)")
        preco = st.session_state.preco_carbono
        moeda = st.session_state.moeda_carbono
        cambio = st.session_state.taxa_cambio
        v_vermi = res['vermi_avoided']
        v_termo = res['thermo_avoided']
        v_std = res['std_avoided']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vermicompostagem (Yang)", f"{formatar_br(v_vermi)} tCO₂eq")
            st.metric("Euro", f"{moeda} {formatar_br(v_vermi*preco)}")
            st.metric("R$", f"R$ {formatar_br(v_vermi*preco*cambio)}")
        with col2:
            st.metric("Termofílica (Yang)", f"{formatar_br(v_termo)} tCO₂eq")
            st.metric("Euro", f"{moeda} {formatar_br(v_termo*preco)}")
            st.metric("R$", f"R$ {formatar_br(v_termo*preco*cambio)}")
        with col3:
            st.metric("Fatores Padrão UNFCCC (TOOL13)", f"{formatar_br(v_std)} tCO₂eq")
            st.metric("Euro", f"{moeda} {formatar_br(v_std*preco)}")
            st.metric("R$", f"R$ {formatar_br(v_std*preco*cambio)}")
        
        razao_vt = v_vermi / v_termo if v_termo != 0 else np.inf
        razao_vs = v_vermi / v_std if v_std != 0 else np.inf
        st.success(f"""
        **💡 Análise financeira:**  
        - A vermicompostagem gera aproximadamente **{formatar_br(razao_vt)}x** mais receita que a termofílica e **{formatar_br(razao_vs)}x** mais que os fatores padrão.  
        - Para cada tonelada de resíduo tratado, o retorno apenas com créditos (sem custos operacionais) é de **{moeda} {formatar_br((v_vermi*preco)/(residuos_kg_dia*365*anos_simulacao/1000))} por t**.
        """)

        st.subheader(f"📊 Comparação Anual das Emissões Evitadas")
        fig, ax = plt.subplots(figsize=(12,6))
        x = np.arange(len(df_anual['Year']))
        width = 0.25
        ax.bar(x - width, df_anual['Evitado_Vermi'], width, label='Vermicompostagem (Yang)', color='forestgreen', edgecolor='black')
        ax.bar(x, df_anual['Evitado_Termo'], width, label='Compostagem Termofílica (Yang)', color='orange', hatch='//', edgecolor='black')
        ax.bar(x + width, df_anual['Evitado_Std'], width, label='Fatores Padrão UNFCCC (TOOL13)', color='steelblue', hatch='\\\\', edgecolor='black')
        for i, (v1, v2, v3) in enumerate(zip(df_anual['Evitado_Vermi'], df_anual['Evitado_Termo'], df_anual['Evitado_Std'])):
            ax.text(i-width, v1+max(v1,v2,v3)*0.01, formatar_br(v1), ha='center', fontsize=8)
            ax.text(i, v2+max(v1,v2,v3)*0.01, formatar_br(v2), ha='center', fontsize=8)
            ax.text(i+width, v3+max(v1,v2,v3)*0.01, formatar_br(v3), ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(df_anual['Year'])
        ax.set_ylabel('tCO₂eq evitadas')
        ax.set_title('Emissões Evitadas por Ano (Baseline UNFCCC: apenas CH₄)')
        ax.legend()
        ax.yaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig)
        plt.close(fig)

        st.subheader(f"📉 Emissões Acumuladas (Baseline vs Tecnologias)")
        fig2, ax2 = plt.subplots(figsize=(11,6))
        ax2.plot(datas, base_acum, 'r-', label='Baseline (Aterro)')
        ax2.plot(datas, vermi_acum, 'g-', label='Vermicompostagem (Yang)')
        ax2.plot(datas, termo_acum, 'orange', label='Termofílica (Yang)')
        ax2.plot(datas, std_acum, 'steelblue', label='Fatores Padrão UNFCCC')
        ax2.fill_between(datas, vermi_acum, base_acum, alpha=0.3, color='lightgreen')
        ax2.set_title(f'Emissões Acumuladas – {anos_simulacao} anos (k={formatar_br(k_ano)} ano⁻¹)')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('tCO₂eq')
        ax2.legend()
        ax2.yaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig2)
        plt.close(fig2)
        
        st.success(f"""
        **📈 Impacto acumulado:**  
        - Em {anos_simulacao} anos, a vermicompostagem evitaria **{formatar_br(base_acum[-1] - vermi_acum[-1])} tCO₂eq** em relação ao aterro.  
        - A termofílica evitaria **{formatar_br(base_acum[-1] - termo_acum[-1])} tCO₂eq**.  
        - Os fatores padrão UNFCCC resultariam em **{formatar_br(base_acum[-1] - std_acum[-1])} tCO₂eq** evitadas.  
        """)

        st.subheader(f"🎯 Análise de Sensibilidade Sobol (com fy como variável)")
        with st.spinner("Sobol em execução..."):
            Si_v, Si_t, Si_s = cached_sobol(n_samples, residuos_kg_dia, k_ano, T, DOC, DOC_f, umidade, anos_simulacao)
        df_sens = pd.DataFrame({
            'Parâmetro': ['k','T','DOC','fy'],
            'S1_Vermi': Si_v['S1'], 'ST_Vermi': Si_v['ST'],
            'S1_Termo': Si_t['S1'], 'ST_Termo': Si_t['ST'],
            'S1_Std': Si_s['S1'], 'ST_Std': Si_s['ST']
        })
        st.dataframe(df_sens.style.format({col: '{:.4f}' for col in df_sens.columns if col != 'Parâmetro'}))

        st.subheader(f"🎲 Monte Carlo e Testes Estatísticos (com fy variável)")
        with st.spinner("Monte Carlo em execução..."):
            arr_v, arr_t, arr_s = cached_montecarlo(n_simulations, residuos_kg_dia, k_ano, T, DOC, DOC_f, umidade, anos_simulacao)

        fig3, ax3 = plt.subplots(figsize=(10,5))
        sns.kdeplot(arr_v, label='Vermicompostagem (Yang)', ax=ax3)
        sns.kdeplot(arr_t, label='Termofílica (Yang)', ax=ax3)
        sns.kdeplot(arr_s, label='Fatores Padrão UNFCCC', ax=ax3)
        ax3.set_title('Distribuição das Emissões Evitadas (Baseline UNFCCC)')
        ax3.set_xlabel('tCO₂eq')
        ax3.xaxis.set_major_formatter(FuncFormatter(br_format))
        st.pyplot(fig3)
        plt.close(fig3)

        stats_df = pd.DataFrame([
            {'Tecnologia': 'Vermicompostagem (Yang)', 'Média': np.mean(arr_v), 'Mediana': np.median(arr_v), 'DP': np.std(arr_v), 'IC95% inf': np.percentile(arr_v,2.5), 'IC95% sup': np.percentile(arr_v,97.5)},
            {'Tecnologia': 'Termofílica (Yang)', 'Média': np.mean(arr_t), 'Mediana': np.median(arr_t), 'DP': np.std(arr_t), 'IC95% inf': np.percentile(arr_t,2.5), 'IC95% sup': np.percentile(arr_t,97.5)},
            {'Tecnologia': 'Fatores Padrão UNFCCC', 'Média': np.mean(arr_s), 'Mediana': np.median(arr_s), 'DP': np.std(arr_s), 'IC95% inf': np.percentile(arr_s,2.5), 'IC95% sup': np.percentile(arr_s,97.5)}
        ])
        st.dataframe(stats_df.style.format({c: lambda x: formatar_br(x) for c in stats_df.columns if c != 'Tecnologia'}))

        cv = (np.std(arr_v)/np.mean(arr_v)*100) if np.mean(arr_v) != 0 else 0
        st.success(f"""
        **📊 Incerteza dos resultados:**  
        - IC 95% da vermicompostagem: **[{formatar_br(np.percentile(arr_v,2.5))}, {formatar_br(np.percentile(arr_v,97.5))}] tCO₂eq**.  
        - Coeficiente de variação: **{cv:.1f}%**.
        """)

        st.write("**Testes de diferença significativa (p-valores):**")
        t_vt = stats.ttest_rel(arr_v, arr_t)[1]
        t_vs = stats.ttest_rel(arr_v, arr_s)[1]
        t_ts = stats.ttest_rel(arr_t, arr_s)[1]
        w_vt = stats.wilcoxon(arr_v, arr_t)[1]
        w_vs = stats.wilcoxon(arr_v, arr_s)[1]
        w_ts = stats.wilcoxon(arr_t, arr_s)[1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Vermi vs Termo**<br>t-test p = {t_vt:.5f}<br>Wilcoxon p = {w_vt:.5f}", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Vermi vs Std**<br>t-test p = {t_vs:.5f}<br>Wilcoxon p = {w_vs:.5f}", unsafe_allow_html=True)
        with col3:
            st.markdown(f"**Termo vs Std**<br>t-test p = {t_ts:.5f}<br>Wilcoxon p = {w_ts:.5f}", unsafe_allow_html=True)

        st.subheader("📋 Resultados Anuais Detalhados")
        df_anual_fmt = df_anual[['Year','Base','Vermi','Termo','Std','Evitado_Vermi','Evitado_Termo','Evitado_Std']].copy()
        df_anual_fmt.columns = ['Ano','Baseline','Vermicompostagem (Yang)','Termofílica (Yang)','Fatores Padrão UNFCCC','Evitado Vermi','Evitado Termo','Evitado Std']
        for col in df_anual_fmt.columns:
            if col != 'Ano':
                df_anual_fmt[col] = df_anual_fmt[col].apply(formatar_br)
        st.dataframe(df_anual_fmt)

    st.session_state.run_simulation = False
else:
    st.info("💡 Ajuste os parâmetros na barra lateral e clique em **Executar Simulação** para ver os resultados.")

st.markdown("---")
with st.expander("📚 Referências Metodológicas Detalhadas"):
    st.markdown("""
    **1. Baseline – Aterro Sanitário (Guatapará, Ribeirão Preto)**  
    - **Modelo UNFCCC A6.4‑AMT‑003 (2025) – apenas CH₄**: MCF=1,0; F=0,5; **OX=0,383**; φ=0,85.  
    - **DOC_f** fixo em 0,7 para resíduos altamente decomponíveis (Tabela 7 da norma).  
    - **f_y (captura de metano)** conforme Data/Parameter table 10.  
    - **N₂O e pré‑descarte NÃO são incluídos**, em conformidade com a metodologia.

    **2. Tecnologias de compostagem**  
    - **Fatores padrão UNFCCC (AMS‑III.F / TOOL13)**: CH₄ = 0,002 t/t; N₂O = 0,0002 t/t.  
    - **Fatores Yang et al. (2017)** para vermicompostagem e termofílica (apenas para comparação).  

    **3. GWP fixo IPCC AR5**: CH₄ = 28, N₂O = 265.
    """)
