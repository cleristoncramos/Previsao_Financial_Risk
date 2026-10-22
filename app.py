"""
=============================================================================
PAINEL ACADÊMICO E PROFISSIONAL — PREVISÃO DE RISCOS FINANCEIROS
Trabalho de Conclusão de Curso (TCC) — Tecnologia em Gestão de Dados
Universidade Federal do Piauí (UFPI) — CEAD
Autora: Lorrana Evelyn de Araújo Pereira
=============================================================================
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Silencia logs e avisos do TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

# Configurações do Diretório e Importações do Projeto
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    ATIVOS,
    DATA_DIR,
    DATA_FIM,
    DATA_INICIO,
    FIGURES_DIR,
    TICKER_REF,
)
from src.data_collection import (
    baixar_ativo,
    carregar_ativo_local,
    coletar_todos_ativos,
    engenharia_atributos,
    salvar_ativo_csv,
)
from src.evaluation import (
    adicionar_baseline_ingenuo,
    calcular_metricas,
    validacao_cruzada_temporal,
)
from src.model_store import (
    carregar_modelos_salvos,
    modelos_existem,
    pasta_modelos_ativo,
    predizer_com_modelos,
)
from src.models import treinar_todos_modelos
from src.preprocessing import DatasetModelagem, preparar_dados_modelagem

# =============================================================================
# CONFIGURAÇÃO GERAL DA PÁGINA STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Previsão de Riscos Financeiros | TCC UFPI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injeção de CSS personalizado
CSS_PATH = BASE_DIR / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =============================================================================
# FUNÇÕES DE CACHE PARA CARREGAMENTO RÁPIDO
# =============================================================================
@st.cache_data(show_spinner=False)
def carregar_dados_ativo_cached(ticker: str) -> Optional[pd.DataFrame]:
    """Carrega dados locais se existirem, ou realiza o download via Yahoo Finance."""
    df = carregar_ativo_local(ticker, pasta_saida=DATA_DIR)
    if df is not None and not df.empty:
        df["Data"] = pd.to_datetime(df["Data"])
        return df.sort_values("Data").reset_index(drop=True)

    # Tenta download se não existir localmente
    df_raw = baixar_ativo(ticker, inicio=DATA_INICIO, fim=DATA_FIM)
    if df_raw is not None and not df_raw.empty:
        df_proc = engenharia_atributos(df_raw)
        categoria = "indices" if ticker.startswith("^") else "acoes_br"
        salvar_ativo_csv(df_proc, ticker, categoria, pasta_saida=DATA_DIR)
        df_proc["Data"] = pd.to_datetime(df_proc["Data"])
        return df_proc.sort_values("Data").reset_index(drop=True)

    return None


@st.cache_data(show_spinner=False)
def carregar_todos_dados_cached() -> Dict[str, pd.DataFrame]:
    """Carrega as séries de todos os ativos monitorados no projeto."""
    todos = {}
    for cat, tickers in ATIVOS.items():
        for t in tickers:
            df = carregar_dados_ativo_cached(t)
            if df is not None:
                todos[t] = df
    return todos


@st.cache_resource(show_spinner=False)
def carregar_modelos_cached(pasta_modelos: str):
    """Carrega os modelos persistidos de um ativo (ver pasta_modelos_ativo)."""
    pasta = Path(pasta_modelos)
    if modelos_existem(pasta):
        return carregar_modelos_salvos(pasta)
    return {}, None, {}


@st.cache_data(show_spinner=False)
def preparar_dataset_cached(df: pd.DataFrame) -> DatasetModelagem:
    """Prepara o dataset com split temporal 80/20 e StandardScaler."""
    return preparar_dados_modelagem(df)


@st.cache_data(show_spinner="Executando validação cruzada temporal...")
def validacao_cruzada_cached(ticker: str, ultima_data: str, _dataset: DatasetModelagem) -> pd.DataFrame:
    """
    Calcula o RMSE do XGBoost em 5 folds do TimeSeriesSplit para o ativo selecionado.
    O cache é indexado por ticker e data final da série (o dataset não é hasheado).
    """
    n_splits = 5
    rmses = validacao_cruzada_temporal(_dataset.X, _dataset.y, n_splits=n_splits)
    datas = pd.to_datetime(_dataset.datas)
    periodos = [
        f"{datas[val_idx[0]]:%m/%Y}–{datas[val_idx[-1]]:%m/%Y}"
        for _, val_idx in TimeSeriesSplit(n_splits=n_splits).split(_dataset.X)
    ]
    return pd.DataFrame(
        {
            "Fold": [f"Fold {i}" for i in range(1, n_splits + 1)],
            "Período de Validação": periodos,
            "RMSE": rmses,
        }
    )


# =============================================================================
# BARRA LATERAL (SIDEBAR)
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1.2rem;">
            <div class="academic-badge">UFPI • CEAD</div>
            <h3 style="margin: 0.3rem 0; font-size: 1.15rem; color: #1e3a8a; font-weight: 700;">
                Gestão de Dados
            </h3>
            <p style="font-size: 0.85rem; color: #64748b; margin: 0;">
                Previsão de Riscos Financeiros
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🎯 Seleção de Ativo de Análise")

    lista_opcoes = [
        "^BVSP (IBOVESPA - Referência TCC)",
        "^GSPC (S&P 500 EUA)",
        "PETR4.SA (Petrobras PN)",
        "VALE3.SA (Vale ON)",
        "ITUB4.SA (Itaú Unibanco PN)",
        "BBDC4.SA (Bradesco PN)",
        "ABEV3.SA (Ambev ON)",
    ]

    opcao_selecionada = st.selectbox(
        "Ativo do Estudo:",
        options=lista_opcoes,
        index=0,
        help="Selecione o ativo financeiro para análise preditiva e exploratória.",
    )

    ticker_selecionado = opcao_selecionada.split(" ")[0]

    # Status dos Modelos: cada ativo tem seus próprios modelos e scaler, pois um
    # modelo treinado em uma série não é válido para outra (escalas e regimes distintos)
    pasta_modelos_sel = pasta_modelos_ativo(ticker_selecionado)
    modelos_dict, scaler_salvo, meta_dict = carregar_modelos_cached(str(pasta_modelos_sel))
    tem_modelos = (
        bool(modelos_dict)
        and scaler_salvo is not None
        and meta_dict.get("ticker_ref") == ticker_selecionado
    )

    st.markdown("---")
    st.markdown("### 🤖 Estado da Modelagem")
    if tem_modelos:
        st.success(f"✅ Modelos de {ticker_selecionado} Carregados")
        data_salvo = meta_dict.get("data_salvamento", "Recente")
        st.caption(f"Última persistência: {data_salvo}")
    else:
        st.warning(f"⚠️ Não há modelos treinados para {ticker_selecionado}.")

    # Botão de Ação: Atualizar Dados (novo download no Yahoo Finance e regravação dos CSVs)
    if st.button("🔄 Atualizar Séries Temporais", use_container_width=True):
        # O parâmetro end do yfinance é exclusivo: amanhã garante a inclusão do pregão de hoje
        fim_coleta = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        with st.spinner("Baixando séries atualizadas do Yahoo Finance..."):
            dados_novos = coletar_todos_ativos(fim=fim_coleta)
        st.cache_data.clear()
        total_ativos = sum(len(t) for t in ATIVOS.values())
        st.session_state["msg_atualizacao"] = (
            f"{len(dados_novos)} de {total_ativos} séries atualizadas."
            if dados_novos
            else "Falha no download; mantidas as séries locais."
        )
        st.rerun()
    if "msg_atualizacao" in st.session_state:
        st.caption(st.session_state.pop("msg_atualizacao"))

    # Informações da Autoria
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #64748b; line-height: 1.5;">
            <strong>Autora:</strong> Lorrana Evelyn de Araújo Pereira<br>
            <strong>Instituição:</strong> Universidade Federal do Piauí<br>
            <strong>Ano:</strong> 2026<br>
            <strong>Metodologia:</strong> Split Temporal 80/20 & XAI
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# CABEÇALHO INSTITUCIONAL PRINCIPAL
# =============================================================================
st.markdown(
    f"""
    <div class="academic-header">
        <div class="academic-badge">Trabalho de Conclusão de Curso (TCC) • UFPI / CEAD</div>
        <div class="academic-title">Previsão de Riscos Financeiros com Data Science e Machine Learning</div>
        <p class="academic-meta">
            <strong>Autora:</strong> Lorrana Evelyn de Araújo Pereira &nbsp;|&nbsp;
            <strong>Curso:</strong> Tecnologia em Gestão de Dados &nbsp;|&nbsp;
            <strong>Ativo em Foco:</strong> <span style="color: #93c5fd; font-family: 'JetBrains Mono', monospace;">{ticker_selecionado}</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Carregamento do ativo em foco
df_ativo = carregar_dados_ativo_cached(ticker_selecionado)

if df_ativo is None or df_ativo.empty:
    st.error(f"Não foi possível carregar os dados para o ativo {ticker_selecionado}.")
    st.stop()

dataset = preparar_dataset_cached(df_ativo)
# A inferência usa o scaler salvo junto aos modelos (o mesmo ajustado no treino)
X_test_scaled = scaler_salvo.transform(dataset.X_test) if tem_modelos else None

# =============================================================================
# NAVEGAÇÃO EM ABAS (ESTRUTURA METODOLÓGICA COMPLETA)
# =============================================================================
aba_visao, aba_eda, aba_modelos, aba_xai, aba_simulador, aba_relatorios = st.tabs(
    [
        "🏛️ Fundamentação & Metodologia",
        "📊 Análise Exploratória (EDA)",
        "⚙️ Desempenho dos Modelos",
        "🔍 Interpretabilidade (XAI)",
        "🔮 Simulador de Risco (t+1) & VaR",
        "📄 Relatórios & Downloads",
    ]
)

# =============================================================================
# ABA 1: FUNDAMENTAÇÃO & METODOLOGIA
# =============================================================================
with aba_visao:
    st.markdown("### 📌 Contextualização Teórica e Metodológica")

    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    with col_res1:
        st.markdown(
            f"""
            <div class="kpi-card kpi-best">
                <div class="kpi-label">Registros Históricos</div>
                <div class="kpi-value">{len(df_ativo):,}</div>
                <div class="kpi-sub">Pregões diários válidos</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_res2:
        dt_inicio = df_ativo["Data"].min().strftime("%d/%m/%Y")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Marco Inicial</div>
                <div class="kpi-value" style="font-size: 1.25rem;">{dt_inicio}</div>
                <div class="kpi-sub">Início da série temporal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_res3:
        dt_fim = df_ativo["Data"].max().strftime("%d/%m/%Y")
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Marco Final</div>
                <div class="kpi-value" style="font-size: 1.25rem;">{dt_fim}</div>
                <div class="kpi-sub">Data da última cotação</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_res4:
        vol_recente = df_ativo["Volatilidade_21d"].iloc[-1] * 100
        st.markdown(
            f"""
            <div class="kpi-card kpi-good">
                <div class="kpi-label">Volatilidade Atual (21d)</div>
                <div class="kpi-value">{vol_recente:.2f}%</div>
                <div class="kpi-sub">Anualizada no último pregão</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_teo1, col_teo2 = st.columns([1.1, 0.9])

    with col_teo1:
        st.markdown("#### 🎯 Problema de Pesquisa & Relevância")
        st.markdown(
            """
            A gestão de riscos e a quantificação da incerteza no mercado de capitais constituem um dos pilares 
            da moderna teoria de finanças e dos acordos regulatórios de Basileia. O presente Trabalho de Conclusão 
            de Curso (TCC) desenvolve e avalia um pipeline de Ciência de Dados ponta a ponta projetado para prever a 
            **volatilidade histórica anualizada no horizonte de 1 dia à frente ($t+1$)**.

            A abordagem soluciona criticidades fundamentais em **Gestão de Dados**:
            1. **Qualidade e Tratamento:** Ingestão de séries históricas higienizadas de proventos e desdobramentos.
            2. **Temporalidade Estrita:** Eliminação de viés de antecipação (*look-ahead bias*) e vazamento de dados (*data leakage*).
            3. **Transparência Algorítmica:** Aplicação de Inteligência Artificial Explicável (XAI) para auditoria e governança regulatória.
            """
        )

        st.markdown("#### 📐 Formulação Matemática")
        st.latex(r"R_t = \frac{P_t - P_{t-1}}{P_{t-1}} \quad \text{(Retorno Diário Aritmético)}")
        st.latex(
            r"\sigma_{t, 21d} = \sqrt{\frac{1}{20} \sum_{i=0}^{20} (R_{t-i} - \bar{R}_t)^2} \times \sqrt{252} \quad \text{(Volatilidade Anualizada)}"
        )
        st.latex(r"Y_t = \sigma_{t+1, 21d} \quad \text{(Variável-Alvo Modelada via shift(-1))}")

    with col_teo2:
        st.markdown("#### 🔄 Arquitetura do Fluxo Metodológico")
        st.markdown(
            """
            <div class="theory-box">
                <div class="theory-title">1. Coleta e Ingestão</div>
                Dados diários automáticos via Yahoo Finance API (2016 até hoje).
            </div>
            <div class="theory-box">
                <div class="theory-title">2. Análise Exploratória (EDA)</div>
                Fatos estilizados, testes de curtose leptocúrtica e matrizes de correlação.
            </div>
            <div class="theory-box">
                <div class="theory-title">3. Pré-processamento Temporal</div>
                Split sequencial 80% treino / 20% teste sem shuffle. StandardScaler ajustado apenas no treino.
            </div>
            <div class="theory-box">
                <div class="theory-title">4. Modelagem Supervisionada Multimodelo</div>
                XGBoost, LightGBM, Rede Neural MLP (Keras) e Baseline Ingênuo de persistência.
            </div>
            <div class="theory-box">
                <div class="theory-title">5. Validação e Avaliação Robusta</div>
                RMSE, MAE, R² e Validação Cruzada Temporal (TimeSeriesSplit com 5 folds).
            </div>
            <div class="theory-box">
                <div class="theory-title">6. Governança e XAI</div>
                Explicabilidade global e local com SHAP (Shapley Additive exPlanations) e LIME.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Tabela de Dicionário de Dados
    with st.expander("📖 Dicionário de Dados & Features Preditivas", expanded=False):
        dicionario = [
            {"Coluna": "Data", "Tipo": "datetime64", "Papel": "Eixo Temporal", "Interpretação": "Data oficial de encerramento do pregão da B3."},
            {"Coluna": "Close", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Preço de fechamento ajustado para dividendos e desdobramentos."},
            {"Coluna": "Volume", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Volume financeiro negociado no dia."},
            {"Coluna": "Retorno_Diario", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Variação percentual diária de preço."},
            {"Coluna": "Volatilidade_21d", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Volatilidade histórica anualizada em janela móvel de 21 dias úteis."},
            {"Coluna": "MM_20d", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Média móvel de 20 pregões (tendência de curto prazo)."},
            {"Coluna": "MM_50d", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Média móvel de 50 pregões (suporte/resistência de médio prazo)."},
            {"Coluna": "RSI_14d", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Índice de Força Relativa (oscilador de momentum [0, 100])."},
            {"Coluna": "Volume_MM20", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Média móvel de volume negociado em 20 dias."},
            {"Coluna": "Target_Vol", "Tipo": "float64", "Papel": "Variável-Alvo", "Interpretação": "Volatilidade observada no pregão seguinte (t+1)."},
        ]
        st.dataframe(pd.DataFrame(dicionario), use_container_width=True, hide_index=True)


# =============================================================================
# ABA 2: ANÁLISE EXPLORATÓRIA (EDA) & FATOS ESTILIZADOS
# =============================================================================
with aba_eda:
    st.markdown(f"### 📊 Análise Exploratória de Dados — {ticker_selecionado}")

    # Gráfico 1: Preço de Fechamento e Médias Móveis
    st.markdown("#### 1. Série Temporal de Preços e Tendências")
    fig_fech = go.Figure()
    fig_fech.add_trace(
        go.Scatter(
            x=df_ativo["Data"],
            y=df_ativo["Close"],
            name="Fechamento Ajustado",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    if "MM_20d" in df_ativo.columns:
        fig_fech.add_trace(
            go.Scatter(
                x=df_ativo["Data"],
                y=df_ativo["MM_20d"],
                name="Média Móvel 20d",
                line=dict(color="#ff7f0e", dash="dash"),
            )
        )
    if "MM_50d" in df_ativo.columns:
        fig_fech.add_trace(
            go.Scatter(
                x=df_ativo["Data"],
                y=df_ativo["MM_50d"],
                name="Média Móvel 50d",
                line=dict(color="#2ca02c", dash="dot"),
            )
        )

    fig_fech.update_layout(
        title=f"{ticker_selecionado} — Preço de Fechamento com Médias Móveis (20d e 50d)",
        xaxis_title="Data",
        yaxis_title="Preço / Pontos",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_fech, use_container_width=True)

    # Gráfico 2: Volatilidade Histórica Anualizada
    st.markdown("#### 2. Evolução da Volatilidade Histórica Anualizada (Janela Móvel 21d)")
    fig_vol = px.area(
        df_ativo,
        x="Data",
        y="Volatilidade_21d",
        title=f"Volatilidade Anualizada — {ticker_selecionado}",
        labels={"Volatilidade_21d": "Volatilidade (anualizada)"},
        color_discrete_sequence=["#d62728"],
        template="plotly_white",
        height=380,
    )
    fig_vol.update_traces(fillcolor="rgba(214, 39, 40, 0.15)")
    fig_vol.update_layout(hovermode="x unified")
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 3. Fatos Estilizados: Distribuição de Retornos e Hipótese de Normalidade")

    retornos = df_ativo["Retorno_Diario"].dropna()
    skewness = float(retornos.skew())
    kurtosis = float(retornos.kurtosis())

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Retorno Médio Diário", f"{retornos.mean()*100:.3f}%")
    with col_stat2:
        st.metric("Desvio Padrão Diário", f"{retornos.std()*100:.3f}%")
    with col_stat3:
        st.metric("Assimetria (Skewness)", f"{skewness:.4f}", help="Negativa indica assimetria com cauda à esquerda")
    with col_stat4:
        st.metric(
            "Curtose em Excesso",
            f"{kurtosis:.4f}",
            help="Curtose de Fisher (pandas): a Normal vale 0. Valores > 0 indicam distribuição leptocúrtica / caudas pesadas.",
        )

    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        # Histograma Interativo com Curva Normal de Referência
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=retornos,
                nbinsx=70,
                histnorm="probability density",
                name="Retornos Reais",
                marker_color="#2563eb",
                opacity=0.65,
            )
        )
        # Linha normal teórica
        x_norm = np.linspace(retornos.min(), retornos.max(), 200)
        y_norm = stats.norm.pdf(x_norm, loc=retornos.mean(), scale=retornos.std())
        fig_dist.add_trace(
            go.Scatter(
                x=x_norm,
                y=y_norm,
                name="Distribuição Normal Teórica",
                line=dict(color="#dc2626", width=2.5),
            )
        )
        fig_dist.update_layout(
            title=f"Distribuição Empírica vs. Normal Teórica — {ticker_selecionado}",
            xaxis_title="Retorno Diário",
            yaxis_title="Densidade",
            template="plotly_white",
            height=380,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_eda2:
        # Q-Q Plot Teórico
        qq_res = stats.probplot(retornos, dist="norm")
        teoricos = qq_res[0][0]
        ordenados = qq_res[0][1]
        slope, intercept, r = qq_res[1]

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=teoricos,
                y=ordenados,
                mode="markers",
                name="Quantis Amostrais",
                marker=dict(color="#1e3a8a", size=4, opacity=0.7),
            )
        )
        fig_qq.add_trace(
            go.Scatter(
                x=teoricos,
                y=slope * teoricos + intercept,
                mode="lines",
                name="Reta de Ajuste Normal",
                line=dict(color="#dc2626", width=2, dash="dash"),
            )
        )
        fig_qq.update_layout(
            title="Q-Q Plot (Normalidade de Retornos)",
            xaxis_title="Quantis Teóricos (Normal)",
            yaxis_title="Quantis Observados",
            template="plotly_white",
            height=380,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    st.info(
        f"💡 **Interpretação Acadêmica:** A curtose em excesso de **{kurtosis:.2f}** (a distribuição Normal vale 0; equivale a "
        f"uma curtose de {kurtosis + 3:.2f} contra 3 da Normal), associada ao afastamento das caudas no Q-Q Plot, "
        f"evidencia o fato estilizado financeiro das **caudas pesadas (leptocurtose)**: eventos extremos ocorrem com "
        f"frequência maior do que a Normal prevê. Por isso, medidas de risco baseadas na Normal, como o VaR paramétrico, "
        f"tendem a subestimar perdas extremas."
    )

    # Matriz de Correlação entre Ativos
    st.markdown("---")
    st.markdown("#### 4. Matriz de Correlação Cruzada entre Ativos")
    todos_dados = carregar_todos_dados_cached()
    if todos_dados:
        ret_dict = {}
        for t_nome, df_t in todos_dados.items():
            if "Retorno_Diario" in df_t.columns:
                nome_limpo = t_nome.replace(".SA", "").replace("^", "")
                ret_dict[nome_limpo] = df_t.set_index("Data")["Retorno_Diario"]

        df_ret_corr = pd.DataFrame(ret_dict).dropna()
        if not df_ret_corr.empty:
            corr_mat = df_ret_corr.corr().round(3)
            fig_corr = px.imshow(
                corr_mat,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlação dos Retornos Diários (Ativos B3 + S&P 500)",
                template="plotly_white",
                height=480,
            )
            st.plotly_chart(fig_corr, use_container_width=True)


# =============================================================================
# ABA 3: DESEMPENHO DOS MODELOS & VALIDAÇÃO
# =============================================================================
with aba_modelos:
    st.markdown("### ⚙️ Desempenho dos Modelos e Validação Temporal")

    if not tem_modelos:
        st.warning(
            f"Não há modelos treinados para **{ticker_selecionado}**. Modelos treinados em outro ativo "
            "não são aplicados aqui, pois a escala e o regime de volatilidade de cada série são distintos. "
            "Clique abaixo para treinar XGBoost, LightGBM e MLP com o split temporal 80/20 deste ativo."
        )
        if st.button("🚀 Treinar Modelos para este Ativo"):
            with st.spinner("Treinando XGBoost, LightGBM e Rede Neural MLP... Aguarde."):
                treinar_todos_modelos(
                    dataset,
                    salvar=True,
                    pasta_modelos=pasta_modelos_sel,
                    ticker_ref=ticker_selecionado,
                )
            st.cache_resource.clear()
            st.rerun()

    else:
        # Gera previsões usando os modelos salvos
        predicoes = predizer_com_modelos(modelos_dict, X_test_scaled)
        predicoes = adicionar_baseline_ingenuo(dataset.features, dataset.X_test, predicoes)

        # Cálculo das Métricas
        linhas_metricas = []
        for nome, y_hat in predicoes.items():
            m = calcular_metricas(nome, dataset.y_test, y_hat)
            linhas_metricas.append(m)

        df_metricas = pd.DataFrame(linhas_metricas).sort_values("RMSE").reset_index(drop=True)

        # Identifica o melhor modelo de ML
        eh_baseline = df_metricas["Modelo"].str.contains("Baseline", case=False)
        df_ml_only = df_metricas[~eh_baseline]
        melhor_ml = df_ml_only.iloc[0] if not df_ml_only.empty else df_metricas.iloc[0]
        rmse_baseline = float(df_metricas.loc[eh_baseline, "RMSE"].iloc[0]) if eh_baseline.any() else None
        melhor_geral = df_metricas.iloc[0]["Modelo"]

        # Exibição de KPIs dos Modelos
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.markdown(
                f"""
                <div class="kpi-card kpi-best">
                    <div class="kpi-label">Melhor Modelo ML</div>
                    <div class="kpi-value" style="font-size: 1.35rem;">{melhor_ml['Modelo']}</div>
                    <div class="kpi-sub">Menor erro médio de projeção</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_kpi2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">RMSE (Melhor ML)</div>
                    <div class="kpi-value">{melhor_ml['RMSE']:.5f}</div>
                    <div class="kpi-sub">{f"Baseline: {rmse_baseline:.5f}" if rmse_baseline is not None else "Root Mean Squared Error"}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_kpi3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">MAE (Melhor ML)</div>
                    <div class="kpi-value">{melhor_ml['MAE']:.5f}</div>
                    <div class="kpi-sub">Mean Absolute Error</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_kpi4:
            st.markdown(
                f"""
                <div class="kpi-card kpi-good">
                    <div class="kpi-label">Aderência R²</div>
                    <div class="kpi-value">{melhor_ml['R2']:.4f}</div>
                    <div class="kpi-sub">Coeficiente de determinação</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparação explícita com o benchmark de persistência (sigma_{t+1} = sigma_t)
        if rmse_baseline is not None:
            ganho_pct = (rmse_baseline - melhor_ml["RMSE"]) / rmse_baseline * 100
            if ganho_pct > 0:
                st.success(
                    f"O melhor modelo de ML (**{melhor_ml['Modelo']}**) reduziu o RMSE em **{ganho_pct:.1f}%** "
                    f"em relação ao Baseline Ingênuo de persistência ({rmse_baseline:.5f})."
                )
            else:
                st.warning(
                    f"O **Baseline Ingênuo de persistência** (σₜ₊₁ = σₜ) obteve RMSE de **{rmse_baseline:.5f}**, "
                    f"**{-ganho_pct:.1f}% menor** que o melhor modelo de ML ({melhor_ml['Modelo']}, "
                    f"{melhor_ml['RMSE']:.5f}). Neste conjunto de teste, os modelos de ML não superaram a regra de "
                    "persistência, o que é coerente com a forte autocorrelação da volatilidade em janela móvel de "
                    "21 dias (de um pregão para o seguinte, 20 dos 21 retornos da janela se repetem)."
                )

        modelos_r2_negativo = df_metricas.loc[df_metricas["R2"] < 0, "Modelo"].tolist()
        if modelos_r2_negativo:
            st.error(
                f"R² negativo em: {', '.join(modelos_r2_negativo)}. Esses modelos erram mais do que prever a "
                "média do período e não devem ser usados para conclusões."
            )

        st.markdown("#### 1. Tabela Comparativa de Métricas de Regressão")

        # Formatação da Tabela
        def classificar_modelo(nome: str) -> str:
            rotulos = []
            if nome == melhor_geral:
                rotulos.append("🏆 Menor RMSE Geral")
            if nome == melhor_ml["Modelo"]:
                rotulos.append("⭐ Melhor Modelo ML")
            if "Baseline" in nome:
                rotulos.append("📌 Benchmark de Persistência")
            return " • ".join(rotulos) if rotulos else "—"

        df_disp = df_metricas.copy()
        df_disp["Classificação"] = df_disp["Modelo"].apply(classificar_modelo)
        df_disp["RMSE"] = df_disp["RMSE"].map(lambda x: f"{x:.5f}")
        df_disp["MAE"] = df_disp["MAE"].map(lambda x: f"{x:.5f}")
        df_disp["R2"] = df_disp["R2"].map(lambda x: f"{x:.4f}")

        st.dataframe(df_disp, use_container_width=True, hide_index=True)

        # Gráfico Comparativo: Real vs Previsões
        st.markdown("---")
        st.markdown("#### 2. Visualização Comparativa da Série Temporal: Real vs. Modelos")

        modelos_disponiveis = list(predicoes.keys())
        modelos_selecionados = st.multiselect(
            "Selecione os modelos a exibir no gráfico:",
            options=modelos_disponiveis,
            default=[melhor_ml["Modelo"], "Baseline Ingênuo (Persistência)"],
        )

        datas_dt = pd.to_datetime(dataset.datas_test)
        fig_comp = go.Figure()

        # Linha Real
        fig_comp.add_trace(
            go.Scatter(
                x=datas_dt,
                y=dataset.y_test,
                name="Volatilidade Real Observada",
                line=dict(color="#0f172a", width=2.5),
            )
        )

        estilos_graf = {
            "XGBoost": dict(color="#2563eb", dash="solid", width=1.8),
            "LightGBM": dict(color="#10b981", dash="dash", width=1.8),
            "Rede Neural (MLP)": dict(color="#8b5cf6", dash="dot", width=1.8),
            "Baseline Ingênuo (Persistência)": dict(color="#94a3b8", dash="dashdot", width=1.5),
        }

        for m_nome in modelos_selecionados:
            y_p = predicoes[m_nome]
            est = estilos_graf.get(m_nome, dict(width=1.5))
            fig_comp.add_trace(
                go.Scatter(
                    x=datas_dt,
                    y=y_p,
                    name=m_nome,
                    line=est,
                )
            )

        fig_comp.update_layout(
            title="Projeção da Volatilidade — Real vs. Modelos Preditivos (Conjunto de Teste - 20%)",
            xaxis_title="Data do Pregão",
            yaxis_title="Volatilidade Anualizada",
            hovermode="x unified",
            template="plotly_white",
            height=480,
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Validação Cruzada Temporal e Curva de Treino
        col_val1, col_val2 = st.columns(2)
        with col_val1:
            st.markdown("#### 3. Validação Cruzada Temporal (TimeSeriesSplit)")
            st.markdown(
                """
                Para avaliar se o modelo de Gradient Boosting possui consistência ao longo de diferentes 
                ciclos e crises financeiras, aplicou-se a divisão progressiva em **5 folds temporais (sem shuffle)**:
                """
            )
            folds_data = validacao_cruzada_cached(
                ticker_selecionado, str(df_ativo["Data"].max().date()), dataset
            )
            fig_folds = px.bar(
                folds_data,
                x="Fold",
                y="RMSE",
                text="RMSE",
                hover_data=["Período de Validação"],
                title=f"RMSE por Fold Temporal (TimeSeriesSplit 5) — {ticker_selecionado}",
                color="RMSE",
                color_continuous_scale="Blues",
                template="plotly_white",
                height=320,
            )
            fig_folds.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig_folds, use_container_width=True)

            fold_pior = folds_data.loc[folds_data["RMSE"].idxmax()]
            st.caption(
                f"RMSE médio: {folds_data['RMSE'].mean():.5f} ± {folds_data['RMSE'].std(ddof=0):.5f}. "
                f"Maior erro no {fold_pior['Fold']} ({fold_pior['Período de Validação']}). "
                "Os folds percorrem a série completa (treino + teste), sempre treinando no passado e validando no futuro."
            )

        with col_val2:
            st.markdown("#### 4. Curva de Aprendizado — Rede Neural MLP")
            historico_nn = meta_dict.get("historico_nn") or {}
            if historico_nn.get("loss") and historico_nn.get("val_loss"):
                epocas = list(range(1, len(historico_nn["loss"]) + 1))
                fig_nn = go.Figure()
                fig_nn.add_trace(
                    go.Scatter(x=epocas, y=historico_nn["loss"], name="Treino (MSE)", line=dict(color="#2563eb", width=1.8))
                )
                fig_nn.add_trace(
                    go.Scatter(x=epocas, y=historico_nn["val_loss"], name="Validação (MSE)", line=dict(color="#ea580c", width=1.8))
                )
                melhor_epoca = int(np.argmin(historico_nn["val_loss"])) + 1
                fig_nn.add_vline(x=melhor_epoca, line=dict(color="#94a3b8", dash="dash"))
                fig_nn.update_layout(
                    title=f"Perda MSE por Época — {ticker_selecionado}",
                    xaxis_title="Época",
                    yaxis_title="MSE",
                    yaxis_type="log",
                    template="plotly_white",
                    height=320,
                    legend=dict(orientation="h", y=1.08),
                )
                st.plotly_chart(fig_nn, use_container_width=True)
                st.caption(
                    f"Early Stopping restaurou os pesos da época {melhor_epoca} (menor perda de validação, "
                    "medida nos últimos 15% do conjunto de treino)."
                )
            else:
                st.info(
                    "Histórico de treino da MLP não registrado para este ativo. "
                    "Treine os modelos novamente para gerar a curva."
                )


# =============================================================================
# ABA 4: INTERPRETABILIDADE & GOVERNANÇA (XAI)
# =============================================================================
with aba_xai:
    st.markdown("### 🔍 Inteligência Artificial Explicável (XAI) & Governança")
    st.markdown(
        """
        Em conformidade com as diretrizes regulatórias do mercado de capitais e princípios de **Governança de Dados**, 
        modelos preditivos aplicados a riscos financeiros não podem operar como "caixas-pretas". 
        Utilizam-se as abordagens **SHAP (*SHapley Additive exPlanations*)** e **LIME** para auditoria quantitativa.
        """
    )

    # As figuras SHAP/LIME são geradas por main.py apenas para o ativo de referência
    figuras_xai_validas = ticker_selecionado == TICKER_REF
    if not figuras_xai_validas:
        st.info(
            f"As figuras SHAP e LIME foram geradas pelo pipeline (main.py) para o ativo de referência "
            f"**{TICKER_REF}** e não são exibidas para {ticker_selecionado}."
        )

    col_xai1, col_xai2 = st.columns(2)
    with col_xai1:
        st.markdown("#### 1. Importância Global das Variáveis (SHAP Bar)")
        img_shap_bar = FIGURES_DIR / "shap_importancia_global_bar.png"
        if figuras_xai_validas and img_shap_bar.exists():
            st.image(str(img_shap_bar), caption="Média absoluta dos valores SHAP (Impacto Global)", use_container_width=True)
        else:
            st.caption("Gráfico SHAP Bar indisponível para este ativo.")

    with col_xai2:
        st.markdown("#### 2. Direção e Dispersão do Impacto (SHAP Beeswarm)")
        img_shap_bee = FIGURES_DIR / "shap_beeswarm.png"
        if figuras_xai_validas and img_shap_bee.exists():
            st.image(str(img_shap_bee), caption="Distribuição do impacto: valores altos (vermelho) vs baixos (azul)", use_container_width=True)
        else:
            st.caption("Gráfico SHAP Beeswarm indisponível para este ativo.")

    st.markdown("---")
    st.markdown("#### 3. Simulador Local Interativo: Decomposição da Previsão de um Pregão Específico")

    datas_test_dt = pd.to_datetime(dataset.datas_test)

    idx_selecionado = st.slider(
        "Selecione o índice da amostra de teste:",
        min_value=0,
        max_value=len(dataset.datas_test) - 1,
        value=10,
        help="Permite inspecionar a previsão individual do modelo para qualquer dia da base de teste.",
    )

    data_amostra = datas_test_dt[idx_selecionado].strftime("%d/%m/%Y")
    val_real = dataset.y_test[idx_selecionado]

    # Previsão pontual do XGBoost
    if tem_modelos and "XGBoost" in modelos_dict:
        m_xgb = modelos_dict["XGBoost"]
        amostra_scaled = X_test_scaled[idx_selecionado].reshape(1, -1)
        pred_xgb_dia = float(m_xgb.predict(amostra_scaled)[0])

        col_loc1, col_loc2, col_loc3 = st.columns(3)
        with col_loc1:
            st.metric("Data do Pregão Analisado", data_amostra)
        with col_loc2:
            st.metric("Volatilidade Real (t+1)", f"{val_real:.4f} ({val_real*100:.2f}%)")
        with col_loc3:
            # Erro de previsão em qualquer direção não é "bom" nem "ruim": delta em cor neutra
            st.metric(
                "Previsão do XGBoost",
                f"{pred_xgb_dia:.4f} ({pred_xgb_dia*100:.2f}%)",
                delta=f"Erro (previsto − real): {(pred_xgb_dia - val_real):+.4f}",
                delta_color="off",
            )

    # Exibe imagem do Waterfall e LIME se for o índice 10 pré-gerado
    col_wf, col_lime = st.columns(2)
    with col_wf:
        img_wf = FIGURES_DIR / f"shap_waterfall_amostra_10.png"
        if figuras_xai_validas and img_wf.exists():
            st.image(str(img_wf), caption=f"SHAP Waterfall Plot (Índice de Amostra #10 - {datas_test_dt[10]:%d/%m/%Y})", use_container_width=True)
        else:
            st.caption("SHAP Waterfall indisponível para este ativo.")

    with col_lime:
        img_lime = FIGURES_DIR / f"lime_explicacao_amostra_10.png"
        if figuras_xai_validas and img_lime.exists():
            st.image(str(img_lime), caption="LIME - Explicação Linear Local (Índice #10)", use_container_width=True)
        else:
            st.caption("Gráfico LIME indisponível para este ativo.")


# =============================================================================
# ABA 5: SIMULADOR DE RISCO (t+1) & VALUE AT RISK (VaR)
# =============================================================================
with aba_simulador:
    st.markdown("### 🔮 Projeção de Risco para o Próximo Pregão ($t+1$) & Value at Risk (VaR)")
    st.markdown(
        """
        Esta ferramenta utiliza os **dados mais recentes do mercado** disponíveis na base de dados para projetar 
        a volatilidade esperada para o próximo pregão e quantificar a perda máxima provável (VaR) sob a ótica 
        de gestão de portfólio.
        """
    )

    if not (tem_modelos and "XGBoost" in modelos_dict):
        st.info(f"Treine os modelos de {ticker_selecionado} na aba **Desempenho dos Modelos** para habilitar o simulador.")
    else:
        m_xgb = modelos_dict["XGBoost"]
        # Último pregão com todas as features calculadas. O conjunto de teste não serve aqui:
        # o shift(-1) do alvo descarta justamente a última linha da série, que ainda não tem t+1.
        df_features_atuais = df_ativo[["Data"] + dataset.features].dropna()
        ultima_linha = df_features_atuais.iloc[-1]
        x_ultimo = ultima_linha[dataset.features].to_numpy(dtype=float).reshape(1, -1)
        vol_predita = float(m_xgb.predict(scaler_salvo.transform(x_ultimo))[0])
        data_base = pd.Timestamp(ultima_linha["Data"]).strftime("%d/%m/%Y")
        vol_pct = vol_predita * 100

        # Classificação do Regime de Risco relativa ao histórico do próprio ativo:
        # limites fixos não servem para séries com níveis de volatilidade diferentes (índice x ação)
        vol_hist_pct = df_ativo["Volatilidade_21d"].dropna() * 100
        p25, p75, p95 = np.percentile(vol_hist_pct, [25, 75, 95])
        percentil_atual = float((vol_hist_pct < vol_pct).mean() * 100)
        if vol_pct < p25:
            regime = "Baixo Risco (Mercado Calmo)"
            estilo_regime = "risk-low"
            desc_regime = f"Abaixo do percentil 25 histórico do ativo ({p25:.1f}%). Oscilação inferior à usual."
        elif vol_pct < p75:
            regime = "Risco Moderado (Normalidade Histórica)"
            estilo_regime = "risk-moderate"
            desc_regime = f"Entre os percentis 25 e 75 históricos do ativo ({p25:.1f}% a {p75:.1f}%)."
        elif vol_pct < p95:
            regime = "Risco Elevado (Atenção / Estresse)"
            estilo_regime = "risk-high"
            desc_regime = (
                f"Entre os percentis 75 e 95 históricos do ativo ({p75:.1f}% a {p95:.1f}%). "
                "Sugere recalibração de stop loss e cautela em alavancagem."
            )
        else:
            regime = "Risco Extremo (Regime de Crise)"
            estilo_regime = "risk-extreme"
            desc_regime = (
                f"Acima do percentil 95 histórico do ativo ({p95:.1f}%). "
                "Típico de choques macroeconômicos e crises agudas."
            )
        desc_regime += f" A projeção supera {percentil_atual:.0f}% dos pregões desde {df_ativo['Data'].min():%Y}."

        col_sim1, col_sim2 = st.columns([1, 1.2])

        with col_sim1:
            st.markdown(
                f"""
                <div class="kpi-card kpi-best">
                    <div class="kpi-label">Volatilidade Projetada (t+1)</div>
                    <div class="kpi-value">{vol_pct:.2f}%</div>
                    <div class="kpi-sub">Projeção anualizada para o pregão seguinte a {data_base}</div>
                    <br>
                    <div class="risk-pill {estilo_regime}">{regime}</div>
                    <p style="font-size: 0.85rem; color: #475569; margin-top: 0.8rem;">{desc_regime}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_sim2:
            st.markdown("#### 💰 Calculadora de Value at Risk (VaR Paramétrico)")
            patrimonio = st.number_input(
                "Patrimônio Alocado no Ativo (R$):",
                min_value=1000.0,
                max_value=100_000_000.0,
                value=100_000.0,
                step=5000.0,
                format="%.2f",
            )

            # Volatilidade diária (desanualizada dividindo por sqrt(252))
            vol_diaria = vol_predita / np.sqrt(252)

            # VaR paramétrico para 1 dia: VaR = Z * sigma_diaria * Patrimonio
            z_95 = 1.644853  # Nível 95%
            z_99 = 2.326348  # Nível 99%

            var_95 = z_95 * vol_diaria * patrimonio
            var_99 = z_99 * vol_diaria * patrimonio

            col_var1, col_var2 = st.columns(2)
            with col_var1:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">VaR Diário (95% Confiança)</div>
                        <div class="kpi-value" style="color: #ea580c; font-size: 1.45rem;">R$ {var_95:,.2f}</div>
                        <div class="kpi-sub">Perda máxima diária esperada em 95% dos pregões</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_var2:
                st.markdown(
                    f"""
                    <div class="kpi-card kpi-alert">
                        <div class="kpi-label">VaR Diário (99% Confiança)</div>
                        <div class="kpi-value" style="color: #dc2626; font-size: 1.45rem;">R$ {var_99:,.2f}</div>
                        <div class="kpi-sub">Perda máxima esperada em 99% dos pregões (cenário adverso)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "Nota Metodológica: O cálculo do Value at Risk paramétrico utiliza o fator de desanualização padrão do mercado financeiro nacional "
            f"(252 dias úteis) e pressupõe normalidade dos retornos em curtíssimo prazo, calibrada pela volatilidade dinâmica estimada pelo XGBoost."
        )


# =============================================================================
# ABA 6: RELATÓRIOS & DOWNLOADS
# =============================================================================
with aba_relatorios:
    st.markdown("### 📄 Relatório Executivo e Exportação de Artefatos")

    # Exibição do Relatório em Texto
    caminho_rel = FIGURES_DIR / "relatorio_final_modelos.txt"
    if caminho_rel.exists():
        with open(caminho_rel, "r", encoding="utf-8") as f:
            conteudo_rel = f.read()

        st.markdown("#### 1. Relatório Consolidado de Desempenho")
        st.code(conteudo_rel, language="text")
    else:
        conteudo_rel = "Relatório ainda não gerado."

    st.markdown("---")
    st.markdown("#### 2. Download Centralizado de Arquivos do Projeto")

    col_dl1, col_dl2, col_dl3 = st.columns(3)

    with col_dl1:
        st.download_button(
            label="📥 Baixar Relatório Final (.txt)",
            data=conteudo_rel,
            file_name="relatorio_final_modelos.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col_dl2:
        caminho_csv = FIGURES_DIR / "metricas_modelos.csv"
        if caminho_csv.exists():
            with open(caminho_csv, "rb") as f:
                csv_bytes = f.read()
            st.download_button(
                label="📥 Baixar Tabela de Métricas (.csv)",
                data=csv_bytes,
                file_name="metricas_modelos.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Tabela CSV Indisponível", disabled=True, use_container_width=True)

    with col_dl3:
        csv_ativo = df_ativo.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label=f"📥 Baixar Dados de {ticker_selecionado} (.csv)",
            data=csv_ativo,
            file_name=f"dados_enriquecidos_{ticker_selecionado}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Catálogo de Imagens e Figuras Salvas
    st.markdown("---")
    st.markdown("#### 3. Catálogo de Figuras Geradas em Alta Resolução")
    imagens_salvas = list(FIGURES_DIR.glob("*.png"))
    if imagens_salvas:
        cols_img = st.columns(3)
        for i, img_p in enumerate(imagens_salvas):
            with cols_img[i % 3]:
                st.caption(f"**{img_p.name}**")
                st.image(str(img_p), use_container_width=True)
                with open(img_p, "rb") as f_img:
                    st.download_button(
                        label=f"Baixar {img_p.name}",
                        data=f_img.read(),
                        file_name=img_p.name,
                        mime="image/png",
                        key=f"dl_img_{i}",
                        use_container_width=True,
                    )

# =============================================================================
# RODAPÉ ACADÊMICO
# =============================================================================
st.markdown(
    """
    <div class="academic-footer">
        Universidade Federal do Piauí (UFPI) • Centro de Educação Aberta e a Distância (CEAD)<br>
        Curso Superior de Tecnologia em Gestão de Dados • TCC Lorrana Evelyn de Araújo Pereira • 2026<br>
        <em>Desenvolvido com Python, Streamlit, Scikit-Learn, XGBoost, LightGBM, Keras e SHAP</em>
    </div>
    """,
    unsafe_allow_html=True,
)
