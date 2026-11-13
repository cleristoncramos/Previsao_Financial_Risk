"""
=============================================================================
PAINEL DE ANÁLISE — PREVISÃO DE RISCOS FINANCEIROS
Página do painel (rota /painel). A configuração da página, o estilo e a
navegação ficam em app.py; as informações acadêmicas, na página inicial.
=============================================================================
"""

import base64
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
import plotly.io as pio
import seaborn as sns
import streamlit as st
import lime.lime_tabular
import shap
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

# Configurações do Diretório e Importações do Projeto (este arquivo fica em paginas/)
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    ATIVOS,
    DATA_DIR,
    DATA_FIM,
    DATA_INICIO,
    FIGURES_DIR,
    RANDOM_STATE,
    agora_brasilia,
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
from src.figuras import (
    COR_REAL,
    CORES_MODELOS,
    figuras_avaliacao,
    figuras_exploratorias,
    figuras_interpretabilidade,
    formatar_condicao_lime,
    triangulo_inferior,
)
from src.textos_graficos import texto_grafico
from src.model_store import (
    carregar_modelos_salvos,
    modelos_existem,
    pasta_modelos_ativo,
    predizer_com_modelos,
)
from src.models import treinar_todos_modelos
from src.preprocessing import DatasetModelagem, preparar_dados_modelagem
from src.report import dados_para_csv, metricas_para_csv, montar_texto_relatorio

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


@st.cache_data(show_spinner="Calculando valores SHAP...")
def shap_cached(ticker: str, versao_modelo: str, _modelo, _X_test_scaled: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Valores SHAP (TreeExplainer) do XGBoost no conjunto de teste do ativo.
    O cache é indexado por ticker e data de salvamento do modelo.
    """
    explainer = shap.TreeExplainer(_modelo)
    valores = np.asarray(explainer.shap_values(_X_test_scaled))
    base = explainer.expected_value
    base = float(base[0]) if isinstance(base, (list, np.ndarray)) else float(base)
    return valores, base


@st.cache_resource(show_spinner=False)
def explicador_lime_cached(ticker: str, versao_modelo: str, _X_train: np.ndarray, features: Tuple[str, ...]):
    """Explicador LIME ajustado nos dados de treino em escala original (rótulos legíveis)."""
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=_X_train,
        feature_names=list(features),
        mode="regression",
        random_state=RANDOM_STATE,
    )


@st.cache_data(show_spinner="Calculando explicação LIME...")
def lime_cached(ticker: str, versao_modelo: str, idx: int, _explicador, _modelo, _scaler, _x_bruto: np.ndarray) -> List[Tuple[str, float]]:
    """Pesos LIME de uma amostra; o modelo recebe os dados já padronizados pelo scaler salvo."""
    exp = _explicador.explain_instance(
        _x_bruto,
        lambda X: _modelo.predict(_scaler.transform(X)),
        num_features=len(_x_bruto),
    )
    return exp.as_list()


@st.cache_data(show_spinner="Gerando figuras exploratórias do ativo...")
def figuras_exploratorias_cached(ticker: str, ultima_data: str, _df: pd.DataFrame, _retornos: Dict[str, pd.Series]):
    """Figuras PNG da EDA do ativo; o cache é indexado por ticker e data final da série."""
    return figuras_exploratorias(_df, ticker, _retornos)


@st.cache_data(show_spinner="Gerando figuras de avaliação dos modelos...")
def figuras_avaliacao_cached(ticker: str, ultima_data: str, versao_modelo: str, _datas, _y, _predicoes, _folds, _historico):
    """Figuras PNG de desempenho; o cache é indexado por ticker, série e versão do modelo."""
    return figuras_avaliacao(ticker, _datas, _y, _predicoes, _folds, _historico)


@st.cache_data(show_spinner="Gerando figuras SHAP e LIME...")
def figuras_interpretabilidade_cached(
    ticker: str, ultima_data: str, versao_modelo: str,
    _shap, _base, _X_scaled, _X, _features, idx: int, data_amostra: str, _lime,
):
    """Figuras PNG de interpretabilidade; o cache é indexado por ticker, série, modelo e pregão escolhido."""
    return figuras_interpretabilidade(ticker, _shap, _base, _X_scaled, _X, _features, idx, data_amostra, _lime)


# =============================================================================
# PADRÕES VISUAIS: PAINÉIS DE SEÇÃO E GRÁFICOS
# =============================================================================
def secao(chave: str):
    """Painel com borda fina que agrupa uma seção (estilizado por [class*="st-key-secao"])."""
    return st.container(border=True, key=f"secao_{chave}")


def linha(chave: str):
    """Agrupamento sem borda (linha de KPIs, avisos) com espaço abaixo."""
    return st.container(key=f"linha_{chave}")


# Template Plotly único: fundo branco, grade discreta, eixos com linha fina escura
CORES_GRAFICO = ["#1e3a8a", "#0f766e", "#b45309", "#7c3aed", "#6b7280", "#9f1239"]
_eixo_padrao = dict(
    showline=True,
    linecolor="#3f4654",
    linewidth=1,
    mirror=False,
    gridcolor="#eceef2",
    zeroline=False,
    ticks="outside",
    tickcolor="#3f4654",
    title=dict(font=dict(size=12, color="#5b6472")),
    # Números por extenso com ponto no milhar (50.000), sem a abreviação inglesa "50k"
    exponentformat="none",
    separatethousands=True,
)
pio.templates["painel"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, sans-serif", size=12.5, color="#1f2937"),
        title=dict(font=dict(size=14.5, color="#172554"), x=0, xanchor="left", y=0.97, yanchor="top"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        colorway=CORES_GRAFICO,
        xaxis=_eixo_padrao,
        yaxis=_eixo_padrao,
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, xanchor="left", bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=64, r=24, t=64, b=96),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#3f4654", font=dict(family="Inter, sans-serif")),
        # Padrão brasileiro em eixos, rótulos e tooltips: vírgula decimal e ponto no milhar
        separators=",.",
    )
)
pio.templates.default = "plotly_white+painel"


def exibir_grafico(fig: go.Figure) -> None:
    """Exibe o gráfico com o template do painel (theme=None evita o tema padrão do Streamlit)."""
    # O Streamlit aplica a cor de fundo da página ao gráfico; o painel exige fundo branco
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    # Eixo de datas em formato numérico brasileiro (o Plotly só traz nomes de meses em inglês)
    eixo_x = getattr(fig.data[0], "x", None) if fig.data else None
    if eixo_x is not None and len(eixo_x) and isinstance(
        pd.Series(eixo_x).iloc[0], (pd.Timestamp, np.datetime64, datetime)
    ):
        fig.update_xaxes(tickformat="%m/%Y", hoverformat="%d/%m/%Y")
    st.plotly_chart(fig, use_container_width=True, theme=None)


def formatar_numero(valor: float, casas: int = 2) -> str:
    """Número no padrão brasileiro: ponto no milhar e vírgula nos decimais (ex.: 1.234,56)."""
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def como_ler(chave: str, **valores) -> None:
    """Botão discreto que abre a explicação do gráfico em linguagem simples (src/textos_graficos.py)."""
    with st.popover("Como ler este gráfico", icon=":material/lightbulb:", type="tertiary"):
        st.markdown(texto_grafico(chave, **valores))


def formatar_reais(valor: float) -> str:
    """Valor monetário no padrão brasileiro: R$ 1.234.567,89."""
    return f"R$ {formatar_numero(valor, 2)}"


# =============================================================================
# EVENTOS HISTÓRICOS: marcadores nos gráficos de série temporal
# =============================================================================
EVENTOS_CSV = BASE_DIR / "assets" / "eventos.csv"
CATEGORIA_USUARIO = "Seus eventos"
CORES_EVENTOS = {
    "Mercado e crises": "#b91c1c",
    "Saúde": "#c2410c",
    "Política": "#7c3aed",
    "Mundo": "#0f766e",
    CATEGORIA_USUARIO: "#111827",
}
COLUNAS_EVENTOS = ["data", "categoria", "titulo", "descricao"]


@st.cache_data(show_spinner=False)
def carregar_eventos_cached() -> pd.DataFrame:
    """Base de eventos de assets/eventos.csv (data, categoria, título e descrição)."""
    if not EVENTOS_CSV.exists():
        return pd.DataFrame(columns=COLUNAS_EVENTOS)
    eventos = pd.read_csv(EVENTOS_CSV, parse_dates=["data"], encoding="utf-8")
    return eventos[COLUNAS_EVENTOS]


def eventos_do_usuario() -> pd.DataFrame:
    """Eventos adicionados pelo usuário nesta sessão do navegador."""
    registros = st.session_state.get("eventos_usuario", [])
    if not registros:
        return pd.DataFrame(columns=COLUNAS_EVENTOS)
    eventos = pd.DataFrame(registros, columns=COLUNAS_EVENTOS)
    eventos["data"] = pd.to_datetime(eventos["data"])
    return eventos


def adicionar_eventos(fig: go.Figure, eventos: pd.DataFrame, y_min: float, y_max: float) -> None:
    """
    Desenha cada evento como uma linha vertical pontilhada, agrupada por categoria: cada
    categoria é uma única série na legenda, que liga e desliga com um clique, e o título
    e a descrição aparecem ao passar o mouse sobre a linha.
    """
    # Ordem fixa das categorias na legenda (a de CORES_EVENTOS)
    for categoria in sorted(eventos["categoria"].unique(), key=lambda c: list(CORES_EVENTOS).index(c) if c in CORES_EVENTOS else 99):
        grupo = eventos[eventos["categoria"] == categoria]
        xs, ys, textos = [], [], []
        for _, ev in grupo.sort_values("data").iterrows():
            texto = f"<b>{ev['titulo']}</b>"
            if isinstance(ev["descricao"], str) and ev["descricao"].strip():
                texto += f"<br>{ev['descricao']}"
            xs += [ev["data"], ev["data"], None]
            ys += [y_min, y_max, None]
            textos += [texto, texto, ""]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=categoria,
                legendgroup=f"evento_{categoria}",
                line=dict(color=CORES_EVENTOS.get(categoria, "#111827"), width=1.2, dash="dot"),
                opacity=0.75,
                customdata=textos,
                hovertemplate="%{customdata}<extra></extra>",
                connectgaps=False,
            )
        )


ICONES_DIR = BASE_DIR / "assets" / "ativos"
TIPOS_ICONE = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml"}


def icone_ativo_html(ticker: str) -> str:
    """
    Logotipo do ativo (<img> embutida em base64) a partir de assets/ativos/<TICKER>.<ext>.
    Os logotipos vêm da Wikimedia Commons / Wikipédia (o do Ibovespa é o da B3, que calcula o índice).
    Procura PNG, depois JPG, depois SVG.
    """
    nome = ticker.replace("^", "").replace(".SA", "")
    for extensao in (".png", ".jpg", ".svg"):
        caminho = ICONES_DIR / f"{nome}{extensao}"
        if caminho.exists():
            conteudo = base64.b64encode(caminho.read_bytes()).decode("ascii")
            return (
                f'<img class="ativo-icone" src="data:{TIPOS_ICONE[extensao]};base64,{conteudo}" '
                f'alt="Miniatura de {ticker}">'
            )
    return ""


# =============================================================================
# BARRA LATERAL (SIDEBAR)
# =============================================================================
with st.sidebar:
    # Ícone discreto de início: volta à página inicial (funciona como "sair" do painel)
    st.page_link("paginas/inicio.py", label="Início", icon=":material/home:")
    st.markdown('<div class="sidebar-titulo">Seleção de Ativo</div>', unsafe_allow_html=True)

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
    # Nome entre parênteses na opção, ex.: "PETR4.SA (Petrobras PN)" -> "Petrobras PN"
    nome_ativo_selecionado = opcao_selecionada.split("(", 1)[1].rstrip(")")

    # Status dos Modelos: cada ativo tem seus próprios modelos e scaler, pois um
    # modelo treinado em uma série não é válido para outra (escalas e regimes distintos)
    pasta_modelos_sel = pasta_modelos_ativo(ticker_selecionado)
    modelos_dict, scaler_salvo, meta_dict = carregar_modelos_cached(str(pasta_modelos_sel))
    tem_modelos = (
        bool(modelos_dict)
        and scaler_salvo is not None
        and meta_dict.get("ticker_ref") == ticker_selecionado
    )

    # Recursos técnicos ficam ocultos por padrão, para o usuário leigo não esbarrar neles
    modo_pesquisador = st.toggle(
        "Modo pesquisador",
        key="modo_pesquisador",
        help="Mostra os recursos técnicos: validação cruzada temporal, curva de aprendizado da rede neural, "
        "SHAP beeswarm e explicação LIME.",
    )

    st.markdown('<div class="sidebar-titulo" style="margin-top: 1.75rem;">Estado da Modelagem</div>', unsafe_allow_html=True)
    if tem_modelos:
        st.success(f"Modelos de {ticker_selecionado} carregados")
        data_salvo = meta_dict.get("data_salvamento", "Recente")
        st.caption(f"Última persistência: {data_salvo}")
    else:
        st.warning(f"Não há modelos treinados para {ticker_selecionado}.")

    # Botão de Ação: Atualizar Dados (novo download no Yahoo Finance e regravação dos CSVs)
    if st.button("Atualizar Séries Temporais", use_container_width=True):
        # O parâmetro end do yfinance é exclusivo: amanhã garante a inclusão do pregão de hoje
        fim_coleta = (agora_brasilia() + timedelta(days=1)).strftime("%Y-%m-%d")
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

# =============================================================================
# SAIBA MAIS: resumo teórico e metodológico, fora das abas
# =============================================================================
URL_REPOSITORIO = "https://github.com/cleristoncramos/Previsao_Financial_Risk"


@st.dialog("Saiba mais", width="large")
def saiba_mais():
    st.markdown(
        """
        **O que o painel faz.** Prevê a volatilidade do pregão seguinte ($t+1$) de ações da B3 e índices de
        mercado, compara modelos de Machine Learning com uma regra simples de referência, explica cada
        previsão (XAI) e traduz o risco projetado em Value at Risk.
        """
    )

    col_formulas, col_fluxo = st.columns(2, gap="large")
    with col_formulas:
        st.markdown("##### Como a volatilidade é calculada")
        st.caption("Retorno diário")
        st.latex(r"R_t = \frac{P_t - P_{t-1}}{P_{t-1}}")
        st.caption("Volatilidade anualizada (janela de 21 pregões)")
        st.latex(r"\sigma_{t} = \text{desvio-padrão}(R_{t-20}, \dots, R_t) \times \sqrt{252}")
        st.caption("O que os modelos preveem")
        st.latex(r"Y_t = \sigma_{t+1}")
    with col_fluxo:
        st.markdown("##### Etapas")
        st.markdown(
            """
            1. **Coleta:** séries diárias do Yahoo Finance, de 2016 até hoje.
            2. **Análise exploratória:** tendência, volatilidade, caudas pesadas e correlação.
            3. **Preparação:** 80% iniciais para treino e 20% finais para teste, sem embaralhar.
            4. **Modelos:** XGBoost, LightGBM e Rede Neural MLP, comparados à regra de persistência
               (amanhã igual a hoje).
            5. **Avaliação:** RMSE, MAE, R² e validação cruzada temporal.
            6. **Explicabilidade:** SHAP e LIME mostram o peso de cada variável.
            """
        )

    with st.expander("Dicionário de dados"):
        dicionario = [
            {"Coluna": "Close", "Significado": "Preço de fechamento ajustado por proventos e desdobramentos"},
            {"Coluna": "Volume", "Significado": "Quantidade de ações (ou contratos) negociada no pregão"},
            {"Coluna": "Retorno_Diario", "Significado": "Variação percentual do fechamento em relação ao pregão anterior"},
            {"Coluna": "Volatilidade_21d", "Significado": "Volatilidade anualizada dos últimos 21 pregões"},
            {"Coluna": "MM_20d / MM_50d", "Significado": "Médias móveis de 20 e 50 pregões (tendência de curto e médio prazo)"},
            {"Coluna": "RSI_14d", "Significado": "Índice de Força Relativa, de 0 a 100 (acima de 70, sobrecompra; abaixo de 30, sobrevenda)"},
            {"Coluna": "Volume_MM20", "Significado": "Média do volume dos últimos 20 pregões"},
            {"Coluna": "Target_Vol", "Significado": "Volatilidade do pregão seguinte: o valor que os modelos preveem"},
        ]
        st.dataframe(pd.DataFrame(dicionario), use_container_width=True, hide_index=True)

    st.markdown(
        f"Para ler mais: [README]({URL_REPOSITORIO}#readme) · "
        f"[Documentação técnica]({URL_REPOSITORIO}/blob/main/DOCUMENTACAO.md) · "
        f"[Notebook passo a passo]({URL_REPOSITORIO}/blob/main/TCC_Previsao_Riscos_Financeiros.ipynb)"
    )


# =============================================================================
# CABEÇALHO PRINCIPAL
# =============================================================================
st.markdown(
    f"""
    <div class="academic-header">
        <div class="academic-title">Previsão de Riscos Financeiros com Data Science e Machine Learning</div>
        <div class="academic-meta">
            <span>Ativo em foco</span>
            {icone_ativo_html(ticker_selecionado)}
            <div class="ativo-identificacao">
                <span class="ticker-tag">{ticker_selecionado}</span>
                <span class="ativo-nome">{nome_ativo_selecionado}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(key="barra_saiba_mais"):
    if st.button("Saiba mais", icon=":material/menu_book:", type="tertiary", key="btn_saiba_mais"):
        saiba_mais()

# Carregamento do ativo em foco
df_ativo = carregar_dados_ativo_cached(ticker_selecionado)

if df_ativo is None or df_ativo.empty:
    st.error(f"Não foi possível carregar os dados para o ativo {ticker_selecionado}.")
    st.stop()

dataset = preparar_dataset_cached(df_ativo)
# A inferência usa o scaler salvo junto aos modelos (o mesmo ajustado no treino)
X_test_scaled = scaler_salvo.transform(dataset.X_test) if tem_modelos else None

# =============================================================================
# NAVEGAÇÃO EM ABAS (o painel abre direto na análise exploratória)
# =============================================================================
aba_eda, aba_modelos, aba_xai, aba_simulador, aba_relatorios = st.tabs(
    [
        "Análise Exploratória",
        "Desempenho dos Modelos",
        "Interpretabilidade (XAI)",
        "Simulador de Risco e VaR",
        "Relatórios e Downloads",
    ]
)

# =============================================================================
# ABA FUNDAMENTAÇÃO & METODOLOGIA (DESATIVADA)
# O conteúdo teórico saiu das abas para o botão "Saiba mais" (janela saiba_mais()), para
# que o painel abra direto nas análises. O código original fica preservado abaixo, comentado,
# para eventual reaproveitamento. Os indicadores da série foram para a aba Análise Exploratória.
# =============================================================================
# with aba_visao:
#     st.markdown("### Contextualização Teórica e Metodológica")
#
#     with linha("kpi_visao"):
#         col_res1, col_res2, col_res3, col_res4 = st.columns(4, gap="large")
#         with col_res1:
#             st.markdown(
#                 f"""
#                 <div class="kpi-card kpi-best">
#                     <div class="kpi-label">Registros Históricos</div>
#                     <div class="kpi-value">{formatar_numero(len(df_ativo), 0)}</div>
#                     <div class="kpi-sub">Pregões diários válidos</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#         with col_res2:
#             dt_inicio = df_ativo["Data"].min().strftime("%d/%m/%Y")
#             st.markdown(
#                 f"""
#                 <div class="kpi-card">
#                     <div class="kpi-label">Marco Inicial</div>
#                     <div class="kpi-value" style="font-size: 1.25rem;">{dt_inicio}</div>
#                     <div class="kpi-sub">Início da série temporal</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#         with col_res3:
#             dt_fim = df_ativo["Data"].max().strftime("%d/%m/%Y")
#             st.markdown(
#                 f"""
#                 <div class="kpi-card">
#                     <div class="kpi-label">Marco Final</div>
#                     <div class="kpi-value" style="font-size: 1.25rem;">{dt_fim}</div>
#                     <div class="kpi-sub">Data da última cotação</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#         with col_res4:
#             vol_recente = df_ativo["Volatilidade_21d"].iloc[-1] * 100
#             st.markdown(
#                 f"""
#                 <div class="kpi-card kpi-good">
#                     <div class="kpi-label">Volatilidade Atual (21d)</div>
#                     <div class="kpi-value">{formatar_numero(vol_recente, 2)}%</div>
#                     <div class="kpi-sub">Anualizada no último pregão</div>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#
#     col_teo1, col_teo2 = st.columns([1.1, 0.9], gap="large")
#
#     with col_teo1:
#         with secao("problema"):
#             st.markdown("#### Problema de Pesquisa e Relevância")
#             st.markdown(
#                 """
#                 A gestão de riscos e a quantificação da incerteza no mercado de capitais constituem um dos pilares
#                 da moderna teoria de finanças e dos acordos regulatórios de Basileia. O presente Trabalho de Conclusão
#                 de Curso (TCC) desenvolve e avalia um pipeline de Ciência de Dados ponta a ponta projetado para prever a
#                 **volatilidade histórica anualizada no horizonte de 1 dia à frente ($t+1$)**.
#
#                 A abordagem soluciona criticidades fundamentais em **Gestão de Dados**:
#                 1. **Qualidade e Tratamento:** Ingestão de séries históricas higienizadas de proventos e desdobramentos.
#                 2. **Temporalidade Estrita:** Eliminação de viés de antecipação (*look-ahead bias*) e vazamento de dados (*data leakage*).
#                 3. **Transparência Algorítmica:** Aplicação de Inteligência Artificial Explicável (XAI) para auditoria e governança regulatória.
#                 """
#             )
#
#         with secao("formulacao"):
#             st.markdown("#### Formulação Matemática")
#             # Rótulos fora do LaTeX para as fórmulas caberem na largura da coluna
#             st.caption("Retorno diário aritmético")
#             st.latex(r"R_t = \frac{P_t - P_{t-1}}{P_{t-1}}")
#             st.caption("Volatilidade anualizada (janela de 21 pregões)")
#             st.latex(r"\sigma_{t, 21d} = \sqrt{\frac{1}{20} \sum_{i=0}^{20} (R_{t-i} - \bar{R}_t)^2} \times \sqrt{252}")
#             st.caption("Variável-alvo modelada via shift(-1)")
#             st.latex(r"Y_t = \sigma_{t+1, 21d}")
#
#     with col_teo2:
#         with secao("fluxo"):
#             st.markdown("#### Arquitetura do Fluxo Metodológico")
#             st.markdown(
#                 """
#                 <div class="theory-box">
#                     <div class="theory-title">1. Coleta e Ingestão</div>
#                     Dados diários automáticos via Yahoo Finance API (2016 até hoje).
#                 </div>
#                 <div class="theory-box">
#                     <div class="theory-title">2. Análise Exploratória (EDA)</div>
#                     Fatos estilizados, testes de curtose leptocúrtica e matrizes de correlação.
#                 </div>
#                 <div class="theory-box">
#                     <div class="theory-title">3. Pré-processamento Temporal</div>
#                     Split sequencial 80% treino / 20% teste sem shuffle. StandardScaler ajustado apenas no treino.
#                 </div>
#                 <div class="theory-box">
#                     <div class="theory-title">4. Modelagem Supervisionada Multimodelo</div>
#                     XGBoost, LightGBM, Rede Neural MLP (Keras) e Baseline Ingênuo de persistência.
#                 </div>
#                 <div class="theory-box">
#                     <div class="theory-title">5. Validação e Avaliação Robusta</div>
#                     RMSE, MAE, R² e Validação Cruzada Temporal (TimeSeriesSplit com 5 folds).
#                 </div>
#                 <div class="theory-box">
#                     <div class="theory-title">6. Governança e XAI</div>
#                     Explicabilidade global e local com SHAP (Shapley Additive exPlanations) e LIME.
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#
#     # Tabela de Dicionário de Dados
#     with secao("dicionario"):
#         st.markdown("#### Dicionário de Dados e Features Preditivas")
#         dicionario = [
#             {"Coluna": "Data", "Tipo": "datetime64", "Papel": "Eixo Temporal", "Interpretação": "Data oficial de encerramento do pregão da B3."},
#             {"Coluna": "Close", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Preço de fechamento ajustado para dividendos e desdobramentos."},
#             {"Coluna": "Volume", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Volume financeiro negociado no dia."},
#             {"Coluna": "Retorno_Diario", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Variação percentual diária de preço."},
#             {"Coluna": "Volatilidade_21d", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Volatilidade histórica anualizada em janela móvel de 21 dias úteis."},
#             {"Coluna": "MM_20d", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Média móvel de 20 pregões (tendência de curto prazo)."},
#             {"Coluna": "MM_50d", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Média móvel de 50 pregões (suporte/resistência de médio prazo)."},
#             {"Coluna": "RSI_14d", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Índice de Força Relativa (oscilador de momentum [0, 100])."},
#             {"Coluna": "Volume_MM20", "Tipo": "float64", "Papel": "Feature", "Interpretação": "Média móvel de volume negociado em 20 dias."},
#             {"Coluna": "Target_Vol", "Tipo": "float64", "Papel": "Variável-Alvo", "Interpretação": "Volatilidade observada no pregão seguinte (t+1)."},
#         ]
#         with st.expander("Exibir dicionário de dados", expanded=False):
#             st.dataframe(pd.DataFrame(dicionario), use_container_width=True, hide_index=True)


# =============================================================================
# ABA 1: ANÁLISE EXPLORATÓRIA (EDA) & FATOS ESTILIZADOS
# =============================================================================
with aba_eda:
    st.markdown(f"### Análise Exploratória de Dados — {ticker_selecionado}")

    # Período analisado: atalhos ou datas livres. Vale só para esta aba; os modelos
    # continuam avaliados no período de teste fixo (20% finais da série)
    data_min = df_ativo["Data"].min().date()
    data_max = df_ativo["Data"].max().date()
    ATALHOS_PERIODO = {"1 ano": 1, "2 anos": 2, "5 anos": 5, "Tudo": None}
    with linha("eda_periodo"):
        col_atalho, col_datas = st.columns([1.4, 1], gap="large", vertical_alignment="bottom")
        with col_atalho:
            atalho = st.segmented_control(
                "Período analisado",
                options=list(ATALHOS_PERIODO) + ["Personalizado"],
                default="Tudo",
                key="eda_atalho_periodo",
            ) or "Tudo"
        with col_datas:
            if atalho == "Personalizado":
                datas_escolhidas = st.date_input(
                    "Datas (início e fim)",
                    value=(max(data_min, data_max - timedelta(days=365)), data_max),
                    min_value=data_min,
                    max_value=data_max,
                    format="DD/MM/YYYY",
                    key="eda_datas_periodo",
                )
                # Enquanto o usuário escolhe só a primeira data, o período vai até o fim da série
                inicio_periodo = datas_escolhidas[0] if len(datas_escolhidas) > 0 else data_min
                fim_periodo = datas_escolhidas[1] if len(datas_escolhidas) > 1 else data_max
            else:
                anos = ATALHOS_PERIODO[atalho]
                inicio_periodo = (
                    max(data_min, (pd.Timestamp(data_max) - pd.DateOffset(years=anos)).date()) if anos else data_min
                )
                fim_periodo = data_max

    df_periodo = df_ativo[
        (df_ativo["Data"].dt.date >= inicio_periodo) & (df_ativo["Data"].dt.date <= fim_periodo)
    ].reset_index(drop=True)
    if len(df_periodo) < 30:
        # Sem st.stop(): as demais abas são montadas na mesma execução e precisam continuar
        st.warning("O período escolhido tem menos de 30 pregões; exibindo a série completa.")
        df_periodo = df_ativo.reset_index(drop=True)
        inicio_periodo, fim_periodo = data_min, data_max
    st.caption(
        f"Exibindo {formatar_numero(len(df_periodo), 0)} pregões, de {df_periodo['Data'].min():%d/%m/%Y} "
        f"a {df_periodo['Data'].max():%d/%m/%Y}. O período vale para todos os gráficos desta aba."
    )

    # Eventos históricos: base do sistema mais os eventos do próprio usuário
    eventos_todos = pd.concat([carregar_eventos_cached(), eventos_do_usuario()], ignore_index=True)
    # A lista vazia de eventos do usuário faz o concat devolver a coluna como texto
    eventos_todos["data"] = pd.to_datetime(eventos_todos["data"])
    categorias_eventos = [c for c in CORES_EVENTOS if c in set(eventos_todos["categoria"])]
    with linha("eda_eventos"):
        col_cat, col_novo = st.columns([1.4, 1], gap="large", vertical_alignment="bottom")
        with col_cat:
            categorias_exibidas = st.multiselect(
                "Eventos históricos nos gráficos",
                options=categorias_eventos,
                default=categorias_eventos,
                help="Marcadores de acontecimentos que mexeram com o mercado. Passe o mouse sobre a linha "
                "pontilhada para ver o evento; clique na legenda do gráfico para ocultar uma categoria.",
                key="eda_categorias_eventos",
            )
        with col_novo:
            with st.popover("Adicionar evento próprio", icon=":material/add_location_alt:", use_container_width=True):
                with st.form("form_evento_usuario", clear_on_submit=True, border=False):
                    data_evento = st.date_input(
                        "Data", value=data_max, min_value=data_min, max_value=data_max, format="DD/MM/YYYY"
                    )
                    titulo_evento = st.text_input("Título", max_chars=60, placeholder="Ex.: Queda das ações de tecnologia")
                    descricao_evento = st.text_input("Descrição (opcional)", max_chars=160)
                    if st.form_submit_button("Adicionar", type="primary", use_container_width=True):
                        if titulo_evento.strip():
                            st.session_state.setdefault("eventos_usuario", []).append(
                                {
                                    "data": pd.Timestamp(data_evento),
                                    "categoria": CATEGORIA_USUARIO,
                                    "titulo": titulo_evento.strip(),
                                    "descricao": descricao_evento.strip(),
                                }
                            )
                            # O novo evento deve aparecer já marcado no filtro de categorias
                            st.session_state.pop("eda_categorias_eventos", None)
                            st.rerun()
                        else:
                            st.warning("Informe um título para o evento.")

                arquivo_eventos = st.file_uploader(
                    "Ou carregue um CSV de eventos salvo antes", type="csv", key="eda_upload_eventos"
                )
                if arquivo_eventos is not None and st.session_state.get("eda_upload_lido") != arquivo_eventos.file_id:
                    try:
                        carregados = pd.read_csv(arquivo_eventos, sep=";", encoding="utf-8-sig", dtype=str)
                        carregados.columns = [c.strip().lower() for c in carregados.columns]
                        carregados["data"] = pd.to_datetime(carregados["data"], dayfirst=True)
                        if "descricao" not in carregados.columns:
                            carregados["descricao"] = ""
                        carregados["descricao"] = carregados["descricao"].fillna("")
                        carregados["categoria"] = CATEGORIA_USUARIO
                        st.session_state.setdefault("eventos_usuario", []).extend(
                            carregados[COLUNAS_EVENTOS].to_dict("records")
                        )
                        st.session_state["eda_upload_lido"] = arquivo_eventos.file_id
                        st.session_state.pop("eda_categorias_eventos", None)
                        st.rerun()
                    except (KeyError, ValueError):
                        st.error("O arquivo precisa das colunas data e titulo (e, opcionalmente, descricao), separadas por ponto e vírgula.")

                meus_eventos = eventos_do_usuario()
                if not meus_eventos.empty:
                    st.caption(f"{len(meus_eventos)} evento(s) seu(s) nesta sessão.")
                    csv_eventos = meus_eventos.assign(data=meus_eventos["data"].dt.strftime("%d/%m/%Y"))[
                        ["data", "titulo", "descricao"]
                    ].to_csv(sep=";", index=False).encode("utf-8-sig")
                    st.download_button(
                        "Baixar meus eventos (CSV)",
                        data=csv_eventos,
                        file_name="meus_eventos.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    if st.button("Remover meus eventos", use_container_width=True):
                        st.session_state.pop("eventos_usuario", None)
                        st.session_state.pop("eda_categorias_eventos", None)
                        st.rerun()

    eventos_periodo = eventos_todos[
        eventos_todos["categoria"].isin(categorias_exibidas)
        & (eventos_todos["data"].dt.date >= inicio_periodo)
        & (eventos_todos["data"].dt.date <= fim_periodo)
    ]

    # Indicadores do período
    with linha("kpi_serie"):
        col_res1, col_res2, col_res3, col_res4 = st.columns(4, gap="large")
        with col_res1:
            st.markdown(
                f"""
                <div class="kpi-card kpi-best">
                    <div class="kpi-label">Registros no Período</div>
                    <div class="kpi-value">{formatar_numero(len(df_periodo), 0)}</div>
                    <div class="kpi-sub">Pregões diários válidos</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_res2:
            dt_inicio = df_periodo["Data"].min().strftime("%d/%m/%Y")
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Marco Inicial</div>
                    <div class="kpi-value" style="font-size: 1.25rem;">{dt_inicio}</div>
                    <div class="kpi-sub">Primeiro pregão do período</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_res3:
            dt_fim = df_periodo["Data"].max().strftime("%d/%m/%Y")
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Marco Final</div>
                    <div class="kpi-value" style="font-size: 1.25rem;">{dt_fim}</div>
                    <div class="kpi-sub">Último pregão do período</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_res4:
            vol_recente = df_periodo["Volatilidade_21d"].iloc[-1] * 100
            st.markdown(
                f"""
                <div class="kpi-card kpi-good">
                    <div class="kpi-label">Volatilidade (21d)</div>
                    <div class="kpi-value">{formatar_numero(vol_recente, 2)}%</div>
                    <div class="kpi-sub">Anualizada no último pregão do período</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Gráfico 1: Preço de Fechamento e Médias Móveis
    with secao("eda_precos"):
        st.markdown("#### 1. Série Temporal de Preços e Tendências")
        preco_ini, preco_fim = float(df_periodo["Close"].iloc[0]), float(df_periodo["Close"].iloc[-1])
        como_ler(
            "eda_precos",
            ativo=ticker_selecionado,
            preco_inicio=formatar_numero(preco_ini, 2),
            preco_fim=formatar_numero(preco_fim, 2),
            variacao=f"{(preco_fim / preco_ini - 1) * 100:+.1f}%".replace(".", ","),
        )
        fig_fech = go.Figure()
        fig_fech.add_trace(
            go.Scatter(
                x=df_periodo["Data"],
                y=df_periodo["Close"],
                name="Fechamento Ajustado",
                line=dict(color="#1e3a8a", width=1.5),
            )
        )
        if "MM_20d" in df_periodo.columns:
            fig_fech.add_trace(
                go.Scatter(
                    x=df_periodo["Data"],
                    y=df_periodo["MM_20d"],
                    name="Média Móvel 20d",
                    line=dict(color="#b45309", width=1.3, dash="dash"),
                )
            )
        if "MM_50d" in df_periodo.columns:
            fig_fech.add_trace(
                go.Scatter(
                    x=df_periodo["Data"],
                    y=df_periodo["MM_50d"],
                    name="Média Móvel 50d",
                    line=dict(color="#0f766e", width=1.3, dash="dot"),
                )
            )
        precos = df_periodo[["Close", "MM_20d", "MM_50d"]]
        adicionar_eventos(fig_fech, eventos_periodo, float(precos.min().min()), float(precos.max().max()))

        fig_fech.update_layout(
            title=f"{ticker_selecionado} — Preço de Fechamento com Médias Móveis (20d e 50d)",
            xaxis_title="Data",
            yaxis_title="Preço / Pontos",
            hovermode="x unified",
            height=470,
        )
        exibir_grafico(fig_fech)

    # Gráfico 2: Volatilidade Histórica Anualizada
    with secao("eda_volatilidade"):
        st.markdown("#### 2. Evolução da Volatilidade Histórica Anualizada (Janela Móvel 21d)")
        linha_pico = df_periodo.loc[df_periodo["Volatilidade_21d"].idxmax()]
        como_ler(
            "eda_volatilidade",
            pico=f"{formatar_numero(linha_pico['Volatilidade_21d'] * 100, 1)}% ao ano",
            data_pico=f"{linha_pico['Data']:%d/%m/%Y}",
            atual=f"{formatar_numero(df_periodo['Volatilidade_21d'].iloc[-1] * 100, 1)}% ao ano",
        )
        fig_vol = px.area(
            df_periodo,
            x="Data",
            y="Volatilidade_21d",
            title=f"Volatilidade Anualizada — {ticker_selecionado}",
            labels={"Volatilidade_21d": "Volatilidade (anualizada)"},
            color_discrete_sequence=["#9f1239"],
            height=400,
        )
        fig_vol.update_traces(
            fillcolor="rgba(159, 18, 57, 0.12)", line=dict(width=1.3), name="Volatilidade", showlegend=True
        )
        adicionar_eventos(fig_vol, eventos_periodo, 0.0, float(df_periodo["Volatilidade_21d"].max()))
        fig_vol.update_layout(hovermode="x unified", margin=dict(b=56))
        exibir_grafico(fig_vol)

    with secao("eda_fatos"):
        st.markdown("#### 3. Fatos Estilizados: Distribuição de Retornos e Hipótese de Normalidade")

        retornos = df_periodo["Retorno_Diario"].dropna()
        datas_retornos = df_periodo.loc[retornos.index, "Data"]
        como_ler(
            "eda_fatos",
            pior_dia=f"{formatar_numero(retornos.min() * 100, 2)}%",
            data_pior=f"{datas_retornos[retornos.idxmin()]:%d/%m/%Y}",
            melhor_dia=f"{formatar_numero(retornos.max() * 100, 2)}%",
            data_melhor=f"{datas_retornos[retornos.idxmax()]:%d/%m/%Y}",
        )
        skewness = float(retornos.skew())
        kurtosis = float(retornos.kurtosis())

        with linha("eda_metricas"):
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4, gap="large")
            with col_stat1:
                st.metric("Retorno Médio Diário", f"{formatar_numero(retornos.mean() * 100, 3)}%")
            with col_stat2:
                st.metric("Desvio Padrão Diário", f"{formatar_numero(retornos.std() * 100, 3)}%")
            with col_stat3:
                st.metric("Assimetria (Skewness)", formatar_numero(skewness, 4), help="Negativa indica assimetria com cauda à esquerda")
            with col_stat4:
                st.metric(
                    "Curtose em Excesso",
                    formatar_numero(kurtosis, 4),
                    help="Curtose de Fisher (pandas): a Normal vale 0. Valores > 0 indicam distribuição leptocúrtica / caudas pesadas.",
                )

        with linha("eda_graficos_distribuicao"):
            col_eda1, col_eda2 = st.columns(2, gap="large")

            with col_eda1:
                # Histograma Interativo com Curva Normal de Referência
                fig_dist = go.Figure()
                fig_dist.add_trace(
                    go.Histogram(
                        x=retornos,
                        nbinsx=70,
                        histnorm="probability density",
                        name="Retornos Reais",
                        marker=dict(color="#1e3a8a", line=dict(color="#ffffff", width=0.5)),
                        opacity=0.7,
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
                        line=dict(color="#b91c1c", width=2.2),
                    )
                )
                fig_dist.update_layout(
                    title=f"Distribuição Empírica vs. Normal Teórica — {ticker_selecionado}",
                    xaxis_title="Retorno Diário",
                    yaxis_title="Densidade",
                    height=400,
                )
                exibir_grafico(fig_dist)

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
                        line=dict(color="#b91c1c", width=2, dash="dash"),
                    )
                )
                fig_qq.update_layout(
                    title="Q-Q Plot (Normalidade de Retornos)",
                    xaxis_title="Quantis Teóricos (Normal)",
                    yaxis_title="Quantis Observados",
                    height=400,
                )
                exibir_grafico(fig_qq)

        st.info(
            f"**Interpretação Acadêmica:** A curtose em excesso de **{formatar_numero(kurtosis, 2)}** (a distribuição Normal vale 0; equivale a "
            f"uma curtose de {formatar_numero(kurtosis + 3, 2)} contra 3 da Normal), associada ao afastamento das caudas no Q-Q Plot, "
            f"evidencia o fato estilizado financeiro das **caudas pesadas (leptocurtose)**: eventos extremos ocorrem com "
            f"frequência maior do que a Normal prevê. Por isso, medidas de risco baseadas na Normal, como o VaR paramétrico, "
            f"tendem a subestimar perdas extremas."
        )

    # Matriz de Correlação entre Ativos
    with secao("eda_correlacao"):
        st.markdown("#### 4. Matriz de Correlação Cruzada entre Ativos")
        todos_dados = carregar_todos_dados_cached()
        if todos_dados:
            ret_dict = {}
            for t_nome, df_t in todos_dados.items():
                if "Retorno_Diario" in df_t.columns:
                    nome_limpo = t_nome.replace(".SA", "").replace("^", "")
                    ret_dict[nome_limpo] = df_t.set_index("Data")["Retorno_Diario"]

            df_ret_corr = pd.DataFrame(ret_dict).dropna()
            # Mesmo período escolhido no topo da aba
            datas_corr = df_ret_corr.index.to_series().dt.date
            df_ret_corr = df_ret_corr[(datas_corr >= inicio_periodo) & (datas_corr <= fim_periodo)]
            if not df_ret_corr.empty:
                # Só o triângulo inferior: cada par de ativos aparece uma vez, sem a diagonal (sempre 1)
                corr_mat = triangulo_inferior(df_ret_corr.corr())
                pares = corr_mat.stack()
                par_max, par_min = pares.idxmax(), pares.idxmin()
                como_ler(
                    "eda_correlacao",
                    par_max=f"{par_max[0]} × {par_max[1]}",
                    corr_max=formatar_numero(pares.max(), 2),
                    par_min=f"{par_min[0]} × {par_min[1]}",
                    corr_min=formatar_numero(pares.min(), 2),
                )
                textos = corr_mat.map(lambda v: "" if pd.isna(v) else formatar_numero(v, 2))
                fig_corr = go.Figure(
                    go.Heatmap(
                        z=corr_mat.values,
                        x=list(corr_mat.columns),
                        y=list(corr_mat.index),
                        text=textos.values,
                        texttemplate="%{text}",
                        textfont=dict(size=12),
                        colorscale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        hoverongaps=False,
                        hovertemplate="%{y} × %{x}<br>Correlação: %{z:.3f}<extra></extra>",
                        colorbar=dict(thickness=12, outlinewidth=0),
                    )
                )
                fig_corr.update_layout(
                    title="Correlação dos Retornos Diários (Ativos B3 + S&P 500)",
                    height=500,
                    margin=dict(b=40),
                )
                # Primeira linha no topo, como numa tabela
                fig_corr.update_yaxes(autorange="reversed", showline=False, ticks="", showgrid=False)
                fig_corr.update_xaxes(showline=False, ticks="", showgrid=False)
                exibir_grafico(fig_corr)


# =============================================================================
# ABA 2: DESEMPENHO DOS MODELOS & VALIDAÇÃO
# =============================================================================
with aba_modelos:
    st.markdown("### Desempenho dos Modelos e Validação Temporal")

    if not tem_modelos:
        with secao("sem_modelos"):
            st.markdown("#### Modelos não treinados para este ativo")
            st.warning(
                f"Não há modelos treinados para **{ticker_selecionado}**. Modelos treinados em outro ativo "
                "não são aplicados aqui, pois a escala e o regime de volatilidade de cada série são distintos. "
                "Clique abaixo para treinar XGBoost, LightGBM e MLP com o split temporal 80/20 deste ativo."
            )
            if st.button("Treinar Modelos para este Ativo"):
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
        with linha("kpi_modelos"):
            col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4, gap="large")
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
                        <div class="kpi-value">{formatar_numero(melhor_ml['RMSE'], 5)}</div>
                        <div class="kpi-sub">{f"Baseline: {formatar_numero(rmse_baseline, 5)}" if rmse_baseline is not None else "Root Mean Squared Error"}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_kpi3:
                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">MAE (Melhor ML)</div>
                        <div class="kpi-value">{formatar_numero(melhor_ml['MAE'], 5)}</div>
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
                        <div class="kpi-value">{formatar_numero(melhor_ml['R2'], 4)}</div>
                        <div class="kpi-sub">Coeficiente de determinação</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Comparação explícita com o benchmark de persistência (sigma_{t+1} = sigma_t)
        with linha("avisos_modelos"):
            if rmse_baseline is not None:
                ganho_pct = (rmse_baseline - melhor_ml["RMSE"]) / rmse_baseline * 100
                if ganho_pct > 0:
                    st.success(
                        f"O melhor modelo de ML (**{melhor_ml['Modelo']}**) reduziu o RMSE em **{formatar_numero(ganho_pct, 1)}%** "
                        f"em relação ao Baseline Ingênuo de persistência ({formatar_numero(rmse_baseline, 5)})."
                    )
                else:
                    # -ganho_pct = (RMSE do modelo - RMSE do baseline) / RMSE do baseline: quanto o modelo ficou acima
                    st.warning(
                        f"O **Baseline Ingênuo de persistência** (σₜ₊₁ = σₜ) obteve o menor erro, com RMSE de "
                        f"**{formatar_numero(rmse_baseline, 5)}**. O RMSE do melhor modelo de ML ({melhor_ml['Modelo']}, "
                        f"{formatar_numero(melhor_ml['RMSE'], 5)}) ficou **{formatar_numero(-ganho_pct, 1)}% acima** do baseline. "
                        "Neste conjunto de teste, os modelos de ML não superaram a regra de "
                        "persistência, o que é coerente com a forte autocorrelação da volatilidade em janela móvel de "
                        "21 dias (de um pregão para o seguinte, 20 dos 21 retornos da janela se repetem)."
                    )

            modelos_r2_negativo = df_metricas.loc[df_metricas["R2"] < 0, "Modelo"].tolist()
            if modelos_r2_negativo:
                st.error(
                    f"R² negativo em: {', '.join(modelos_r2_negativo)}. Esses modelos erram mais do que prever a "
                    "média do período e não devem ser usados para conclusões."
                )

        with secao("tabela_metricas"):
            st.markdown("#### 1. Tabela Comparativa de Métricas de Regressão")
            como_ler(
                "tabela_metricas",
                melhor_modelo=df_metricas.iloc[0]["Modelo"],
                melhor_rmse=formatar_numero(df_metricas.iloc[0]["RMSE"], 5),
            )

            # Formatação da Tabela
            def classificar_modelo(nome: str) -> str:
                rotulos = []
                if nome == melhor_geral:
                    rotulos.append("Menor RMSE Geral")
                if nome == melhor_ml["Modelo"]:
                    rotulos.append("Melhor Modelo ML")
                if "Baseline" in nome:
                    rotulos.append("Benchmark de Persistência")
                return " • ".join(rotulos) if rotulos else "—"

            df_disp = df_metricas.copy()
            df_disp["Classificação"] = df_disp["Modelo"].apply(classificar_modelo)
            df_disp["RMSE"] = df_disp["RMSE"].map(lambda x: formatar_numero(x, 5))
            df_disp["MAE"] = df_disp["MAE"].map(lambda x: formatar_numero(x, 5))
            df_disp["R2"] = df_disp["R2"].map(lambda x: formatar_numero(x, 4))

            st.dataframe(df_disp, use_container_width=True, hide_index=True)

        # Gráfico Comparativo: Real vs Previsões
        with secao("real_vs_modelos"):
            st.markdown("#### 2. Visualização Comparativa da Série Temporal: Real vs. Modelos")
            como_ler("real_vs_modelos")

            modelos_disponiveis = list(predicoes.keys())
            modelos_selecionados = st.multiselect(
                "Selecione os modelos a exibir no gráfico:",
                options=modelos_disponiveis,
                # O baseline continua na lista; vem marcado só no modo pesquisador
                default=[melhor_ml["Modelo"]] + (["Baseline Ingênuo (Persistência)"] if modo_pesquisador else []),
            )

            datas_dt = pd.to_datetime(dataset.datas_test)
            fig_comp = go.Figure()

            # Linha Real
            fig_comp.add_trace(
                go.Scatter(
                    x=datas_dt,
                    y=dataset.y_test,
                    name="Volatilidade Real Observada",
                    line=dict(color=COR_REAL, width=2.6),
                )
            )

            # Série real em preto e cada modelo numa cor bem distinta (paleta de src/figuras.py)
            estilos_graf = {
                "XGBoost": dict(color=CORES_MODELOS["XGBoost"], dash="solid", width=1.8),
                "LightGBM": dict(color=CORES_MODELOS["LightGBM"], dash="dash", width=1.8),
                "Rede Neural (MLP)": dict(color=CORES_MODELOS["Rede Neural (MLP)"], dash="dot", width=2),
                "Baseline Ingênuo (Persistência)": dict(color=CORES_MODELOS["Baseline Ingênuo (Persistência)"], dash="dashdot", width=1.6),
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
                height=500,
            )
            exibir_grafico(fig_comp)

        if modo_pesquisador:
            # Validação Cruzada Temporal e Curva de Treino: painéis lado a lado com a mesma altura
            with st.container(key="pares_validacao_curva"):
                col_val1, col_val2 = st.columns(2, gap="large")
                with col_val1:
                    with secao("validacao_cruzada"):
                        st.markdown("#### 3. Validação Cruzada Temporal (TimeSeriesSplit)")
                        como_ler("validacao_cruzada")
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
                            color_discrete_sequence=["#1e3a8a"],
                            height=360,
                        )
                        fig_folds.update_traces(texttemplate="%{text:.4f}", textposition="outside", cliponaxis=False)
                        fig_folds.update_layout(margin=dict(b=56))
                        exibir_grafico(fig_folds)

                        fold_pior = folds_data.loc[folds_data["RMSE"].idxmax()]
                        st.caption(
                            f"RMSE médio: {formatar_numero(folds_data['RMSE'].mean(), 5)} ± {formatar_numero(folds_data['RMSE'].std(ddof=0), 5)}. "
                            f"Maior erro no {fold_pior['Fold']} ({fold_pior['Período de Validação']}). "
                            "Os folds percorrem a série completa (treino + teste), sempre treinando no passado e validando no futuro."
                        )

                with col_val2:
                    with secao("curva_mlp"):
                        st.markdown("#### 4. Curva de Aprendizado — Rede Neural MLP")
                        como_ler("curva_mlp")
                        historico_nn = meta_dict.get("historico_nn") or {}
                        if historico_nn.get("loss") and historico_nn.get("val_loss"):
                            epocas = list(range(1, len(historico_nn["loss"]) + 1))
                            fig_nn = go.Figure()
                            fig_nn.add_trace(
                                go.Scatter(x=epocas, y=historico_nn["loss"], name="Treino (MSE)", line=dict(color="#1e3a8a", width=1.8))
                            )
                            fig_nn.add_trace(
                                go.Scatter(x=epocas, y=historico_nn["val_loss"], name="Validação (MSE)", line=dict(color="#b45309", width=1.8))
                            )
                            melhor_epoca = int(np.argmin(historico_nn["val_loss"])) + 1
                            fig_nn.add_vline(x=melhor_epoca, line=dict(color="#6b7280", dash="dash"))
                            fig_nn.update_layout(
                                title=f"Perda MSE por Época — {ticker_selecionado}",
                                xaxis_title="Época",
                                yaxis_title="MSE",
                                yaxis_type="log",
                                yaxis_dtick=1,  # um rótulo por década (0,0001; 0,001; ...)
                                height=360,
                            )
                            exibir_grafico(fig_nn)
                            st.caption(
                                f"Early Stopping restaurou os pesos da época {melhor_epoca} (menor perda de validação, "
                                "medida nos últimos 15% do conjunto de treino)."
                            )
                        else:
                            st.info(
                                "Histórico de treino da MLP não registrado para este ativo. "
                                "Treine os modelos novamente para gerar a curva."
                            )

        else:
            st.caption(
                "A validação cruzada temporal e a curva de aprendizado da rede neural ficam no "
                "**Modo pesquisador**, na barra lateral."
            )


# =============================================================================
# ABA 3: INTERPRETABILIDADE & GOVERNANÇA (XAI)
# =============================================================================
with aba_xai:
    st.markdown("### Inteligência Artificial Explicável (XAI) e Governança")
    with linha("intro_xai"):
        st.markdown(
            """
            Em conformidade com as diretrizes regulatórias do mercado de capitais e princípios de **Governança de Dados**,
            modelos preditivos aplicados a riscos financeiros não podem operar como "caixas-pretas".
            Utilizam-se as abordagens **SHAP (*SHapley Additive exPlanations*)** e **LIME** para auditoria quantitativa
            do XGBoost treinado para o ativo selecionado.
            """
        )

    if not (tem_modelos and "XGBoost" in modelos_dict):
        st.info(
            f"Treine os modelos de {ticker_selecionado} na aba **Desempenho dos Modelos** "
            "para gerar as explicações SHAP e LIME deste ativo."
        )
    else:
        m_xgb = modelos_dict["XGBoost"]
        versao_modelo = str(meta_dict.get("data_salvamento", ""))
        nomes_features = list(dataset.features)
        datas_test_dt = pd.to_datetime(dataset.datas_test)

        valores_shap, base_shap = shap_cached(ticker_selecionado, versao_modelo, m_xgb, X_test_scaled)
        importancia = pd.DataFrame(
            {"Feature": nomes_features, "Impacto Médio |SHAP|": np.abs(valores_shap).mean(axis=0)}
        ).sort_values("Impacto Médio |SHAP|")

        # Sem o modo pesquisador, a importância global ocupa a largura toda (o beeswarm fica oculto)
        col_xai1, col_xai2 = st.columns(2, gap="large") if modo_pesquisador else (st.container(), None)
        with col_xai1:
            with secao("shap_bar"):
                st.markdown("#### 1. Importância Global das Variáveis (SHAP)")
                como_ler("shap_bar", top_feature=importancia.iloc[-1]["Feature"])
                fig_imp = go.Figure(
                    go.Bar(
                        x=importancia["Impacto Médio |SHAP|"],
                        y=importancia["Feature"],
                        orientation="h",
                        marker_color="#1e3a8a",
                        hovertemplate="%{y}: %{x:.5f}<extra></extra>",
                    )
                )
                fig_imp.update_layout(
                    title=f"Média de |SHAP| no Conjunto de Teste — {ticker_selecionado}",
                    xaxis_title="Impacto médio na volatilidade prevista",
                    height=400,
                    margin=dict(l=130, b=56),
                )
                exibir_grafico(fig_imp)
                st.caption("Quanto maior a barra, mais a variável altera, em média, a previsão do XGBoost.")

        if modo_pesquisador:
            with col_xai2:
                with secao("shap_beeswarm"):
                    st.markdown("#### 2. Direção e Dispersão do Impacto (SHAP Beeswarm)")
                    como_ler("shap_beeswarm")
                    # Cor = posição do valor da feature entre as amostras de teste (0 = menor, 1 = maior)
                    gerador = np.random.default_rng(RANDOM_STATE)
                    fig_bee = go.Figure()
                    for pos, feat in enumerate(importancia["Feature"]):
                        j = nomes_features.index(feat)
                        valores_feat = pd.Series(dataset.X_test[:, j])
                        fig_bee.add_trace(
                            go.Scatter(
                                x=valores_shap[:, j],
                                y=pos + gerador.uniform(-0.28, 0.28, len(valores_feat)),
                                mode="markers",
                                name=feat,
                                showlegend=False,
                                customdata=valores_feat,
                                hovertemplate=f"{feat} = %{{customdata:.4g}}<br>SHAP = %{{x:.5f}}<extra></extra>",
                                marker=dict(
                                    size=5,
                                    opacity=0.75,
                                    color=valores_feat.rank(pct=True),
                                    colorscale=[[0, "#1e3a8a"], [0.5, "#a3a8b8"], [1, "#b91c1c"]],
                                    cmin=0,
                                    cmax=1,
                                    showscale=pos == 0,
                                    colorbar=dict(
                                        title=dict(text="Valor da<br>feature", side="right"),
                                        tickvals=[0, 1],
                                        ticktext=["Baixo", "Alto"],
                                        thickness=12,
                                        outlinewidth=0,
                                    ),
                                ),
                            )
                        )
                    fig_bee.add_vline(x=0, line=dict(color="#3f4654", width=1))
                    fig_bee.update_layout(
                        title=f"Impacto de Cada Amostra de Teste — {ticker_selecionado}",
                        xaxis_title="Valor SHAP (efeito na volatilidade prevista)",
                        yaxis=dict(
                            tickmode="array",
                            tickvals=list(range(len(importancia))),
                            ticktext=list(importancia["Feature"]),
                            showgrid=False,
                        ),
                        height=400,
                        margin=dict(l=130, b=56),
                    )
                    exibir_grafico(fig_bee)
                    st.caption("Pontos à direita aumentam a previsão; vermelho indica valor alto da variável, azul indica valor baixo.")

        with secao("previsao_local"):
            st.markdown(f"#### {3 if modo_pesquisador else 2}. Simulador Local Interativo: Previsão de um Pregão Específico")
            como_ler("previsao_local")

            # Seleção por data (e não por índice): o pregão escolhido também define a decomposição
            # SHAP/LIME da seção 4 e as figuras locais do catálogo na aba Relatórios
            datas_opcoes = [d.strftime("%d/%m/%Y") for d in datas_test_dt]
            data_amostra = st.select_slider(
                "Data do pregão analisado:",
                options=datas_opcoes,
                value=datas_opcoes[-1],
                key=f"data_pregao_xai_{ticker_selecionado}",
                help=(
                    "Arraste para escolher qualquer pregão do conjunto de teste (20% finais da série). "
                    "Também é possível clicar no controle e usar as setas do teclado."
                ),
            )
            st.caption(
                f"Conjunto de teste: {datas_opcoes[0]} a {datas_opcoes[-1]} ({len(datas_opcoes)} pregões). "
                "A decomposição abaixo e as figuras SHAP Waterfall e LIME do catálogo acompanham esta data."
            )
            idx_selecionado = datas_opcoes.index(data_amostra)

            val_real = dataset.y_test[idx_selecionado]
            amostra_scaled = X_test_scaled[idx_selecionado].reshape(1, -1)
            pred_xgb_dia = float(m_xgb.predict(amostra_scaled)[0])

            # Linha de KPIs: os três cards ficam com a mesma altura (o terceiro tem a linha do erro)
            with linha("kpi_previsao_local"):
                col_loc1, col_loc2, col_loc3 = st.columns(3, gap="large")
                with col_loc1:
                    st.metric("Data do Pregão Analisado", data_amostra)
                with col_loc2:
                    st.metric("Volatilidade Real (t+1)", f"{formatar_numero(val_real * 100, 2)}%")
                with col_loc3:
                    # Erro de previsão em qualquer direção não é "bom" nem "ruim": delta em cor neutra
                    st.metric(
                        "Previsão do XGBoost",
                        f"{formatar_numero(pred_xgb_dia * 100, 2)}%",
                        delta=f"Erro (previsto − real): {'+' if pred_xgb_dia >= val_real else ''}"
                        f"{formatar_numero((pred_xgb_dia - val_real) * 100, 2)} p.p.",
                        delta_color="off",
                    )

        # Decomposição local do pregão escolhido no slider
        with secao("explicacoes_locais"):
            st.markdown(f"#### {4 if modo_pesquisador else 3}. Decomposição Local da Previsão — Pregão de {data_amostra}")
            como_ler("explicacoes_locais")
            # Os dois gráficos ficam empilhados, com a largura toda, para os rótulos longos caberem
            with linha("xai_waterfall"):
                contrib = valores_shap[idx_selecionado]
                ordem = np.argsort(-np.abs(contrib))
                def formatar_valor(v: float) -> str:
                    # Preço, médias e volume em milhares; retornos, volatilidade e RSI com 4 dígitos
                    return formatar_numero(v, 0) if abs(v) >= 1000 else f"{v:.4g}".replace(".", ",")

                rotulos = [f"{nomes_features[j]} = {formatar_valor(dataset.X_test[idx_selecionado, j])}" for j in ordem]
                fig_wf = go.Figure(
                    go.Waterfall(
                        orientation="h",
                        measure=["absolute"] + ["relative"] * len(ordem) + ["total"],
                        y=["Valor esperado E[f(x)]"] + rotulos + ["Previsão f(x)"],
                        x=[base_shap] + [float(contrib[j]) for j in ordem] + [0],
                        text=[formatar_numero(base_shap, 4)]
                        + [("+" if contrib[j] >= 0 else "") + formatar_numero(contrib[j], 4) for j in ordem]
                        + [formatar_numero(pred_xgb_dia, 4)],
                        textposition="outside",
                        cliponaxis=False,
                        increasing=dict(marker=dict(color="#b91c1c")),
                        decreasing=dict(marker=dict(color="#1e3a8a")),
                        totals=dict(marker=dict(color="#6b7280")),
                        connector=dict(line=dict(color="#a3a8b8", width=1)),
                    )
                )
                # Eixo aproximado da faixa percorrida pelas contribuições (como no waterfall do SHAP)
                trajetoria = base_shap + np.concatenate([[0], np.cumsum(contrib[ordem])])
                folga = max(trajetoria.max() - trajetoria.min(), 1e-4) * 0.35
                fig_wf.update_layout(
                    title="SHAP Waterfall",
                    xaxis_title="Volatilidade prevista (anualizada)",
                    xaxis_range=[trajetoria.min() - folga, trajetoria.max() + folga],
                    yaxis=dict(autorange="reversed", showgrid=False),
                    showlegend=False,
                    height=440,
                    margin=dict(l=220, r=70, b=56),
                )
                exibir_grafico(fig_wf)
                st.caption(
                    "Vermelho aumenta e azul reduz a volatilidade prevista a partir do valor esperado do modelo. "
                    "O eixo foi aproximado da faixa das contribuições; as barras cinza partem de valores fora dela."
                )

            # LIME: segunda explicação local, mais técnica, só no modo pesquisador
            if modo_pesquisador:
                explicador_lime = explicador_lime_cached(
                    ticker_selecionado, versao_modelo, dataset.X_train, tuple(nomes_features)
                )
                pesos_lime = lime_cached(
                    ticker_selecionado,
                    versao_modelo,
                    idx_selecionado,
                    explicador_lime,
                    m_xgb,
                    scaler_salvo,
                    dataset.X_test[idx_selecionado],
                )
                df_lime = pd.DataFrame(pesos_lime, columns=["Condição", "Peso"]).iloc[::-1]
                df_lime["Condição"] = df_lime["Condição"].map(formatar_condicao_lime)
                fig_lime = go.Figure(
                    go.Bar(
                        x=df_lime["Peso"],
                        y=df_lime["Condição"],
                        orientation="h",
                        marker_color=np.where(df_lime["Peso"] >= 0, "#b91c1c", "#1e3a8a"),
                        hovertemplate="%{y}<br>Peso = %{x:.5f}<extra></extra>",
                    )
                )
                fig_lime.add_vline(x=0, line=dict(color="#3f4654", width=1))
                fig_lime.update_layout(
                    title="LIME — Explicação Linear Local",
                    xaxis_title="Peso na volatilidade prevista",
                    yaxis=dict(showgrid=False),
                    height=400,
                    margin=dict(l=260, b=56),
                )
                exibir_grafico(fig_lime)
                st.caption("Condições em escala original das variáveis; o peso indica o efeito local estimado pelo LIME.")


# =============================================================================
# ABA 4: SIMULADOR DE RISCO (t+1) & VALUE AT RISK (VaR)
# =============================================================================
with aba_simulador:
    st.markdown("### Projeção de Risco para o Próximo Pregão ($t+1$) e Value at Risk (VaR)")
    with linha("intro_simulador"):
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
            desc_regime = f"Abaixo do percentil 25 histórico do ativo ({formatar_numero(p25, 1)}%). Oscilação inferior à usual."
        elif vol_pct < p75:
            regime = "Risco Moderado (Normalidade Histórica)"
            estilo_regime = "risk-moderate"
            desc_regime = f"Entre os percentis 25 e 75 históricos do ativo ({formatar_numero(p25, 1)}% a {formatar_numero(p75, 1)}%)."
        elif vol_pct < p95:
            regime = "Risco Elevado (Atenção / Estresse)"
            estilo_regime = "risk-high"
            desc_regime = (
                f"Entre os percentis 75 e 95 históricos do ativo ({formatar_numero(p75, 1)}% a {formatar_numero(p95, 1)}%). "
                "Sugere recalibração de stop loss e cautela em alavancagem."
            )
        else:
            regime = "Risco Extremo (Regime de Crise)"
            estilo_regime = "risk-extreme"
            desc_regime = (
                f"Acima do percentil 95 histórico do ativo ({formatar_numero(p95, 1)}%). "
                "Típico de choques macroeconômicos e crises agudas."
            )
        desc_regime += f" A projeção supera {percentil_atual:.0f}% dos pregões desde {df_ativo['Data'].min():%Y}."

        col_sim1, col_sim2 = st.columns([1, 1.2], gap="large")

        with col_sim1:
            with secao("projecao"):
                st.markdown("#### Volatilidade Projetada")
                como_ler("projecao")
                st.markdown(
                    f"""
                    <div class="kpi-card kpi-best">
                        <div class="kpi-label">Volatilidade Projetada (t+1)</div>
                        <div class="kpi-value">{formatar_numero(vol_pct, 2)}%</div>
                        <div class="kpi-sub">Projeção anualizada para o pregão seguinte a {data_base}</div>
                    </div>
                    <div style="margin-top: 1.25rem;">
                        <div class="risk-pill {estilo_regime}">{regime}</div>
                        <p style="font-size: 0.88rem; color: var(--cor-texto-suave); margin-top: 0.9rem;">{desc_regime}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_sim2:
            with secao("var"):
                st.markdown("#### Calculadora de Value at Risk (VaR Paramétrico)")
                como_ler("var")
                patrimonio = st.number_input(
                    "Patrimônio Alocado no Ativo (R$):",
                    min_value=1000.0,
                    max_value=100_000_000.0,
                    value=100_000.0,
                    step=5000.0,
                    format="%.2f",
                )
                # O campo numérico não aceita separador de milhar; o valor formatado fica logo abaixo
                st.caption(f"Patrimônio considerado: {formatar_reais(patrimonio)}")

                # Volatilidade diária (desanualizada dividindo por sqrt(252))
                vol_diaria = vol_predita / np.sqrt(252)

                # VaR paramétrico para 1 dia: VaR = Z * sigma_diaria * Patrimonio
                z_95 = 1.644853  # Nível 95%
                z_99 = 2.326348  # Nível 99%

                var_95 = z_95 * vol_diaria * patrimonio
                var_99 = z_99 * vol_diaria * patrimonio

                col_var1, col_var2 = st.columns(2, gap="large")
                with col_var1:
                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">VaR Diário (95% Confiança)</div>
                            <div class="kpi-value" style="color: #c2410c; font-size: 1.45rem;">{formatar_reais(var_95)}</div>
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
                            <div class="kpi-value" style="color: #b91c1c; font-size: 1.45rem;">{formatar_reais(var_99)}</div>
                            <div class="kpi-sub">Perda máxima esperada em 99% dos pregões (cenário adverso)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.caption(
            "Nota Metodológica: O cálculo do Value at Risk paramétrico utiliza o fator de desanualização padrão do mercado financeiro nacional "
            f"(252 dias úteis) e pressupõe normalidade dos retornos em curtíssimo prazo, calibrada pela volatilidade dinâmica estimada pelo XGBoost."
        )


# =============================================================================
# ABA 5: RELATÓRIOS & DOWNLOADS
# =============================================================================
with aba_relatorios:
    st.markdown("### Relatório Executivo e Exportação de Artefatos")

    # Relatório e arquivos montados na hora para o ativo selecionado, com as mesmas métricas
    # da aba Desempenho dos Modelos (df_metricas), a data e hora da exibição e o período real da série
    momento_relatorio = agora_brasilia()
    prefixo_ativo = ticker_selecionado.replace("^", "").replace(".", "_")
    carimbo_arquivo = f"{prefixo_ativo}_{momento_relatorio:%Y%m%d_%H%M}"
    with secao("relatorio"):
        st.markdown("#### 1. Relatório Consolidado de Desempenho")
        if tem_modelos:
            conteudo_rel = montar_texto_relatorio(
                df_metricas,
                ticker_ref=ticker_selecionado,
                data_inicio=df_ativo["Data"].min(),
                data_fim=df_ativo["Data"].max(),
                n_train=len(dataset.X_train),
                n_test=len(dataset.X_test),
                gerado_em=momento_relatorio,
                periodo_teste=(dataset.datas_test[0], dataset.datas_test[-1]),
                nome_ativo=nome_ativo_selecionado,
                periodo_treino=(dataset.datas_train[0], dataset.datas_train[-1]),
                folds=validacao_cruzada_cached(ticker_selecionado, str(df_ativo["Data"].max().date()), dataset),
            )
            st.code(conteudo_rel, language="text")
            st.caption(
                "Relatório gerado no momento da exibição, a partir da série e dos modelos atuais do ativo selecionado."
            )
        else:
            conteudo_rel = ""
            st.info(
                f"Treine os modelos de {ticker_selecionado} na aba **Desempenho dos Modelos** "
                "para gerar o relatório consolidado deste ativo."
            )

    with secao("downloads"):
        st.markdown("#### 2. Download Centralizado de Arquivos do Projeto")

        col_dl1, col_dl2, col_dl3 = st.columns(3, gap="large")

        with col_dl1:
            st.download_button(
                label="Baixar Relatório Final (.txt)",
                data=conteudo_rel,
                file_name=f"relatorio_desempenho_{carimbo_arquivo}.txt",
                mime="text/plain",
                disabled=not conteudo_rel,
                use_container_width=True,
            )
            st.caption("Relatório da seção 1: dados, desempenho, comparação com o benchmark, validação cruzada e conclusão.")

        with col_dl2:
            if tem_modelos:
                st.download_button(
                    label="Baixar Tabela de Métricas (.csv)",
                    data=metricas_para_csv(df_metricas, ticker_selecionado, momento_relatorio),
                    file_name=f"metricas_modelos_{carimbo_arquivo}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button("Tabela CSV Indisponível", disabled=True, use_container_width=True)
            st.caption("Uma linha por modelo, ordenada por RMSE, com RMSE, MAE, R² e classificação.")

        with col_dl3:
            st.download_button(
                label=f"Baixar Dados de {ticker_selecionado} (.csv)",
                data=dados_para_csv(df_ativo),
                file_name=f"dados_{carimbo_arquivo}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Série diária com preços, volume, indicadores calculados e a volatilidade t+1 (alvo).")

        st.caption(
            "Os arquivos CSV usam o padrão do Excel em português: ponto e vírgula entre colunas, "
            "vírgula decimal e datas dd/mm/aaaa."
        )

    # Catálogo de figuras do ativo selecionado, geradas em memória a partir dos dados e modelos atuais
    with secao("catalogo"):
        st.markdown("#### 3. Catálogo de Figuras Geradas em Alta Resolução")
        st.caption(
            f"Figuras de **{ticker_selecionado}** geradas a partir da série e dos modelos atuais, "
            "todas em PNG de 1600 × 1000 px."
        )
        ultima_data = str(df_ativo["Data"].max().date())
        retornos_ativos = {
            t.replace(".SA", "").replace("^", ""): d.set_index("Data")["Retorno_Diario"]
            for t, d in carregar_todos_dados_cached().items()
            if "Retorno_Diario" in d.columns
        }
        figuras_catalogo = list(
            figuras_exploratorias_cached(ticker_selecionado, ultima_data, df_ativo, retornos_ativos)
        )

        if tem_modelos:
            versao_modelo = str(meta_dict.get("data_salvamento", ""))
            predicoes_catalogo = adicionar_baseline_ingenuo(
                dataset.features, dataset.X_test, predizer_com_modelos(modelos_dict, X_test_scaled)
            )
            figuras_catalogo += figuras_avaliacao_cached(
                ticker_selecionado,
                ultima_data,
                versao_modelo,
                dataset.datas_test,
                dataset.y_test,
                predicoes_catalogo,
                validacao_cruzada_cached(ticker_selecionado, ultima_data, dataset),
                meta_dict.get("historico_nn"),
            )
            if "XGBoost" in modelos_dict:
                # Mesmo pregão escolhido na aba Interpretabilidade (XAI)
                idx_ref = idx_selecionado
                valores_shap_cat, base_shap_cat = shap_cached(
                    ticker_selecionado, versao_modelo, modelos_dict["XGBoost"], X_test_scaled
                )
                pesos_lime_cat = lime_cached(
                    ticker_selecionado,
                    versao_modelo,
                    idx_ref,
                    explicador_lime_cached(ticker_selecionado, versao_modelo, dataset.X_train, tuple(dataset.features)),
                    modelos_dict["XGBoost"],
                    scaler_salvo,
                    dataset.X_test[idx_ref],
                )
                figuras_catalogo += figuras_interpretabilidade_cached(
                    ticker_selecionado,
                    ultima_data,
                    versao_modelo,
                    valores_shap_cat,
                    base_shap_cat,
                    X_test_scaled,
                    dataset.X_test,
                    list(dataset.features),
                    idx_ref,
                    pd.Timestamp(dataset.datas_test[idx_ref]).strftime("%d/%m/%Y"),
                    pesos_lime_cat,
                )
        else:
            st.info(
                f"Treine os modelos de {ticker_selecionado} na aba **Desempenho dos Modelos** para incluir "
                "as figuras de avaliação e interpretabilidade no catálogo."
            )

        # Grade de 3 colunas montada linha a linha: as figuras de uma mesma linha ficam alinhadas
        prefixo_arquivo = ticker_selecionado.replace("^", "").replace(".", "_")
        for inicio in range(0, len(figuras_catalogo), 3):
            with linha(f"catalogo_{inicio}"):
                colunas_fig = st.columns(3, gap="large")
                for coluna, (arquivo, titulo, png) in zip(colunas_fig, figuras_catalogo[inicio:inicio + 3]):
                    with coluna:
                        st.markdown(f'<div class="catalogo-titulo">{titulo}</div>', unsafe_allow_html=True)
                        st.image(png, use_container_width=True)
                        st.download_button(
                            label="Baixar PNG",
                            data=png,
                            file_name=f"{prefixo_arquivo}_{arquivo}",
                            mime="image/png",
                            key=f"dl_fig_{arquivo}",
                            use_container_width=True,
                        )
