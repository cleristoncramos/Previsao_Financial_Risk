"""
=============================================================================
PÁGINA INICIAL (LANDING PAGE) — rota raiz "/"
Apresentação do sistema: frase de efeito, botão "Get Started", vídeo curto,
tutorial breve, primeiros passos e as informações acadêmicas do TCC
(concentradas aqui, numa janela aberta por um botão discreto).
=============================================================================
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import ATIVOS, TICKER_REF
from src.data_collection import carregar_ativo_local

VIDEO_APRESENTACAO = BASE_DIR / "assets" / "video" / "apresentacao.gif"


@st.cache_data(show_spinner=False)
def periodo_da_serie() -> tuple:
    """Primeiro e último ano da série do ativo de referência (para o resumo da página)."""
    df = carregar_ativo_local(TICKER_REF)
    if df is None or df.empty:
        return None, None
    datas = pd.to_datetime(df["Data"])
    return datas.min().year, datas.max().year


@st.dialog("Informações acadêmicas", width="large")
def informacoes_academicas():
    st.markdown(
        """
        <div class="info-academica">
            <div class="info-academica-tipo">Trabalho de Conclusão de Curso</div>
            <div class="info-academica-titulo">
                Previsão de Riscos Financeiros com Data Science e Machine Learning
            </div>
            <table>
                <tr><th>Autora</th><td>Lorrana Evelyn de Araújo Pereira</td></tr>
                <tr><th>Orientador</th><td>Prof. Dr. Arlino Henrique Magalhães de Araújo</td></tr>
                <tr><th>Curso</th><td>Curso Superior de Tecnologia em Gestão de Dados</td></tr>
                <tr><th>Unidade</th><td>Centro de Educação Aberta e a Distância (CEAD)</td></tr>
                <tr><th>Instituição</th><td>Universidade Federal do Piauí (UFPI)</td></tr>
                <tr><th>Ano</th><td>2026</td></tr>
                <tr><th>Metodologia</th><td>Split temporal 80/20, modelos de Machine Learning e explicabilidade (XAI)</td></tr>
                <tr><th>Tecnologias</th><td>Python, Streamlit, Scikit-Learn, XGBoost, LightGBM, Keras, SHAP e LIME</td></tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ir_para_painel():
    st.switch_page("paginas/painel.py")


# =============================================================================
# BARRA SUPERIOR: identificação do sistema e acesso discreto às informações acadêmicas
# =============================================================================
col_marca, col_info = st.columns([4, 1.2], vertical_alignment="center")
with col_marca:
    st.markdown(
        '<div class="landing-marca">Previsão de Riscos Financeiros</div>',
        unsafe_allow_html=True,
    )
with col_info:
    if st.button("Informações acadêmicas", icon=":material/school:", type="tertiary", key="btn_info_academica"):
        informacoes_academicas()

# =============================================================================
# DESTAQUE: frase de efeito, chamada principal e vídeo curto
# =============================================================================
with st.container(key="landing_hero"):
    col_texto, col_video = st.columns([1, 1.15], gap="large", vertical_alignment="center")
    with col_texto:
        st.markdown(
            """
            <div class="landing-selo">Ciência de Dados aplicada à gestão de risco</div>
            <h1 class="landing-titulo">Antecipe a volatilidade.<br>Decida com dados.</h1>
            <p class="landing-subtitulo">
                Um painel que prevê a volatilidade do dia seguinte de ações da B3 e índices de mercado,
                compara modelos de Machine Learning com uma regra de referência, explica cada previsão
                e traduz o risco em Value at Risk.
            </p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Get Started", type="primary", icon=":material/arrow_forward:", key="btn_comecar_topo"):
            ir_para_painel()
        st.markdown('<p class="landing-nota">Acesso direto ao painel, sem cadastro.</p>', unsafe_allow_html=True)

    with col_video:
        st.markdown('<div class="landing-video-rotulo">Apresentação em 20 segundos</div>', unsafe_allow_html=True)
        if VIDEO_APRESENTACAO.exists():
            st.image(str(VIDEO_APRESENTACAO), use_container_width=True)
        else:
            st.info("Vídeo de apresentação indisponível.")

# Números do projeto
ano_inicio, ano_fim = periodo_da_serie()
total_ativos = sum(len(t) for t in ATIVOS.values())
periodo_txt = f"{ano_inicio}–{ano_fim}" if ano_inicio else "Diária"
st.markdown(
    f"""
    <div class="landing-numeros">
        <div><strong>{total_ativos}</strong><span>ativos monitorados (ações da B3 e índices)</span></div>
        <div><strong>{periodo_txt}</strong><span>série histórica diária do Yahoo Finance</span></div>
        <div><strong>4</strong><span>abordagens comparadas: XGBoost, LightGBM, MLP e baseline</span></div>
        <div><strong>SHAP + LIME</strong><span>explicação global e local de cada previsão</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# TUTORIAL BREVE: o que cada aba do painel oferece
# =============================================================================
st.markdown(
    """
    <div class="landing-secao">
        <div class="landing-secao-rotulo">Tutorial breve</div>
        <h2>O que você encontra no painel</h2>
        <div class="landing-grade">
            <div class="landing-cartao">
                <div class="landing-cartao-num">01</div>
                <h3>Fundamentação e Metodologia</h3>
                <p>Problema de pesquisa, fórmulas da volatilidade, fluxo metodológico e dicionário de dados.</p>
            </div>
            <div class="landing-cartao">
                <div class="landing-cartao-num">02</div>
                <h3>Análise Exploratória</h3>
                <p>Preços e médias móveis, volatilidade histórica, caudas pesadas e correlação entre ativos.</p>
            </div>
            <div class="landing-cartao">
                <div class="landing-cartao-num">03</div>
                <h3>Desempenho dos Modelos</h3>
                <p>RMSE, MAE e R² no período de teste, comparação com a regra de persistência e validação cruzada.</p>
            </div>
            <div class="landing-cartao">
                <div class="landing-cartao-num">04</div>
                <h3>Interpretabilidade (XAI)</h3>
                <p>Quais variáveis mais pesam na previsão e a decomposição de um pregão escolhido por data.</p>
            </div>
            <div class="landing-cartao">
                <div class="landing-cartao-num">05</div>
                <h3>Simulador de Risco e VaR</h3>
                <p>Volatilidade projetada para o próximo pregão, regime de risco e perda máxima provável em reais.</p>
            </div>
            <div class="landing-cartao">
                <div class="landing-cartao-num">06</div>
                <h3>Relatórios e Downloads</h3>
                <p>Relatório consolidado, tabela de métricas, dados do ativo e catálogo de figuras em alta resolução.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# PRIMEIROS PASSOS
# =============================================================================
st.markdown(
    """
    <div class="landing-secao">
        <div class="landing-secao-rotulo">Primeiros passos</div>
        <h2>Comece em cinco passos</h2>
        <ol class="landing-passos">
            <li><strong>Entre no painel</strong> pelo botão <em>Get Started</em>.</li>
            <li><strong>Escolha o ativo</strong> na barra lateral. O IBOVESPA (^BVSP) é a referência do estudo e já vem com modelos treinados.</li>
            <li><strong>Treine os modelos</strong> do ativo, se o painel indicar que ainda não existem: basta um clique na aba <em>Desempenho dos Modelos</em>.</li>
            <li><strong>Explore as análises</strong>: compare os modelos, escolha um pregão na aba de interpretabilidade e simule o VaR com o seu patrimônio.</li>
            <li><strong>Baixe os resultados</strong> na aba <em>Relatórios e Downloads</em>. Para voltar a esta página, use o ícone de início no topo da barra lateral.</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chamada final
with st.container(key="landing_cta"):
    col_cta_texto, col_cta_botao = st.columns([3, 1], vertical_alignment="center")
    with col_cta_texto:
        st.markdown(
            '<div class="landing-cta-texto">Pronto para analisar o risco do próximo pregão?</div>',
            unsafe_allow_html=True,
        )
    with col_cta_botao:
        if st.button("Get Started", type="primary", icon=":material/arrow_forward:", key="btn_comecar_rodape",
                     use_container_width=True):
            ir_para_painel()
