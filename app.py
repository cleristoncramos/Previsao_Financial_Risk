"""
=============================================================================
PREVISÃO DE RISCOS FINANCEIROS — PONTO DE ENTRADA DO STREAMLIT
Configura a página, aplica o estilo e define a navegação entre:
- Página inicial (rota raiz "/"): apresentação, tutorial e informações acadêmicas
- Painel (rota "/painel"): análise, modelos, XAI, simulador e relatórios
=============================================================================
"""

import os
import sys
from pathlib import Path

# Silencia logs e avisos do TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

st.set_page_config(
    page_title="Previsão de Riscos Financeiros",
    page_icon=":material/show_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo compartilhado pelas duas páginas
CSS_PATH = BASE_DIR / "assets" / "style.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

pagina_inicio = st.Page(
    "paginas/inicio.py",
    title="Início | Previsão de Riscos Financeiros",
    icon=":material/home:",
    default=True,
)
pagina_painel = st.Page(
    "paginas/painel.py",
    title="Painel | Previsão de Riscos Financeiros",
    icon=":material/monitoring:",
    url_path="painel",
)

# Menu padrão de páginas oculto: a navegação é feita pelo botão "Get Started" e pelo ícone de início
st.navigation([pagina_inicio, pagina_painel], position="hidden").run()
