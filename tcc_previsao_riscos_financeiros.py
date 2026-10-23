# -*- coding: utf-8 -*-
"""
Previsão de Riscos Financeiros com Data Science e Machine Learning

Autora: Lorrana Evelyn de Araújo Pereira
Orientador: Prof. Dr. Arlino Henrique Magalhães de Araújo
Curso: Tecnologia em Gestão de Dados (UFPI / CEAD)

Versão em script do notebook TCC_Previsao_Riscos_Financeiros.ipynb. Reproduz, passo a
passo, a metodologia do trabalho com as mesmas funções do pacote src/ usadas pelo
pipeline (main.py) e pelo painel Streamlit, de modo que os resultados coincidem.

Etapas:
    1. Ambiente e importações
    2. Coleta de dados (Yahoo Finance) e engenharia de atributos
    3. Análise exploratória
    4. Pré-processamento: alvo t+1, divisão temporal 80/20 e padronização
    5. Modelagem: XGBoost, LightGBM e Rede Neural MLP
    6. Avaliação: comparação com o baseline de persistência e validação cruzada temporal
    7. Interpretabilidade: SHAP e LIME
    8. Risco projetado: volatilidade do próximo pregão, regime de risco e Value at Risk
    9. Relatório consolidado

Uso (na raiz do projeto ou de qualquer pasta):
    python tcc_previsao_riscos_financeiros.py
    python tcc_previsao_riscos_financeiros.py --nao-mostrar-graficos
    python tcc_previsao_riscos_financeiros.py --baixar-dados
"""

# =============================================================================
# 1. AMBIENTE E IMPORTAÇÕES
# =============================================================================
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

parser = argparse.ArgumentParser(description="Previsão de Riscos Financeiros (TCC - Lorrana Pereira)")
parser.add_argument(
    "--baixar-dados",
    action="store_true",
    help="Baixa novamente as séries no Yahoo Finance e substitui os CSVs de dados_coletados/",
)
parser.add_argument(
    "--nao-mostrar-graficos",
    action="store_true",
    help="Executa sem abrir as janelas e abas dos gráficos",
)
args = parser.parse_args()
MOSTRAR_GRAFICOS = not args.nao_mostrar_graficos
USAR_DADOS_LOCAIS = not args.baixar_dados

import matplotlib

# src.figuras fixa o backend "Agg" (exigido pelo painel); guarda o padrão para reabrir as janelas
BACKEND_PADRAO = matplotlib.get_backend() if MOSTRAR_GRAFICOS else "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
import shap
import lime.lime_tabular
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb

from src.config import ATIVOS, TICKER_REF, FEATURES, RANDOM_STATE, agora_brasilia
from src.data_collection import carregar_ativo_local, coletar_todos_ativos
from src.preprocessing import preparar_dados_modelagem
from src.models import treinar_todos_modelos
from src.evaluation import avaliar_todos_modelos, adicionar_baseline_ingenuo, validacao_cruzada_temporal
from src.figuras import triangulo_inferior
from src.report import montar_texto_relatorio

plt.switch_backend(BACKEND_PADRAO)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110


def br(valor, casas=2):
    """Formata um número no padrão brasileiro (milhar com ponto e decimal com vírgula)."""
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def mostrar_plotly(fig):
    """Abre o gráfico Plotly no navegador, se a exibição estiver habilitada."""
    if MOSTRAR_GRAFICOS:
        fig.show()


def mostrar_matplotlib():
    """Exibe a figura Matplotlib atual (ou apenas a fecha, sem exibição)."""
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close("all")


def titulo_etapa(texto):
    print("\n" + "=" * 70)
    print(f" {texto}")
    print("=" * 70)


print(f"TensorFlow {tf.__version__} | XGBoost {xgb.__version__} | LightGBM {lgb.__version__} | SHAP {shap.__version__}")

# =============================================================================
# 2. COLETA DE DADOS
# Séries diárias do Yahoo Finance (preços ajustados) de cinco ações da B3 e de dois
# índices. Por padrão usa os CSVs de dados_coletados/ (os mesmos do painel), o que
# torna os resultados reprodutíveis; --baixar-dados baixa as séries novamente.
# =============================================================================
titulo_etapa("2. COLETA DE DADOS")

if USAR_DADOS_LOCAIS:
    dados = {}
    for categoria, tickers in ATIVOS.items():
        for ticker in tickers:
            df = carregar_ativo_local(ticker, categoria)
            if df is not None:
                dados[ticker] = df
    if len(dados) < sum(len(t) for t in ATIVOS.values()):
        print("CSVs incompletos; baixando as séries no Yahoo Finance.")
        dados = coletar_todos_ativos()
else:
    dados = coletar_todos_ativos()

resumo = pd.DataFrame(
    {
        "Registros": [len(df) for df in dados.values()],
        "Início": [f"{df['Data'].min():%d/%m/%Y}" for df in dados.values()],
        "Fim": [f"{df['Data'].max():%d/%m/%Y}" for df in dados.values()],
    },
    index=list(dados),
)
print(resumo.to_string())

# =============================================================================
# 3. ANÁLISE EXPLORATÓRIA
# Ativo de referência: IBOVESPA (^BVSP). As primeiras linhas, sem histórico suficiente
# para as janelas móveis (até 50 pregões), são descartadas.
# =============================================================================
titulo_etapa("3. ANÁLISE EXPLORATÓRIA")

df_ref = dados[TICKER_REF].dropna(subset=FEATURES).reset_index(drop=True)
print(f"Pregões com todos os atributos: {br(len(df_ref), 0)}")
print(f"Período: {df_ref['Data'].min():%d/%m/%Y} a {df_ref['Data'].max():%d/%m/%Y}")

colunas_num = ["Close", "Volume", "Retorno_Diario", "Volatilidade_21d", "MM_20d", "MM_50d", "RSI_14d"]
print("\nEstatísticas descritivas:")
print(df_ref[colunas_num].describe().round(4).to_string())

# O volume zero corresponde a pregões em que o Yahoo Finance não informa o volume do índice
print(f"\nPregões com volume zero: {(df_ref['Volume'] == 0).sum()}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_ref["Data"], y=df_ref["Close"], name="Fechamento", line=dict(color="#1e3a8a", width=1.2)))
fig.add_trace(go.Scatter(x=df_ref["Data"], y=df_ref["MM_20d"], name="MM 20d", line=dict(color="#c2410c", dash="dash")))
fig.add_trace(go.Scatter(x=df_ref["Data"], y=df_ref["MM_50d"], name="MM 50d", line=dict(color="#15803d", dash="dot")))
fig.update_layout(
    title="IBOVESPA: preço de fechamento e médias móveis",
    xaxis_title="Data", yaxis_title="Pontos", hovermode="x unified",
    template="plotly_white", height=420, separators=",.", legend=dict(orientation="h", y=1.02),
)
mostrar_plotly(fig)

fig = px.area(
    df_ref, x="Data", y="Volatilidade_21d",
    title="IBOVESPA: volatilidade histórica anualizada (janela de 21 pregões)",
    labels={"Volatilidade_21d": "Volatilidade anualizada"},
    color_discrete_sequence=["#b91c1c"], template="plotly_white", height=380,
)
fig.update_layout(separators=",.", yaxis_tickformat=".0%")
mostrar_plotly(fig)

pico = df_ref.loc[df_ref["Volatilidade_21d"].idxmax()]
print(f"Volatilidade máxima: {br(pico['Volatilidade_21d'] * 100, 1)}% em {pico['Data']:%d/%m/%Y}")

# Todos os retornos da série, como no painel (inclui os pregões anteriores às janelas de 50 dias)
retornos_ref = dados[TICKER_REF]["Retorno_Diario"].dropna()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(retornos_ref, bins=80, kde=True, color="#1e3a8a", ax=axes[0])
axes[0].axvline(0, color="#b91c1c", linestyle="--", linewidth=1)
axes[0].set_title("Distribuição dos retornos diários do IBOVESPA")
axes[0].set_xlabel("Retorno diário")
stats.probplot(retornos_ref, plot=axes[1])
axes[1].set_title("Q-Q plot: retornos contra a distribuição Normal")
plt.tight_layout()
mostrar_matplotlib()

assimetria = retornos_ref.skew()
curtose_excesso = retornos_ref.kurtosis()  # Fisher: vale 0 para a Normal
print(f"Assimetria: {br(assimetria, 4)}")
print(f"Curtose em excesso: {br(curtose_excesso, 4)} (curtose de {br(curtose_excesso + 3, 2)}, contra 3 da Normal)")
print("A curtose em excesso positiva indica caudas pesadas (distribuição leptocúrtica), típicas de séries financeiras.")

# Correlação dos retornos entre os ativos, alinhados pela data do pregão: as séries têm
# calendários diferentes, e só entram os dias em que todos os ativos negociaram
retornos = pd.DataFrame(
    {t.replace(".SA", "").replace("^", ""): df.set_index("Data")["Retorno_Diario"] for t, df in dados.items()}
).dropna()
corr = triangulo_inferior(retornos.corr())
print(f"\nPregões em comum entre os ativos: {br(len(retornos), 0)}")

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1, ax=ax,
            linewidths=.5, cbar_kws={"shrink": .8})
for texto in ax.texts:
    texto.set_text(texto.get_text().replace(".", ","))
ax.set_title("Correlação dos retornos diários entre os ativos", fontsize=13, pad=15)
plt.tight_layout()
mostrar_matplotlib()

# =============================================================================
# 4. PRÉ-PROCESSAMENTO
# Alvo: volatilidade do pregão seguinte (shift(-1)). Divisão temporal 80/20, sem
# embaralhamento. StandardScaler ajustado apenas no treino (sem vazamento do teste).
# =============================================================================
dataset = preparar_dados_modelagem(dados[TICKER_REF])

datas_treino = pd.to_datetime(dataset.datas_train)
datas_teste = pd.to_datetime(dataset.datas_test)
print(f"Treino: {br(len(dataset.X_train), 0)} amostras ({datas_treino[0]:%d/%m/%Y} a {datas_treino[-1]:%d/%m/%Y})")
print(f"Teste : {br(len(dataset.X_test), 0)} amostras ({datas_teste[0]:%d/%m/%Y} a {datas_teste[-1]:%d/%m/%Y})")

# =============================================================================
# 5. MODELAGEM
# XGBoost e LightGBM (500 árvores, taxa 0,05) e Rede Neural MLP (128-64-32, Dropout de
# 10%, saída com Rescaling). O Early Stopping da MLP monitora os últimos 15% do treino,
# nunca o teste. Os modelos não são salvos, para não substituir os usados pelo painel.
# As métricas da MLP variam entre execuções (o TensorFlow não é totalmente determinístico);
# XGBoost, LightGBM e o baseline são determinísticos e coincidem com o painel.
# =============================================================================
modelos, predicoes, extras = treinar_todos_modelos(dataset, salvar=False)
modelo_xgb = modelos["XGBoost"]
history = extras["history_nn"]

# =============================================================================
# 6. AVALIAÇÃO
# Baseline de persistência: repete a volatilidade do pregão atual como previsão do
# seguinte. É a régua mínima que um modelo precisa superar.
# =============================================================================
predicoes_todas = adicionar_baseline_ingenuo(dataset.features, dataset.X_test, predicoes)
df_resultados = avaliar_todos_modelos(dataset.y_test, predicoes_todas)

tabela = df_resultados.copy()
for coluna, casas in [("RMSE", 5), ("MAE", 5), ("R2", 4)]:
    tabela[coluna] = tabela[coluna].map(lambda v: br(v, casas))
print("\n" + tabela.rename(columns={"R2": "R²"}).set_index("Modelo").to_string())

eh_baseline = df_resultados["Modelo"].str.contains("Baseline")
rmse_baseline = df_resultados.loc[eh_baseline, "RMSE"].iloc[0]
melhor_ml = df_resultados[~eh_baseline].iloc[0]
diferenca = (melhor_ml["RMSE"] - rmse_baseline) / rmse_baseline * 100

print(f"\nMelhor modelo de ML: {melhor_ml['Modelo']} (RMSE {br(melhor_ml['RMSE'], 5)})")
print(f"Baseline de persistência: RMSE {br(rmse_baseline, 5)}")
if diferenca > 0:
    print(f"O RMSE do modelo ficou {br(diferenca, 1)}% acima do baseline: neste horizonte, a persistência não foi superada.")
else:
    print(f"O RMSE do modelo ficou {br(-diferenca, 1)}% abaixo do baseline.")

cores = {"XGBoost": "#c2410c", "LightGBM": "#15803d", "Rede Neural (MLP)": "#7c3aed", "Baseline Ingênuo (Persistência)": "#6b7280"}
tracos = {"XGBoost": "dash", "LightGBM": "dot", "Rede Neural (MLP)": "dashdot", "Baseline Ingênuo (Persistência)": "longdash"}

fig = go.Figure()
fig.add_trace(go.Scatter(x=datas_teste, y=dataset.y_test, name="Real", line=dict(color="#1e3a8a", width=2)))
for nome, previsao in predicoes_todas.items():
    fig.add_trace(go.Scatter(x=datas_teste, y=previsao, name=nome, line=dict(color=cores[nome], dash=tracos[nome], width=1.3)))
fig.update_layout(
    title="Volatilidade do pregão seguinte: real contra previsões (conjunto de teste)",
    xaxis_title="Data", yaxis_title="Volatilidade anualizada", yaxis_tickformat=".0%",
    hovermode="x unified", template="plotly_white", height=450, separators=",.",
    legend=dict(orientation="h", y=1.02),
)
mostrar_plotly(fig)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history["loss"], label="Treino (MSE)", color="#1e3a8a")
ax.plot(history.history["val_loss"], label="Validação: últimos 15% do treino (MSE)", color="#c2410c")
ax.set_title("Curva de aprendizado da Rede Neural MLP")
ax.set_xlabel("Época")
ax.set_ylabel("Perda (MSE)")
ax.legend()
plt.tight_layout()
mostrar_matplotlib()
print(f"Épocas executadas: {len(history.epoch)}")

# Validação cruzada temporal: TimeSeriesSplit com 5 folds sobre a série completa; cada
# fold treina com todo o passado disponível e valida no bloco seguinte
rmses = validacao_cruzada_temporal(dataset.X, dataset.y, n_splits=5)

datas_serie = pd.to_datetime(dataset.datas)
linhas_folds = []
for i, (tr_idx, val_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(dataset.X)):
    linhas_folds.append({
        "Fold": f"Fold {i + 1}",
        "Treino": f"{datas_serie[tr_idx[0]]:%d/%m/%Y} a {datas_serie[tr_idx[-1]]:%d/%m/%Y} ({br(len(tr_idx), 0)})",
        "Período de Validação": f"{datas_serie[val_idx[0]]:%m/%Y}–{datas_serie[val_idx[-1]]:%m/%Y}",
        "RMSE": rmses[i],
    })
folds = pd.DataFrame(linhas_folds)

print(f"\nMédia: {br(np.mean(rmses), 5)} ± {br(np.std(rmses), 5)}")
print(folds.assign(RMSE=folds["RMSE"].map(lambda v: br(v, 5))).set_index("Fold").to_string())
# O fold 2 concentra o erro: sua validação inclui o choque da pandemia de 2020

# =============================================================================
# 7. INTERPRETABILIDADE
# SHAP decompõe cada previsão do XGBoost na soma das contribuições das variáveis.
# LIME aproxima o modelo, em torno de um pregão, por um modelo linear interpretável.
# =============================================================================
titulo_etapa("7. INTERPRETABILIDADE (SHAP E LIME)")

explainer_shap = shap.TreeExplainer(modelo_xgb)
shap_values = explainer_shap.shap_values(dataset.X_test_scaled)
base_shap = float(np.ravel(explainer_shap.expected_value)[0])

importancia = pd.Series(np.abs(shap_values).mean(axis=0), index=dataset.features).sort_values(ascending=False)
print("Média de |SHAP| no conjunto de teste:")
for variavel, valor in importancia.items():
    print(f"  {variavel:<17} {br(valor, 5)}")

plt.figure(figsize=(10, 5))
shap.summary_plot(shap_values, dataset.X_test_scaled, feature_names=dataset.features, plot_type="bar", show=False)
plt.title("SHAP: importância global das variáveis (XGBoost)", pad=15)
plt.tight_layout()
mostrar_matplotlib()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, dataset.X_test_scaled, feature_names=dataset.features, show=False)
plt.title("SHAP beeswarm: impacto de cada variável nas previsões", pad=15)
plt.tight_layout()
mostrar_matplotlib()

# Explicação de um pregão: o waterfall e o LIME exibem as variáveis na escala original;
# no LIME, o scaler é aplicado dentro da função de previsão
idx_exemplo = 10
print(f"\nPregão explicado: índice {idx_exemplo} do teste, {datas_teste[idx_exemplo]:%d/%m/%Y}")

explicacao = shap.Explanation(
    values=shap_values[idx_exemplo],
    base_values=base_shap,
    data=dataset.X_test[idx_exemplo],
    feature_names=dataset.features,
)
shap.plots.waterfall(explicacao, show=False)
plt.title("SHAP waterfall: decomposição da previsão do pregão", pad=12)
plt.tight_layout()
mostrar_matplotlib()

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=dataset.X_train,
    feature_names=dataset.features,
    mode="regression",
    random_state=RANDOM_STATE,
)
exp_lime = explainer_lime.explain_instance(
    dataset.X_test[idx_exemplo],
    lambda X: modelo_xgb.predict(dataset.scaler.transform(X)),
    num_features=len(dataset.features),
)

print(f"Previsão do XGBoost: {br(predicoes['XGBoost'][idx_exemplo] * 100, 2)}%")
print(f"Valor real         : {br(dataset.y_test[idx_exemplo] * 100, 2)}%")
print("Pesos LIME:")
for condicao, peso in exp_lime.as_list():
    print(f"  {condicao:<45} {br(peso, 5)}")
exp_lime.as_pyplot_figure()
plt.title(f"LIME: explicação local do pregão de {datas_teste[idx_exemplo]:%d/%m/%Y}", pad=12)
plt.tight_layout()
mostrar_matplotlib()

# =============================================================================
# 8. RISCO PROJETADO E VALUE AT RISK
# VaR paramétrico de 1 dia: VaR = z × (σ anual / √252) × patrimônio, com z de 1,644853
# (95%) e 2,326348 (99%). Supõe retornos normais e tende a subestimar perdas extremas.
# =============================================================================
titulo_etapa("8. RISCO PROJETADO E VALUE AT RISK")

PATRIMONIO = 100_000.0

# Último pregão com todos os atributos: ele não entra na modelagem, pois ainda não tem t+1
ultima_linha = dados[TICKER_REF][["Data"] + FEATURES].dropna().iloc[-1]
x_ultimo = ultima_linha[FEATURES].to_numpy(dtype=float).reshape(1, -1)
vol_projetada = float(modelo_xgb.predict(dataset.scaler.transform(x_ultimo))[0])

vol_hist = dados[TICKER_REF]["Volatilidade_21d"].dropna()
p25, p75, p95 = np.percentile(vol_hist, [25, 75, 95])
if vol_projetada < p25:
    regime = "Baixo"
elif vol_projetada < p75:
    regime = "Moderado"
elif vol_projetada < p95:
    regime = "Elevado"
else:
    regime = "Extremo"

vol_diaria = vol_projetada / np.sqrt(252)
var_95 = 1.644853 * vol_diaria * PATRIMONIO
var_99 = 2.326348 * vol_diaria * PATRIMONIO

print(f"Pregão base             : {ultima_linha['Data']:%d/%m/%Y}")
print(f"Volatilidade projetada  : {br(vol_projetada * 100, 2)}% ao ano")
print(f"Limites históricos      : P25 {br(p25 * 100, 2)}% | P75 {br(p75 * 100, 2)}% | P95 {br(p95 * 100, 2)}%")
print(f"Regime de risco         : {regime}")
print(f"VaR de 1 dia (95%)      : R$ {br(var_95)} para R$ {br(PATRIMONIO)}")
print(f"VaR de 1 dia (99%)      : R$ {br(var_99)} para R$ {br(PATRIMONIO)}")

# =============================================================================
# 9. RELATÓRIO CONSOLIDADO
# Mesmo relatório da aba Relatórios e Downloads do painel, no horário de Brasília.
# =============================================================================
df_serie = dados[TICKER_REF]
relatorio = montar_texto_relatorio(
    df_resultados,
    ticker_ref=TICKER_REF,
    data_inicio=df_serie["Data"].min(),
    data_fim=df_serie["Data"].max(),
    n_train=len(dataset.X_train),
    n_test=len(dataset.X_test),
    gerado_em=agora_brasilia(),
    periodo_teste=(dataset.datas_test[0], dataset.datas_test[-1]),
    nome_ativo="IBOVESPA",
    periodo_treino=(dataset.datas_train[0], dataset.datas_train[-1]),
    folds=folds,
)
print("\n" + relatorio)
print("\nPainel interativo: streamlit run app.py")
