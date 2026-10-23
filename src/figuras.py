"""
Módulo do Catálogo de Figuras por Ativo.

Gera em memória (PNG) as figuras do ativo selecionado no painel:
- Análise exploratória: preços e médias, volatilidade, distribuição, Q-Q plot e correlação
- Avaliação: real vs. modelos, validação cruzada temporal e curva de aprendizado da MLP
- Interpretabilidade: importância SHAP, beeswarm SHAP, waterfall SHAP e LIME

Todas as figuras têm o mesmo tamanho (10 x 6,25 pol. a 160 dpi = 1600 x 1000 px),
o que mantém a grade do catálogo alinhada.
"""

import io
import re
import threading
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats

TAMANHO_FIGURA = (10, 6.25)
DPI_FIGURA = 160

COR_MARINHO = "#1e3a8a"
COR_VERDE = "#0f766e"
COR_AMBAR = "#b45309"
COR_VIOLETA = "#7c3aed"
COR_CINZA = "#6b7280"
COR_VERMELHO = "#b91c1c"
COR_TEXTO = "#1f2937"

# (nome do arquivo, título exibido, conteúdo PNG)
Figura = Tuple[str, str, bytes]

# O pyplot (usado também por SHAP) não é thread-safe e o Streamlit atende sessões em threads
_trava_pyplot = threading.Lock()


def _nova_figura():
    fig, ax = plt.subplots(figsize=TAMANHO_FIGURA)
    return fig, ax


def _titulo(fig, texto: str) -> None:
    """
    Título preso à borda esquerda da figura (não do eixo): com rótulos longos no eixo Y,
    um título alinhado ao eixo era empurrado para a direita e saía cortado na exportação.
    """
    fig.suptitle(texto, x=0.015, y=0.975, ha="left", va="top", fontsize=14, fontweight="bold", color=COR_MARINHO)


def formatar_condicao_lime(condicao: str) -> str:
    """Números >= 1000 nas condições do LIME com separador de milhar brasileiro (ex.: 22.568.590)."""
    def _formatar(m: re.Match) -> str:
        valor = float(m.group(0))
        return f"{valor:,.0f}".replace(",", ".") if abs(valor) >= 1000 else m.group(0)

    return re.sub(r"-?\d+(?:\.\d+)?", _formatar, condicao)


def _estilizar(ax, titulo: str, rotulo_x: str = "", rotulo_y: str = "") -> None:
    """Aplica o padrão visual do painel: fundo branco, grade discreta e eixos finos."""
    _titulo(ax.figure, titulo)
    ax.set_xlabel(rotulo_x, fontsize=11, color="#5b6472")
    ax.set_ylabel(rotulo_y, fontsize=11, color="#5b6472")
    ax.grid(True, color="#eceef2", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#3f4654")
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=COR_TEXTO, labelsize=10)


def _para_png(fig) -> bytes:
    """Fixa o tamanho padrão, salva a figura em PNG e libera a memória."""
    fig.set_size_inches(*TAMANHO_FIGURA)
    # Reserva a faixa superior para o título da figura e mantém margem nas bordas
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.92))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI_FIGURA, facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def _legenda(ax) -> None:
    ax.legend(frameon=False, fontsize=10, loc="upper left")


def figuras_exploratorias(
    df: pd.DataFrame,
    ticker: str,
    retornos_ativos: Dict[str, pd.Series],
) -> List[Figura]:
    """Figuras da análise exploratória do ativo (não dependem de modelos)."""
    figuras: List[Figura] = []
    retornos = df["Retorno_Diario"].dropna()

    with _trava_pyplot:
        # 1. Preço de fechamento com médias móveis
        fig, ax = _nova_figura()
        ax.plot(df["Data"], df["Close"], color=COR_MARINHO, linewidth=1.1, label="Fechamento Ajustado")
        ax.plot(df["Data"], df["MM_20d"], color=COR_AMBAR, linewidth=1, linestyle="--", label="Média Móvel 20d")
        ax.plot(df["Data"], df["MM_50d"], color=COR_VERDE, linewidth=1, linestyle=":", label="Média Móvel 50d")
        _estilizar(ax, f"{ticker} — Preço de Fechamento com Médias Móveis", "Data", "Preço / Pontos")
        _legenda(ax)
        figuras.append(("eda_fechamento_medias.png", "Preço de Fechamento e Médias Móveis", _para_png(fig)))

        # 2. Volatilidade histórica anualizada
        fig, ax = _nova_figura()
        ax.plot(df["Data"], df["Volatilidade_21d"], color="#9f1239", linewidth=1)
        ax.fill_between(df["Data"], df["Volatilidade_21d"], color="#9f1239", alpha=0.12)
        _estilizar(ax, f"{ticker} — Volatilidade Anualizada (Janela Móvel 21d)", "Data", "Volatilidade anualizada")
        figuras.append(("eda_volatilidade.png", "Volatilidade Histórica Anualizada", _para_png(fig)))

        # 3. Distribuição dos retornos vs. Normal teórica
        fig, ax = _nova_figura()
        ax.hist(retornos, bins=70, density=True, color=COR_MARINHO, alpha=0.7, edgecolor="white", linewidth=0.4, label="Retornos Reais")
        x_norm = np.linspace(retornos.min(), retornos.max(), 300)
        ax.plot(x_norm, stats.norm.pdf(x_norm, retornos.mean(), retornos.std()), color=COR_VERMELHO, linewidth=2, label="Normal Teórica")
        _estilizar(ax, f"{ticker} — Distribuição Empírica vs. Normal Teórica", "Retorno diário", "Densidade")
        _legenda(ax)
        figuras.append(("eda_distribuicao_retornos.png", "Distribuição dos Retornos Diários", _para_png(fig)))

        # 4. Q-Q plot de normalidade
        fig, ax = _nova_figura()
        (teoricos, ordenados), (inclinacao, intercepto, _) = stats.probplot(retornos, dist="norm")
        ax.scatter(teoricos, ordenados, s=8, color=COR_MARINHO, alpha=0.6, label="Quantis Amostrais")
        ax.plot(teoricos, inclinacao * teoricos + intercepto, color=COR_VERMELHO, linestyle="--", linewidth=1.8, label="Reta de Ajuste Normal")
        _estilizar(ax, f"{ticker} — Q-Q Plot (Normalidade dos Retornos)", "Quantis teóricos (Normal)", "Quantis observados")
        _legenda(ax)
        figuras.append(("eda_qq_plot.png", "Q-Q Plot de Normalidade", _para_png(fig)))

        # 5. Matriz de correlação entre os ativos monitorados
        df_ret = pd.DataFrame(retornos_ativos).dropna()
        if not df_ret.empty:
            corr = df_ret.corr()
            fig, ax = _nova_figura()
            # aspect="auto" faz o mapa ocupar a largura da figura (sem faixa vazia à esquerda)
            mapa = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(corr)), corr.columns, rotation=0, fontsize=10)
            ax.set_yticks(range(len(corr)), corr.index, fontsize=10)
            for i in range(len(corr)):
                for j in range(len(corr)):
                    valor = corr.values[i, j]
                    ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(valor) > 0.6 else COR_TEXTO)
            fig.colorbar(mapa, ax=ax, fraction=0.035, pad=0.02)
            _titulo(fig, "Correlação dos Retornos Diários entre Ativos")
            for lado in ax.spines.values():
                lado.set_visible(False)
            figuras.append(("eda_matriz_correlacao.png", "Matriz de Correlação entre Ativos", _para_png(fig)))

    return figuras


def figuras_avaliacao(
    ticker: str,
    datas_test: np.ndarray,
    y_test: np.ndarray,
    predicoes: Dict[str, np.ndarray],
    folds: pd.DataFrame,
    historico_nn: Optional[Dict[str, List[float]]],
) -> List[Figura]:
    """Figuras de desempenho dos modelos treinados para o ativo."""
    figuras: List[Figura] = []
    estilos = {
        "XGBoost": dict(color=COR_MARINHO, linestyle="-"),
        "LightGBM": dict(color=COR_VERDE, linestyle="--"),
        "Rede Neural (MLP)": dict(color=COR_VIOLETA, linestyle=":"),
        "Baseline Ingênuo (Persistência)": dict(color="#9ca3af", linestyle="-."),
    }

    with _trava_pyplot:
        # 1. Volatilidade real vs. previsões no conjunto de teste
        fig, ax = _nova_figura()
        datas = pd.to_datetime(datas_test)
        ax.plot(datas, y_test, color="#111827", linewidth=1.8, label="Volatilidade Real")
        for nome, y_hat in predicoes.items():
            ax.plot(datas, y_hat, linewidth=1.1, label=nome, **estilos.get(nome, {}))
        _estilizar(ax, f"{ticker} — Volatilidade Real vs. Modelos (Teste 20%)", "Data do pregão", "Volatilidade anualizada")
        _legenda(ax)
        figuras.append(("avaliacao_real_vs_modelos.png", "Real vs. Modelos no Conjunto de Teste", _para_png(fig)))

        # 2. RMSE por fold da validação cruzada temporal
        fig, ax = _nova_figura()
        barras = ax.bar(folds["Fold"], folds["RMSE"], color=COR_MARINHO, width=0.6)
        for barra, periodo in zip(barras, folds["Período de Validação"]):
            ax.annotate(f"{barra.get_height():.4f}\n{periodo}", (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                        ha="center", va="bottom", fontsize=9, color=COR_TEXTO, xytext=(0, 4), textcoords="offset points")
        ax.set_ylim(0, folds["RMSE"].max() * 1.25)
        _estilizar(ax, f"{ticker} — RMSE por Fold (TimeSeriesSplit 5)", "Fold", "RMSE")
        figuras.append(("avaliacao_validacao_cruzada.png", "Validação Cruzada Temporal", _para_png(fig)))

        # 3. Curva de aprendizado da MLP
        if historico_nn and historico_nn.get("loss") and historico_nn.get("val_loss"):
            fig, ax = _nova_figura()
            epocas = np.arange(1, len(historico_nn["loss"]) + 1)
            ax.semilogy(epocas, historico_nn["loss"], color=COR_MARINHO, linewidth=1.5, label="Treino (MSE)")
            ax.semilogy(epocas, historico_nn["val_loss"], color=COR_AMBAR, linewidth=1.5, label="Validação (MSE)")
            melhor = int(np.argmin(historico_nn["val_loss"])) + 1
            ax.axvline(melhor, color=COR_CINZA, linestyle="--", linewidth=1, label=f"Melhor época ({melhor})")
            _estilizar(ax, f"{ticker} — Curva de Aprendizado da Rede Neural MLP", "Época", "MSE (escala log)")
            _legenda(ax)
            figuras.append(("avaliacao_curva_aprendizado_mlp.png", "Curva de Aprendizado da MLP", _para_png(fig)))

    return figuras


def figuras_interpretabilidade(
    ticker: str,
    valores_shap: np.ndarray,
    base_shap: float,
    X_test_scaled: np.ndarray,
    X_test: np.ndarray,
    features: List[str],
    idx_amostra: int,
    data_amostra: str,
    pesos_lime: List[Tuple[str, float]],
) -> List[Figura]:
    """Figuras SHAP e LIME do XGBoost do ativo."""
    figuras: List[Figura] = []

    with _trava_pyplot:
        # 1. Importância global (média de |SHAP|)
        importancia = pd.Series(np.abs(valores_shap).mean(axis=0), index=features).sort_values()
        fig, ax = _nova_figura()
        ax.barh(importancia.index, importancia.values, color=COR_MARINHO, height=0.6)
        _estilizar(ax, f"{ticker} — Importância Global das Variáveis (média |SHAP|)", "Impacto médio na volatilidade prevista")
        ax.grid(axis="y", visible=False)
        figuras.append(("shap_importancia_global.png", "SHAP — Importância Global", _para_png(fig)))

        # 2. Beeswarm (gráfico canônico da biblioteca SHAP)
        shap.summary_plot(valores_shap, X_test, feature_names=features, show=False, plot_size=TAMANHO_FIGURA)
        fig = plt.gcf()
        _titulo(fig, f"{ticker} — SHAP Beeswarm: Impacto de Cada Amostra")
        # Traduz os rótulos que a biblioteca SHAP gera em inglês
        fig.axes[0].set_xlabel("Valor SHAP (efeito na volatilidade prevista)", fontsize=11, color="#5b6472")
        barra_cores = fig.axes[-1]
        barra_cores.set_ylabel("Valor da variável", fontsize=11, color="#5b6472")
        barra_cores.set_yticks(barra_cores.get_yticks(), ["Baixo", "Alto"])
        figuras.append(("shap_beeswarm.png", "SHAP — Beeswarm", _para_png(fig)))

        # 3. Waterfall da amostra de referência (valores das variáveis na escala original)
        explicacao = shap.Explanation(
            values=valores_shap[idx_amostra],
            base_values=base_shap,
            data=X_test[idx_amostra],
            feature_names=features,
        )
        shap.plots.waterfall(explicacao, show=False)
        fig = plt.gcf()
        _titulo(fig, f"{ticker} — SHAP Waterfall (Pregão de {data_amostra})")
        figuras.append(("shap_waterfall.png", f"SHAP — Waterfall ({data_amostra})", _para_png(fig)))

        # 4. LIME da mesma amostra
        df_lime = pd.DataFrame(pesos_lime, columns=["Condição", "Peso"]).iloc[::-1]
        df_lime["Condição"] = df_lime["Condição"].map(formatar_condicao_lime)
        fig, ax = _nova_figura()
        cores = [COR_VERMELHO if p >= 0 else COR_MARINHO for p in df_lime["Peso"]]
        ax.barh(df_lime["Condição"], df_lime["Peso"], color=cores, height=0.6)
        ax.axvline(0, color="#3f4654", linewidth=0.8)
        _estilizar(ax, f"{ticker} — LIME: Explicação Local (Pregão de {data_amostra})", "Peso na volatilidade prevista")
        ax.grid(axis="y", visible=False)
        figuras.append(("lime_explicacao_local.png", f"LIME — Explicação Local ({data_amostra})", _para_png(fig)))

    return figuras
