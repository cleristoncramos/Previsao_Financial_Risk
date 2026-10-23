"""
Gera o vídeo curto de apresentação da página inicial (assets/video/apresentacao.gif).

Animação de ~20 segundos com dados reais do ativo de referência (^BVSP): a série de
volatilidade, o RMSE dos modelos salvos, a importância SHAP do XGBoost e o VaR
projetado para o próximo pregão. Formato GIF para tocar automaticamente no navegador
sem dependências externas (ffmpeg).

Uso (na raiz do projeto):
    python scripts/gerar_video_apresentacao.py
"""

import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import MODELS_DIR, TICKER_REF
from src.data_collection import carregar_ativo_local
from src.model_store import carregar_modelos_salvos

SAIDA = BASE_DIR / "assets" / "video" / "apresentacao.gif"
LARGURA, ALTURA, DPI = 9.6, 5.4, 100  # 960 x 540 px
QUADROS_POR_SEGUNDO = 10

MARINHO = "#1e3a8a"
MARINHO_ESCURO = "#172554"
TEXTO = "#1f2937"
SUAVE = "#5b6472"
BORDA = "#3f4654"
FUNDO = "#f4f5f7"
VINHO = "#9f1239"

ETAPAS = [
    "Escolha o ativo",
    "Explore a volatilidade",
    "Compare os modelos",
    "Entenda a previsão",
    "Projete o risco",
]


def numero_br(valor: float, casas: int = 2) -> str:
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def suavizar(t: float) -> float:
    """Curva de animação (entrada e saída suaves) para t em [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# -----------------------------------------------------------------------------
# Dados reais usados nas cenas
# -----------------------------------------------------------------------------
def carregar_dados():
    df = carregar_ativo_local(TICKER_REF)
    df["Data"] = pd.to_datetime(df["Data"])
    df = df.sort_values("Data").reset_index(drop=True)

    metadados = json.loads((MODELS_DIR / "metadados_modelos.json").read_text(encoding="utf-8"))
    metricas = pd.DataFrame(metadados["metricas"]).sort_values("RMSE")

    modelos, scaler, _ = carregar_modelos_salvos(MODELS_DIR)
    features = metadados["features"]
    base = df.dropna(subset=features).copy()
    corte = int(len(base) * 0.8)
    X_teste = scaler.transform(base[features].values[corte:])

    import shap

    valores_shap = np.asarray(shap.TreeExplainer(modelos["XGBoost"]).shap_values(X_teste))
    importancia = pd.Series(np.abs(valores_shap).mean(axis=0), index=features).sort_values()

    ultima = scaler.transform(base[features].values[-1:])
    vol_prevista = float(modelos["XGBoost"].predict(ultima)[0])
    var_95 = 1.644853 * vol_prevista / np.sqrt(252) * 100_000

    return df, metricas, importancia, vol_prevista, var_95


# -----------------------------------------------------------------------------
# Moldura comum a todas as cenas
# -----------------------------------------------------------------------------
def nova_cena(etapa: int = None):
    fig = plt.figure(figsize=(LARGURA, ALTURA), dpi=DPI)
    fig.patch.set_facecolor(FUNDO)
    # Painel branco com borda fina e barra azul-marinho à esquerda (atrás dos gráficos: zorder negativo)
    fig.add_artist(Rectangle((0.03, 0.06), 0.94, 0.88, transform=fig.transFigure,
                             facecolor="white", edgecolor=BORDA, linewidth=1, zorder=-10))
    fig.add_artist(Rectangle((0.03, 0.06), 0.012, 0.88, transform=fig.transFigure,
                             facecolor=MARINHO, edgecolor="none"))
    if etapa is not None:
        fig.text(0.07, 0.86, f"PASSO {etapa + 1} DE {len(ETAPAS)}", fontsize=10, fontweight="bold",
                 color=MARINHO, family="DejaVu Sans")
        fig.text(0.07, 0.79, ETAPAS[etapa], fontsize=21, fontweight="bold", color=MARINHO_ESCURO)
        # Barra de progresso das etapas
        for i in range(len(ETAPAS)):
            fig.add_artist(Rectangle((0.07 + i * 0.172, 0.095), 0.16, 0.012, transform=fig.transFigure,
                                     facecolor=MARINHO if i <= etapa else "#d4d7dd", edgecolor="none"))
    return fig


def para_imagem(fig) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def eixo_grafico(fig, posicao=(0.12, 0.2, 0.8, 0.52)):
    ax = fig.add_axes(posicao)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(BORDA)
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=TEXTO, labelsize=9)
    ax.grid(True, color="#eceef2", linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


# -----------------------------------------------------------------------------
# Cenas
# -----------------------------------------------------------------------------
def cena_abertura(progresso: float):
    fig = nova_cena()
    alfa = suavizar(progresso * 2)
    fig.text(0.08, 0.66, "PREVISÃO DE RISCOS FINANCEIROS", fontsize=12, fontweight="bold",
             color=MARINHO, alpha=alfa)
    fig.text(0.08, 0.50, "Antecipe a volatilidade.", fontsize=32, fontweight="bold",
             color=MARINHO_ESCURO, alpha=alfa)
    fig.text(0.08, 0.38, "Decida com dados.", fontsize=32, fontweight="bold",
             color=MARINHO_ESCURO, alpha=suavizar(progresso * 2 - 0.4))
    fig.text(0.08, 0.24, "Machine Learning, explicabilidade e Value at Risk para ativos da B3",
             fontsize=13, color=SUAVE, alpha=suavizar(progresso * 2 - 0.8))
    return para_imagem(fig)


def cena_ativos(progresso: float):
    fig = nova_cena(0)
    ativos = ["^BVSP", "^GSPC", "PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"]
    escolhido = min(int(suavizar(progresso * 1.3) * len(ativos)), len(ativos) - 1)
    # O destaque percorre a lista e termina no ativo de referência
    destaque = 0 if progresso > 0.75 else escolhido
    for i, nome in enumerate(ativos):
        x, y = 0.08 + (i % 4) * 0.215, 0.53 - (i // 4) * 0.2
        ativo = i == destaque
        fig.add_artist(Rectangle((x, y), 0.19, 0.14, transform=fig.transFigure,
                                 facecolor="#eef2fb" if ativo else "white",
                                 edgecolor=MARINHO if ativo else "#d4d7dd", linewidth=2 if ativo else 1))
        fig.text(x + 0.095, y + 0.07, nome, ha="center", va="center", fontsize=15, fontweight="bold",
                 color=MARINHO if ativo else SUAVE, family="DejaVu Sans Mono")
    fig.text(0.08, 0.2, "Na barra lateral, selecione uma ação da B3 ou um índice de mercado.",
             fontsize=12, color=SUAVE)
    return para_imagem(fig)


def cena_volatilidade(progresso: float, df: pd.DataFrame):
    fig = nova_cena(1)
    ax = eixo_grafico(fig)
    serie = df.dropna(subset=["Volatilidade_21d"])
    n = max(2, int(len(serie) * suavizar(progresso * 1.15)))
    parte = serie.iloc[:n]
    ax.plot(parte["Data"], parte["Volatilidade_21d"] * 100, color=VINHO, linewidth=1.2)
    ax.fill_between(parte["Data"], parte["Volatilidade_21d"] * 100, color=VINHO, alpha=0.12)
    ax.set_xlim(serie["Data"].iloc[0], serie["Data"].iloc[-1])
    ax.set_ylim(0, serie["Volatilidade_21d"].max() * 105)
    ax.set_ylabel("Volatilidade anualizada (%)", fontsize=9, color=SUAVE)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: numero_br(v, 0)))
    return para_imagem(fig)


def cena_modelos(progresso: float, metricas: pd.DataFrame):
    fig = nova_cena(2)
    ax = eixo_grafico(fig, (0.3, 0.2, 0.6, 0.5))
    nomes = [m.replace("Baseline Ingênuo (Persistência)", "Baseline (persistência)") for m in metricas["Modelo"]][::-1]
    valores = metricas["RMSE"].values[::-1]
    fator = suavizar(progresso * 1.4)
    cores = [MARINHO if "Baseline" not in n else "#9ca3af" for n in nomes]
    barras = ax.barh(nomes, valores * fator, color=cores, height=0.55)
    ax.set_xlim(0, valores.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSE no conjunto de teste (menor é melhor)", fontsize=9, color=SUAVE)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: numero_br(v, 3)))
    if fator > 0.95:
        for barra, valor in zip(barras, valores):
            ax.text(barra.get_width() + valores.max() * 0.02, barra.get_y() + barra.get_height() / 2,
                    numero_br(valor, 5), va="center", fontsize=9, color=TEXTO)
    return para_imagem(fig)


def cena_xai(progresso: float, importancia: pd.Series):
    fig = nova_cena(3)
    ax = eixo_grafico(fig, (0.3, 0.2, 0.6, 0.5))
    fator = suavizar(progresso * 1.4)
    ax.barh(importancia.index, importancia.values * fator, color=MARINHO, height=0.6)
    ax.set_xlim(0, importancia.max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Impacto médio na previsão (média de |SHAP|)", fontsize=9, color=SUAVE)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: numero_br(v, 2)))
    return para_imagem(fig)


def cena_risco(progresso: float, vol_prevista: float, var_95: float):
    fig = nova_cena(4)
    fator = suavizar(progresso * 1.5)
    fig.add_artist(Rectangle((0.08, 0.3), 0.38, 0.38, transform=fig.transFigure,
                             facecolor="white", edgecolor=BORDA, linewidth=1))
    fig.add_artist(Rectangle((0.08, 0.66), 0.38, 0.02, transform=fig.transFigure,
                             facecolor=MARINHO, edgecolor="none"))
    fig.text(0.1, 0.58, "VOLATILIDADE PROJETADA (t+1)", fontsize=10, fontweight="bold", color=SUAVE)
    fig.text(0.1, 0.42, f"{numero_br(vol_prevista * 100 * fator, 2)}%", fontsize=34, fontweight="bold",
             color=TEXTO, family="DejaVu Sans Mono")
    fig.add_artist(Rectangle((0.52, 0.3), 0.4, 0.38, transform=fig.transFigure,
                             facecolor="white", edgecolor=BORDA, linewidth=1))
    fig.add_artist(Rectangle((0.52, 0.66), 0.4, 0.02, transform=fig.transFigure,
                             facecolor="#b91c1c", edgecolor="none"))
    fig.text(0.54, 0.58, "VaR DIÁRIO 95% (R$ 100.000)", fontsize=10, fontweight="bold", color=SUAVE)
    fig.text(0.54, 0.42, f"R$ {numero_br(var_95 * fator, 2)}", fontsize=30, fontweight="bold",
             color="#c2410c", family="DejaVu Sans Mono")
    fig.text(0.08, 0.2, "Clique em Get Started e comece a sua análise.", fontsize=13,
             fontweight="bold", color=MARINHO, alpha=suavizar(progresso * 2 - 0.8))
    return para_imagem(fig)


def main():
    df, metricas, importancia, vol_prevista, var_95 = carregar_dados()
    roteiro = [
        (30, cena_abertura, ()),
        (32, cena_ativos, ()),
        (38, cena_volatilidade, (df,)),
        (34, cena_modelos, (metricas,)),
        (34, cena_xai, (importancia,)),
        (40, cena_risco, (vol_prevista, var_95)),
    ]
    quadros = []
    for total, cena, argumentos in roteiro:
        for i in range(total):
            quadros.append(cena(i / (total - 1), *argumentos))

    # Paleta adaptativa única para todo o vídeo (cores estáveis entre quadros)
    paleta = quadros[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quadros_gif = [q.quantize(palette=paleta, dither=Image.Dither.NONE) for q in quadros]

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    quadros_gif[0].save(
        SAIDA,
        save_all=True,
        append_images=quadros_gif[1:],
        duration=int(1000 / QUADROS_POR_SEGUNDO),
        loop=0,
        optimize=True,
    )
    print(f"{len(quadros)} quadros ({len(quadros) / QUADROS_POR_SEGUNDO:.0f} s) salvos em {SAIDA}")
    print(f"Tamanho: {SAIDA.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
