"""
Gera o vídeo curto de apresentação da página inicial (assets/video/apresentacao.gif).

Animação de ~25 segundos com dados reais: cartões com os logotipos e a minissérie de
preço de cada ativo, a volatilidade do ativo de referência (^BVSP) com o pico da
pandemia, o RMSE dos modelos salvos, a importância SHAP do XGBoost e um medidor do
regime de risco com o VaR projetado. Transições suaves entre as cenas e elementos
com entrada em sequência. Formato GIF para tocar automaticamente no navegador sem
dependências externas (ffmpeg).

Pré-requisito: logotipos em PNG (python scripts/rasterizar_logos.py).

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
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge
from matplotlib.ticker import FuncFormatter
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import MODELS_DIR, TICKER_REF
from src.data_collection import carregar_ativo_local
from src.model_store import carregar_modelos_salvos

SAIDA = BASE_DIR / "assets" / "video" / "apresentacao.gif"
LOGOS = BASE_DIR / "assets" / "ativos" / "png"
LARGURA, ALTURA, DPI = 12.8, 7.2, 100  # 1280 x 720 px
QPS = 12  # quadros por segundo
QUADROS_TRANSICAO = 6

MARINHO = "#1e3a8a"
MARINHO_ESCURO = "#172554"
TEXTO = "#1f2937"
SUAVE = "#5b6472"
BORDA = "#3f4654"
BORDA_SUAVE = "#d4d7dd"
FUNDO = "#f4f5f7"
VINHO = "#9f1239"
CORES_REGIME = ["#047857", "#a16207", "#c2410c", "#b91c1c"]

ATIVOS = [
    ("^BVSP", "BVSP", "Ibovespa"),
    ("^GSPC", "GSPC", "S&P 500"),
    ("PETR4.SA", "PETR4", "Petrobras"),
    ("VALE3.SA", "VALE3", "Vale"),
    ("ITUB4.SA", "ITUB4", "Itaú Unibanco"),
    ("BBDC4.SA", "BBDC4", "Bradesco"),
    ("ABEV3.SA", "ABEV3", "Ambev"),
]
ETAPAS = ["Ativo", "Volatilidade", "Modelos", "Explicação", "Risco"]


# -----------------------------------------------------------------------------
# Utilitários
# -----------------------------------------------------------------------------
def numero_br(valor: float, casas: int = 2) -> str:
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def suave(t: float) -> float:
    """Curva de animação com entrada e saída suaves, para t em [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def atraso(progresso: float, inicio: float, duracao: float = 0.25) -> float:
    """Progresso local de um elemento que começa em `inicio` (entrada em sequência)."""
    return suave((progresso - inicio) / duracao)


def carregar_logo(arquivo: str) -> np.ndarray:
    return np.asarray(Image.open(LOGOS / f"{arquivo}.png").convert("RGBA"))


def desenhar_logo(fig, imagem: np.ndarray, x, y, largura, altura, alfa=1.0):
    """Logotipo centralizado numa caixa (coordenadas da figura), preservando a proporção."""
    if alfa <= 0:
        return
    proporcao_img = imagem.shape[1] / imagem.shape[0]
    proporcao_caixa = (largura * LARGURA) / (altura * ALTURA)
    if proporcao_img > proporcao_caixa:
        nova_altura = largura * LARGURA / proporcao_img / ALTURA
        y, altura = y + (altura - nova_altura) / 2, nova_altura
    else:
        nova_largura = altura * ALTURA * proporcao_img / LARGURA
        x, largura = x + (largura - nova_largura) / 2, nova_largura
    ax = fig.add_axes([x, y, largura, altura])
    ax.imshow(imagem, alpha=alfa, interpolation="lanczos")
    ax.axis("off")


def retangulo(fig, x, y, largura, altura, **estilo):
    # Por padrão atrás dos eixos (logotipos e gráficos), que ficam em zorder 0
    estilo.setdefault("zorder", -5)
    fig.add_artist(Rectangle((x, y), largura, altura, transform=fig.transFigure, **estilo))


def para_imagem(fig) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def eixo(fig, posicao):
    ax = fig.add_axes(posicao)
    ax.set_facecolor("none")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(BORDA)
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=TEXTO, labelsize=10)
    ax.grid(True, color="#eceef2", linewidth=0.8)
    ax.set_axisbelow(True)
    return ax


# -----------------------------------------------------------------------------
# Moldura: barra superior com as etapas e barra de progresso contínua
# -----------------------------------------------------------------------------
def moldura(etapa, progresso_total):
    fig = plt.figure(figsize=(LARGURA, ALTURA), dpi=DPI)
    fig.patch.set_facecolor(FUNDO)
    retangulo(fig, 0.025, 0.045, 0.95, 0.91, facecolor="white", edgecolor=BORDA, linewidth=1, zorder=-10)
    retangulo(fig, 0.025, 0.045, 0.008, 0.91, facecolor=MARINHO, edgecolor="none", zorder=-9)
    fig.text(0.055, 0.905, "PREVISÃO DE RISCOS FINANCEIROS", fontsize=11, fontweight="bold", color=MARINHO)
    if etapa is not None:
        for i, nome in enumerate(ETAPAS):
            x = 0.42 + i * 0.108
            ativo = i == etapa
            concluido = i < etapa
            fig.add_artist(Circle((x, 0.912), 0.011, transform=fig.transFigure,
                                  facecolor=MARINHO if (ativo or concluido) else "white",
                                  edgecolor=MARINHO if (ativo or concluido) else BORDA_SUAVE, linewidth=1.2))
            fig.text(x, 0.912, str(i + 1), ha="center", va="center", fontsize=8.5, fontweight="bold",
                     color="white" if (ativo or concluido) else SUAVE)
            fig.text(x + 0.016, 0.912, nome, va="center", fontsize=9.5,
                     fontweight="bold" if ativo else "normal", color=MARINHO_ESCURO if ativo else SUAVE)
    # Progresso contínuo do vídeo inteiro
    retangulo(fig, 0.055, 0.075, 0.89, 0.006, facecolor="#eceef2", edgecolor="none")
    retangulo(fig, 0.055, 0.075, 0.89 * progresso_total, 0.006, facecolor=MARINHO, edgecolor="none")
    return fig


def titulo_cena(fig, rotulo, titulo, progresso):
    alfa = atraso(progresso, 0.0, 0.18)
    deslocamento = 0.02 * (1 - alfa)
    fig.text(0.055, 0.815 - deslocamento, rotulo, fontsize=11, fontweight="bold", color=MARINHO, alpha=alfa)
    fig.text(0.055, 0.745 - deslocamento, titulo, fontsize=26, fontweight="bold", color=MARINHO_ESCURO, alpha=alfa)


# -----------------------------------------------------------------------------
# Dados reais usados nas cenas
# -----------------------------------------------------------------------------
def carregar_dados():
    dados = {}
    for ticker, arquivo, nome in ATIVOS:
        df = carregar_ativo_local(ticker)
        df["Data"] = pd.to_datetime(df["Data"])
        df = df.sort_values("Data").reset_index(drop=True)
        ultimo_ano = df.tail(252)
        dados[ticker] = {
            "nome": nome,
            "logo": carregar_logo(arquivo),
            "precos": ultimo_ano["Close"].values,
            "variacao": (ultimo_ano["Close"].iloc[-1] / ultimo_ano["Close"].iloc[0] - 1) * 100,
        }

    df_ref = carregar_ativo_local(TICKER_REF)
    df_ref["Data"] = pd.to_datetime(df_ref["Data"])
    df_ref = df_ref.sort_values("Data").reset_index(drop=True)

    metadados = json.loads((MODELS_DIR / "metadados_modelos.json").read_text(encoding="utf-8"))
    metricas = pd.DataFrame(metadados["metricas"]).sort_values("RMSE")

    modelos, scaler, _ = carregar_modelos_salvos(MODELS_DIR)
    features = metadados["features"]
    base = df_ref.dropna(subset=features).copy()
    corte = int(len(base) * 0.8)
    X_teste = scaler.transform(base[features].values[corte:])

    import shap

    valores_shap = np.asarray(shap.TreeExplainer(modelos["XGBoost"]).shap_values(X_teste))
    importancia = pd.Series(np.abs(valores_shap).mean(axis=0), index=features).sort_values()

    vol_prevista = float(modelos["XGBoost"].predict(scaler.transform(base[features].values[-1:]))[0]) * 100
    var_95 = 1.644853 * vol_prevista / 100 / np.sqrt(252) * 100_000
    vol_hist = df_ref["Volatilidade_21d"].dropna() * 100
    limites = np.percentile(vol_hist, [25, 75, 95])

    return dados, df_ref, metricas, importancia, vol_prevista, var_95, limites


# -----------------------------------------------------------------------------
# Cenas (cada uma recebe o progresso local 0..1 e o progresso total do vídeo)
# -----------------------------------------------------------------------------
def cena_abertura(p, total, dados):
    fig = moldura(None, total)
    # Logotipos entram em sequência numa faixa
    for i, (ticker, _, _) in enumerate(ATIVOS):
        alfa = atraso(p, 0.02 + i * 0.06, 0.2)
        x = 0.07 + i * 0.125
        retangulo(fig, x, 0.62 - 0.03 * (1 - alfa), 0.105, 0.13, facecolor="white",
                  edgecolor=BORDA_SUAVE if alfa > 0 else "none", linewidth=1, alpha=alfa)
        desenhar_logo(fig, dados[ticker]["logo"], x + 0.012, 0.635 - 0.03 * (1 - alfa), 0.081, 0.1, alfa)
    a1, a2, a3 = atraso(p, 0.45, 0.2), atraso(p, 0.55, 0.2), atraso(p, 0.68, 0.2)
    fig.text(0.07, 0.43 + 0.02 * (1 - a1), "Antecipe a volatilidade.", fontsize=40, fontweight="bold",
             color=MARINHO_ESCURO, alpha=a1)
    fig.text(0.07, 0.31 + 0.02 * (1 - a2), "Decida com dados.", fontsize=40, fontweight="bold",
             color=MARINHO, alpha=a2)
    fig.text(0.07, 0.2, "Machine Learning, explicabilidade e Value at Risk para ações da B3 e índices",
             fontsize=15, color=SUAVE, alpha=a3)
    return para_imagem(fig)


def cena_ativos(p, total, dados):
    fig = moldura(0, total)
    titulo_cena(fig, "PASSO 1", "Escolha o ativo", p)
    # Destaque percorre os cartões e para no ativo de referência
    ordem = [1, 2, 3, 4, 5, 6, 0]
    if p < 0.35:
        destaque = None
    elif p < 0.78:
        destaque = ordem[min(int((p - 0.35) / 0.43 * len(ordem)), len(ordem) - 1)]
    else:
        destaque = 0
    for i, (ticker, _, _) in enumerate(ATIVOS):
        alfa = atraso(p, 0.08 + i * 0.035, 0.18)
        coluna, linha = i % 4, i // 4
        x, y = 0.055 + coluna * 0.225, 0.4 - linha * 0.24 + 0.03 * (1 - alfa)
        selecionado = destaque == i
        retangulo(fig, x, y, 0.205, 0.205, facecolor="#eef2fb" if selecionado else "white",
                  edgecolor=MARINHO if selecionado else BORDA_SUAVE, linewidth=2.2 if selecionado else 1,
                  alpha=alfa)
        if selecionado:
            retangulo(fig, x, y + 0.199, 0.205, 0.006, facecolor=MARINHO, edgecolor="none")
        info = dados[ticker]
        desenhar_logo(fig, info["logo"], x + 0.015, y + 0.115, 0.1, 0.07, alfa)
        fig.text(x + 0.015, y + 0.085, info["nome"], fontsize=10.5, fontweight="bold", color=TEXTO, alpha=alfa)
        cor = "#047857" if info["variacao"] >= 0 else "#b91c1c"
        sinal = "+" if info["variacao"] >= 0 else ""
        fig.text(x + 0.19, y + 0.085, f"{sinal}{numero_br(info['variacao'], 1)}% 12m", ha="right",
                 fontsize=9, color=cor, alpha=alfa)
        # Minissérie de preço dos últimos 12 meses
        if alfa > 0:
            ax = fig.add_axes([x + 0.015, y + 0.018, 0.175, 0.055])
            precos = info["precos"]
            n = max(2, int(len(precos) * atraso(p, 0.15 + i * 0.035, 0.3)))
            ax.plot(range(n), precos[:n], color=cor, linewidth=1.4, alpha=alfa)
            ax.fill_between(range(n), precos[:n], precos.min(), color=cor, alpha=0.08 * alfa)
            ax.set_xlim(0, len(precos) - 1)
            ax.set_ylim(precos.min(), precos.max())
            ax.axis("off")
    if destaque == 0:
        fig.text(0.055, 0.618, "Ativo de referência do estudo", fontsize=9.5, fontweight="bold",
                 color=MARINHO, alpha=atraso(p, 0.8, 0.12))
    return para_imagem(fig)


def cena_volatilidade(p, total, dados, df):
    fig = moldura(1, total)
    titulo_cena(fig, "PASSO 2", "Explore a volatilidade", p)
    desenhar_logo(fig, dados[TICKER_REF]["logo"], 0.82, 0.73, 0.1, 0.07, atraso(p, 0.05, 0.2))
    ax = eixo(fig, (0.1, 0.17, 0.83, 0.5))
    serie = df.dropna(subset=["Volatilidade_21d"])
    datas, vol = serie["Data"].values, serie["Volatilidade_21d"].values * 100
    n = max(2, int(len(serie) * atraso(p, 0.1, 0.7)))
    ax.plot(datas[:n], vol[:n], color=VINHO, linewidth=1.3)
    ax.fill_between(datas[:n], vol[:n], color=VINHO, alpha=0.1)
    ax.scatter([datas[n - 1]], [vol[n - 1]], s=40, color=VINHO, zorder=5)
    ax.set_xlim(datas[0], datas[-1])
    ax.set_ylim(0, vol.max() * 1.12)
    ax.set_ylabel("Volatilidade anualizada (%)", fontsize=10, color=SUAVE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: numero_br(v, 0)))
    pico = int(np.argmax(vol))
    if n > pico:
        alfa = atraso(p, 0.1 + 0.7 * pico / len(serie), 0.15)
        ax.annotate(
            f"Pandemia: {numero_br(vol[pico], 0)}%\n({pd.Timestamp(datas[pico]):%m/%Y})",
            xy=(datas[pico], vol[pico]), xytext=(datas[pico] + np.timedelta64(420, "D"), vol[pico] * 0.92),
            fontsize=10.5, fontweight="bold", color=VINHO, alpha=alfa,
            arrowprops=dict(arrowstyle="-", color=VINHO, alpha=alfa, linewidth=1),
        )
    return para_imagem(fig)


def cena_modelos(p, total, metricas):
    fig = moldura(2, total)
    titulo_cena(fig, "PASSO 3", "Compare os modelos", p)
    ax = eixo(fig, (0.3, 0.2, 0.6, 0.46))
    nomes = [m.replace("Baseline Ingênuo (Persistência)", "Baseline (persistência)") for m in metricas["Modelo"]]
    valores = metricas["RMSE"].values
    ax.set_xlim(0, valores.max() * 1.32)
    ax.set_ylim(-0.6, len(nomes) - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(range(len(nomes)), nomes, fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RMSE no conjunto de teste (menor é melhor)", fontsize=10, color=SUAVE)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero_br(v, 3)))
    for i, (nome, valor) in enumerate(zip(nomes, valores)):
        fator = atraso(p, 0.12 + i * 0.1, 0.3)
        cor = MARINHO if i == 0 else "#9ca3af" if "Baseline" in nome else "#7d8fc4"
        ax.barh(i, valor * fator, color=cor, height=0.58)
        if fator > 0.9:
            ax.text(valor + valores.max() * 0.02, i, numero_br(valor, 5), va="center", fontsize=10.5,
                    color=TEXTO, alpha=suave((fator - 0.9) * 10))
    alfa = atraso(p, 0.7, 0.15)
    fig.text(0.3, 0.685, f"Menor erro: {nomes[0]}", fontsize=12, fontweight="bold", color=MARINHO, alpha=alfa)
    return para_imagem(fig)


def cena_xai(p, total, importancia):
    fig = moldura(3, total)
    titulo_cena(fig, "PASSO 4", "Entenda cada previsão", p)
    ax = eixo(fig, (0.3, 0.2, 0.6, 0.46))
    nomes, valores = list(importancia.index)[::-1], importancia.values[::-1]
    ax.set_xlim(0, valores.max() * 1.18)
    ax.set_ylim(-0.6, len(nomes) - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(range(len(nomes)), nomes, fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Impacto médio na previsão (média de |SHAP|)", fontsize=10, color=SUAVE)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: numero_br(v, 2)))
    for i, valor in enumerate(valores):
        fator = atraso(p, 0.1 + i * 0.05, 0.3)
        ax.barh(i, valor * fator, color=MARINHO if i == 0 else "#7d8fc4", height=0.62)
    fig.text(0.3, 0.685, f"{nomes[0]} é a variável que mais pesa na previsão",
             fontsize=12, fontweight="bold", color=MARINHO, alpha=atraso(p, 0.7, 0.15))
    return para_imagem(fig)


def cena_risco(p, total, vol_prevista, var_95, limites):
    fig = moldura(4, total)
    titulo_cena(fig, "PASSO 5", "Projete o risco", p)
    # Medidor do regime de risco (faixas pelos percentis históricos do ativo)
    ax = fig.add_axes([0.06, 0.13, 0.42, 0.58])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.35, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    escala_max = max(limites[2] * 1.35, vol_prevista * 1.2)
    bordas = [0, limites[0], limites[1], limites[2], escala_max]
    nomes_faixas = ["Baixo", "Moderado", "Elevado", "Extremo"]
    angulo = lambda v: 180 - 180 * min(v, escala_max) / escala_max
    for i in range(4):
        alfa = atraso(p, 0.08 + i * 0.06, 0.15)
        ax.add_patch(Wedge((0, 0), 1, angulo(bordas[i + 1]), angulo(bordas[i]), width=0.24,
                           facecolor=CORES_REGIME[i], alpha=0.85 * alfa, edgecolor="white", linewidth=2))
        meio = np.radians((angulo(bordas[i]) + angulo(bordas[i + 1])) / 2)
        ax.text(1.17 * np.cos(meio), 1.17 * np.sin(meio), nomes_faixas[i], ha="center", va="center",
                fontsize=9.5, color=CORES_REGIME[i], fontweight="bold", alpha=alfa)
    valor_agulha = vol_prevista * atraso(p, 0.3, 0.4)
    teta = np.radians(angulo(valor_agulha))
    ax.add_patch(Polygon([(0.78 * np.cos(teta), 0.78 * np.sin(teta)),
                          (0.05 * np.cos(teta + np.pi / 2), 0.05 * np.sin(teta + np.pi / 2)),
                          (0.05 * np.cos(teta - np.pi / 2), 0.05 * np.sin(teta - np.pi / 2))],
                         facecolor=MARINHO_ESCURO, alpha=atraso(p, 0.25, 0.1)))
    ax.add_patch(Circle((0, 0), 0.07, facecolor=MARINHO_ESCURO, alpha=atraso(p, 0.25, 0.1)))
    ax.text(0, -0.2, f"{numero_br(valor_agulha, 2)}%", ha="center", va="center", fontsize=26,
            fontweight="bold", color=TEXTO, family="DejaVu Sans Mono", alpha=atraso(p, 0.25, 0.1))
    fig.text(0.27, 0.15, "Volatilidade projetada para o próximo pregão", ha="center", fontsize=10.5,
             color=SUAVE, alpha=atraso(p, 0.25, 0.1))

    # Cartão do VaR
    alfa = atraso(p, 0.45, 0.18)
    retangulo(fig, 0.56, 0.3, 0.36, 0.34, facecolor="white", edgecolor=BORDA, linewidth=1, alpha=alfa)
    retangulo(fig, 0.56, 0.632, 0.36, 0.008, facecolor="#b91c1c", edgecolor="none", alpha=alfa)
    fig.text(0.58, 0.565, "VaR DIÁRIO 95%", fontsize=11, fontweight="bold", color=SUAVE, alpha=alfa)
    fig.text(0.58, 0.525, "para R$ 100.000 aplicados", fontsize=10, color=SUAVE, alpha=alfa)
    fig.text(0.58, 0.39, f"R$ {numero_br(var_95 * atraso(p, 0.5, 0.35), 2)}", fontsize=34, fontweight="bold",
             color="#c2410c", family="DejaVu Sans Mono", alpha=alfa)
    return para_imagem(fig)


def cena_final(p, total, dados):
    fig = moldura(None, total)
    a1 = atraso(p, 0.0, 0.25)
    fig.text(0.5, 0.66, "Pronto para analisar o risco do próximo pregão?", ha="center", fontsize=28,
             fontweight="bold", color=MARINHO_ESCURO, alpha=a1)
    for i, (ticker, _, _) in enumerate(ATIVOS):
        alfa = atraso(p, 0.12 + i * 0.04, 0.2)
        desenhar_logo(fig, dados[ticker]["logo"], 0.07 + i * 0.125, 0.43, 0.105, 0.11, alfa)
    # Botão "Get Started" com pulso suave
    a2 = atraso(p, 0.45, 0.2)
    pulso = 1 + 0.025 * np.sin(p * 4 * np.pi) * (p > 0.6)
    largura, altura = 0.2 * pulso, 0.085 * pulso
    fig.add_artist(FancyBboxPatch((0.5 - largura / 2, 0.22 - altura / 2), largura, altura,
                                  boxstyle="square,pad=0", transform=fig.transFigure,
                                  facecolor=MARINHO, edgecolor="none", alpha=a2))
    fig.text(0.5, 0.22, "Get Started  →", ha="center", va="center", fontsize=17, fontweight="bold",
             color="white", alpha=a2)
    return para_imagem(fig)


# -----------------------------------------------------------------------------
# Montagem
# -----------------------------------------------------------------------------
def transicao(a: Image.Image, b: Image.Image, n: int) -> list:
    """Quadros intermediários com dissolução entre o fim de uma cena e o início da seguinte."""
    return [Image.blend(a, b, suave((i + 1) / (n + 1))) for i in range(n)]


def main():
    dados, df_ref, metricas, importancia, vol_prevista, var_95, limites = carregar_dados()
    roteiro = [
        (3.2, lambda p, t: cena_abertura(p, t, dados)),
        (4.4, lambda p, t: cena_ativos(p, t, dados)),
        (4.0, lambda p, t: cena_volatilidade(p, t, dados, df_ref)),
        (3.4, lambda p, t: cena_modelos(p, t, metricas)),
        (3.4, lambda p, t: cena_xai(p, t, importancia)),
        (4.2, lambda p, t: cena_risco(p, t, vol_prevista, var_95, limites)),
        (2.8, lambda p, t: cena_final(p, t, dados)),
    ]
    total_quadros = sum(int(segundos * QPS) for segundos, _ in roteiro)
    cenas, contador = [], 0
    for segundos, cena in roteiro:
        n = int(segundos * QPS)
        quadros = []
        for i in range(n):
            quadros.append(cena(i / (n - 1), contador / (total_quadros - 1)))
            contador += 1
        cenas.append(quadros)

    video = list(cenas[0])
    for anterior, seguinte in zip(cenas, cenas[1:]):
        video += transicao(anterior[-1], seguinte[0], QUADROS_TRANSICAO) + seguinte

    # Paleta única para o vídeo inteiro (arquivo leve e cores estáveis entre os quadros).
    # A amostra usa o fim de cada cena e repete a faixa de logotipos da cena final, para que
    # as cores das marcas (amarelos da Vale e da Petrobras) tenham peso suficiente na paleta.
    largura, altura = video[0].width, video[0].height
    faixa_logos = cenas[-1][-1].crop((0, int(altura * 0.4), largura, int(altura * 0.62)))
    amostras = [quadros[-1] for quadros in cenas] + [faixa_logos] * 6
    amostra = Image.new("RGB", (largura, sum(a.height for a in amostras)))
    y = 0
    for a in amostras:
        amostra.paste(a, (0, y))
        y += a.height
    paleta = amostra.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quadros_gif = [q.quantize(palette=paleta, dither=Image.Dither.NONE) for q in video]

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    quadros_gif[0].save(
        SAIDA, save_all=True, append_images=quadros_gif[1:],
        duration=round(1000 / QPS), loop=0, optimize=True,
    )
    print(f"{len(video)} quadros ({len(video) / QPS:.1f} s, {QPS} quadros/s) salvos em {SAIDA}")
    print(f"Tamanho: {SAIDA.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
