"""
Módulo de Geração de Relatórios e Resumos Executivos.

Consolida os resultados numéricos do TCC, destacando o melhor modelo preditivo.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATA_FIM, DATA_INICIO, FIGURES_DIR, TICKER_REF, agora_brasilia

LARGURA = 80


def _numero_br(valor: float, casas: int) -> str:
    """Número no padrão brasileiro: ponto no milhar e vírgula nos decimais."""
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _data_br(data) -> str:
    """Data em dd/mm/aaaa (aceita string ISO, datetime ou Timestamp)."""
    return pd.Timestamp(data).strftime("%d/%m/%Y")


def _titulo_secao(titulo: str) -> list:
    return ["", f"  {titulo}", "  " + "-" * (LARGURA - 2)]


def classificar_modelos(df_resultados: pd.DataFrame) -> pd.Series:
    """Rótulo de cada modelo: Menor RMSE geral, Melhor modelo ML e/ou Benchmark de persistência."""
    eh_baseline = df_resultados["Modelo"].str.contains("Baseline", case=False)
    melhor_geral = df_resultados["Modelo"].iloc[0]
    modelos_ml = df_resultados.loc[~eh_baseline, "Modelo"]
    melhor_ml = modelos_ml.iloc[0] if not modelos_ml.empty else None

    def _rotulo(nome: str) -> str:
        rotulos = []
        if nome == melhor_geral:
            rotulos.append("Menor RMSE geral")
        if nome == melhor_ml:
            rotulos.append("Melhor modelo ML")
        if "Baseline" in nome:
            rotulos.append("Benchmark de persistência")
        return " / ".join(rotulos) if rotulos else "-"

    return df_resultados["Modelo"].map(_rotulo)


def montar_texto_relatorio(
    df_resultados: pd.DataFrame,
    ticker_ref: str,
    data_inicio,
    data_fim,
    n_train: int,
    n_test: int,
    gerado_em: Optional[datetime] = None,
    periodo_teste: Optional[tuple] = None,
    nome_ativo: Optional[str] = None,
    periodo_treino: Optional[tuple] = None,
    folds: Optional[pd.DataFrame] = None,
) -> str:
    """
    Monta o texto do relatório consolidado de desempenho no padrão brasileiro.

    Seções: identificação, dados e metodologia, desempenho no teste, comparação com o
    benchmark, validação cruzada temporal (se `folds` for informado) e conclusão.

    Args:
        df_resultados: Métricas por modelo (colunas Modelo, RMSE, MAE, R2), ordenadas por RMSE.
        ticker_ref: Ativo analisado.
        data_inicio, data_fim: Primeira e última data da série utilizada.
        n_train, n_test: Quantidade de amostras de treino e teste.
        gerado_em: Momento da geração (padrão: agora).
        periodo_teste, periodo_treino: (início, fim) de cada conjunto, opcionais.
        nome_ativo: Nome por extenso do ativo, opcional.
        folds: Resultado da validação cruzada (colunas Fold, Período de Validação, RMSE), opcional.
    """
    if df_resultados.empty:
        return ""

    gerado_em = gerado_em or agora_brasilia()
    eh_baseline = df_resultados["Modelo"].str.contains("Baseline", case=False)
    modelos_ml = df_resultados[~eh_baseline]
    melhor_ml = modelos_ml.iloc[0] if not modelos_ml.empty else df_resultados.iloc[0]
    rmse_baseline = float(df_resultados.loc[eh_baseline, "RMSE"].iloc[0]) if eh_baseline.any() else None
    melhor_geral = df_resultados.iloc[0]["Modelo"]

    ativo = f"{ticker_ref} ({nome_ativo})" if nome_ativo else ticker_ref
    linhas = [
        "=" * LARGURA,
        "  RELATÓRIO CONSOLIDADO DE DESEMPENHO DOS MODELOS",
        "  TCC — Previsão de Riscos Financeiros com Data Science e Machine Learning",
        "  Autora: Lorrana Evelyn de Araújo Pereira",
        "  Curso: Tecnologia em Gestão de Dados — UFPI / CEAD",
        f"  Gerado em: {gerado_em.strftime('%d/%m/%Y às %H:%M')}",
        "=" * LARGURA,
    ]

    # 1. Dados e metodologia
    linhas += _titulo_secao("1. DADOS E METODOLOGIA")
    linhas += [
        f"  Ativo analisado   : {ativo}",
        f"  Período da série  : {_data_br(data_inicio)} a {_data_br(data_fim)}",
    ]
    treino = f"{_numero_br(n_train, 0)} amostras"
    if periodo_treino is not None:
        treino += f" ({_data_br(periodo_treino[0])} a {_data_br(periodo_treino[1])})"
    teste = f"{_numero_br(n_test, 0)} amostras"
    if periodo_teste is not None:
        teste += f" ({_data_br(periodo_teste[0])} a {_data_br(periodo_teste[1])})"
    linhas += [
        f"  Conjunto de treino: {treino}",
        f"  Conjunto de teste : {teste}",
        "  Divisão           : temporal 80/20, sem embaralhamento",
        "  Variável-alvo     : volatilidade histórica anualizada (21 dias) em t+1",
    ]

    # 2. Desempenho no conjunto de teste
    linhas += _titulo_secao("2. DESEMPENHO NO CONJUNTO DE TESTE (ordenado por RMSE)")
    linhas.append(f"  {'Modelo':<33} {'RMSE':>9} {'MAE':>9} {'R²':>8}   Classificação")
    linhas.append("  " + "-" * (LARGURA - 2))
    for _, row in df_resultados.iterrows():
        # Rótulo curto para caber na linha; o asterisco marca o menor RMSE geral
        rotulo = "Benchmark" if "Baseline" in row["Modelo"] else ("Melhor ML" if row["Modelo"] == melhor_ml["Modelo"] else "-")
        if row["Modelo"] == melhor_geral:
            rotulo += " *"
        linhas.append(
            f"  {row['Modelo']:<33} {_numero_br(row['RMSE'], 5):>9} {_numero_br(row['MAE'], 5):>9} "
            f"{_numero_br(row['R2'], 4):>8}   {rotulo}"
        )
    linhas += [
        "  * Menor RMSE geral no conjunto de teste.",
        "  RMSE e MAE na escala da volatilidade anualizada (0,01 = 1 ponto percentual).",
    ]

    # 3. Comparação com o benchmark de persistência
    if rmse_baseline is not None and not modelos_ml.empty:
        ganho = (rmse_baseline - melhor_ml["RMSE"]) / rmse_baseline * 100
        linhas += _titulo_secao("3. COMPARAÇÃO COM O BENCHMARK DE PERSISTÊNCIA")
        linhas += [
            f"  Melhor modelo ML      : {melhor_ml['Modelo']} (RMSE = {_numero_br(melhor_ml['RMSE'], 5)})",
            f"  Baseline (σt+1 = σt)  : RMSE = {_numero_br(rmse_baseline, 5)}",
        ]
        if ganho > 0:
            linhas.append(f"  Resultado             : RMSE {_numero_br(ganho, 1)}% menor que o do baseline.")
        else:
            linhas.append(
                f"  Resultado             : baseline com RMSE {_numero_br(-ganho, 1)}% menor que o melhor ML."
            )

    # 4. Validação cruzada temporal
    if folds is not None and not folds.empty:
        linhas += _titulo_secao("4. VALIDAÇÃO CRUZADA TEMPORAL (XGBoost, TimeSeriesSplit com 5 folds)")
        linhas.append(f"  {'Fold':<8} {'Período de validação':<24} {'RMSE':>9}")
        for _, f in folds.iterrows():
            linhas.append(f"  {f['Fold']:<8} {f['Período de Validação']:<24} {_numero_br(f['RMSE'], 5):>9}")
        linhas.append(
            f"  Média ± desvio-padrão: {_numero_br(folds['RMSE'].mean(), 5)} ± {_numero_br(folds['RMSE'].std(ddof=0), 5)}"
        )

    # 5. Conclusão
    numero_conclusao = 5 if folds is not None and not folds.empty else 4
    linhas += _titulo_secao(f"{numero_conclusao}. CONCLUSÃO")
    if "Baseline" in melhor_geral:
        linhas += [
            "  No conjunto de teste, nenhum modelo de Machine Learning superou a regra de",
            "  persistência, coerente com a forte autocorrelação da volatilidade em janela",
            "  móvel de 21 dias.",
        ]
    else:
        linhas += [
            f"  O {melhor_geral} obteve o menor erro no conjunto de teste, superando a regra",
            "  de persistência usada como benchmark.",
        ]
    linhas.append("=" * LARGURA)
    return "\n".join(linhas)


def _csv_brasileiro(tabela: pd.DataFrame) -> bytes:
    """CSV para Excel em português: ';' entre colunas, vírgula decimal e UTF-8 com BOM."""
    return tabela.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def metricas_para_csv(df_resultados: pd.DataFrame, ticker: str, gerado_em: Optional[datetime] = None) -> bytes:
    """Tabela de métricas do ativo (uma linha por modelo, ordenada por RMSE) em CSV brasileiro."""
    gerado_em = gerado_em or agora_brasilia()
    tabela = pd.DataFrame(
        {
            "Ativo": ticker,
            "Posição": range(1, len(df_resultados) + 1),
            "Modelo": df_resultados["Modelo"].values,
            "RMSE": df_resultados["RMSE"].round(6).values,
            "MAE": df_resultados["MAE"].round(6).values,
            "R²": df_resultados["R2"].round(4).values,
            "Classificação": classificar_modelos(df_resultados).values,
            "Gerado em": gerado_em.strftime("%d/%m/%Y %H:%M"),
        }
    )
    return _csv_brasileiro(tabela)


# Colunas exportadas da série do ativo: (coluna original, nome no arquivo, casas decimais)
COLUNAS_DADOS = [
    ("Open", "Abertura", 4),
    ("High", "Máxima", 4),
    ("Low", "Mínima", 4),
    ("Close", "Fechamento ajustado", 4),
    ("Volume", "Volume", 0),
    ("Retorno_Diario", "Retorno diário (decimal)", 6),
    ("Volatilidade_21d", "Volatilidade anualizada 21d (decimal)", 6),
    ("Target_Vol", "Volatilidade t+1 - alvo (decimal)", 6),
    ("MM_20d", "Média móvel 20d", 4),
    ("MM_50d", "Média móvel 50d", 4),
    ("RSI_14d", "RSI 14d", 2),
    ("Volume_MM20", "Volume médio 20d", 0),
]


def dados_para_csv(df_ativo: pd.DataFrame) -> bytes:
    """
    Série do ativo com os atributos calculados, em CSV brasileiro: datas dd/mm/aaaa,
    nomes de colunas em português, casas decimais padronizadas e a variável-alvo (t+1).
    Células vazias no início da série correspondem ao aquecimento das janelas móveis.
    """
    base = df_ativo.sort_values("Data").reset_index(drop=True).copy()
    base["Target_Vol"] = base["Volatilidade_21d"].shift(-1)

    tabela = pd.DataFrame({"Data": pd.to_datetime(base["Data"]).dt.strftime("%d/%m/%Y")})
    for original, nome, casas in COLUNAS_DADOS:
        if original not in base.columns:
            continue
        coluna = base[original].round(casas)
        tabela[nome] = coluna.astype("Int64") if casas == 0 else coluna
    return _csv_brasileiro(tabela)


def gerar_relatorio_final(
    df_resultados: pd.DataFrame,
    ticker_ref: str = TICKER_REF,
    data_inicio: str = DATA_INICIO,
    data_fim: str = DATA_FIM,
    n_train: int = 0,
    n_test: int = 0,
    salvar: bool = True,
    pasta_saida: Path = FIGURES_DIR,
) -> str:
    """
    Formata, exibe e (opcionalmente) salva o relatório final de desempenho dos modelos.
    """
    texto_relatorio = montar_texto_relatorio(df_resultados, ticker_ref, data_inicio, data_fim, n_train, n_test)
    if not texto_relatorio:
        return ""

    print("\n" + texto_relatorio)

    if salvar:
        caminho = pasta_saida / "relatorio_final_modelos.txt"
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(texto_relatorio)
        print(f"\n  [Relatório Salvo] {caminho}")

        caminho_csv = pasta_saida / "metricas_modelos.csv"
        df_resultados.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
        print(f"  [Métricas CSV Salvas] {caminho_csv}")

    return texto_relatorio
