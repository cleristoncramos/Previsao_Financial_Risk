"""
Módulo de Avaliação e Validação de Modelos.

Fornece:
- Cálculo de métricas de regressão: RMSE, MAE e R²
- Validação Cruzada Temporal (TimeSeriesSplit)
- Visualização interativa Real vs. Modelos (Plotly)
- Curva de aprendizado da Rede Neural (Matplotlib)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.config import FIGURES_DIR, RANDOM_STATE


def calcular_metricas(nome_modelo: str, y_real: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calcula as métricas RMSE, MAE e R² para um modelo preditivo.
    """
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    return {
        "Modelo": nome_modelo,
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
    }


def avaliar_todos_modelos(y_real: np.ndarray, dict_predicoes: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Consolida as métricas de todos os modelos em um DataFrame ordenado pelo menor RMSE.
    """
    resultados: List[Dict[str, Any]] = []
    print("\n" + "=" * 60)
    print(" ETAPA 6: AVALIAÇÃO DE DESEMPENHO DOS MODELOS")
    print("=" * 60)

    for nome, y_pred in dict_predicoes.items():
        res = calcular_metricas(nome, y_real, y_pred)
        resultados.append(res)
        print(f"  [{nome:<18}]  RMSE: {res['RMSE']:.5f}  |  MAE: {res['MAE']:.5f}  |  R²: {res['R2']:.4f}")

    df_res = pd.DataFrame(resultados).sort_values("RMSE").reset_index(drop=True)
    return df_res


def adicionar_baseline_ingenuo(
    features: List[str],
    X_test_unscaled: np.ndarray,
    dict_predicoes: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Adiciona o Modelo de Persistência (Random Walk / Naive: sigma_{t+1} = sigma_t)
    ao dicionário de predições para servir de benchmarking de referência.
    """
    preds = dict(dict_predicoes)
    col_vol = [f for f in features if f.startswith("Volatilidade_")]
    if col_vol:
        idx_vol = features.index(col_vol[0])
        preds["Baseline Ingênuo (Persistência)"] = X_test_unscaled[:, idx_vol]
    return preds


def validacao_cruzada_temporal(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> List[float]:
    """
    Executa validação cruzada temporal (TimeSeriesSplit) com XGBoost
    para assegurar robustez temporal e ausência de viés de overfitting.
    """
    print(f"\n-> Validação Cruzada Temporal ({n_splits} folds) — XGBoost...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses: List[float] = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]

        sc = StandardScaler().fit(Xtr)
        m = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_state,
            verbosity=0,
        )
        m.fit(sc.transform(Xtr), ytr, verbose=False)
        pred = m.predict(sc.transform(Xval))
        rmse = np.sqrt(mean_squared_error(yval, pred))
        rmses.append(rmse)
        print(f"   Fold {fold}: RMSE = {rmse:.5f}")

    print(f"  ==> Média RMSE CV: {np.mean(rmses):.5f}  ±  {np.std(rmses):.5f}")
    return rmses


def plotar_real_vs_predicoes(
    datas_test: np.ndarray,
    y_real: np.ndarray,
    dict_predicoes: Dict[str, np.ndarray],
    salvar: bool = True,
    mostrar: bool = True,
    pasta_figuras: Path = FIGURES_DIR,
) -> go.Figure:
    """
    Gera gráfico interativo comparativo de séries temporais: Valores Reais vs. Modelos.
    """
    datas_dt = pd.to_datetime(datas_test)
    fig = go.Figure()

    # Linha Real
    fig.add_trace(
        go.Scatter(
            x=datas_dt,
            y=y_real,
            name="Real",
            line=dict(color="#1f77b4", width=2),
        )
    )

    estilos = {
        "XGBoost": dict(color="#ff7f0e", dash="dash"),
        "LightGBM": dict(color="#2ca02c", dash="dot"),
        "Rede Neural (MLP)": dict(color="#9467bd", dash="dashdot"),
        "Baseline Ingênuo (Persistência)": dict(color="#7f7f7f", dash="dot"),
    }

    for nome, y_pred in dict_predicoes.items():
        estilo = estilos.get(nome, dict(dash="dash"))
        fig.add_trace(
            go.Scatter(
                x=datas_dt,
                y=y_pred,
                name=nome,
                line=estilo,
            )
        )

    fig.update_layout(
        title="Previsão de Volatilidade — Real vs. Modelos (Conjunto de Teste)",
        xaxis_title="Data",
        yaxis_title="Volatilidade (anualizada)",
        hovermode="x unified",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", y=1.02),
    )

    if salvar:
        caminho = pasta_figuras / "avaliacao_real_vs_modelos.html"
        fig.write_html(str(caminho))
        print(f"  [Gráfico Salvo] {caminho}")

    if mostrar:
        fig.show()

    return fig


# Alias de compatibilidade
plotar_comparativo_real_vs_predicoes = plotar_real_vs_predicoes


def plotar_curva_aprendizado_nn(
    history: Any,
    salvar: bool = True,
    mostrar: bool = True,
    pasta_figuras: Path = FIGURES_DIR,
) -> Optional[plt.Figure]:
    """
    Exibe as curvas de perda (MSE) de treino e validação da Rede Neural MLP.
    """
    if history is None or not hasattr(history, "history"):
        return None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(history.history["loss"], label="Treino (MSE)", color="steelblue", linewidth=1.5)
    ax.plot(history.history["val_loss"], label="Validação (MSE)", color="coral", linewidth=1.5)
    ax.set_title("Curva de Aprendizado — Rede Neural MLP")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    plt.tight_layout()

    if salvar:
        caminho = pasta_figuras / "avaliacao_curva_aprendizado_nn.png"
        fig.savefig(caminho, bbox_inches="tight")
        print(f"  [Gráfico Salvo] {caminho}")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

    return fig
