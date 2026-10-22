"""
Módulo de Interpretabilidade de Modelos (Explainable AI - XAI).

Implementa técnicas globais e locais de explicabilidade:
- SHAP (SHapley Additive exPlanations): TreeExplainer, Beeswarm e Waterfall
- LIME (Local Interpretable Model-agnostic Explanations): Explicação local tabular
"""

from pathlib import Path
from typing import Any, List, Optional

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import FIGURES_DIR, RANDOM_STATE


def explicar_com_shap(
    modelo_arvore: Any,
    X_test_scaled: np.ndarray,
    feature_names: List[str],
    idx_exemplo: int = 10,
    datas_test: Optional[np.ndarray] = None,
    salvar: bool = True,
    mostrar: bool = True,
    pasta_figuras: Path = FIGURES_DIR,
) -> Any:
    """
    Gera as análises SHAP para modelos baseados em árvore (ex: XGBoost).

    - Summary plot de barras (importância média dos valores absolutos)
    - Beeswarm plot (direção e intensidade do impacto)
    - Waterfall plot (decomposição aditiva da previsão para um dia específico)
    """
    print("\n" + "=" * 60)
    print(" ETAPA 7.1: INTERPRETABILIDADE GLOBAL & LOCAL COM SHAP")
    print("=" * 60)
    print("  -> Calculando valores SHAP (TreeExplainer)...")

    explainer = shap.TreeExplainer(modelo_arvore)
    shap_values = explainer.shap_values(X_test_scaled)

    # 1. Bar Summary Plot
    plt.figure(figsize=(10, 5))
    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP — Importância Global das Features (XGBoost)", pad=15)
    plt.tight_layout()

    if salvar:
        caminho_bar = pasta_figuras / "shap_importancia_global_bar.png"
        plt.savefig(caminho_bar, bbox_inches="tight")
        print(f"  [Gráfico Salvo] {caminho_bar}")

    if mostrar:
        plt.show()
    else:
        plt.close()

    # 2. Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_test_scaled,
        feature_names=feature_names,
        show=False,
    )
    plt.title("SHAP Beeswarm — Impacto de Cada Feature nas Previsões", pad=15)
    plt.tight_layout()

    if salvar:
        caminho_bee = pasta_figuras / "shap_beeswarm.png"
        plt.savefig(caminho_bee, bbox_inches="tight")
        print(f"  [Gráfico Salvo] {caminho_bee}")

    if mostrar:
        plt.show()
    else:
        plt.close()

    # 3. Waterfall Plot para uma instância de teste
    if idx_exemplo < len(X_test_scaled):
        data_str = ""
        if datas_test is not None:
            data_str = f" — Data: {pd.to_datetime(datas_test[idx_exemplo]).date()}"
        print(f"  -> Gerando SHAP Waterfall para o índice [{idx_exemplo}]{data_str}...")

        expected_val = (
            explainer.expected_value[0]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

        shap_exp = shap.Explanation(
            values=shap_values[idx_exemplo],
            base_values=expected_val,
            data=X_test_scaled[idx_exemplo],
            feature_names=feature_names,
        )

        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(shap_exp, show=False)
        plt.tight_layout()

        if salvar:
            caminho_wf = pasta_figuras / f"shap_waterfall_amostra_{idx_exemplo}.png"
            plt.savefig(caminho_wf, bbox_inches="tight")
            print(f"  [Gráfico Salvo] {caminho_wf}")

        if mostrar:
            plt.show()
        else:
            plt.close()

    return explainer, shap_values


def explicar_com_lime(
    modelo: Any,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    idx_exemplo: int = 10,
    salvar: bool = True,
    mostrar: bool = True,
    pasta_figuras: Path = FIGURES_DIR,
) -> Any:
    """
    Gera a explicação local agnóstica a modelo com LIME para uma instância específica.
    """
    print("\n" + "=" * 60)
    print(" ETAPA 7.2: INTERPRETABILIDADE LOCAL COM LIME")
    print("=" * 60)

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        mode="regression",
        random_state=RANDOM_STATE,
    )

    instancia = X_test_scaled[idx_exemplo]
    pred = modelo.predict(instancia.reshape(1, -1))[0]
    real = y_test[idx_exemplo]

    print(f"  Instância analisada : Índice [{idx_exemplo}]")
    print(f"  Previsão do modelo  : {pred:.5f}")
    print(f"  Valor real          : {real:.5f}")

    exp = explainer_lime.explain_instance(
        instancia,
        modelo.predict,
        num_features=len(feature_names),
    )

    fig = exp.as_pyplot_figure()
    plt.title(f"LIME — Explicação Local (Instância #{idx_exemplo})", pad=12)
    plt.tight_layout()

    if salvar:
        caminho_lime = pasta_figuras / f"lime_explicacao_amostra_{idx_exemplo}.png"
        fig.savefig(caminho_lime, bbox_inches="tight")
        print(f"  [Gráfico Salvo] {caminho_lime}")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

    return exp
