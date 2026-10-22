"""
Módulo de Modelagem Preditiva.

Implementa três arquiteturas para previsão de volatilidade:
1. XGBoost Regressor
2. LightGBM Regressor
3. Rede Neural MLP (Perceptron Multicamadas via Keras / TensorFlow)
"""

import os
import sys

# Configura codificação de saída para UTF-8 compatível com Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import tensorflow as tf
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    LIGHTGBM_PARAMS,
    MODELS_DIR,
    NEURAL_NETWORK_PARAMS,
    RANDOM_STATE,
    TICKER_REF,
    XGBOOST_PARAMS,
)
from src.model_store import salvar_modelos_treinados
from src.preprocessing import DatasetModelagem


def treinar_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any] = XGBOOST_PARAMS,
) -> Tuple[xgb.XGBRegressor, np.ndarray]:
    """
    Treina o modelo de Gradient Boosting XGBoost.
    """
    print("  -> [1/3] Treinando XGBoost...")
    modelo = xgb.XGBRegressor(**params)
    modelo.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    predicoes = modelo.predict(X_val)
    print("  [OK] XGBoost concluído com sucesso!")
    return modelo, predicoes


def treinar_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any] = LIGHTGBM_PARAMS,
) -> Tuple[lgb.LGBMRegressor, np.ndarray]:
    """
    Treina o modelo LightGBM Regressor.
    """
    print("  -> [2/3] Treinando LightGBM...")
    modelo = lgb.LGBMRegressor(**params)
    try:
        modelo.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
        )
    except TypeError:
        modelo.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )
    predicoes = modelo.predict(X_val)
    print("  [OK] LightGBM concluído com sucesso!")
    return modelo, predicoes


def construir_rede_neural(
    input_dim: int,
    lr: float = 1e-3,
    alvo_media: float = 0.0,
    alvo_desvio: float = 1.0,
) -> keras.Model:
    """
    Constrói a arquitetura MLP (Multilayer Perceptron) com Regularização:
    - Camadas densas com ativação ReLU
    - Dropout leve para controle de overfitting
    - Saída linear reescalada para a escala original do alvo

    A camada final de Rescaling converte a saída padronizada da rede em
    volatilidade (y = z * desvio + média), de modo que a rede aprenda em uma
    escala próxima de N(0, 1) e o modelo salvo já devolva valores no domínio real.
    Batch Normalization foi removida: com alvo de baixa magnitude ela degradava
    fortemente a generalização no período de teste.
    """
    modelo = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),  # Saída contínua padronizada
            layers.Rescaling(scale=alvo_desvio, offset=alvo_media),
        ],
        name="MLP_Previsao_Volatilidade",
    )

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return modelo


def treinar_rede_neural(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any] = NEURAL_NETWORK_PARAMS,
) -> Tuple[keras.Model, keras.callbacks.History, np.ndarray]:
    """
    Treina o modelo de Deep Learning MLP com Early Stopping.

    O Early Stopping monitora os últimos 15% do conjunto de TREINO (validation_split
    do Keras, que preserva a ordem temporal). O conjunto de teste não participa da
    seleção de épocas, evitando vazamento de informação na avaliação final.
    X_val / y_val são usados apenas para gerar as previsões de teste.
    """
    print("  -> [3/3] Treinando Rede Neural (MLP)...")
    tf.random.set_seed(params.get("random_state", RANDOM_STATE))

    input_dim = X_train.shape[1]
    modelo = construir_rede_neural(
        input_dim,
        lr=params.get("learning_rate", 1e-3),
        alvo_media=float(np.mean(y_train)),
        alvo_desvio=float(np.std(y_train)),
    )

    paciencia = params.get("early_stopping_patience", 20)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=paciencia,
        restore_best_weights=True,
    )

    history = modelo.fit(
        X_train,
        y_train,
        validation_split=params.get("validation_split", 0.15),
        epochs=params.get("epochs", 200),
        batch_size=params.get("batch_size", 32),
        callbacks=[early_stop],
        verbose=0,
    )

    predicoes = modelo.predict(X_val, verbose=0).flatten()
    print(f"  [OK] Rede Neural concluída! ({len(history.epoch)} épocas executadas)")
    return modelo, history, predicoes


def treinar_todos_modelos(
    dados: DatasetModelagem,
    salvar: bool = True,
    pasta_modelos: Path = MODELS_DIR,
    ticker_ref: str = TICKER_REF,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Executa o treinamento de todas as arquiteturas (XGBoost, LightGBM, Rede Neural).

    Args:
        dados: Objeto tipado DatasetModelagem com dados preparados.
        salvar: Se True, persiste os modelos e o scaler em disco via model_store.
        pasta_modelos: Diretório para salvar os modelos treinados.
        ticker_ref: Ticker cujos dados foram usados no treino (gravado nos metadados).

    Returns:
        Tupla com (dicionario_modelos, dicionario_predicoes, extras_como_historico)
    """
    print("\n" + "=" * 60)
    print(" ETAPA 5: MODELAGEM PREDITIVA (MACHINE LEARNING & DEEP LEARNING)")
    print("=" * 60)

    # 1. XGBoost
    modelo_xgb, pred_xgb = treinar_xgboost(
        dados.X_train_scaled, dados.y_train, dados.X_test_scaled, dados.y_test
    )

    # 2. LightGBM
    modelo_lgb, pred_lgb = treinar_lightgbm(
        dados.X_train_scaled, dados.y_train, dados.X_test_scaled, dados.y_test
    )

    # 3. Rede Neural MLP
    modelo_nn, history_nn, pred_nn = treinar_rede_neural(
        dados.X_train_scaled, dados.y_train, dados.X_test_scaled, dados.y_test
    )

    modelos = {
        "XGBoost": modelo_xgb,
        "LightGBM": modelo_lgb,
        "Rede Neural (MLP)": modelo_nn,
    }

    predicoes = {
        "XGBoost": pred_xgb,
        "LightGBM": pred_lgb,
        "Rede Neural (MLP)": pred_nn,
    }

    extras = {
        "history_nn": history_nn,
    }

    if salvar:
        salvar_modelos_treinados(
            modelos=modelos,
            scaler=dados.scaler,
            features=dados.features,
            n_train=len(dados.X_train),
            n_test=len(dados.X_test),
            ticker_ref=ticker_ref,
            pasta_modelos=pasta_modelos,
            historico_nn=history_nn,
        )

    return modelos, predicoes, extras
