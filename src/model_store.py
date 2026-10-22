"""
Módulo de Persistência e Gerenciamento de Modelos Treinados.

Responsável por:
- Salvar artefatos binários dos modelos (XGBoost, LightGBM, MLP Keras) e Scaler
- Carregar modelos persistidos para inferência rápida (sem necessidade de retreino)
- Salvar e carregar metadados de treino, métricas consolidadas e features
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, TICKER_REF

# Nomes dos arquivos de persistência
FILE_XGB = "xgboost_model.joblib"
FILE_LGB = "lightgbm_model.joblib"
FILE_MLP = "mlp_model.keras"
FILE_SCALER = "scaler.joblib"
FILE_METADATA = "metadados_modelos.json"


def pasta_modelos_ativo(ticker: str, pasta_base: Path = MODELS_DIR) -> Path:
    """
    Retorna o diretório de modelos de um ativo.

    O ativo de referência (TICKER_REF) usa a raiz de modelos_salvos/; os demais
    ativos ficam em modelos_salvos/por_ativo/<ticker>/, pois cada modelo e scaler
    só é válido para a série em que foi treinado.
    """
    if ticker == TICKER_REF:
        return pasta_base
    slug = ticker.replace("^", "").replace(".", "_")
    return pasta_base / "por_ativo" / slug


def modelos_existem(pasta_modelos: Path = MODELS_DIR) -> bool:
    """
    Verifica se os principais artefatos de modelo e scaler já foram persistidos em disco.
    """
    xgb_path = pasta_modelos / FILE_XGB
    scaler_path = pasta_modelos / FILE_SCALER
    return xgb_path.exists() and scaler_path.exists()


def salvar_modelos_treinados(
    modelos: Dict[str, Any],
    scaler: Any,
    df_resultados: Optional[pd.DataFrame] = None,
    ticker_ref: str = TICKER_REF,
    features: Optional[List[str]] = None,
    n_train: int = 0,
    n_test: int = 0,
    pasta_modelos: Path = MODELS_DIR,
    historico_nn: Optional[Any] = None,
) -> Dict[str, Path]:
    """
    Salva os modelos treinados, o scaler e o arquivo de metadados em disco.

    Args:
        modelos: Dicionário contendo os objetos de modelo treinados.
        scaler: Instância de StandardScaler ajustada nos dados de treino.
        df_resultados: DataFrame com as métricas de avaliação (RMSE, MAE, R²).
        ticker_ref: Ticker utilizado no treinamento.
        features: Lista com os nomes dos preditores.
        n_train: Quantidade de amostras de treino.
        n_test: Quantidade de amostras de teste.
        pasta_modelos: Diretório de destino.
        historico_nn: History do Keras (ou dict) do treino da MLP. Se omitido,
            preserva o histórico já gravado nos metadados existentes.

    Returns:
        Dicionário com os caminhos dos arquivos salvos.
    """
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    caminhos_salvos: Dict[str, Path] = {}

    print("\n" + "=" * 60)
    print(" PERSISTÊNCIA: SALVANDO ARTEFATOS DOS MODELOS EM DISCO")
    print("=" * 60)

    # 1. Salvar Scaler
    if scaler is not None:
        p_scaler = pasta_modelos / FILE_SCALER
        joblib.dump(scaler, p_scaler)
        caminhos_salvos["scaler"] = p_scaler
        print(f"  [OK] Scaler salvo: {p_scaler.name}")

    # 2. Salvar XGBoost
    if "XGBoost" in modelos and modelos["XGBoost"] is not None:
        p_xgb = pasta_modelos / FILE_XGB
        joblib.dump(modelos["XGBoost"], p_xgb)
        caminhos_salvos["XGBoost"] = p_xgb
        print(f"  [OK] XGBoost salvo: {p_xgb.name}")

    # 3. Salvar LightGBM
    if "LightGBM" in modelos and modelos["LightGBM"] is not None:
        p_lgb = pasta_modelos / FILE_LGB
        joblib.dump(modelos["LightGBM"], p_lgb)
        caminhos_salvos["LightGBM"] = p_lgb
        print(f"  [OK] LightGBM salvo: {p_lgb.name}")

    # 4. Salvar Rede Neural MLP (Keras)
    if "Rede Neural (MLP)" in modelos and modelos["Rede Neural (MLP)"] is not None:
        try:
            p_mlp = pasta_modelos / FILE_MLP
            modelos["Rede Neural (MLP)"].save(p_mlp)
            caminhos_salvos["Rede Neural (MLP)"] = p_mlp
            print(f"  [OK] Rede Neural MLP salva: {p_mlp.name}")
        except Exception as e:
            print(f"  [AVISO] Não foi possível persistir MLP via Keras: {e}")

    # 5. Salvar Metadados JSON
    metricas_dict = {}
    if df_resultados is not None and not df_resultados.empty:
        metricas_dict = df_resultados.to_dict(orient="records")

    p_meta = pasta_modelos / FILE_METADATA
    historico_dict: Dict[str, List[float]] = {}
    if historico_nn is not None:
        bruto = getattr(historico_nn, "history", historico_nn)
        historico_dict = {k: [float(v) for v in vals] for k, vals in bruto.items() if k in ("loss", "val_loss")}
    elif p_meta.exists():
        try:
            with open(p_meta, "r", encoding="utf-8") as f:
                historico_dict = json.load(f).get("historico_nn", {})
        except Exception:
            historico_dict = {}

    metadados = {
        "ticker_ref": ticker_ref,
        "data_salvamento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": features or [],
        "n_train": n_train,
        "n_test": n_test,
        "metricas": metricas_dict,
        "arquivos": {k: str(v.name) for k, v in caminhos_salvos.items()},
        "historico_nn": historico_dict,
    }

    with open(p_meta, "w", encoding="utf-8") as f:
        json.dump(metadados, f, indent=2, ensure_ascii=False)
    caminhos_salvos["metadados"] = p_meta
    print(f"  [OK] Metadados salvos: {p_meta.name}")

    return caminhos_salvos


def carregar_modelos_salvos(
    pasta_modelos: Path = MODELS_DIR,
) -> Tuple[Dict[str, Any], Optional[Any], Dict[str, Any]]:
    """
    Carrega todos os modelos persistidos, o scaler e o dicionário de metadados.

    Returns:
        Tupla com (modelos_dict, scaler, metadados_dict)
    """
    modelos: Dict[str, Any] = {}
    scaler = None
    metadados: Dict[str, Any] = {}

    p_meta = pasta_modelos / FILE_METADATA
    if p_meta.exists():
        try:
            with open(p_meta, "r", encoding="utf-8") as f:
                metadados = json.load(f)
        except Exception as e:
            print(f"  [AVISO] Falha ao ler metadados JSON: {e}")

    # Scaler
    p_scaler = pasta_modelos / FILE_SCALER
    if p_scaler.exists():
        try:
            scaler = joblib.load(p_scaler)
        except Exception as e:
            print(f"  [AVISO] Falha ao carregar scaler: {e}")

    # XGBoost
    p_xgb = pasta_modelos / FILE_XGB
    if p_xgb.exists():
        try:
            modelos["XGBoost"] = joblib.load(p_xgb)
        except Exception as e:
            print(f"  [AVISO] Falha ao carregar XGBoost: {e}")

    # LightGBM
    p_lgb = pasta_modelos / FILE_LGB
    if p_lgb.exists():
        try:
            modelos["LightGBM"] = joblib.load(p_lgb)
        except Exception as e:
            print(f"  [AVISO] Falha ao carregar LightGBM: {e}")

    # MLP
    p_mlp = pasta_modelos / FILE_MLP
    if p_mlp.exists():
        try:
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            from tensorflow import keras
            modelos["Rede Neural (MLP)"] = keras.models.load_model(p_mlp)
        except Exception as e:
            print(f"  [AVISO] Falha ao carregar Rede Neural MLP: {e}")

    return modelos, scaler, metadados


def predizer_com_modelos(
    modelos: Dict[str, Any],
    X_scaled: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Gera as predições para cada modelo disponível a partir de uma matriz escalonada.

    Returns:
        Dicionário com nome_modelo -> array de previsões
    """
    predicoes: Dict[str, np.ndarray] = {}

    for nome, modelo in modelos.items():
        try:
            if nome == "Rede Neural (MLP)":
                pred = modelo.predict(X_scaled, verbose=0).flatten()
            else:
                pred = modelo.predict(X_scaled)
            predicoes[nome] = np.asarray(pred, dtype=float)
        except Exception as e:
            print(f"  [ERRO] Falha ao inferir com {nome}: {e}")

    return predicoes
