

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Fuso horário oficial de Brasília. Servidores (ex.: Streamlit Cloud) costumam operar em UTC,
# por isso todo horário exibido ou gravado pelo projeto deve vir de agora_brasilia().
try:
    FUSO_HORARIO = ZoneInfo("America/Sao_Paulo")
except ZoneInfoNotFoundError:
    # Sem base de fusos no sistema: UTC-3 fixo (o Brasil não adota horário de verão desde 2019)
    FUSO_HORARIO = timezone(timedelta(hours=-3), "BRT")


def agora_brasilia() -> datetime:
    """Data e hora atuais no fuso de São Paulo, independentemente do fuso do servidor."""
    return datetime.now(FUSO_HORARIO)


# Diretórios base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dados_coletados"
FIGURES_DIR = BASE_DIR / "relatorios_graficos"
MODELS_DIR = BASE_DIR / "modelos_salvos"

# Garantir existência de pastas
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Ativos monitorados no Yahoo Finance
ATIVOS = {
    "acoes_br": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"],
    "indices": ["^BVSP", "^GSPC"],
}

# Parâmetros temporais para coleta
DATA_INICIO = "2016-01-01"
DATA_FIM = agora_brasilia().strftime("%Y-%m-%d")
INTERVALO = "1d"

# Ativo de referência para o estudo preditivo principal
TICKER_REF = "^BVSP"

# Engenharia de atributos e janelas
JANELA_VOLATILIDADE = 21  # Dias úteis para volatilidade histórica anualizada
JANELA_MEDIA_CURTA = 20
JANELA_MEDIA_LONGA = 50
JANELA_RSI = 14
JANELA_VOLUME_MM = 20

# Features utilizadas nos modelos
FEATURES = [
    "Close",
    "Volume",
    "Retorno_Diario",
    f"Volatilidade_{JANELA_VOLATILIDADE}d",
    f"MM_{JANELA_MEDIA_CURTA}d",
    f"MM_{JANELA_MEDIA_LONGA}d",
    f"RSI_{JANELA_RSI}d",
    f"Volume_MM{JANELA_VOLUME_MM}",
]

# Variável-alvo (Volatilidade deslocada para prever 1 dia à frente: t+1)
TARGET = "Target_Vol"

# Parâmetros de Divisão de Dados e Reprodutibilidade
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Configurações de Modelos
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}

NEURAL_NETWORK_PARAMS = {
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "early_stopping_patience": 20,
    "validation_split": 0.15,  # Fração final do treino usada no Early Stopping
    "random_state": RANDOM_STATE,
}
