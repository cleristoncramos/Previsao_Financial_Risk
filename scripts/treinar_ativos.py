"""
Treina e salva os modelos (XGBoost, LightGBM e MLP) de cada ativo do sistema.

Usa os CSVs de dados_coletados/ e grava cada conjunto em modelos_salvos/por_ativo/<ATIVO>/,
com o scaler e os metadados (métricas no teste e histórico de perda da MLP). Assim, todos
os ativos já abrem no painel com modelos pré-treinados.

Por padrão, o ativo de referência do TCC (^BVSP) não é retreinado: seus modelos são os que
sustentam os resultados documentados. Use --incluir-referencia para retreiná-lo também.

Uso (na raiz do projeto):
    python scripts/treinar_ativos.py
    python scripts/treinar_ativos.py --ticker PETR4.SA
    python scripts/treinar_ativos.py --incluir-referencia
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import ATIVOS, TICKER_REF
from src.data_collection import carregar_ativo_local
from src.evaluation import adicionar_baseline_ingenuo, avaliar_todos_modelos
from src.model_store import pasta_modelos_ativo, salvar_modelos_treinados
from src.models import treinar_todos_modelos
from src.preprocessing import preparar_dados_modelagem


def treinar_ativo(ticker: str) -> None:
    df = carregar_ativo_local(ticker)
    if df is None or df.empty:
        print(f"[ERRO] Sem dados locais para {ticker}; execute a coleta antes.")
        return

    dataset = preparar_dados_modelagem(df)
    pasta = pasta_modelos_ativo(ticker)
    modelos, predicoes, extras = treinar_todos_modelos(dataset, salvar=False, ticker_ref=ticker)

    predicoes = adicionar_baseline_ingenuo(dataset.features, dataset.X_test, predicoes)
    df_resultados = avaliar_todos_modelos(dataset.y_test, predicoes)

    salvar_modelos_treinados(
        modelos=modelos,
        scaler=dataset.scaler,
        df_resultados=df_resultados,
        ticker_ref=ticker,
        features=dataset.features,
        n_train=len(dataset.X_train),
        n_test=len(dataset.X_test),
        pasta_modelos=pasta,
        historico_nn=extras.get("history_nn"),
    )
    print(f"[OK] {ticker}: modelos salvos em {pasta}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina os modelos de cada ativo do sistema")
    parser.add_argument("--ticker", help="Treina apenas este ativo (ex.: PETR4.SA)")
    parser.add_argument(
        "--incluir-referencia",
        action="store_true",
        help=f"Retreina também o ativo de referência ({TICKER_REF})",
    )
    args = parser.parse_args()

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [t for lista in ATIVOS.values() for t in lista]
        if not args.incluir_referencia:
            tickers = [t for t in tickers if t != TICKER_REF]

    for ticker in tickers:
        treinar_ativo(ticker)


if __name__ == "__main__":
    main()
