

import argparse
import os
import sys
from pathlib import Path

# Configura codificação de saída para UTF-8 compatível com Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Silencia logs e avisos desnecessários do TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Adiciona o diretório raiz ao PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    ATIVOS,
    DATA_FIM,
    DATA_INICIO,
    FIGURES_DIR,
    MODELS_DIR,
    TICKER_REF,
)
from src.data_collection import (
    carregar_ativo_local,
    coletar_todos_ativos,
)
from src.eda import executar_eda_completa
from src.evaluation import (
    adicionar_baseline_ingenuo,
    avaliar_todos_modelos,
    plotar_curva_aprendizado_nn,
    plotar_real_vs_predicoes,
    validacao_cruzada_temporal,
)
from src.interpretability import explicar_com_lime, explicar_com_shap
from src.model_store import (
    carregar_modelos_salvos,
    modelos_existem,
    pasta_modelos_ativo,
    predizer_com_modelos,
    salvar_modelos_treinados,
)
from src.models import treinar_todos_modelos
from src.preprocessing import preparar_dados_modelagem
from src.report import gerar_relatorio_final


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de Previsão de Riscos Financeiros (TCC - Lorrana Pereira)"
    )
    parser.add_argument(
        "--etapa",
        type=str,
        default="todas",
        choices=["todas", "coleta", "eda", "treino", "avaliar", "interpretar"],
        help="Etapa específica a ser executada (padrão: todas)",
    )
    parser.add_argument(
        "--usar-dados-locais",
        action="store_true",
        help="Utiliza os arquivos CSV existentes na pasta dados_coletados sem refazer o download",
    )
    parser.add_argument(
        "--nao-mostrar-graficos",
        action="store_true",
        help="Salva os gráficos em disco sem abrir janelas interativas (ideal para execuções não interativas)",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=TICKER_REF,
        help=f"Ticker de referência para o estudo preditivo (padrão: {TICKER_REF})",
    )
    parser.add_argument(
        "--carregar-modelos",
        action="store_true",
        help="Carrega os modelos persistidos em disco sem necessidade de retreino",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mostrar_graficos = not args.nao_mostrar_graficos
    ticker_ref = args.ticker

    print("\n" + "=" * 65)
    print("  TCC: PREVISÃO DE RISCOS FINANCEIROS COM DATA SCIENCE E ML")
    print("  Autora: Lorrana Evelyn de Araújo Pereira | UFPI / CEAD")
    print("=" * 65)

    # -------------------------------------------------------------
    # 1 & 2. Coleta / Carregamento de Dados
    # -------------------------------------------------------------
    dados = {}
    if args.usar_dados_locais:
        print("\n[INFO] Carregando dados a partir do armazenamento local...")
        df_ref = carregar_ativo_local(ticker_ref)
        if df_ref is not None:
            dados[ticker_ref] = df_ref
            print(f"  Ativo de referência {ticker_ref} carregado com sucesso.")
        else:
            print(f"  [AVISO] {ticker_ref} não encontrado localmente. Iniciando coleta...")
            dados = coletar_todos_ativos()
    else:
        print("\n[INFO] Iniciando coleta de dados atualizada via Yahoo Finance...")
        dados = coletar_todos_ativos()

    if ticker_ref not in dados or dados[ticker_ref] is None:
        # Tenta carregar localmente caso a coleta não tenha retornado
        df_ref = carregar_ativo_local(ticker_ref)
        if df_ref is not None:
            dados[ticker_ref] = df_ref
        else:
            print(f"[ERRO CRÍTICO] Não foi possível obter dados para {ticker_ref}.")
            sys.exit(1)

    df_ref = dados[ticker_ref].dropna().copy()

    if args.etapa == "coleta":
        print("\n[FIM] Etapa de Coleta concluída com êxito.")
        return

    # -------------------------------------------------------------
    # 3. Análise Exploratória dos Dados (EDA)
    # -------------------------------------------------------------
    if args.etapa in ["todas", "eda"]:
        executar_eda_completa(
            df_ref=df_ref,
            dados_todos=dados,
            ticker_ref=ticker_ref,
            salvar=True,
            mostrar=mostrar_graficos,
        )

    if args.etapa == "eda":
        print("\n[FIM] Etapa de EDA concluída com êxito.")
        return

    # -------------------------------------------------------------
    # 4. Pré-processamento
    # -------------------------------------------------------------
    dataset = preparar_dados_modelagem(df_ref)

    # -------------------------------------------------------------
    # 5. Modelagem (Treinamento ou Carregamento Persistido)
    # -------------------------------------------------------------
    extras = {}
    pasta_modelos = pasta_modelos_ativo(ticker_ref)
    if args.carregar_modelos and modelos_existem(pasta_modelos):
        print(f"\n[INFO] Carregando modelos persistidos em disco ({pasta_modelos})...")
        modelos, _, _ = carregar_modelos_salvos(pasta_modelos)
        predicoes = predizer_com_modelos(modelos, dataset.X_test_scaled)
        print("  [OK] Modelos carregados e inferências geradas com sucesso!")
    else:
        modelos, predicoes, extras = treinar_todos_modelos(
            dataset, salvar=True, pasta_modelos=pasta_modelos, ticker_ref=ticker_ref
        )

    if args.etapa == "treino":
        print("\n[FIM] Etapa de Treinamento concluída com êxito.")
        return

    # -------------------------------------------------------------
    # 6. Avaliação de Modelos (com Baseline Acadêmico de Persistência)
    # -------------------------------------------------------------
    predicoes = adicionar_baseline_ingenuo(dataset.features, dataset.X_test, predicoes)
    df_resultados = avaliar_todos_modelos(dataset.y_test, predicoes)

    # Gráfico Comparativo Real vs Previsões
    plotar_real_vs_predicoes(
        datas_test=dataset.datas_test,
        y_real=dataset.y_test,
        dict_predicoes=predicoes,
        salvar=True,
        mostrar=mostrar_graficos,
    )

    # Curva de Aprendizado da Rede Neural
    plotar_curva_aprendizado_nn(
        history=extras.get("history_nn"),
        salvar=True,
        mostrar=mostrar_graficos,
    )

    # Validação Cruzada Temporal (TimeSeriesSplit)
    validacao_cruzada_temporal(dataset.X, dataset.y, n_splits=5)

    if args.etapa == "avaliar":
        gerar_relatorio_final(
            df_resultados,
            ticker_ref=ticker_ref,
            n_train=len(dataset.X_train),
            n_test=len(dataset.X_test),
        )
        print("\n[FIM] Etapa de Avaliação concluída com êxito.")
        return

    # -------------------------------------------------------------
    # 7. Interpretabilidade (XAI: SHAP & LIME)
    # -------------------------------------------------------------
    if args.etapa in ["todas", "interpretar"]:
        # SHAP com o modelo XGBoost
        if "XGBoost" in modelos:
            explicar_com_shap(
                modelo_arvore=modelos["XGBoost"],
                X_test_scaled=dataset.X_test_scaled,
                feature_names=dataset.features,
                idx_exemplo=10,
                datas_test=dataset.datas_test,
                salvar=True,
                mostrar=mostrar_graficos,
            )

            # LIME com o modelo XGBoost
            explicar_com_lime(
                modelo=modelos["XGBoost"],
                X_train_scaled=dataset.X_train_scaled,
                X_test_scaled=dataset.X_test_scaled,
                y_test=dataset.y_test,
                feature_names=dataset.features,
                idx_exemplo=10,
                salvar=True,
                mostrar=mostrar_graficos,
            )

    # -------------------------------------------------------------
    # 8. Relatório Final de Desempenho & Metadados
    # -------------------------------------------------------------
    gerar_relatorio_final(
        df_resultados,
        ticker_ref=ticker_ref,
        data_inicio=DATA_INICIO,
        data_fim=DATA_FIM,
        n_train=len(dataset.X_train),
        n_test=len(dataset.X_test),
    )

    # Persiste metadados atualizados com a tabela final de métricas
    salvar_modelos_treinados(
        modelos=modelos,
        scaler=dataset.scaler,
        df_resultados=df_resultados,
        ticker_ref=ticker_ref,
        features=dataset.features,
        n_train=len(dataset.X_train),
        n_test=len(dataset.X_test),
        pasta_modelos=pasta_modelos,
        historico_nn=extras.get("history_nn"),
    )

    print("\n[SUCESSO] Execução completa do pipeline finalizada com sucesso!")


if __name__ == "__main__":
    main()
