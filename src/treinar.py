"""Pipeline didático de preparação dos dados e treinamento dos modelos da Olist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ARQUIVOS_OBRIGATORIOS = {
    "pedidos": "olist_orders_dataset.csv",
    "itens": "olist_order_items_dataset.csv",
    "pagamentos": "olist_order_payments_dataset.csv",
    "avaliacoes": "olist_order_reviews_dataset.csv",
    "clientes": "olist_customers_dataset.csv",
}


def ler_csv(pasta_dados: Path, nome_arquivo: str) -> pd.DataFrame:
    """Lê um CSV e informa claramente quando o arquivo não existe."""
    caminho_arquivo = pasta_dados / nome_arquivo
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    return pd.read_csv(caminho_arquivo)


def preparar_base(pasta_dados: Path) -> pd.DataFrame:
    """Lê as tabelas da Olist e cria a base usada pelos modelos."""
    pedidos = ler_csv(pasta_dados, ARQUIVOS_OBRIGATORIOS["pedidos"])
    itens = ler_csv(pasta_dados, ARQUIVOS_OBRIGATORIOS["itens"])
    pagamentos = ler_csv(pasta_dados, ARQUIVOS_OBRIGATORIOS["pagamentos"])
    avaliacoes = ler_csv(pasta_dados, ARQUIVOS_OBRIGATORIOS["avaliacoes"])
    clientes = ler_csv(pasta_dados, ARQUIVOS_OBRIGATORIOS["clientes"])

    # Converte as colunas de texto para datas reconhecidas pelo pandas.
    colunas_de_data = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for coluna in colunas_de_data:
        pedidos[coluna] = pd.to_datetime(pedidos[coluna], errors="coerce")

    # Só pedidos entregues possuem uma data real para comparação.
    pedidos = pedidos[pedidos["order_status"] == "delivered"]
    pedidos = pedidos.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    )

    # Calcula o atraso: entrega real menos entrega estimada, em dias.
    pedidos["atraso_entrega_dias"] = (
        pedidos["order_delivered_customer_date"]
        - pedidos["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0

    # Resume os vários itens de cada pedido em uma linha por pedido.
    resumo_itens = (
        itens.groupby("order_id", as_index=False)
        .agg(
            quantidade_itens=("order_item_id", "count"),
            preco_total=("price", "sum"),
            frete_total=("freight_value", "sum"),
        )
    )

    # Resume os pagamentos de cada pedido.
    resumo_pagamentos = (
        pagamentos.groupby("order_id", as_index=False)
        .agg(
            valor_pagamento=("payment_value", "sum"),
            parcelas_pagamento=("payment_installments", "max"),
            tipo_pagamento=(
                "payment_type",
                lambda valores: (
                    valores.mode().iloc[0] if not valores.mode().empty else np.nan
                ),
            ),
        )
    )

    # Calcula a nota média quando há mais de uma avaliação para o pedido.
    resumo_avaliacoes = (
        avaliacoes.groupby("order_id", as_index=False)
        .agg(nota_avaliacao=("review_score", "mean"))
    )

    dados_clientes = clientes[["customer_id", "customer_state"]].drop_duplicates()

    # Junta as tabelas usando order_id e customer_id.
    base = (
        pedidos[
            [
                "order_id",
                "customer_id",
                "order_purchase_timestamp",
                "atraso_entrega_dias",
            ]
        ]
        .merge(resumo_itens, on="order_id", how="left")
        .merge(resumo_pagamentos, on="order_id", how="left")
        .merge(resumo_avaliacoes, on="order_id", how="inner")
        .merge(dados_clientes, on="customer_id", how="left")
    )

    # Cria o segundo alvo: 1 significa avaliação de uma ou duas estrelas.
    base["avaliacao_negativa"] = (base["nota_avaliacao"] <= 2).astype(int)

    # Extrai partes úteis da data da compra.
    base["mes_compra"] = base["order_purchase_timestamp"].dt.month
    base["dia_semana_compra"] = base["order_purchase_timestamp"].dt.dayofweek
    base["hora_compra"] = base["order_purchase_timestamp"].dt.hour

    # Remove identificadores e colunas já transformadas em variáveis úteis.
    base = base.drop(
        columns=["order_purchase_timestamp", "nota_avaliacao", "customer_id"]
    )
    return base.dropna(subset=["atraso_entrega_dias", "avaliacao_negativa"])


def treinar_modelos(base: pd.DataFrame) -> tuple[dict, Pipeline, Pipeline]:
    """Treina os dois modelos e calcula suas métricas no conjunto de teste."""
    colunas_de_entrada = [
        "quantidade_itens",
        "preco_total",
        "frete_total",
        "valor_pagamento",
        "parcelas_pagamento",
        "tipo_pagamento",
        "customer_state",
        "mes_compra",
        "dia_semana_compra",
        "hora_compra",
    ]

    entradas = base[colunas_de_entrada]
    alvo_atraso = base["atraso_entrega_dias"]
    alvo_avaliacao = base["avaliacao_negativa"]

    # Reserva 20% dos dados para testar o que os modelos aprenderam.
    (
        entradas_treino,
        entradas_teste,
        atraso_treino,
        atraso_teste,
        avaliacao_treino,
        avaliacao_teste,
    ) = train_test_split(
        entradas,
        alvo_atraso,
        alvo_avaliacao,
        test_size=0.2,
        random_state=42,
    )

    colunas_numericas = [
        "quantidade_itens",
        "preco_total",
        "frete_total",
        "valor_pagamento",
        "parcelas_pagamento",
        "mes_compra",
        "dia_semana_compra",
        "hora_compra",
    ]
    colunas_categoricas = ["tipo_pagamento", "customer_state"]

    # Prepara números e textos antes de entregá-los aos modelos.
    preparador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                Pipeline([("preencher", SimpleImputer(strategy="median"))]),
                colunas_numericas,
            ),
            (
                "categoricas",
                Pipeline(
                    [
                        (
                            "preencher",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "transformar_em_numeros",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                colunas_categoricas,
            ),
        ]
    )

    modelo_atraso = Pipeline(
        steps=[
            ("preparador", preparador),
            (
                "modelo",
                RandomForestRegressor(
                    n_estimators=250,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    modelo_avaliacao = Pipeline(
        steps=[
            ("preparador", preparador),
            (
                "modelo",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # O método fit é o momento em que os modelos aprendem com os dados.
    modelo_atraso.fit(entradas_treino, atraso_treino)
    modelo_avaliacao.fit(entradas_treino, avaliacao_treino)

    previsao_atraso = modelo_atraso.predict(entradas_teste)
    probabilidade_avaliacao = modelo_avaliacao.predict_proba(entradas_teste)[:, 1]
    previsao_avaliacao = (probabilidade_avaliacao >= 0.5).astype(int)

    metricas = {
        "regressao_atraso": {
            "mae": float(mean_absolute_error(atraso_teste, previsao_atraso)),
            "rmse": float(
                np.sqrt(mean_squared_error(atraso_teste, previsao_atraso))
            ),
        },
        "classificacao_avaliacao": {
            "roc_auc": float(
                roc_auc_score(avaliacao_teste, probabilidade_avaliacao)
            ),
            "f1": float(f1_score(avaliacao_teste, previsao_avaliacao)),
            "precisao": float(
                precision_score(
                    avaliacao_teste,
                    previsao_avaliacao,
                    zero_division=0,
                )
            ),
            "revocacao": float(
                recall_score(
                    avaliacao_teste,
                    previsao_avaliacao,
                    zero_division=0,
                )
            ),
            "taxa_avaliacao_negativa": float(avaliacao_teste.mean()),
        },
        "linhas_utilizadas": int(len(base)),
    }

    return metricas, modelo_atraso, modelo_avaliacao


def ler_argumentos() -> argparse.Namespace:
    """Define os caminhos que podem ser alterados pelo terminal."""
    parser = argparse.ArgumentParser(
        description="Treina modelos com os dados da Olist."
    )
    parser.add_argument(
        "--pasta-dados",
        default="data/raw",
        help="Pasta com os CSVs originais do Kaggle.",
    )
    parser.add_argument(
        "--caminho-processado",
        default="data/processed/modeling_dataset.csv",
        help="Arquivo da base pronta para modelagem.",
    )
    parser.add_argument(
        "--pasta-modelos",
        default="models",
        help="Pasta onde os modelos serão salvos.",
    )
    parser.add_argument(
        "--caminho-metricas",
        default="reports/metrics.json",
        help="Arquivo JSON onde as métricas serão salvas.",
    )
    return parser.parse_args()


def principal() -> None:
    """Orquestra todas as etapas do pipeline."""
    argumentos = ler_argumentos()
    pasta_dados = Path(argumentos.pasta_dados)
    caminho_processado = Path(argumentos.caminho_processado)
    pasta_modelos = Path(argumentos.pasta_modelos)
    caminho_metricas = Path(argumentos.caminho_metricas)

    # Cria as pastas de saída caso ainda não existam.
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    caminho_processado.parent.mkdir(parents=True, exist_ok=True)
    caminho_metricas.parent.mkdir(parents=True, exist_ok=True)

    base = preparar_base(pasta_dados)
    metricas, modelo_atraso, modelo_avaliacao = treinar_modelos(base)

    base.to_csv(caminho_processado, index=False)
    joblib.dump(modelo_atraso, pasta_modelos / "modelo_atraso_entrega.joblib")
    joblib.dump(
        modelo_avaliacao,
        pasta_modelos / "modelo_avaliacao_negativa.joblib",
    )

    with caminho_metricas.open("w", encoding="utf-8") as arquivo:
        json.dump(metricas, arquivo, indent=2, ensure_ascii=False)

    print("Treinamento concluído com sucesso.")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    principal()
