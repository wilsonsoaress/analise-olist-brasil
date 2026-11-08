from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "processed" / "modeling_dataset.csv"
DEFAULT_METRICS = ROOT / "reports" / "metrics.json"
DEFAULT_REGRESSOR = ROOT / "models" / "delivery_delay_regressor.joblib"
DEFAULT_CLASSIFIER = ROOT / "models" / "negative_review_classifier.joblib"


@st.cache_data
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Carrega a base processada usada pelos gráficos."""
    return pd.read_csv(dataset_path)


@st.cache_data
def load_metrics(metrics_path: Path) -> dict:
    """Carrega as métricas geradas pelo treinamento."""
    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_delay_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Agrupa os atrasos em faixas para deixar o gráfico legível."""
    delay_days = data["delivery_delay_days"].clip(lower=-15, upper=30)
    bins = [-15, -10, -5, 0, 5, 10, 15, 20, 30]
    labels = [
        "15 a 10 dias antes",
        "10 a 5 dias antes",
        "5 a 0 dias antes",
        "0 a 5 dias de atraso",
        "5 a 10 dias de atraso",
        "10 a 15 dias de atraso",
        "15 a 20 dias de atraso",
        "20 a 30 dias de atraso",
    ]

    counts = (
        pd.cut(
            delay_days,
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        .value_counts()
        .reindex(labels, fill_value=0)
    )
    return pd.DataFrame(
        {
            "Faixa": labels,
            "Quantidade de pedidos": counts.to_numpy(),
            "Tipo": [
                "Entrega antecipada",
                "Entrega antecipada",
                "Entrega antecipada",
                "Atraso",
                "Atraso",
                "Atraso",
                "Atraso",
                "Atraso",
            ],
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="Olist - Atrasos e avaliações",
        page_icon="📦",
        layout="wide",
    )
    st.title("📦 Olist Brasil: atraso de entrega e avaliações negativas")
    st.caption(
        "Visão histórica dos pedidos e simulador de risco baseado nos modelos treinados."
    )

    if not DEFAULT_DATASET.exists() or not DEFAULT_METRICS.exists():
        st.warning(
            "Execute o treinamento antes de abrir o dashboard:\n\n"
            "`python src/train.py --raw-dir data/raw`"
        )
        return

    data = load_dataset(DEFAULT_DATASET)
    metrics = load_metrics(DEFAULT_METRICS)

    st.subheader("Desempenho dos modelos")
    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)
    metric_column_1.metric(
        "Regressão - MAE (dias)",
        f"{metrics['regression']['mae']:.2f}",
    )
    metric_column_2.metric(
        "Regressão - RMSE (dias)",
        f"{metrics['regression']['rmse']:.2f}",
    )
    metric_column_3.metric(
        "Classificação - ROC AUC",
        f"{metrics['classification']['roc_auc']:.3f}",
    )

    st.subheader("Distribuição do atraso de entrega")
    st.write(
        "Cada barra representa a quantidade de pedidos entregues antes, "
        "no prazo ou depois da data estimada."
    )
    delay_distribution = build_delay_distribution(data)
    delay_chart = px.bar(
        delay_distribution,
        x="Quantidade de pedidos",
        y="Faixa",
        color="Tipo",
        orientation="h",
        text="Quantidade de pedidos",
        color_discrete_map={
            "Entrega antecipada": "#2ca02c",
            "Atraso": "#d62728",
        },
        labels={
            "Quantidade de pedidos": "Pedidos",
            "Faixa": "Diferença para a data estimada",
            "Tipo": "Situação",
        },
    )
    delay_chart.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )
    delay_chart.update_layout(
        height=430,
        showlegend=True,
        yaxis={
            "categoryorder": "array",
            "categoryarray": delay_distribution["Faixa"].tolist(),
        },
        margin={"l": 10, "r": 30, "t": 20, "b": 10},
    )
    st.plotly_chart(delay_chart, use_container_width=True)

    st.subheader("Avaliações negativas por estado")
    st.write(
        "Mostra os 15 estados com maior proporção de avaliações de 1 ou 2 estrelas."
    )
    negative_reviews_by_state = (
        data.groupby("customer_state", dropna=False)["is_negative_review"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    negative_reviews_by_state = negative_reviews_by_state.rename(
        "Taxa de avaliações negativas"
    )
    st.bar_chart(negative_reviews_by_state)

    st.subheader("Simulador de risco para um novo pedido")
    st.write(
        "Informe as características de um pedido hipotético para estimar "
        "o atraso e o risco de uma avaliação negativa."
    )

    with st.form("prediction_form"):
        input_column_1, input_column_2, input_column_3 = st.columns(3)

        items_count = input_column_1.number_input(
            "Quantidade de itens",
            min_value=1,
            value=1,
        )
        total_price = input_column_1.number_input(
            "Preço total (R$)",
            min_value=0.0,
            value=100.0,
        )
        total_freight = input_column_1.number_input(
            "Frete total (R$)",
            min_value=0.0,
            value=15.0,
        )

        payment_value = input_column_2.number_input(
            "Valor pago (R$)",
            min_value=0.0,
            value=115.0,
        )
        payment_installments = input_column_2.number_input(
            "Parcelas",
            min_value=1,
            value=1,
        )
        payment_type = input_column_2.selectbox(
            "Tipo de pagamento",
            sorted(data["payment_type"].dropna().astype(str).unique()),
        )

        customer_state = input_column_3.selectbox(
            "Estado do cliente",
            sorted(data["customer_state"].dropna().astype(str).unique()),
        )
        purchase_month = input_column_3.slider(
            "Mês da compra",
            min_value=1,
            max_value=12,
            value=6,
        )
        purchase_day_of_week = input_column_3.slider(
            "Dia da semana (0 = segunda-feira)",
            min_value=0,
            max_value=6,
            value=2,
        )
        purchase_hour = input_column_3.slider(
            "Hora da compra",
            min_value=0,
            max_value=23,
            value=12,
        )

        submit = st.form_submit_button("Prever")

    if not submit:
        return

    if not DEFAULT_REGRESSOR.exists() or not DEFAULT_CLASSIFIER.exists():
        st.error(
            "Modelos não encontrados. Execute "
            "`python src/train.py --raw-dir data/raw` primeiro."
        )
        return

    regressor = joblib.load(DEFAULT_REGRESSOR)
    classifier = joblib.load(DEFAULT_CLASSIFIER)

    feature_row = pd.DataFrame(
        [
            {
                "items_count": items_count,
                "total_price": total_price,
                "total_freight": total_freight,
                "payment_value": payment_value,
                "payment_installments": payment_installments,
                "payment_type": payment_type,
                "customer_state": customer_state,
                "purchase_month": purchase_month,
                "purchase_day_of_week": purchase_day_of_week,
                "purchase_hour": purchase_hour,
            }
        ]
    )

    predicted_delay = float(regressor.predict(feature_row)[0])
    negative_review_probability = float(
        classifier.predict_proba(feature_row)[0, 1]
    )

    st.success(f"Atraso previsto: **{predicted_delay:.2f} dias**")
    st.info(
        "Probabilidade de avaliação negativa (1-2 estrelas): "
        f"**{negative_review_probability:.1%}**"
    )


if __name__ == "__main__":
    main()
