from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "processed" / "modeling_dataset.csv"
DEFAULT_METRICS = ROOT / "reports" / "metrics.json"
DEFAULT_REGRESSOR = ROOT / "models" / "delivery_delay_regressor.joblib"
DEFAULT_CLASSIFIER = ROOT / "models" / "negative_review_classifier.joblib"


@st.cache_data
def load_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


@st.cache_data
def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    st.set_page_config(page_title="Olist - Atraso e Avaliação", layout="wide")
    st.title("📦 Olist Brasil: atraso de entrega e avaliações negativas")

    if not DEFAULT_DATASET.exists() or not DEFAULT_METRICS.exists():
        st.warning(
            "Execute o treinamento antes de abrir o dashboard:\n\n"
            "`python src/train.py --raw-dir data/raw`"
        )
        return

    data = load_dataset(DEFAULT_DATASET)
    metrics = load_metrics(DEFAULT_METRICS)

    col1, col2, col3 = st.columns(3)
    col1.metric("Regressão - MAE (dias)", f"{metrics['regression']['mae']:.2f}")
    col2.metric("Regressão - RMSE (dias)", f"{metrics['regression']['rmse']:.2f}")
    col3.metric("Classificação - ROC AUC", f"{metrics['classification']['roc_auc']:.3f}")

    st.subheader("Distribuição do atraso de entrega")
    st.bar_chart(data["delivery_delay_days"].clip(lower=-15, upper=30).value_counts().sort_index())

    st.subheader("Avaliações negativas por estado")
    by_state = (
        data.groupby("customer_state", dropna=False)["is_negative_review"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    st.bar_chart(by_state)

    st.subheader("Simulador de risco para um novo pedido")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        items_count = c1.number_input("Quantidade de itens", min_value=1, value=1)
        total_price = c1.number_input("Preço total (R$)", min_value=0.0, value=100.0)
        total_freight = c1.number_input("Frete total (R$)", min_value=0.0, value=15.0)

        payment_value = c2.number_input("Valor pago (R$)", min_value=0.0, value=115.0)
        payment_installments = c2.number_input("Parcelas", min_value=1, value=1)
        payment_type = c2.selectbox(
            "Tipo de pagamento",
            sorted(data["payment_type"].dropna().astype(str).unique()),
        )

        customer_state = c3.selectbox(
            "Estado do cliente",
            sorted(data["customer_state"].dropna().astype(str).unique()),
        )
        purchase_month = c3.slider("Mês da compra", 1, 12, 6)
        purchase_day_of_week = c3.slider("Dia da semana (0=seg)", 0, 6, 2)
        purchase_hour = c3.slider("Hora da compra", 0, 23, 12)

        submit = st.form_submit_button("Prever")

    if submit:
        if not DEFAULT_REGRESSOR.exists() or not DEFAULT_CLASSIFIER.exists():
            st.error("Modelos não encontrados. Execute `python src/train.py` primeiro.")
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
        prob_negative = float(classifier.predict_proba(feature_row)[0, 1])

        st.success(f"Atraso previsto: **{predicted_delay:.2f} dias**")
        st.info(f"Probabilidade de avaliação negativa (1-2 estrelas): **{prob_negative:.1%}**")


if __name__ == "__main__":
    main()
