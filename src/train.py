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


REQUIRED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}


def _read_csv(raw_dir: Path, filename: str) -> pd.DataFrame:
    file_path = raw_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    return pd.read_csv(file_path)


def load_training_frame(raw_dir: Path) -> pd.DataFrame:
    orders = _read_csv(raw_dir, REQUIRED_FILES["orders"])
    order_items = _read_csv(raw_dir, REQUIRED_FILES["order_items"])
    payments = _read_csv(raw_dir, REQUIRED_FILES["order_payments"])
    reviews = _read_csv(raw_dir, REQUIRED_FILES["order_reviews"])
    customers = _read_csv(raw_dir, REQUIRED_FILES["customers"])

    orders = orders.copy()
    date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_columns:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders = orders[orders["order_status"] == "delivered"]
    orders = orders.dropna(
        subset=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    )

    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0

    item_agg = (
        order_items.groupby("order_id", as_index=False)
        .agg(
            items_count=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
        )
    )

    payment_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        )
    )

    review_agg = (
        reviews.groupby("order_id", as_index=False)
        .agg(review_score=("review_score", "mean"))
    )

    customer_features = customers[["customer_id", "customer_state"]].drop_duplicates()

    frame = (
        orders[["order_id", "customer_id", "order_purchase_timestamp", "delivery_delay_days"]]
        .merge(item_agg, on="order_id", how="left")
        .merge(payment_agg, on="order_id", how="left")
        .merge(review_agg, on="order_id", how="inner")
        .merge(customer_features, on="customer_id", how="left")
    )

    frame["is_negative_review"] = (frame["review_score"] <= 2).astype(int)
    frame["purchase_month"] = frame["order_purchase_timestamp"].dt.month
    frame["purchase_day_of_week"] = frame["order_purchase_timestamp"].dt.dayofweek
    frame["purchase_hour"] = frame["order_purchase_timestamp"].dt.hour

    frame = frame.drop(columns=["order_purchase_timestamp", "review_score", "customer_id"])
    frame = frame.dropna(subset=["delivery_delay_days", "is_negative_review"])

    return frame


def train_models(frame: pd.DataFrame) -> tuple[dict, Pipeline, Pipeline]:
    feature_columns = [
        "items_count",
        "total_price",
        "total_freight",
        "payment_value",
        "payment_installments",
        "payment_type",
        "customer_state",
        "purchase_month",
        "purchase_day_of_week",
        "purchase_hour",
    ]
    target_delay = "delivery_delay_days"
    target_negative = "is_negative_review"

    X = frame[feature_columns]
    y_delay = frame[target_delay]
    y_negative = frame[target_negative]

    (
        X_train,
        X_test,
        y_delay_train,
        y_delay_test,
        y_negative_train,
        y_negative_test,
    ) = train_test_split(X, y_delay, y_negative, test_size=0.2, random_state=42)

    numeric_features = [
        "items_count",
        "total_price",
        "total_freight",
        "payment_value",
        "payment_installments",
        "purchase_month",
        "purchase_day_of_week",
        "purchase_hour",
    ]
    categorical_features = ["payment_type", "customer_state"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    regressor = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1),
            ),
        ]
    )
    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    regressor.fit(X_train, y_delay_train)
    classifier.fit(X_train, y_negative_train)

    delay_pred = regressor.predict(X_test)
    prob_negative = classifier.predict_proba(X_test)[:, 1]
    pred_negative = (prob_negative >= 0.5).astype(int)

    metrics = {
        "regression": {
            "mae": float(mean_absolute_error(y_delay_test, delay_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_delay_test, delay_pred))),
        },
        "classification": {
            "roc_auc": float(roc_auc_score(y_negative_test, prob_negative)),
            "f1": float(f1_score(y_negative_test, pred_negative)),
            "precision": float(precision_score(y_negative_test, pred_negative, zero_division=0)),
            "recall": float(recall_score(y_negative_test, pred_negative, zero_division=0)),
            "negative_review_rate": float(y_negative_test.mean()),
        },
        "rows_used": int(len(frame)),
    }

    return metrics, regressor, classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina modelos para atraso de entrega e avaliação negativa no dataset Olist."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Diretório com os CSVs brutos do Kaggle.",
    )
    parser.add_argument(
        "--processed-path",
        default="data/processed/modeling_dataset.csv",
        help="Caminho para salvar base processada.",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Diretório para salvar os modelos treinados.",
    )
    parser.add_argument(
        "--metrics-path",
        default="reports/metrics.json",
        help="Arquivo JSON de métricas dos modelos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_path = Path(args.processed_path)
    model_dir = Path(args.model_dir)
    metrics_path = Path(args.metrics_path)

    model_dir.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    frame = load_training_frame(raw_dir)
    metrics, regressor, classifier = train_models(frame)

    frame.to_csv(processed_path, index=False)
    joblib.dump(regressor, model_dir / "delivery_delay_regressor.joblib")
    joblib.dump(classifier, model_dir / "negative_review_classifier.joblib")

    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)

    print("Treinamento concluído com sucesso.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
