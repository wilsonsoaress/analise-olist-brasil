# Análise Olist Brasil

Este é o primeiro projeto da série de Data Science usando o dataset público da Olist (Kaggle). O objetivo é construir um fluxo de ponta a ponta para:

1. **Prever o tempo de atraso de entrega** (em dias) por pedido.
2. **Estimar a probabilidade de avaliação negativa** (1 ou 2 estrelas).
3. **Apresentar resultados em um dashboard interativo**.
4. **Documentar o passo a passo** desde dados brutos até deploy.

## Objetivos de negócio

- Identificar perfis de pedidos com maior risco de atraso.
- Antecipar experiências ruins do cliente para ações proativas.
- Entregar uma visão executiva e operacional via dashboard.

## Stack e habilidades demonstradas

- Manipulação de dados relacionais com `pandas` (joins entre múltiplas tabelas da Olist)
- Engenharia de features temporais, logísticas e de pagamento
- Modelagem preditiva com `scikit-learn`:
  - Regressão para atraso (`RandomForestRegressor`)
  - Classificação para avaliação negativa (`RandomForestClassifier`)
- Comunicação de resultados via dashboard com `streamlit`
- Organização para deploy reprodutível

## Estrutura do repositório

```text
.
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/               # CSVs originais do Kaggle (não versionados)
│   └── processed/         # Base modelada gerada pelo pipeline
├── models/                # Modelos treinados (.joblib)
├── reports/               # Métricas em JSON
├── src/
│   └── train.py           # Pipeline de preparação + treino + avaliação
├── requirements.txt
└── README.md
```

## Passo a passo do projeto

### 1) Coleta e organização dos dados

Baixe o dataset da Olist no Kaggle e coloque os arquivos CSV em:

- `data/raw/olist_orders_dataset.csv`
- `data/raw/olist_order_items_dataset.csv`
- `data/raw/olist_order_payments_dataset.csv`
- `data/raw/olist_order_reviews_dataset.csv`
- `data/raw/olist_customers_dataset.csv`

### 2) Treinamento e avaliação dos modelos

Instale dependências:

```bash
pip install -r requirements.txt
```

Execute o pipeline:

```bash
python src/train.py --raw-dir data/raw
```

Para estudar o pipeline com nomes de funções, variáveis e comentários em português,
execute a versão didática:

```bash
python src/treinar.py --pasta-dados data/raw
```

Essa versão mantém as mesmas etapas do `train.py`, mas usa nomes em português.
Os arquivos de saída dos modelos recebem nomes diferentes para não sobrescrever
os modelos gerados pelo pipeline original.

Saídas geradas:

- `data/processed/modeling_dataset.csv`
- `models/delivery_delay_regressor.joblib`
- `models/negative_review_classifier.joblib`
- `reports/metrics.json`

### 3) Dashboard interativo

Com os artefatos já gerados, execute:

```bash
streamlit run dashboard/app.py
```

O dashboard apresenta:

- Métricas dos modelos (MAE, RMSE, ROC AUC)
- Distribuição dos atrasos
- Taxa de avaliações negativas por estado
- Simulador para prever atraso e risco de review negativa

## Metodologia (documentação do fluxo)

1. **Integração relacional** por `order_id` e `customer_id`.
2. **Definição de alvo de regressão**: `delivery_delay_days = delivered_date - estimated_delivery_date`.
3. **Definição de alvo de classificação**: `is_negative_review = review_score <= 2`.
4. **Engenharia de features**:
   - Quantidade de itens
   - Valor/frete total
   - Tipo e parcelas de pagamento
   - Estado do cliente
   - Mês, dia da semana e hora da compra
5. **Treino/teste** com holdout (80/20).
6. **Avaliação**:
   - Regressão: MAE e RMSE
   - Classificação: ROC AUC, F1, precisão e recall
7. **Publicação de resultados** no dashboard.

## Próximos passos

- Incluir validação temporal para reduzir risco de leakage.
- Testar modelos adicionais (XGBoost/LightGBM/CatBoost).
- Adicionar monitoramento de drift e retraining.
- Publicar dashboard em ambiente cloud (ex.: Streamlit Community Cloud).
