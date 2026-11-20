from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.preprocessing import build_preprocessor


def train_xgboost_early_stopping(
    X_train,
    y_train,
    params,
    early_stopping_rounds=30
):

    # 1. Split training data again
    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # 2. Build preprocessor
    preprocessor = build_preprocessor(
        X_fit,
        scale_numeric=False
    )

    # 3. Fit preprocessing on X_fit only
    X_fit_processed = preprocessor.fit_transform(X_fit)

    # 4. Transform early-stopping validation data
    X_es_processed = preprocessor.transform(X_es)

    # 5. Configure XGBoost
    model_params = {
        **params,
        "n_estimators": 1500,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "early_stopping_rounds": early_stopping_rounds
    }

    model = XGBClassifier(**model_params)

    # 6. Train with early stopping
    model.fit(
        X_fit_processed,
        y_fit,
        eval_set=[
            (X_fit_processed, y_fit),
            (X_es_processed, y_es)
        ],
        verbose=False
    )

    return model, preprocessor