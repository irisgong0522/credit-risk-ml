import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.preprocessing import build_preprocessor
from src.models import build_xgboost_pipeline

def tune_xgboost(X_train, y_train, n_trials=5):

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    def objective(trial):

        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 5),

            "min_child_weight": trial.suggest_int(
                "min_child_weight", 5, 30
            ),

            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),

            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1.0, 30.0, log=True
            ),

            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-3, 10.0, log=True
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.08, log=True
            ),

            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 800, step=100
            )
        }

        preprocessor = build_preprocessor(
            X_train,
            scale_numeric=False
        )

        pipeline = build_xgboost_pipeline(
            preprocessor,
            params=params
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1
        )

        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    return study


