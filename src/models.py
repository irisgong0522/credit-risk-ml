from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# logistic regression pipeline
def build_baseline_pipeline(preprocessor):

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline

# xgboost pipeline
def build_xgboost_pipeline(preprocessor, params=None):

    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 4,
        "importance_type": "gain"
    }
    
    if params is not None:
        default_params.update(params)

    model = XGBClassifier(**default_params)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline

# random forest pipeline
def build_random_forest_pipeline(preprocessor):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20, # constraint model complexity, every leaf node must have at least 20 samples
        n_jobs=-1, # utilize all available CPU cores for parallel processing
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline

# lightgbm pipeline
def build_lightgbm_pipeline(preprocessor):

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=30,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline

# catboost pipeline
def build_catboost_pipeline(preprocessor):

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        thread_count=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline