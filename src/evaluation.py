import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)


def plot_roc_curve(y_true, y_prob):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob
    )

    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"XGBoost (AUC = {auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="grey",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — XGBoost")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)


def plot_precision_recall_curve(y_true, y_prob):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_prob
    )

    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"XGBoost (AP = {ap:.4f})"
    )

    plt.axhline(
        y_true.mean(),
        color="grey",
        linestyle="--",
        label="Positive Class Prevalence"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve — XGBoost")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

def plot_confusion_matrix(y_true, y_prob, threshold=0.5):

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Difficulty", "Difficulty"],
        yticklabels=["No Difficulty", "Difficulty"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (Threshold = {threshold})")

    plt.tight_layout()
    plt.show()

    print(classification_report(
        y_true,
        y_pred,
        zero_division=0
    ))


def plot_feature_importance(pipeline, top_n=20, save_path=None):

    # Extract fitted preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]

    # Extract fitted XGBoost model
    model = pipeline.named_steps["classifier"]

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Get feature importance scores
    importance = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    # Sort by importance
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Select top features
    top_features = importance_df.head(top_n)

    # Plot
    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Feature Importance (Gain)")
    plt.ylabel("Features")
    plt.title(f"Top {top_n} XGBoost Feature Importances")

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/xgboost_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return importance_df