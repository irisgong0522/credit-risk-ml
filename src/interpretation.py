import numpy as np
import shap
import matplotlib.pyplot as plt


def plot_shap_summary(pipeline, X, sample_size=1000):

    # Extract fitted components
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    # Sample observations for faster computation
    X_sample = X.sample(
        n=min(sample_size, len(X)),
        random_state=42
    )

    # Apply fitted preprocessing
    X_processed = preprocessor.transform(X_sample)

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert sparse matrix to dense if necessary
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer(X_processed)

    # Plot summary
    shap.summary_plot(
        shap_values.values,
        X_processed,
        feature_names=feature_names,
        max_display=20,
        show=False
    )

    plt.title("XGBoost SHAP Summary Plot")
    plt.tight_layout()
    plt.show()

    return shap_values


