
from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "supplier_sales_model.pkl"
)


def load_supplier_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Supplier model not found: {MODEL_PATH}"
        )

    saved = joblib.load(MODEL_PATH)

    if not isinstance(saved, dict):
        raise TypeError("Expected a saved model dictionary.")

    if "model" not in saved:
        raise KeyError("Saved model key is missing.")

    if "feature_columns" not in saved:
        raise KeyError("Feature columns are missing.")

    return saved


def predict_supplier_sales(input_data):
    saved = load_supplier_model()

    model = saved["model"]
    feature_columns = list(saved["feature_columns"])

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    if not isinstance(input_data, pd.DataFrame):
        raise TypeError(
            "Input must be a dictionary or pandas DataFrame."
        )

    missing = [
        feature
        for feature in feature_columns
        if feature not in input_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    X = input_data.loc[:, feature_columns]

    expected_count = getattr(
        model, "n_features_in_", None
    )

    if (
        expected_count is not None
        and X.shape[1] != expected_count
    ):
        raise ValueError(
            "Feature count does not match trained model."
        )

    prediction = model.predict(X)

    return prediction


if __name__ == "__main__":
    saved = load_supplier_model()

    print("=" * 50)
    print("MARKETMIND AI - SUPPLIER SALES")
    print("=" * 50)

    print("Model:", type(saved["model"]).__name__)

    features = list(saved["feature_columns"])

    print("Total features:", len(features))

    print("\nFirst 30 features:")
    for feature in features[:30]:
        print("-", feature)

    print("\nSupplier model loaded successfully.")
    print("Prediction engine ready for validated inputs.")