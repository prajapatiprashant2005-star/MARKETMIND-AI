
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "trained"

def load_supplier_model():
    path = MODEL_DIR / "supplier_sales_model.pkl"
    return joblib.load(path)

def load_customer_model():
    path = MODEL_DIR / "online_retail_segmentation_model.pkl"
    return joblib.load(path)

if __name__ == "__main__":
    supplier = load_supplier_model()
    customer = load_customer_model()

    print("SUPPLIER MODEL LOADED")
    print("Saved keys:", list(supplier.keys()))

    print("\nCUSTOMER MODEL LOADED")
    print("Saved keys:", list(customer.keys()))

    print("\nMARKETMIND AI MODELS READY")
