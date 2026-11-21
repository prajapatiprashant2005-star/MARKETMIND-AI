import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MARKETMIND AI", page_icon="🧠", layout="wide")

# --- File Paths (Update these if your filenames differ) ---
BASE_DIR = Path(__file__).resolve().parent
SUPPLIER_FILE = BASE_DIR / "supplier_model.pkl"
CUSTOMER_FILE = BASE_DIR / "customer_model.pkl"
METRICS_FILE = BASE_DIR / "supplier_metrics.json"


# --- Model Loaders ---
@st.cache_resource
def supplier_bundle():
    bundle = joblib.load(SUPPLIER_FILE)
    if isinstance(bundle, dict):
        return bundle["model"], bundle["features"]
    features = list(getattr(bundle, "feature_names_in_", []))
    return bundle, features


@st.cache_resource
def customer_bundle():
    bundle = joblib.load(CUSTOMER_FILE)
    if isinstance(bundle, dict):
        return (
            bundle["pipeline"],
            bundle.get("features", ["Recency", "Frequency", "Monetary"]),
            bundle.get("labels", {}),
        )
    features = list(
        getattr(bundle, "feature_names_in_", ["Recency", "Frequency", "Monetary"])
    )
    return bundle, features, {}


# --- Workspaces ---
def supplier_workspace(prefix: str):
    st.subheader("📈 Supplier Sales AI")
    if not SUPPLIER_FILE.is_file():
        st.error(f"Missing supplier model: {SUPPLIER_FILE}")
        return
    try:
        model, features = supplier_bundle()
        st.success("Supplier regression model connected")

        if not features:
            st.warning("No feature schema saved inside the supplier model.")
            return

        inputs = {}
        cols = st.columns(min(3, len(features)))
        for idx, feat in enumerate(features):
            inputs[feat] = cols[idx % len(cols)].number_input(
                feat, value=0.0, key=f"{prefix}_{feat}"
            )

        if st.button("Predict supplier sales", type="primary", key=f"{prefix}_run"):
            frame = pd.DataFrame([{col: 0.0 for col in features}])
            for col, value in inputs.items():
                if col in frame.columns:
                    frame.loc[0, col] = value
            result = float(model.predict(frame)[0])
            st.metric("Predicted model output (units not verified)", f"{result:,.4f}")
            st.caption(
                "This result uses a simplified input vector, not the full validated production preprocessing pipeline."
            )

        with st.expander("View supplier feature schema"):
            st.dataframe(pd.DataFrame({"Feature": features}), use_container_width=True)
    except Exception as exc:
        st.error(f"Supplier model error: {exc}")


def customer_workspace(prefix: str):
    st.subheader("👥 Customer Segmentation AI")
    if not CUSTOMER_FILE.is_file():
        st.error(f"Missing customer model: {CUSTOMER_FILE}")
        return
    try:
        pipeline, features, labels = customer_bundle()
        st.success("Customer segmentation pipeline connected")
        a, b = st.columns(2)
        a.metric("RFM features", len(features))
        b.metric("Saved segments", len(labels))

        x, y, z = st.columns(3)
        recency = x.number_input("Recency (days)", min_value=0, value=30, key=f"{prefix}_recency")
        frequency = y.number_input("Frequency", min_value=1, value=5, key=f"{prefix}_frequency")
        monetary = z.number_input("Monetary value", min_value=0.0, value=1000.0, key=f"{prefix}_monetary")

        if st.button("Analyze customer", type="primary", key=f"{prefix}_run"):
            row = {"Recency": recency, "Frequency": frequency, "Monetary": monetary}
            frame = pd.DataFrame([row])
            missing = [col for col in features if col not in frame.columns]
            if missing:
                st.error(f"Unsupported customer features: {missing}")
            else:
                frame = frame[features]
                cluster = int(pipeline.predict(frame)[0])
                name = labels.get(cluster, labels.get(str(cluster), f"Segment {cluster}"))
                st.success(f"Predicted segment: {name}")
                a, b = st.columns(2)
                a.metric("Customer segment", name)
                b.metric("Cluster ID", cluster)
                st.dataframe(frame, use_container_width=True)
                suggestions = {
                    "Premium Customers": "Prioritize personalized retention and premium service.",
                    "Loyal Customers": "Consider loyalty rewards and relevant cross-selling.",
                    "Regular Customers": "Encourage repeat purchases through relevant offers.",
                    "Low Value Customers": "Consider cost-effective re-engagement campaigns.",
                }
                st.info(suggestions.get(name, "Review customer purchasing behavior before taking action."))

        with st.expander("View customer schema and segment labels"):
            st.write("Input features:", features)
            st.json({str(k): v for k, v in labels.items()})
    except Exception as exc:
        st.error(f"Customer model error: {exc}")


# --- Sidebar & Navigation ---
with st.sidebar:
    st.title("🧠 MARKETMIND AI")
    st.caption("ENTERPRISE INTELLIGENCE SUITE")
    page = st.radio(
        "WORKSPACES",
        [
            "Executive Overview",
            "Supplier Sales AI",
            "Customer Segmentation AI",
            "Model Performance",
            "Prediction Center",
        ],
    )
    st.divider()
    st.caption("Real Market Sales Analysis | ML research project")

st.title("🧠 MARKETMIND AI")
st.caption("ENTERPRISE SALES & CUSTOMER INTELLIGENCE PLATFORM")
st.divider()

if page == "Executive Overview":
    st.header("Executive Intelligence Overview")
    st.write(
        "Explore two independently trained machine-learning models: supplier sales regression and customer RFM segmentation."
    )
    a, b, c, d = st.columns(4)
    a.metric("Saved model files", sum(p.is_file() for p in [SUPPLIER_FILE, CUSTOMER_FILE]))
    try:
        _, cols = supplier_bundle()
        b.metric("Supplier input features", len(cols))
    except Exception:
        b.metric("Supplier input features", "Unavailable")
    try:
        _, cols, names = customer_bundle()
        c.metric("Customer segments", len(names))
        d.metric("RFM inputs", len(cols))
    except Exception:
        c.metric("Customer segments", "Unavailable")
        d.metric("RFM inputs", "Unavailable")

    st.subheader("AI Workspaces")
    left, right = st.columns(2)
    with left:
        st.info("📈 **Supplier Sales AI** — Random Forest regression and saved evaluation metrics")
    with right:
        st.info("👥 **Customer Segmentation AI** — RFM-based customer clustering")
    st.caption("These datasets and models are analyzed separately; no cross-dataset relationship is assumed.")

elif page == "Supplier Sales AI":
    supplier_workspace("supplier_page")

elif page == "Customer Segmentation AI":
    customer_workspace("customer_page")

elif page == "Model Performance":
    st.header("Model Performance & Evaluation")
    st.subheader("Supplier regression")
    if METRICS_FILE.is_file():
        try:
            with METRICS_FILE.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            st.success("Saved supplier evaluation metrics loaded")
            st.json(metrics)
            st.caption("Interpret these metrics only with the original test split, target definition, and units.")
        except Exception as exc:
            st.error(f"Unable to read supplier metrics: {exc}")
    else:
        st.warning("Saved supplier metrics file was not found.")

    st.divider()
    st.subheader("Customer segmentation")
    try:
        _, features, names = customer_bundle()
        a, b = st.columns(2)
        a.metric("Input features", len(features))
        b.metric("Saved segments", len(names))
        st.write("Feature schema:", features)
        st.json({str(k): v for k, v in names.items()})
        st.info("Clustering quality metrics have not been verified. Evaluate using original customer records.")
    except Exception as exc:
        st.error(f"Customer model unavailable: {exc}")

elif page == "Prediction Center":
    st.header("Unified AI Prediction Center")
    tab1, tab2 = st.tabs(["📈 Supplier prediction", "👥 Customer segmentation"])
    with tab1:
        supplier_workspace("supplier_center")
    with tab2:
        customer_workspace("customer_center")