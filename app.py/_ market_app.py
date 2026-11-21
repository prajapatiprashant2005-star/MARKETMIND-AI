import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="MARKETMIND AI | Enterprise Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "trained"
SUPPLIER_PATH = MODEL_DIR / "supplier_sales_model.pkl"
CUSTOMER_PATH = MODEL_DIR / "online_retail_segmentation_model.pkl"
METRICS_PATH = ROOT / "reports" / "models" / "supplier_sales" / "model_metrics.json"

st.markdown(
    """
<style>
:root { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"] { background: #0b1424; color: #eaf2ff; }
[data-testid="stSidebar"] { background: #111f35; }
[data-testid="stSidebar"] * { color: #eaf2ff !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp label, .stApp small,
[data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p,
[data-testid="stMarkdownContainer"] p { color: #eaf2ff !important; }
.stApp [data-testid="stCaptionContainer"] p { color: #a8bfdc !important; }
.block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 2rem; }
[data-testid="stMetric"] { background: #172941; padding: 18px; border: 1px solid #2c4360; border-radius: 14px; }
[data-testid="stMetricValue"] { color: #63e6d0 !important; }
[data-testid="stMetricLabel"] * { color: #bcd0e8 !important; }
[data-testid="stAlert"] p { color: #152238 !important; }
div.stButton > button[kind="primary"] { background: #16a6a0; color: white !important; border: none; }
div.stButton > button { border-radius: 9px; font-weight: 600; }
div.stButton > button:hover { border-color: #63e6d0; }
.mm-card { background: #172941; border: 1px solid #2c4360; border-radius: 14px; padding: 22px; min-height: 165px; }
.mm-card h3 { color: #ffffff !important; margin-top: 0; }
.mm-card p { color: #bcd0e8 !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle(path):
    return joblib.load(str(path))


def supplier_bundle():
    saved = load_bundle(SUPPLIER_PATH)
    return saved["model"], list(saved["feature_columns"])


def customer_bundle():
    saved = load_bundle(CUSTOMER_PATH)
    return saved["pipeline"], list(saved["feature_names"]), saved["segment_names"]


def customer_prediction(recency, frequency, monetary):
    pipeline, columns, names = customer_bundle()
    frame = pd.DataFrame([{"Recency": recency, "Frequency": frequency, "Monetary": monetary}])
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Unsupported customer features: {missing}")
    frame = frame[columns]
    segment = int(pipeline.predict(frame)[0])
    name = names.get(segment, names.get(str(segment), f"Segment {segment}"))
    return segment, name, frame


def supplier_prediction(supplier, year, month, transfers):
    model, features = supplier_bundle()
    frame = pd.DataFrame(0.0, index=[0], columns=features)
    if supplier in frame.columns:
        frame.loc[0, supplier] = 1.0
    for column, value in {"YEAR": year, "MONTH": month, "RETAIL TRANSFERS": transfers}.items():
        if column in frame.columns:
            frame.loc[0, column] = value
    return float(model.predict(frame)[0])


def supplier_workspace(key_prefix="supplier"):
    st.subheader("📈 Supplier Sales Prediction Engine")
    if not SUPPLIER_PATH.is_file():
        st.error(f"Supplier model not found: {SUPPLIER_PATH}")
        return
    try:
        model, features = supplier_bundle()
        st.success("Random Forest model connected")
        a, b = st.columns(2)
        a.metric("Model", type(model).__name__)
        b.metric("Input Features", len(features))

        suppliers = [x for x in features if x.startswith("SUPPLIER_")]
        if not suppliers:
            st.warning("No SUPPLIER_ columns were found in the saved model features.")
            return

        st.markdown("### Configure Prediction")
        supplier = st.selectbox(
            "Select Supplier",
            suppliers,
            format_func=lambda x: x.replace("SUPPLIER_", ""),
            key=f"{key_prefix}_supplier",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            year = st.number_input(
                "Year",
                min_value=2000,
                max_value=2100,
                value=2026,
                key=f"{key_prefix}_year",
            )
        with c2:
            month = st.selectbox(
                "Month",
                list(range(1, 13)),
                key=f"{key_prefix}_month",
            )
        with c3:
            transfers = st.number_input(
                "Retail Transfers",
                min_value=0.0,
                value=0.0,
                key=f"{key_prefix}_transfers",
            )

        if st.button("Predict supplier sales", type="primary", key=f"{key_prefix}_run"):
            result = supplier_prediction(supplier, year, month, transfers)
            st.metric("Predicted model output (units not verified)", f"{result:,.4f}")
            st.caption(
                "This result uses a simplified input vector, not the full validated production preprocessing pipeline."
            )

        with st.expander("View supplier feature schema"):
            st.dataframe(pd.DataFrame({"Feature": features}), use_container_width=True)
    except Exception as exc:
        st.error(f"Supplier model error: {exc}")


def customer_workspace(prefix="customer"):
    st.subheader("👥 Customer Segmentation AI")
    if not CUSTOMER_PATH.is_file():
        st.error(f"Missing customer model: {CUSTOMER_PATH}")
        return
    try:
        _, features, labels = customer_bundle()
        st.success("Customer segmentation pipeline connected")
        a, b = st.columns(2)
        a.metric("RFM features", len(features))
        b.metric("Saved segments", len(labels))

        x, y, z = st.columns(3)
        recency = x.number_input("Recency (days)", min_value=0, value=30, key=f"{prefix}_recency")
        frequency = y.number_input("Frequency", min_value=1, value=5, key=f"{prefix}_frequency")
        monetary = z.number_input("Monetary value", min_value=0.0, value=1000.0, key=f"{prefix}_monetary")

        if st.button("Analyze customer", type="primary", key=f"{prefix}_run"):
            cluster, name, frame = customer_prediction(recency, frequency, monetary)
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
    a.metric("Saved model files", sum(p.is_file() for p in [SUPPLIER_PATH, CUSTOMER_PATH]))
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
    if METRICS_PATH.is_file():
        try:
            with METRICS_PATH.open("r", encoding="utf-8") as f:
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