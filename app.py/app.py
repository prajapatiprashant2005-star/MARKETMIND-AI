
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ==========================================
# MARKETMIND AI - ENTERPRISE APPLICATION
# ==========================================

st.set_page_config(
    page_title="MARKETMIND AI | Enterprise Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "trained"

SUPPLIER_PATH = MODEL_DIR / "supplier_sales_model.pkl"
CUSTOMER_PATH = MODEL_DIR / "online_retail_segmentation_model.pkl"
METRICS_PATH = (
    ROOT / "reports" / "models"
    / "supplier_sales" / "model_metrics.json"
)

# ==========================================
# PROFESSIONAL DARK THEME
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1424;
        color: #eaf2ff;
    }

    [data-testid="stSidebar"] {
        background-color: #111f35;
    }

    h1, h2, h3 {
        color: #eaf2ff !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    div[data-testid="stMetric"] {
        background: #172941;
        padding: 18px;
        border: 1px solid #2c4360;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a8bfdc;
    }

    div[data-testid="stMetricValue"] {
        color: #63e6d0;
    }

    div.stButton > button {
        background: #16a6a0;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: #118985;
        color: white;
    }

    .mm-card {
        background: #172941;
        border: 1px solid #2c4360;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 14px;
    }

    .mm-muted {
        color: #a8bfdc;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_bundle(path):
    return joblib.load(str(path))


def supplier_bundle():
    saved = load_bundle(SUPPLIER_PATH)
    return saved["model"], list(saved["feature_columns"])


def customer_bundle():
    saved = load_bundle(CUSTOMER_PATH)
    return (
        saved["pipeline"],
        list(saved["feature_names"]),
        saved["segment_names"]
    )


def customer_prediction(recency, frequency, monetary):
    pipeline, columns, names = customer_bundle()

    values = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }

    frame = pd.DataFrame([values])
    frame = frame[columns]

    segment = int(pipeline.predict(frame)[0])

    name = names.get(
        segment,
        names.get(str(segment), f"Segment {segment}")
    )

    return segment, name, frame


def supplier_prediction(supplier, year, month, transfers):
    model, features = supplier_bundle()

    # Baseline demonstration only.
    # Missing business features remain zero.
    frame = pd.DataFrame(
        0.0,
        index=[0],
        columns=features
    )

    frame.loc[0, supplier] = 1.0

    inputs = {
        "YEAR": year,
        "MONTH": month,
        "RETAIL TRANSFERS": transfers
    }

    for column, value in inputs.items():
        if column in frame.columns:
            frame.loc[0, column] = value

    result = model.predict(frame)

    return float(result[0])


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:
    st.title("🧠 MARKETMIND AI")
    st.caption("ENTERPRISE INTELLIGENCE SUITE")

    st.divider()

    page = st.radio(
        "WORKSPACES",
        [
            "Executive Overview",
            "Supplier Sales AI",
            "Customer Segmentation AI",
            "Model Performance",
            "Prediction Center"
        ]
    )

    st.divider()

    st.success("Application initialized")

    st.caption(
        "Real Market Sales Analysis | "
        "Machine Learning Platform"
    )


# ==========================================
# COMMON HEADER
# ==========================================

st.title("🧠 MARKETMIND AI")
st.caption(
    "ENTERPRISE SALES & CUSTOMER INTELLIGENCE PLATFORM"
)
st.divider()


# ==========================================
# SUPPLIER PREDICTION WORKSPACE
# ==========================================

def supplier_workspace():

    st.subheader("Supplier Sales Prediction Engine")

    if not SUPPLIER_PATH.exists():
        st.error("Supplier model file was not found.")
        return

    try:
        model, features = supplier_bundle()

        st.success("Random Forest model connected")

        a, b = st.columns(2)
        a.metric("Model", type(model).__name__)
        b.metric("Input Features", len(features))

        suppliers = [
            x for x in features
            if x.startswith("SUPPLIER_")
        ]

        if not suppliers:
            st.warning("Supplier feature columns not found.")
            return

        st.markdown("### Configure Prediction")

        supplier = st.selectbox(
            "Select Supplier",
            suppliers,
            format_func=lambda x: x.replace(
                "SUPPLIER_", ""
            )
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            year = st.number_input(
                "Year",
                min_value=2000,
                max_value=2100,
                value=2026
            )

        with c2:
            month = st.selectbox(
                "Month",
                range(1, 13)
            )

        with c3:
            transfers = st.number_input(
                "Retail Transfers",
                min_value=0.0,
                value=0.0
            )

        st.warning(
            "Research demonstration: unspecified model "
            "features are zero-filled. The predicted value "
            "is not a validated business sales forecast."
        )

        if st.button(
            "🚀 Generate Sales Prediction",
            key="supplier_predict"
        ):
            result = supplier_prediction(
                supplier,
                year,
                month,
                transfers
            )

            st.markdown("### Prediction Result")

            st.metric(
                "Predicted Model Output",
                f"{result:,.4f}"
            )

            st.success("Prediction executed successfully")

        with st.expander("View Model Input Schema"):
            st.dataframe(
                pd.DataFrame({"Feature": features}),
                use_container_width=True
            )

    except Exception as error:
        st.error(f"Supplier model error: {error}")


# ==========================================
# CUSTOMER PREDICTION WORKSPACE
# ==========================================

def customer_workspace():

    st.subheader("Customer Segmentation Engine")

    if not CUSTOMER_PATH.exists():
        st.error("Customer model file was not found.")
        return

    try:
        pipeline, columns, names = customer_bundle()

        st.success("Customer clustering pipeline connected")

        a, b = st.columns(2)
        a.metric("Customer Features", len(columns))
        b.metric("Customer Segments", len(names))

        st.markdown("### Customer RFM Analysis")

        c1, c2, c3 = st.columns(3)

        with c1:
            recency = st.number_input(
                "Recency (Days)",
                min_value=0,
                value=30
            )

        with c2:
            frequency = st.number_input(
                "Frequency",
                min_value=1,
                value=5
            )

        with c3:
            monetary = st.number_input(
                "Monetary Value",
                min_value=0.0,
                value=1000.0
            )

        if st.button(
            "🧠 Analyze Customer",
            key="customer_predict"
        ):
            segment, name, frame = customer_prediction(
                recency,
                frequency,
                monetary
            )

            st.success(f"Predicted Segment: {name}")

            a, b = st.columns(2)
            a.metric("Customer Segment", name)
            b.metric("Cluster ID", segment)

            st.dataframe(
                frame,
                use_container_width=True
            )

            st.markdown("### Business Interpretation")

            recommendations = {
                "Premium Customers":
                    "Focus on personalized service and retention.",
                "Loyal Customers":
                    "Offer loyalty rewards and relevant products.",
                "Regular Customers":
                    "Use targeted promotions to encourage repeat purchases.",
                "Low Value Customers":
                    "Consider cost-effective re-engagement campaigns."
            }

            st.info(
                recommendations.get(
                    name,
                    "Review this customer's purchasing behavior."
                )
            )

        with st.expander("View Segment Labels"):
            st.json({
                str(k): v for k, v in names.items()
            })

    except Exception as error:
        st.error(f"Customer model error: {error}")


# ==========================================
# EXECUTIVE OVERVIEW
# ==========================================

if page == "Executive Overview":

    st.header("Enterprise Intelligence Overview")

    st.write(
        "A unified machine learning environment for "
        "supplier sales analysis and customer intelligence."
    )

    a, b, c, d = st.columns(4)

    a.metric("AI Models", "02")
    b.metric("Supplier Features", "549")
    c.metric("Customer Segments", "04")
    d.metric("ML Approaches", "02")

    st.markdown("### Intelligence Workspaces")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="mm-card">
            <h3>📈 Supplier Sales AI</h3>
            <p class="mm-muted">
                Random Forest regression model for
                supplier-related sales analysis.
            </p>
            <p>Supervised Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="mm-card">
            <h3>👥 Customer Intelligence</h3>
            <p class="mm-muted">
                RFM-based customer segmentation using
                a trained clustering pipeline.
            </p>
            <p>Unsupervised Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Model Availability")

    a, b = st.columns(2)

    a.metric(
        "Supplier Model",
        "Available" if SUPPLIER_PATH.exists() else "Missing"
    )

    b.metric(
        "Customer Model",
        "Available" if CUSTOMER_PATH.exists() else "Missing"
    )

    st.info(
        "Select a workspace from the sidebar to "
        "explore your trained machine learning models."
    )


# ==========================================
# SUPPLIER SALES AI PAGE
# ==========================================

elif page == "Supplier Sales AI":

    st.header("Supplier Sales Intelligence")

    supplier_workspace()


# ==========================================
# CUSTOMER SEGMENTATION PAGE
# ==========================================

elif page == "Customer Segmentation AI":

    st.header("Customer Intelligence")

    customer_workspace()


# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    st.header("Model Performance & Evaluation")

    st.markdown("### Supplier Sales Model")

    if METRICS_PATH.exists():
        try:
            with open(
                METRICS_PATH,
                "r",
                encoding="utf-8"
            ) as file:
                metrics = json.load(file)

            st.success("Saved model metrics loaded")

            st.json(metrics)

        except Exception as error:
            st.error(f"Metrics loading failed: {error}")

    else:
        st.warning(
            "Supplier evaluation metrics are unavailable."
        )

    st.divider()

    st.markdown("### Customer Segmentation Model")

    try:
        pipeline, columns, names = customer_bundle()

        a, b = st.columns(2)
        a.metric("Input Features", len(columns))
        b.metric("Segments", len(names))

        st.write("Feature Schema:", columns)

        st.json({
            str(k): v for k, v in names.items()
        })

        st.info(
            "Clustering quality metrics must be "
            "evaluated using the original customer data."
        )

    except Exception as error:
        st.error(f"Customer evaluation error: {error}")


# ==========================================
# PREDICTION CENTER
# ==========================================

elif page == "Prediction Center":

    st.header("Unified AI Prediction Center")

    tab1, tab2 = st.tabs([
        "📈 Supplier Prediction",
        "👥 Customer Segmentation"
    ])

    with tab1:
        supplier_workspace()

    with tab2:
        customer_workspace()


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "MARKETMIND AI | REAL MARKET SALES ANALYSIS "
    "| ENTERPRISE MACHINE LEARNING PROJECT"
)
