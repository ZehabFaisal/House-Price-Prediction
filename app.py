"""Streamlit UI for the house-price regression demo."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "house_price_model.joblib"

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1000px 600px at 12% -10%, rgba(56, 189, 248, 0.14), transparent 60%),
            radial-gradient(900px 600px at 100% 0%, rgba(139, 92, 246, 0.16), transparent 55%),
            linear-gradient(160deg, #0b1120 0%, #070b14 55%, #05080f 100%);
        background-attachment: fixed;
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: transparent;
        backdrop-filter: blur(6px);
    }
    [data-testid="stToolbar"], #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: #f8fafc;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 23, 42, 0.62) !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
        border-radius: 22px !important;
        padding: 1.7rem 1.8rem !important;
        box-shadow: 0 24px 80px rgba(2, 6, 23, 0.55);
        backdrop-filter: blur(16px);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(34, 211, 238, 0.28) !important;
        box-shadow: 0 26px 90px rgba(2, 6, 23, 0.7);
    }

    .st-key-hero h1 {
        font-size: clamp(1.9rem, 3.4vw, 2.8rem);
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0.15rem 0 0.55rem 0;
        color: #ffffff;
    }
    .st-key-hero p {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 680px;
        margin-bottom: 0;
    }
    .eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        color: #67e8f9;
        margin-bottom: 0.4rem;
    }

    .st-key-details h3,
    .st-key-result h3 {
        font-size: 1.03rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #ffffff;
        margin: 0 0 0.2rem 0;
    }
    .section-sub {
        color: #94a3b8;
        font-size: 0.86rem;
        margin-bottom: 1.3rem;
    }

    [data-testid="stWidgetLabel"] p {
        color: #cbd5e1;
        font-size: 0.86rem;
        font-weight: 500;
    }

    div[data-baseweb="input"] {
        background: rgba(2, 6, 23, 0.55) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: rgba(34, 211, 238, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.12) !important;
    }
    div[data-baseweb="input"] input {
        color: #ffffff !important;
        font-weight: 600;
        padding: 0.55rem 0.75rem !important;
    }
    div[data-testid="stNumberInput"] button {
        background: rgba(148, 163, 184, 0.1) !important;
        border-color: rgba(148, 163, 184, 0.18) !important;
        color: #e2e8f0 !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        background: rgba(34, 211, 238, 0.18) !important;
        color: #67e8f9 !important;
    }

    div.stButton > button {
        width: 100%;
        margin-top: 0.6rem;
        padding: 0.78rem 1rem;
        border: none;
        border-radius: 12px;
        background: linear-gradient(90deg, #06b6d4, #6366f1);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.98rem;
        letter-spacing: 0.01em;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.35);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
        box-shadow: 0 14px 36px rgba(79, 70, 229, 0.48);
        color: #ffffff;
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    .result-price {
        font-size: clamp(2rem, 4vw, 2.9rem);
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        background: linear-gradient(90deg, #67e8f9, #a5b4fc);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.35rem 0 0.8rem;
    }
    .result-note {
        color: #94a3b8;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .empty-title {
        color: #e2e8f0;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.35rem 0 0.45rem;
    }
    .divider {
        height: 1px;
        background: rgba(148, 163, 184, 0.12);
        margin: 1.3rem 0;
    }
    .result-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1.2rem;
    }
    .chip {
        font-size: 0.76rem;
        color: #cbd5e1;
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.14);
        padding: 0.38rem 0.68rem;
        border-radius: 999px;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=False, key="hero"):
    st.title("🏠 House Price Prediction")
    st.write("Estimate a house price using its area, bedrooms, bathrooms, and age.")

if not MODEL_PATH.exists():
    st.error("Error! Model not found. Run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    with st.container(border=True, key="details"):
        st.subheader("Property Details")
        st.markdown(
            '<div class="section-sub">Enter the details below to estimate a price.</div>',
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            area = st.number_input(
                "Area (square feet)",
                min_value=100,
                max_value=10000,
                value=1200,
                step=50,
            )
            bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                max_value=10,
                value=2,
            )
        with col_b:
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=1,
                max_value=15,
                value=3,
            )
            age = st.number_input(
                "Property age (years)",
                min_value=0,
                max_value=150,
                value=5,
            )
        estimate = st.button("Estimate price", type="primary", use_container_width=True)

with right:
    with st.container(border=True, key="result"):
        st.subheader("Estimated Price")
        if estimate:
            row = pd.DataFrame(
                {
                    "area_sqft": [area],
                    "bedrooms": [bedrooms],
                    "bathrooms": [bathrooms],
                    "age_years": [age],
                }
            )
            prediction = float(model.predict(row)[0])
            st.markdown(
                f'<div class="result-price">{prediction:,.0f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="result-note">Sample price in arbitrary units, generated by a '
                "Random Forest model trained on synthetic data.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="divider"></div>
                <div class="result-chips">
                    <span class="chip">Area: {area:,.0f} sq ft</span>
                    <span class="chip">Bedrooms: {bedrooms}</span>
                    <span class="chip">Bathrooms: {bathrooms}</span>
                    <span class="chip">Property age: {age} years</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="empty-title">Your estimate will appear here.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="result-note">Enter the property details and click '
                "<b>Estimate price</b> to run the model.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="divider"></div>
                <div class="result-chips">
                    <span class="chip">Model: Random Forest</span>
                    <span class="chip">Library: scikit-learn</span>
                    <span class="chip">Data: Synthetic</span>
                </div>
                """,
                unsafe_allow_html=True,
            )