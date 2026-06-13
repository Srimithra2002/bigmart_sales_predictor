import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
.stApp { background: #0f0f0f; color: #f0f0f0; }
.block-container { padding-top: 2rem; max-width: 720px; }
.header-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 2rem; text-align: center;
}
.header-box h1 { font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0; }
.header-box p  { color: #a0aec0; margin-top: 0.5rem; font-size: 0.95rem; }
.section-label {
    font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase; color: #e94560;
    margin-bottom: 0.8rem; margin-top: 1.5rem;
}
.result-card { border-radius: 14px; padding: 1.8rem 2rem; margin-top: 1.5rem; text-align: center; }
.result-card.success { background: linear-gradient(135deg, #0d2d1a, #0a3d20); border: 1px solid #38a169; }
.result-card.warning { background: linear-gradient(135deg, #2d1a0d, #3d2a0a); border: 1px solid #dd6b20; }
.result-card.info    { background: linear-gradient(135deg, #0d1a2d, #0a203d); border: 1px solid #3182ce; }
.result-card h2 { font-size: 2.8rem; font-weight: 800; margin: 0.3rem 0; color: #ffffff; }
.result-card .label  { font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; color: #a0aec0; font-weight: 600; }
.result-card .status { font-size: 1.1rem; font-weight: 600; margin-top: 0.8rem; }
.result-card.success .status { color: #68d391; }
.result-card.warning .status { color: #f6ad55; }
.result-card.info    .status { color: #63b3ed; }
.stButton > button {
    background: #e94560 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.7rem 2rem !important;
    width: 100% !important; margin-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("retail_sales_model.pkl")

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🛒 BigMart Sales Predictor</h1>
    <p>Enter product & store details to predict expected sales revenue</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ retail_sales_model.pkl not found. Upload it to your GitHub repo.")
    st.stop()

# ── Mappings ──────────────────────────────────────────────────
fat_content_map = {"Low Fat": 0, "Regular": 1}
item_type_map   = {
    "Baking Goods": 0, "Breads": 1, "Breakfast": 2, "Canned": 3,
    "Dairy": 4, "Frozen Foods": 5, "Fruits and Vegetables": 6,
    "Hard Drinks": 7, "Health and Hygiene": 8, "Household": 9,
    "Meat": 10, "Others": 11, "Seafood": 12, "Snack Foods": 13,
    "Soft Drinks": 14, "Starchy Foods": 15
}
outlet_size_map = {"Small": 0, "Medium": 1, "High": 2}
outlet_loc_map  = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
outlet_type_map = {
    "Grocery Store": 0, "Supermarket Type1": 1,
    "Supermarket Type2": 2, "Supermarket Type3": 3
}
outlet_id_map = {
    "OUT010": 0, "OUT013": 1, "OUT017": 2, "OUT018": 3, "OUT019": 4,
    "OUT027": 5, "OUT035": 6, "OUT045": 7, "OUT046": 8, "OUT049": 9
}

# ── Inputs ────────────────────────────────────────────────────
st.markdown('<div class="section-label">📦 Product Details</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    item_weight     = st.number_input("Item Weight (kg)", min_value=1.0, max_value=30.0, value=12.5, step=0.5)
    item_fat        = st.selectbox("Fat Content", list(fat_content_map.keys()))
    item_visibility = st.slider("Item Visibility", 0.0, 0.35, 0.05, step=0.005)
with col2:
    item_type       = st.selectbox("Item Type", list(item_type_map.keys()))
    item_mrp        = st.slider("Item MRP (₹)", 10, 300, 150)
    item_identifier = st.number_input("Item ID (numeric)", min_value=0, max_value=999, value=100)

st.markdown('<div class="section-label">🏪 Store Details</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    outlet_id   = st.selectbox("Outlet", list(outlet_id_map.keys()))
    outlet_size = st.selectbox("Outlet Size", list(outlet_size_map.keys()), index=1)
    outlet_year = st.selectbox("Establishment Year", [1985, 1987, 1999, 2002, 2004, 2007, 2009], index=3)
with col4:
    outlet_loc  = st.selectbox("Location Type", list(outlet_loc_map.keys()), index=1)
    outlet_type = st.selectbox("Outlet Type", list(outlet_type_map.keys()), index=1)

st.markdown('<div class="section-label">📊 Stock Check</div>', unsafe_allow_html=True)
current_stock = st.number_input("Current Stock Units", min_value=0, max_value=5000, value=300, step=10)

# ── Predict ───────────────────────────────────────────────────
if st.button("🔮 Predict Sales"):

    # Base features
    item_type_encoded = item_type_map[item_type]
    outlet_year_val   = outlet_year

    # ✅ Engineered features (must match training)
    outlet_age          = 2024 - outlet_year_val
    mrp_per_visibility  = item_mrp / (item_visibility + 0.001)
    non_food            = [6, 7, 8]
    is_food_item        = 0 if item_type_encoded in non_food else 1

    input_df = pd.DataFrame([{
        "Item_Identifier":           item_identifier,
        "Item_Weight":               item_weight,
        "Item_Fat_Content":          fat_content_map[item_fat],
        "Item_Visibility":           item_visibility,
        "Item_Type":                 item_type_encoded,
        "Item_MRP":                  item_mrp,
        "Outlet_Identifier":         outlet_id_map[outlet_id],
        "Outlet_Establishment_Year": outlet_year_val,
        "Outlet_Size":               outlet_size_map[outlet_size],
        "Outlet_Location_Type":      outlet_loc_map[outlet_loc],
        "Outlet_Type":               outlet_type_map[outlet_type],
        # ✅ 3 engineered features
        "Outlet_Age":                outlet_age,
        "MRP_per_Visibility":        mrp_per_visibility,
        "Is_Food_Item":              is_food_item
    }])

    log_pred        = model.predict(input_df)[0]
    predicted_sales = np.expm1(log_pred)

    # Stock alert
    if predicted_sales > current_stock:
        card_class = "warning"
        status_msg = f"⚠️ Stockout Risk — Only {current_stock} units available"
    elif predicted_sales < current_stock * 0.7:
        card_class = "info"
        excess     = current_stock - predicted_sales
        status_msg = f"📦 Overstock Risk — Excess ~{excess:.0f} units"
    else:
        card_class = "success"
        status_msg = "✅ Optimal Stock Level"

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="label">Predicted Sales Revenue</div>
        <h2>₹ {predicted_sales:,.2f}</h2>
        <div class="status">{status_msg}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Item MRP",        f"₹{item_mrp}")
    m2.metric("Predicted Sales", f"₹{predicted_sales:,.0f}")
    m3.metric("Stock Units",     current_stock)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "BigMart Sales Predictor · XGBoost Model · BigMart Dataset</p>",
    unsafe_allow_html=True
)

