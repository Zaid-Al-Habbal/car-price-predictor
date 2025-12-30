import streamlit as st
from api import predict_price
from config import API_URL
from styles import load_enhanced_neon_theme


# Page Config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚙",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load custom theme
load_enhanced_neon_theme()

# Header with animation
st.markdown("""
    <div class="header-container">
        <h1 class="neon-title">🚙 Car Price Predictor</h1>
        <p class="subtitle">Get instant AI-powered price estimates for your vehicle</p>
    </div>
""", unsafe_allow_html=True)

# Info cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div class="info-card">
            <div class="info-icon">⚡</div>
            <div class="info-text">Fast</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div class="info-card">
            <div class="info-icon">🎯</div>
            <div class="info-text">Accurate</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div class="info-card">
            <div class="info-icon">🔒</div>
            <div class="info-text">Secure</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main form with better organization
st.markdown('<div class="form-container">', unsafe_allow_html=True)

with st.form("car_form"):
    st.markdown("### Vehicle Details")
    
    # Basic Info Section
    st.markdown('<div class="section-header">Basic Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("🚙 Car Name", "Hyundai i20", help="Enter the car model name")
        year = st.number_input("📅 Year", min_value=1990, max_value=2025, value=2018, help="Manufacturing year")
    with col2:
        km_driven = st.number_input("🛣️ KM Driven", min_value=0, step=1000, value=50000, help="Total kilometers driven")
        seats = st.number_input("💺 Seats", min_value=2, max_value=10, value=5, help="Number of seats")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Engine Specifications Section
    st.markdown('<div class="section-header">Engine Specifications</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        mileage = st.text_input("⛽ Mileage", "18.5 kmpl", help="Fuel efficiency")
    with col2:
        engine = st.text_input("🔧 Engine", "1197 CC", help="Engine capacity")
    with col3:
        max_power = st.text_input("⚡ Max Power", "82 bhp", help="Maximum power output")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional Details Section
    st.markdown('<div class="section-header">Additional Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fuel = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
        transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])
    with col2:
        seller_type = st.selectbox("👤 Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
        owner = st.selectbox("🔑 Owner Type", ["First Owner", "Second Owner", "Third & Above Owner"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Submit button with custom styling
    submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction results
if submitted:
    payload = {
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "torque": "",
        "name": name,
    }

    with st.spinner("🔄 Analyzing your vehicle..."):
        try:
            price = predict_price(API_URL, payload) / 73
            
            # Success message with animation
            st.markdown(f"""
                <div class="result-card success-card">
                    <div class="result-title">Estimated Price</div>
                    <div class="result-price">${price:,.0f}</div>
                    <div class="result-subtitle">Based on current market trends</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional info
            st.markdown("""
                <div class="info-box">
                    💡 <b>Tip:</b> This is an estimated price based on machine learning models (random forest). 
                    Actual prices may vary based on vehicle condition and market demand.
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown("""
                <div class="result-card error-card">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title">Prediction Failed</div>
                    <div class="result-subtitle">Unable to connect to the prediction service. Please ensure the API is running.</div>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div class="footer">
        <p>Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)