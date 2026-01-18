import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Real Estate Investment Analyzer",
    page_icon="🏠",
    layout="centered"
)

# Load models
reg_model = joblib.load("models/regressor_pipeline.pkl")
clf_model = joblib.load("models/investment_model.pkl")

st.title("🏠 Real Estate Investment Analyzer")
st.caption("Predict property price & investment potential using Machine Learning")
st.divider()

state = st.selectbox("State", ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu"])
city = st.text_input("City", "Mumbai")
property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Independent House"])

bhk = st.slider("BHK", 1, 6, 2)
area = st.number_input("Area (SqFt)", min_value=300, max_value=10000, value=1000)

furnishing = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
availability = st.selectbox("Availability Status", ["Ready to Move", "Under Construction"])

floor_no = st.number_input("Floor No", 0, 50, 2)
total_floors = st.number_input("Total Floors", 1, 60, 10)

schools = st.slider("Nearby Schools", 0, 10, 3)
hospitals = st.slider("Nearby Hospitals", 0, 10, 2)

transport = st.slider("Public Transport Access (1–5)", 1, 5, 4)
parking = st.selectbox("Parking Available", ["Yes", "No"])
security = st.selectbox("Security", ["Yes", "No"])
amenities = st.slider("Amenities Count", 0, 10, 5)

property_age = st.slider("Property Age (Years)", 0, 50, 10)
st.divider()

if st.button("🔮 Predict Investment", key="predict_btn"):
        input_df = pd.DataFrame([{
        "State": state,
        "City": city,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": area,
        "Furnished_Status": furnishing,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Availability_Status": availability,
        "Property_Age": property_age
    }])
        price = reg_model.predict(input_df)[0]
        price = max(price, 0)  # safety

        invest_pred = clf_model.predict(input_df)[0]
        invest_prob = clf_model.predict_proba(input_df)[0][1]

        if invest_pred == 1:
            st.success("✅ Good Investment")
        else:
            st.error("❌ Not a Good Investment")

        st.metric(
        label="💰 Estimated Price per SqFt",
        value=f"₹ {price:,.0f}"
        )

        years = st.slider("Future Years", 1, 10, 5)

        growth_rate = 0.08  # 8% annual
        future_price = price * ((1 + growth_rate) ** years)

        st.metric(
            label=f"📈 Estimated Price after {years} years",
            value=f"₹ {future_price:,.0f}"
        )

        st.progress(int(invest_prob * 100))
        st.caption(f"Investment Confidence: {invest_prob*100:.1f}%")





