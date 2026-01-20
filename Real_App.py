import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="centered"
)

MODEL_DIR = "models"

# ===============================
# LOAD MODELS (GLOBAL – VERY IMPORTANT)
# ===============================
@st.cache_resource
def load_models():
    clf = joblib.load(os.path.join(MODEL_DIR, "classifier_pipeline.pkl"))
    reg = joblib.load(os.path.join(MODEL_DIR, "regressor_pipeline.pkl"))
    return clf, reg

clf_model, reg_model = load_models()

@st.cache_data
def load_data():
    return pd.read_csv("data/eda_sample.csv")
df = load_data()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview","EDA", "Prediction"]
)

# ===============================
# OVERVIEW
# ===============================
if section == "Overview":
    st.title("🏠 Real Estate Investment Advisor")

    st.markdown("""
    ### 🎯 Objective
    - Predict **Property Price per SqFt**
    - Classify **Good / Not a Good Investment**
    - Estimate **Future Price Growth**

    ### 🧠 ML Models
    - Regression model for price prediction
    - Classification model for investment decision

    ### 📊 Dataset
    - Indian real estate listings
    - Location, size, amenities, age, accessibility

    ### ⚙️ Tools
    - Python, Pandas, NumPy
    - Scikit-learn
    - Streamlit
    """)

    st.success("End-to-end ML-powered real estate advisor")


elif section == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    st.subheader("Visual Analysis")

    chart = st.selectbox(
        "Select an analysis",
        [
            "Price per SqFt Distribution",
            "Price per SqFt by City",
            "BHK vs Price",
            "Furnishing Status vs Price",
            "Good Investment Distribution"
        ]
    )

    if chart == "Price per SqFt Distribution":
        fig, ax = plt.subplots()
        ax.hist(df["Price_per_SqFt"], bins=40)
        ax.set_xlabel("Price per SqFt")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif chart == "Price per SqFt by City":
        fig, ax = plt.subplots()
        df.groupby("City")["Price_per_SqFt"].mean().sort_values().plot(
            kind="barh", ax=ax
        )
        ax.set_xlabel("Average Price per SqFt")
        st.pyplot(fig)

    elif chart == "BHK vs Price":
        fig, ax = plt.subplots()
        df.groupby("BHK")["Price_per_SqFt"].mean().plot(kind="bar", ax=ax)
        ax.set_ylabel("Average Price per SqFt")
        st.pyplot(fig)

    elif chart == "Furnishing Status vs Price":
        fig, ax = plt.subplots()
        df.groupby("Furnished_Status")["Price_per_SqFt"].mean().plot(
            kind="bar", ax=ax
        )
        ax.set_ylabel("Average Price per SqFt")
        st.pyplot(fig)

    elif chart == "Good Investment Distribution":
        fig, ax = plt.subplots()
        df["Good_Investment"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Investment Class")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# ===============================
# PREDICTION
# ===============================
elif section == "Prediction":
    st.title("🔮 Property Prediction")

    # ---------- USER INPUTS ----------
    st.subheader("🏠 Property Details")

    state = st.selectbox(
    "State",
    ["Maharashtra", "Karnataka", "Delhi", "Telangana", "Tamil Nadu"]
    )
    city = st.selectbox(
        "City",
        ["Pune", "Mumbai", "Bangalore", "Delhi", "Hyderabad"]
    )

    property_type = st.selectbox(
        "Property Type",
        ["Apartment", "Villa", "Independent House"]
    )

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2
    )

    area = st.number_input(
        "Size in SqFt",
        min_value=300,
        max_value=10000,
        value=1000,
        step=50
    )

    furnishing = st.selectbox(
        "Furnishing Status",
        ["Unfurnished", "Semi-Furnished", "Furnished"]
    )

    floor_no = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=50,
        value=1
    )

    total_floors = st.number_input(
        "Total Floors",
        min_value=1,
        max_value=60,
        value=10
    )

    nearby_schools = st.number_input(
        "Nearby Schools",
        min_value=0,
        max_value=20,
        value=3
    )

    nearby_hospitals = st.number_input(
        "Nearby Hospitals",
        min_value=0,
        max_value=20,
        value=2
    )

    pta = st.slider(
        "Public Transport Accessibility (1–10)",
        1, 10, 5
    )

    parking = st.selectbox(
        "Parking Space",
        [0, 1]
    )

    security = st.selectbox(
        "Security Available",
        [0, 1]
    )

    amenities = st.slider(
        "Amenities Score (1–10)",
        1, 10, 6
    )

    availability = st.selectbox(
        "Availability Status",
        ["Ready to Move", "Under Construction"]
    )

    property_age = st.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=100,
        value=5
    )

    # ---------- INPUT DATAFRAME ----------
    input_df = pd.DataFrame({
    "State": [state],
    "City": [city],
    "Property_Type": [property_type],
    "BHK": [bhk],
    "Size_in_SqFt": [area],
    "Furnished_Status": [furnishing],
    "Nearby_Schools": [3],
    "Nearby_Hospitals": [3],
    "Availability_Status": [availability],
    "Property_Age": [property_age],

    "Price_per_SqFt": [5000]
    })


    # ---------- PREDICT ----------
    if st.button("Predict"):
        price = reg_model.predict(input_df)[0]
        invest = clf_model.predict(input_df)[0]

        st.success(f"💰 Estimated Price per SqFt: ₹ {price:,.0f}")

        st.info(
            "✅ Good Investment"
            if invest == 1 else
            "❌ Not a Good Investment"
        )

        years = st.slider("Future Years", 1, 10, 5)
        future_price = price * ((1 + 0.08) ** years)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price / SqFt", f"₹ {price:,.0f}")
        with col2:
            st.metric("Future Price", f"₹ {future_price:,.0f}")
