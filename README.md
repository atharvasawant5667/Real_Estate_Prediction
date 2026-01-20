🏠 Real Estate Investment Advisor

An end-to-end Machine Learning–powered web application that helps users analyze real-estate properties by:

Predicting Price per SqFt

Classifying properties as Good / Bad Investment

Estimating future property value

Visualizing EDA insights interactively

Built using Python, Scikit-Learn, and Streamlit.

Features
Predictions

Price per SqFt prediction using Regression

Investment classification using Classification

Future price estimation (compound growth)

EDA Dashboard

Price distribution

City-wise average prices

BHK vs Price

Furnishing vs Price

Machine Learning

Full preprocessing pipelines

Handles missing values & categorical encoding

Production-ready trained models

Web App

Clean UI using Streamlit

Sidebar navigation (Overview | EDA | Prediction)

Real-time predictions

Models Used
Task	Model
Price Prediction	Random Forest Regressor / XGBoost
Investment Classification	Random Forest Classifier

Project Structure
Real_Estate_Advisor/
│
├── data/
│   └── cleaned_data.csv
│
├── models/
│   ├── regressor_pipeline.pkl
│   └── classifier_pipeline.pkl
│
├── notebooks/
│   └── Real_estate.ipynb
│
├── Real_App.py
├── requirements.txt
└── README.md

Input Features Used

The models are trained using the following features:

State
City
Property_Type
BHK
Size_in_SqFt
Furnished_Status
Floor_No
Total_Floors
Nearby_Schools
Nearby_Hospitals
Public_Transport_Accessibility
Parking_Space
Security
Amenities
Availability_Status
Property_Age


All features must be provided during prediction.

Example Output
Good Investment 
Current Price / SqFt: ₹7,921
Estimated Price after 5 years: ₹11,646

Model Performance
Regression

MAE: ₹4,796

R² Score: 0.31

Classification

Accuracy: ~78–82%

ROC-AUC: ~0.85

Metrics depend on dataset distribution and city.

Installation
Clone Repository
git clone https://github.com/your-username/real-estate-advisor.git
cd real-estate-advisor

Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

Install Dependencies
pip install -r requirements.txt

Run the App
streamlit run Real_App.py


Open in browser:

http://localhost:8501

Deployment (Streamlit Cloud)

Push code to GitHub

Exclude large datasets (>25MB)

Load data via:

Git LFS

Cloud storage (Google Drive / S3)

Upload trained models inside models/

Technologies Used

Python

Pandas, NumPy

Scikit-Learn

XGBoost

Matplotlib

Streamlit

Joblib

Future Improvements

City-specific models

Price confidence intervals

User-uploaded property data

MLflow experiment tracking

API deployment (FastAPI)

Author

Atharva Sawant
Machine Learning & Data Science Enthusiast


If you like this project

Give it a ⭐ on GitHub and feel free to fork or contribute!
