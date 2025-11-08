import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("D:\APFrontEnd\car.csv")

# Features and Target
features = ['Year', 'Kms_Driven', 'Fuel_Type', 'Transmission', 'Owner', 'Car_Name']
target = 'Present_Price'

# Prepare Data
X = data[features].copy()
y = data[target]

X['Car_Age'] = 2025 - X['Year']
X = X.drop('Year', axis=1)

categorical_features = ['Fuel_Type', 'Transmission', 'Owner', 'Car_Name']
numeric_features = ['Kms_Driven', 'Car_Age']

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train Model
model.fit(X_train, y_train)

# ----------------------------
# Streamlit UI
# ----------------------------
#st.title("Used Car Price Prediction App")
#st.image("D:\APFrontEnd\car.jpg", caption="Used Car Price Prediction", use_container_width=True)


col1, col2 = st.columns([3, 4])  # left = small, right = large space
with col1:
    st.image("D:\APFrontEnd\car.jpg", width=500)  # small logo size
with col2:
    st.title("Used Car Price Prediction App")

st.write("Enter the details below to estimate the car's market price.")


st.markdown(
    """
     <style>
        /* Main page background */
        [data-testid="stAppViewContainer"] {
            background-color: white;
        }

        /* Header transparent */
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }

        /* Sidebar background */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #cfe2ff 0%, #9ec5fe 100%);
            color: black;
        }
        /* Sidebar text */
        [data-testid="stSidebar"] * {
            color: #000000;
        }
    """,
    unsafe_allow_html=True
    )
# --------------------------------
# 🧩 Sidebar Inputs (Dropdowns)
# --------------------------------
st.sidebar.image("D:\APFrontEnd\car1.jpg", width=320)  # 🚗 small car image at top-left

st.sidebar.markdown("### 📅 Select Year")
years = sorted(data["Year"].unique().tolist())
selected_year = st.sidebar.selectbox("Year", years)
st.sidebar.image("D:\APFrontEnd\car2.jpg", width=320)  # 🚗 small car image at top-left

st.sidebar.markdown("### 🚙 Select Car Model")
car_models = data["Car_Name"].unique().tolist()
selected_model = st.sidebar.selectbox("Car Model", car_models)

# --- Input Widgets ---
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000, step=1000)
year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, value=2018)
fuel_type = st.selectbox("Fuel Type", options=sorted(data['Fuel_Type'].unique()))
transmission = st.selectbox("Transmission", options=sorted(data['Transmission'].unique()))
owner = st.selectbox("Owner Type", options=sorted(data['Owner'].unique()))
car_name = st.selectbox("Car Brand/Name", options=sorted(data['Car_Name'].unique()))

# --- Predict Button ---
if st.button("Predict Price"):
    car_age = 2025 - year
    new_car = pd.DataFrame({
        'Kms_Driven': [kms_driven],
        'Car_Age': [car_age],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Owner': [owner],
        'Car_Name': [car_name]
    })
    
    predicted_price = model.predict(new_car)[0]
    st.success(f"💰 **Estimated Price: ₹{predicted_price:.2f}** Lakhs")

# --- Optional Footer ---
st.markdown("---")

