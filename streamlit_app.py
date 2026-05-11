import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# Page config
# ======================
st.set_page_config(page_title="Lifestyle Data Analysis", layout="wide")

# ======================
# Load data
# ======================
@st.cache_data
def load_data():
    return pd.read_csv("final_data_cleaned.csv")

df = load_data()

# ======================
# ML Model Training
# ======================
@st.cache_resource
def train_model(df):
    target = "calories_burned"

    numeric_features = [
        "age", "weight_(kg)", "height_(m)", "session_duration_(hours)",
        "fat_percentage", "water_intake_(liters)", "bmi"
    ]
    categorical_features = [
        "gender", "experience_level", "workout_type"
    ]

    df_ml = df.dropna(subset=[target]).copy()
    X = df_ml[numeric_features + categorical_features]
    y = df_ml[target]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=250, random_state=42)
    rf_pipeline = Pipeline([
        ("prep", preprocessor),
        ("rf", rf_model)
    ])
    rf_pipeline.fit(X_train, y_train)

    # Optional: print metrics
    preds = rf_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    st.write(f"ML Model Trained! RMSE: {rmse:.2f}, R²: {r2:.2f}")

    return rf_pipeline, X_train.columns

rf_pipeline, feature_columns = train_model(df)

# ======================
# Prediction function
# ======================
def predict_calories(input_dict):
    temp_df = pd.DataFrame([input_dict])
    # Fill missing columns
    for col in feature_columns:
        if col not in temp_df.columns:
            temp_df[col] = np.nan
    temp_df = temp_df[feature_columns]
    return rf_pipeline.predict(temp_df)[0]

# ======================
# Sidebar: Page selection
# ======================
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Machine Learning"])

# ======================
# Dashboard Tab
# ======================
if page == "Dashboard":
    st.title("🏋️ Lifestyle & Fitness Data Dashboard")
    st.write("Interactive data exploration based on your dataset.")

    st.sidebar.header("Filters")
    gender_filter = st.sidebar.selectbox("Select Gender:", options=["All"] + sorted(df['gender'].dropna().unique().tolist()))
    experience_filter = st.sidebar.selectbox("Experience Level:", options=["All"] + sorted(df['experience_level'].dropna().unique().tolist()))
    diet_filter = st.sidebar.selectbox("Diet Type:", options=["All"] + sorted(df['diet_type'].dropna().unique().tolist()))

    df_filtered = df.copy()
    if gender_filter != "All":
        df_filtered = df_filtered[df_filtered['gender'] == gender_filter]
    if experience_filter != "All":
        df_filtered = df_filtered[df_filtered['experience_level'] == experience_filter]
    if diet_filter != "All":
        df_filtered = df_filtered[df_filtered['diet_type'] == diet_filter]

    st.subheader("📊 Dataset Preview")
    st.dataframe(df_filtered.head())

    # Summary stats
    st.subheader("📌 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average BMI", f"{df_filtered['bmi'].mean():.2f}")
    col2.metric("Avg Workout Duration (hrs)", f"{df_filtered['session_duration_(hours)'].mean():.2f}")
    col3.metric("Avg Calories Burned", f"{df_filtered['calories_burned'].mean():.2f}")
    col4.metric("Avg Protein per kg", f"{df_filtered['protein_per_kg'].mean():.2f}")

    # EDA Plots
    st.header("📈 Exploratory Data Analysis")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df_filtered['bmi'], kde=True, ax=ax)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=df_filtered, x='session_duration_(hours)', y='calories_burned', hue='experience_level', ax=ax)
    st.pyplot(fig)

# ======================
# ML Prediction Tab
# ======================
if page == "Machine Learning":
    st.header("🤖 Predict Calories Burned")

    # Inputs
    age = st.number_input("Age", 10, 100, 30)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.75)
    session_duration = st.number_input("Session Duration (hours)", 0.1, 5.0, 1.0)
    fat_percentage = st.number_input("Fat Percentage", 0.0, 50.0, 15.0)
    water_intake = st.number_input("Water Intake (liters)", 0.0, 10.0, 2.0)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    experience_level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
    workout_type = st.selectbox("Workout Type", ["Cardio", "Strength", "Mixed"])

    if st.button("Predict Calories"):
        input_dict = {
            "age": age,
            "weight_(kg)": weight,
            "height_(m)": height,
            "session_duration_(hours)": session_duration,
            "fat_percentage": fat_percentage,
            "water_intake_(liters)": water_intake,
            "bmi": bmi,
            "gender": gender,
            "experience_level": experience_level,
            "workout_type": workout_type
        }
        calories_pred = predict_calories(input_dict)
        st.success(f"Predicted Calories Burned: {calories_pred:.2f}")