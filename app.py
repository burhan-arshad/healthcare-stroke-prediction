import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras

st.set_page_config(
    page_title="Stroke Prediction",
    layout="centered"
)

model = keras.models.load_model("model.keras")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Stroke Prediction using ANN")
st.write("Enter patient information to predict the probability of stroke.")

sample_data = {
    "gender": "Male",
    "age": 67.0,
    "hypertension": "Yes",
    "heart_disease": "Yes",
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

if st.button("Load Sample Data"):
    st.session_state["gender"] = sample_data["gender"]
    st.session_state["age"] = sample_data["age"]
    st.session_state["hypertension"] = sample_data["hypertension"]
    st.session_state["heart_disease"] = sample_data["heart_disease"]
    st.session_state["ever_married"] = sample_data["ever_married"]
    st.session_state["work_type"] = sample_data["work_type"]
    st.session_state["Residence_type"] = sample_data["Residence_type"]
    st.session_state["avg_glucose_level"] = sample_data["avg_glucose_level"]
    st.session_state["bmi"] = sample_data["bmi"]
    st.session_state["smoking_status"] = sample_data["smoking_status"]
    st.rerun()

gender = st.selectbox(
    "Gender",
    ["Female", "Male"],
    key="gender"
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=120.0,
    key="age"
)

hypertension = st.selectbox(
    "Hypertension",
    ["No", "Yes"],
    key="hypertension"
)

heart_disease = st.selectbox(
    "Heart Disease",
    ["No", "Yes"],
    key="heart_disease"
)

ever_married = st.selectbox(
    "Ever Married",
    ["No", "Yes"],
    key="ever_married"
)

work_type = st.selectbox(
    "Work Type",
    [
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked"
    ],
    key="work_type"
)

residence_type = st.selectbox(
    "Residence Type",
    ["Rural", "Urban"],
    key="Residence_type"
)

avg_glucose_level = st.number_input(
    "Average Glucose Level",
    min_value=0.0,
    max_value=500.0,
    key="avg_glucose_level"
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=100.0,
    key="bmi"
)

smoking_status = st.selectbox(
    "Smoking Status",
    [
        "formerly smoked",
        "never smoked",
        "smokes",
        "Unknown"
    ],
    key="smoking_status"
)

if st.button("Predict Stroke Risk", use_container_width=True):

    input_data = pd.DataFrame({
        "gender": [1 if gender == "Male" else 0],
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "ever_married": [1 if ever_married == "Yes" else 0],
        "work_type": [work_type],
        "Residence_type": [1 if residence_type == "Urban" else 0],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    })

    input_data = pd.get_dummies(
        input_data,
        columns=["work_type", "smoking_status"],
        dtype=int
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    numeric_cols = [
        "age",
        "avg_glucose_level",
        "bmi"
    ]

    input_data[numeric_cols] = scaler.transform(
        input_data[numeric_cols]
    )

    probability = model.predict(
        input_data,
        verbose=0
    )[0][0]

    prediction = probability >= 0.5

    st.subheader("Prediction Result")

    st.metric(
        "Stroke Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction:
        st.error("The model predicts a higher risk of stroke.")
    else:
        st.success("The model predicts a lower risk of stroke.")

    st.caption(
        "This application is for educational purposes only "
        "and should not be used as a medical diagnosis."
    )