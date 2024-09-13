import streamlit as st
import pandas as pd
import joblib

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load("random_forest_model.pkl")
    return model

model = load_model()

st.sidebar.header("Input Features")

def user_input_features():
    RaceDesc = st.sidebar.selectbox("Race Description", ["Asian", "Black", "Hispanic", "White", "Other"])
    EmployeeType = st.sidebar.selectbox("Employee Type", ["Zone A", "Zone B", "Zone C"])
    EmployeeClassificationType = st.sidebar.selectbox("Employee Classification Type", ["Part Time", "Full Time", "Temporary"])
    Performance_Score = st.sidebar.slider("Performance Score", 0, 100, 50)
    
    data = {
        'RaceDesc': RaceDesc,
        'EmployeeType': EmployeeType,
        'EmployeeClassificationType': EmployeeClassificationType,
        'Performance Score': Performance_Score
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

df_encoded = pd.get_dummies(input_df)
df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict
prediction = model.predict(df_encoded)
prediction_proba = model.predict_proba(df_encoded)

st.subheader("Prediction Employee")
st.write(f"Employee Status: {prediction[0]}")

st.subheader("Prediction Probability")
st.write(prediction_proba)
