import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("diabetes_dataset.csv")

# Drop columns we don't need for prediction
df = df.drop(["year", "location"], axis=1)

# Encode categorical columns
le_gender = LabelEncoder()
le_smoking = LabelEncoder()

df["gender"] = le_gender.fit_transform(df["gender"])
df["smoking_history"] = le_smoking.fit_transform(df["smoking_history"])

# Features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save feature names for consistent prediction input
expected_features = X.columns.tolist()

# Streamlit UI
st.title("🧠 Diabetes Prediction App (Enhanced Dataset)")

# Input form for user data
gender = st.selectbox("Gender", le_gender.classes_)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking = st.selectbox("Smoking History", le_smoking.classes_)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=110)

# Race columns — you can select one race only (as per dataset encoding)
race_options = ['AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other']
race_selected = st.selectbox("Race", race_options)

# Prepare race data (one-hot encoded)
race_data = {
    'race_AfricanAmerican': 0,
    'race_Asian': 0,
    'race_Caucasian': 0,
    'race_Hispanic': 0,
    'race_Other': 0
}
race_column = f'race_{race_selected}'
race_data[race_column] = 1

if st.button("Predict"):
    # Prepare basic input data
    input_data = {
        'gender': le_gender.transform([gender])[0],
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': le_smoking.transform([smoking])[0],
        'bmi': bmi,
        'hbA1c_level': hba1c,
        'blood_glucose_level': glucose,
    }

    # Combine input data with race data
    input_data.update(race_data)

    # Convert to DataFrame and ensure correct column order
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Make prediction and get probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # Probability of class 1 (diabetes)

    # Display result
    if prediction == 1:
        st.error(f"⚠️ The model predicts that the person **has diabetes**.\n\n🔢 Probability: **{probability:.2%}**")
    else:
        st.success(f"✅ The model predicts that the person **does NOT have diabetes**.\n\n🔢 Probability: **{probability:.2%}**")
