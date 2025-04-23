import streamlit as st
import pandas as pd
import pickle
import os

# Artifact directory
model_dir = 'mediactionproject/working'

# Load model and encoders
with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
with open(os.path.join(model_dir, 'symptom_columns.pkl'), 'rb') as f:
    symptom_columns = pickle.load(f)

# Load static data
description = pd.read_csv('description.csv')
precautions = pd.read_csv('precautions_df.csv')
medications = pd.read_csv('medications.csv')
diets = pd.read_csv('diets.csv')
workout = pd.read_csv('workout_df.csv')

# Helper: predict disease
def predict_disease(selected):
    # Construct input vector
    input_df = pd.DataFrame(0, index=[0], columns=symptom_columns)
    for sym in selected:
        if sym in input_df.columns:
            input_df.at[0, sym] = 1
    # Predict
    idx = model.predict(input_df)[0]
    disease = label_encoder.inverse_transform([idx])[0]
    return disease

# Helper: fetch disease details
def get_details(disease):
    desc = description.loc[description['Disease']==disease, 'Description']
    desc = desc.iloc[0] if not desc.empty else "No description available."
    pre_df = precautions.loc[precautions['Disease']==disease,
                             ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = pre_df.iloc[0].tolist() if not pre_df.empty else []
    med = medications.loc[medications['Disease']==disease, 'Medication'].tolist() or ["No medications available."]
    die = diets.loc[diets['Disease']==disease, 'Diet'].tolist() or ["No diet plan available."]
    wrk = workout.loc[workout['disease']==disease, 'workout'].tolist() or ["No workout recommendation."]
    return desc, pre, med, die, wrk

# Streamlit UI
st.title("Disease Prediction and Health Advice")
st.write("Select your symptoms to predict a possible disease and get recommendations.")

# Multiselect to avoid typos
selected_symptoms = st.multiselect("Choose symptoms:", options=symptom_columns)

if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        disease = predict_disease(selected_symptoms)
        st.subheader("Predicted Disease")
        st.write(disease)
        desc, pre, med, die, wrk = get_details(disease)
        st.subheader("Description")
        st.write(desc)
        st.subheader("Precautions")
        for i, p in enumerate(pre, 1): st.write(f"{i}. {p}")
        st.subheader("Medications")
        for m in med: st.write(m)
        st.subheader("Diet Plan")
        for d in die: st.write(d)
        st.subheader("Recommended Workout")
        for w in wrk: st.write(w)