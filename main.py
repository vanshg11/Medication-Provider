import streamlit as st
import pandas as pd
import pickle
import os

# Load artifacts
model_dir = 'mediactionproject/working'
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

# Helper: predict disease from symptoms
def predict_disease(symptoms):
    df0 = pd.DataFrame(0, index=[0], columns=symptom_columns)
    for s in symptoms:
        if s in df0.columns:
            df0.at[0, s] = 1
    idx = model.predict(df0)[0]
    return label_encoder.inverse_transform([idx])[0]

# Helper: fetch details for a disease
def fetch_details(disease):
    desc = description.loc[description['Disease']==disease, 'Description']
    desc = desc.iloc[0] if not desc.empty else "No description available."
    pre_df = precautions.loc[precautions['Disease']==disease,
                             ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = pre_df.iloc[0].tolist() if not pre_df.empty else []
    med = medications.loc[medications['Disease']==disease, 'Medication'].tolist() or ["No medications available."]
    die = diets.loc[diets['Disease']==disease, 'Diet'].tolist() or ["No diet plan available."]
    wrk = workout.loc[workout['disease']==disease, 'workout'].tolist() or ["No workout recommendation."]
    return desc, pre, med, die, wrk

# UI
st.title("Disease Prediction and Health Advice")
user_input = st.text_input("Enter symptoms separated by commas:")
if st.button("Predict"):
    syms = [s.strip() for s in user_input.split(',') if s.strip()]
    if not syms:
        st.warning("Please enter at least one symptom.")
    else:
        disease = predict_disease(syms)
        st.subheader("Predicted Disease")
        st.write(disease)
        desc, pre, med, die, wrk = fetch_details(disease)
        st.subheader("Description")
        st.write(desc)
        st.subheader("Precautions")
        for p in pre: st.write(f"- {p}")
        st.subheader("Medications")
        for m in med: st.write(f"- {m}")
        st.subheader("Diet Plan")
        for d in die: st.write(f"- {d}")
        st.subheader("Recommended Workout")
        for w in wrk: st.write(f"- {w}")