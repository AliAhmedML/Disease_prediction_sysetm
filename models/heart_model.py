import streamlit as st
import pickle
import numpy as np
import pandas as pd
import cudf


class HeartModel:
    def __init__(self):
        self.model = pickle.load(open("Trained_models_Heart/Heart.pkl", "rb"))
        self.power_transformer = pickle.load(
            open("Trained_models_Heart/Power_Transformer.pkl", "rb")
        )
        self.scaler = pickle.load(open("Trained_models_Heart/Scaler.pkl", "rb"))

    def render_ui(self, container):
        with container:
            st.header("Heart Disease Test")
            st.write("### Please enter the following health parameters:")

            age = st.number_input("Age", min_value=0, key="heart_age")
            sex = st.selectbox("Sex", options=[0, 1], key="heart_sex")
            cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], key="heart_cp")
            trestbps = st.number_input(
                "Resting Blood Pressure", min_value=0, key="heart_trestbps"
            )
            chol = st.number_input("Cholesterol", min_value=0, key="heart_chol")
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl", options=[0, 1], key="heart_fbs"
            )
            restecg = st.selectbox("Rest ECG", options=[0, 1, 2], key="heart_restecg")
            thalach = st.number_input(
                "Max Heart Rate", min_value=0, key="heart_thalach"
            )
            exang = st.selectbox(
                "Exercise Induced Angina", options=[0, 1], key="heart_exang"
            )
            oldpeak = st.number_input("Oldpeak", format="%.3f", key="heart_oldpeak")
            slope = st.selectbox("Slope", options=[0, 1, 2], key="heart_slope")
            ca = st.selectbox(
                "Number of Major Vessels", options=[0, 1, 2, 3], key="heart_ca"
            )
            thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], key="heart_thal")

            if st.button("Test Result", key="heart_button"):

                input_data = pd.DataFrame(
                    [
                        [
                            age,
                            sex,
                            cp,
                            trestbps,
                            chol,
                            fbs,
                            restecg,
                            thalach,
                            exang,
                            oldpeak,
                            slope,
                            ca,
                            thal,
                        ]
                    ],
                    columns=[
                        "age",
                        "sex",
                        "cp",
                        "trestbps",
                        "chol",
                        "fbs",
                        "restecg",
                        "thalach",
                        "exang",
                        "oldpeak",
                        "slope",
                        "ca",
                        "thal",
                    ],
                )
                input_data = input_data.astype("float32")
                transformed = self.power_transformer.transform(
                    input_data[["oldpeak", "ca", "exang"]]
                )
                input_data.loc[
                    :,
                    ["oldpeak", "ca", "exang"],
                ] = transformed

                input_scaled = self.scaler.transform(input_data)
                input_scaled = cudf.DataFrame(input_scaled).astype("float32")
                prediction = self.model.predict(input_scaled)

                if prediction[0] == 1:
                    st.error("⚠️ Positive: Risk of Heart Disease ❤️‍🔥", icon="💔")
                else:
                    st.success("✅ Negative: Your heart is healthy! ❤️", icon="🫀")
