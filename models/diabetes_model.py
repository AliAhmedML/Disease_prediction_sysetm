import streamlit as st
import pickle
import numpy as np
import pandas as pd
import cudf


class DiabetesModel:
    def __init__(self):
        self.model = pickle.load(open("Trained_models_Diabetes/Diabetes.pkl", "rb"))
        self.power_transformer = pickle.load(
            open("Trained_models_Diabetes/Power_Transformer.pkl", "rb")
        )
        self.scaler = pickle.load(open("Trained_models_Diabetes/Scaler.pkl", "rb"))

    def render_ui(self, container):
        with container:
            st.header("Diabetes Test")
            st.write("### Please enter the following health parameters:")

            pregnancies = st.number_input(
                "Pregnancies", min_value=0, key="diabetes_pregnancies"
            )
            glucose = st.number_input(
                "Glucose Level", min_value=0, key="diabetes_glucose"
            )
            bp = st.number_input("Blood Pressure", min_value=0, key="diabetes_bp")
            skin_thickness = st.number_input(
                "Skin Thickness", min_value=0, key="diabetes_skin"
            )
            insulin = st.number_input("Insulin", min_value=0, key="diabetes_insulin")
            bmi = st.number_input(
                "BMI", min_value=0.0, format="%.3f", key="diabetes_bmi"
            )
            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                format="%.3f",
                key="diabetes_dpf",
            )
            age = st.number_input("Age", min_value=0, key="diabetes_age")

            if st.button("Test Result", key="diabetes_button"):

                input_data = pd.DataFrame(
                    [
                        [
                            pregnancies,
                            glucose,
                            bp,
                            skin_thickness,
                            insulin,
                            bmi,
                            dpf,
                            age,
                        ]
                    ],
                    columns=[
                        "Pregnancies",
                        "Glucose",
                        "BloodPressure",
                        "SkinThickness",
                        "Insulin",
                        "BMI",
                        "DiabetesPedigreeFunction",
                        "Age",
                    ],
                )
                input_data = input_data.astype("float32")

                transformed = self.power_transformer.transform(
                    input_data[
                        [
                            "Pregnancies",
                            "Glucose",
                            "Insulin",
                            "DiabetesPedigreeFunction",
                            "Age",
                        ]
                    ]
                )
                input_data.loc[
                    :,
                    [
                        "Pregnancies",
                        "Glucose",
                        "Insulin",
                        "DiabetesPedigreeFunction",
                        "Age",
                    ],
                ] = transformed

                input_scaled = self.scaler.transform(input_data)
                input_scaled = cudf.DataFrame(input_scaled).astype("float32")
                prediction = self.model.predict(input_scaled)

                if prediction[0] == 1:
                    st.error(
                        "\u26a0\ufe0f Positive: High risk of Diabetes 🁬", icon="🚨"
                    )
                else:
                    st.success("✅ Negative: You're healthy! 🥦", icon="💪")
