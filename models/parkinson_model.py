import streamlit as st
import pickle
import numpy as np
import pandas as pd
import cudf


class ParkinsonModel:
    def __init__(self):
        self.model = pickle.load(open("Trained_models_Parkinson/Parkinsons.pkl", "rb"))
        self.power_transformer = pickle.load(
            open("Trained_models_Parkinson/Power_Transformer.pkl", "rb")
        )
        self.scaler = pickle.load(open("Trained_models_Parkinson/Scaler.pkl", "rb"))
        self.pca_1 = pickle.load(open("Trained_models_Parkinson/PCA_1.pkl", "rb"))
        self.pca_2 = pickle.load(open("Trained_models_Parkinson/PCA_2.pkl", "rb"))
        self.pca_3 = pickle.load(open("Trained_models_Parkinson/PCA_3.pkl", "rb"))

    def render_ui(self, container):
        with container:
            st.header("Parkinson's Disease Test")
            st.write("### Please enter the following parameters:")

            name = st.text_input("Patient Name")

            fo = st.number_input("MDVP:Fo(Hz)", format="%.6f", key="parkinson_fo")
            fhi = st.number_input("MDVP:Fhi(Hz)", format="%.6f", key="parkinson_fhi")
            flo = st.number_input("MDVP:Flo(Hz)", format="%.6f", key="parkinson_flo")
            jitter_percent = st.number_input(
                "MDVP:Jitter(%)", format="%.6f", key="parkinson_jitter_percent"
            )
            jitter_abs = st.number_input(
                "MDVP:Jitter(Abs)", format="%.6f", key="parkinson_jitter_abs"
            )
            rap = st.number_input("MDVP:RAP", format="%.6f", key="parkinson_rap")
            ppq = st.number_input("MDVP:PPQ", format="%.6f", key="parkinson_ppq")
            ddp = st.number_input("Jitter:DDP", format="%.6f", key="parkinson_ddp")
            shimmer = st.number_input(
                "MDVP:Shimmer", format="%.6f", key="parkinson_shimmer"
            )
            shimmer_db = st.number_input(
                "MDVP:Shimmer(dB)", format="%.6f", key="parkinson_shimmer_db"
            )
            apq3 = st.number_input("Shimmer:APQ3", format="%.6f", key="parkinson_apq3")
            apq5 = st.number_input("Shimmer:APQ5", format="%.6f", key="parkinson_apq5")
            apq = st.number_input("MDVP:APQ", format="%.6f", key="parkinson_apq")
            dda = st.number_input("Shimmer:DDA", format="%.6f", key="parkinson_dda")
            nhr = st.number_input("NHR", format="%.6f", key="parkinson_nhr")
            hnr = st.number_input("HNR", format="%.6f", key="parkinson_hnr")
            rpde = st.number_input("RPDE", format="%.6f", key="parkinson_rpde")
            dfa = st.number_input("DFA", format="%.6f", key="parkinson_dfa")
            spread1 = st.number_input("spread1", format="%.6f", key="parkinson_spread1")
            spread2 = st.number_input("spread2", format="%.6f", key="parkinson_spread2")
            d2 = st.number_input("D2", format="%.6f", key="parkinson_d2")
            ppe = st.number_input("PPE", format="%.6f", key="parkinson_ppe")

            if st.button("Test Result", key="parkinson_button"):

                input_data = pd.DataFrame(
                    [
                        [
                            fo,
                            fhi,
                            flo,
                            jitter_percent,
                            jitter_abs,
                            rap,
                            ppq,
                            ddp,
                            shimmer,
                            shimmer_db,
                            apq3,
                            apq5,
                            apq,
                            dda,
                            nhr,
                            hnr,
                            rpde,
                            dfa,
                            spread1,
                            spread2,
                            d2,
                            ppe,
                        ]
                    ],
                    columns=[
                        "MDVP:Fo(Hz)",
                        "MDVP:Fhi(Hz)",
                        "MDVP:Flo(Hz)",
                        "MDVP:Jitter(%)",
                        "MDVP:Jitter(Abs)",
                        "MDVP:RAP",
                        "MDVP:PPQ",
                        "Jitter:DDP",
                        "MDVP:Shimmer",
                        "MDVP:Shimmer(dB)",
                        "Shimmer:APQ3",
                        "Shimmer:APQ5",
                        "MDVP:APQ",
                        "Shimmer:DDA",
                        "NHR",
                        "HNR",
                        "RPDE",
                        "DFA",
                        "spread1",
                        "spread2",
                        "D2",
                        "PPE",
                    ],
                )
                input_data = input_data.astype("float32")

                transformed = self.power_transformer.transform(
                    input_data[
                        [
                            "MDVP:Fhi(Hz)",
                            "Jitter:DDP",
                            "MDVP:Jitter(%)",
                            "MDVP:RAP",
                            "MDVP:PPQ",
                            "MDVP:Shimmer",
                            "MDVP:Shimmer(dB)",
                            "Shimmer:APQ5",
                            "MDVP:APQ",
                            "NHR",
                        ]
                    ]
                )
                input_data.loc[
                    :,
                    [
                        "MDVP:Fhi(Hz)",
                        "Jitter:DDP",
                        "MDVP:Jitter(%)",
                        "MDVP:RAP",
                        "MDVP:PPQ",
                        "MDVP:Shimmer",
                        "MDVP:Shimmer(dB)",
                        "Shimmer:APQ5",
                        "MDVP:APQ",
                        "NHR",
                    ],
                ] = transformed

                input_scaled = self.scaler.transform(input_data)
                input_scaled = np.hstack(
                    [
                        input_scaled,
                        self.pca_1.transform(input_scaled[:, [3, 4, 5, 6]]),
                        self.pca_2.transform(
                            input_scaled[:, [7, 8, 9, 10, 11, 12, 13, 14]]
                        ),
                        self.pca_3.transform(input_scaled[:, [18, 21]]),
                    ]
                )
                input_scaled = np.delete(
                    input_scaled,
                    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 21],
                    axis=1,
                )
                input_scaled = cudf.DataFrame(input_scaled).astype("float32")
                prediction = self.model.predict(input_scaled)

                if prediction[0] == 1:
                    st.error(f"⚠️ Positive: Parkinson's likely 🧠", icon="🧓")
                else:
                    st.success(
                        f"✅ Negative: No signs of Parkinson's detected 🧠", icon="🙂"
                    )
