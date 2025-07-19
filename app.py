import streamlit as st
from models import diabetes_model, heart_model, parkinson_model
from utils.load_css import load_custom_css

st.set_page_config(page_title="\U0001f3e5 AI Disease Test Results", layout="wide")
load_custom_css()

tabs = st.tabs(["Diabetes Test", "Heart Disease Test", "Parkinson's Test"])

diabetes_model.render_ui(tabs[0])
heart_model.render_ui(tabs[1])
parkinson_model.render_ui(tabs[2])
