import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu 
from model_predict import *
import tensorflow as tf
from keras.models import load_model
import tensorflow.keras as keras
import librosa
st.set_page_config(layout="wide")
# st.header("Speech Emotion Recognition")

selected = option_menu(
    None,
    options=["Improved Algo", "Baseline", "Performance Comparison"],
    icons=["arrow-up-right-circle","arrow-repeat","window-stack"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#a1b5d6", "display": "inline"},
        "icon": {"color": "black", "font-size": "22px"}, 
        "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#ffe100"},
        "nav-link-selected": {"background-color": "green"},}
)


if selected == "Improved Algo":
    done = False
    col1, col2 = st.columns(2)
    with col1:
        model = load_model("model.hdf5")
        done = True
        run_model(model)
        
    with col2:
        if done == True:
            predict = st.button("Predict")
        
        if predict:
            st.title("Prediction Results")