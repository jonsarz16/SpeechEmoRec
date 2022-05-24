import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu 
from model_predict import *
import tensorflow as tf
from keras.models import load_model
import tensorflow.keras as keras

st.header("Speech Emotion Recognition")
selected = option_menu(
    None,
    options=["Improved Algo", "Baseline", "Performance Comparison"],
    icons=["arrow-up-right-circle","arrow-repeat","terminal-split"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#a1b5d6", "display: flex"},
        "icon": {"color": "black", "font-size": "30px"}, 
        "nav-link": {"font-size": "17px", "text-align": "auto", "margin":"0px", "--hover-color": "#ffe100"},
        "nav-link-selected": {"background-color": "green"},}
)

if selected == "Baseline":
    st.write(f"You have selected {selected}")
    # load models
    model = load_model("model.hdf5")

    # load file
    file_audio = st.file_uploader("", type=['mp3','wav'])

    if file_audio is not None:
        CLASSIFY = st.button("Generate Prediction")
    # preprocess the audio file
        input = model_predict(file_audio)
    else:
        CLASSIFY = st.button("Generate Prediction on Test File")
    # preprocess the audio file
        audio_file = open('YAF_back_angry.wav', 'rb')
        input = model_predict(audio_file)
    
    
    
    if CLASSIFY:
        output = model.predict(input)
        st.write(output)