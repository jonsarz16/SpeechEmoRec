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

def run_model(model):
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
col1, col2 = st.columns(2)
if selected == "Baseline":
    col1.header("Audio waveplot and Spectrogram")
    # col1.image
    col2.header("Prediction Results")

    st.write(f"You have selected {selected}")
    model = load_model("model.hdf5")
    run_model(model)

if selected == "Improved Algo":
    col1.header("Audio waveplot and Spectrogram")
    # col1.image
    col2.header("Prediction Results")
    st.write(f"You have selected {selected}")
    model = load_model("model.hdf5")
    run_model(model)

if selected == "Performance Comparison":
    st.write("Analysis")
    col1.header("DSCNN Baseline")
    # col1.image
    col2.header("DSCNN Improved")

