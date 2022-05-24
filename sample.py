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
def createWaveplot(sample, sr, e, a):
  plt.figure()
  librosa.display.waveplot(sample, sr)
  plt.title("Waveplot for audio with {} emotion ({} voice)".format(e,a))
  plt.xlabel("Time (seconds)")
  plt.ylabel("Amplitude")
  plt.show()


def run_model(model):
    col1, col2 = st.columns(2)
    with col1:
        file_audio = st.file_uploader("", type=['mp3','wav'])
        sample, srate = librosa.load(file_audio) 
        createWaveplot(sample,srate,emotion, actress)
        if file_audio is not None:
            CLASSIFY = st.button("Generate Prediction")
            input = model_predict(file_audio)
        else:
            CLASSIFY = st.button("Generate Prediction on Test File")
            audio_file = open('YAF_back_angry.wav', 'rb')
            input = model_predict(audio_file)
    with col2:    
        if CLASSIFY:
            output = model.predict(input)
            st.write("Prediction Analysis.....")

if selected == "Baseline":
    model = load_model("model.hdf5")
    run_model(model)

if selected == "Improved Algo":
    model = load_model("model.hdf5")
    run_model(model)




if selected == "Performance Comparison":
    col1, col2 = st.columns(2)
    with col1:
        st.header("DSCNN Baseline")
    with col2:
        st.header("DSCNN Improved")