import streamlit as st
from streamlit_pages.streamlit_pages import MultiPage
 
# import cv2
import numpy as np
import streamlit as st
from preprocess_augment import *
from model_predict import model_predict
import tensorflow as tf
from keras.models import load_model
import tensorflow.keras as keras



st.header("Speech Emotion Recognition")
def improved():
    st.write("Welcome")
    if st.button("Click Start"):
        st.write("Improved Algorithm")


def baseline():
    st.write("Welcome")
    if st.button("Click Start"):
        st.write("Baseline Algorithm")


def perf_comparison():
    st.write("Welcome")
    if st.button("Click Contact"):
        st.write("Performance Comparison")


# call app class object
app = MultiPage()
# Add pages
app.add_page("Improved Algorithm",improved)
app.add_page("Baseline Alogrithm",baseline)
app.add_page("Performance Comparison",perf_comparison)
app.run()








# load models
model = load_model("model.hdf5")

### load file
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