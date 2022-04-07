import streamlit as st

st.title("Speech Emotion Recognition")
st.title("Tudtud - Badang - Sarigumba)
# import cv2
import numpy as np
import streamlit as st
from preprocess_augment import *
from model_predict import model_predict
import tensorflow as tf
from keras.models import load_model
import tensorflow.keras as keras

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