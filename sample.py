import streamlit as st

st.title("Speech Emotion Recognition")

# import cv2
import numpy as np
import streamlit as st
from preprocess_augment import *
import tensorflow as tf
from keras.models import load_model
import tensorflow.keras as keras

# load models
model = load_model("model.hdf5")

### load file
file_audio = st.file_uploader("", type=['mp3','wav'])



if file_audio is not None:
    # preprocess the audio file
    sample, srate = librosa.load(file_audio)
    mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=n_fft, hop_length=hop_length, n_mels=256)
    mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)  #power_to_db = amplitude squared to decibel units
    mel_spect1 = cv2.resize(mel_spect, (256, 256))
    # train_x = X_train['MelSpectrogram'].values.tolist()
    input = np.asarray(mel_spect1, dtype=np.float64)

    input = np.array(input)
    
    st.write(input.shape)
    


CLASSIFY = st.button("Generate Prediction")    
if CLASSIFY:
    output = model.predict(input)
    st.write(output)