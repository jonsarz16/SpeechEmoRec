import streamlit as st

st.title("Speech Emotion Recognition")

# import cv2
import numpy as np
import streamlit as st
from preprocess_augment import *
import tensorflow as tf
from keras.models import load_model
import tensorflow.keras as keras
import h5py


# with h5py.File("model.hdf5", "r") as f:
#     # List all groups
#     print("Keys: %s" % f.keys())
#     a_group_key = list(f.keys())[0]

#     # Get the data
#     data = list(f[a_group_key])
#     st.write(data)

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
   
    mel_spect1 = mel_spect1.astype("float")
    mel_spect1 = np.array(mel_spect1)
    input = np.expand_dims(mel_spect1,axis=0) #single photo
    input1 = input[:,:,:,np.newaxis] #single photo

    st.write(input1.shape)
else:
    # preprocess the audio file
    st.title("Try test file")
    sample, srate = librosa.load("YAF_back_angry.wav")
    mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=n_fft, hop_length=hop_length, n_mels=256)
    mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)  #power_to_db = amplitude squared to decibel units
    mel_spect1 = cv2.resize(mel_spect, (256, 256))
   
    mel_spect1 = mel_spect1.astype("float")
    mel_spect1 = np.array(mel_spect1)
    input = np.expand_dims(mel_spect1,axis=0) #single photo
    input1 = input[:,:,:,np.newaxis] #single photo

    st.write(input1.shape)
    


CLASSIFY = st.button("Generate Prediction")    
if CLASSIFY:
    output = model.predict(input1)
    st.write(output)