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

def predict(audiofile):
    sample, srate = librosa.load(audiofile)
    mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=n_fft, hop_length=hop_length, n_mels=256)
    mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)  #power_to_db = amplitude squared to decibel units
    mel_spect1 = cv2.resize(mel_spect, (256, 256))
   
    mel_spect1 = mel_spect1.astype("float")
    mel_spect1 = np.array(mel_spect1)
    input = np.expand_dims(mel_spect1,axis=0) #single photo
    input1 = input[:,:,:,np.newaxis] #single photo
    return input1
# pred_class = []

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
    CLASSIFY = st.button("Generate Prediction")
  
    input = predict(file_audio)
else:
    # preprocess the audio file
    audio_file = open('YAF_back_angry.wav', 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/wav')
    input = predict(audio_file)
    


   
if CLASSIFY:
    output = model.predict(input)
    st.write(output)