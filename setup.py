import streamlit as st
import numpy as np
import librosa, librosa.display
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.models import load_model


st.set_page_config(
     page_title="SER Web App",
     layout="wide",
     initial_sidebar_state="collapsed"
 )


# improved_model
# improved_model = load_model("improved.hdf5")



def data_visual_baseline(audiofile):
  sample, srate = librosa.load(audiofile)
  plt.figure()
  librosa.display.waveplot(sample, srate)
  plt.xlabel("Time (seconds)")
  plt.ylabel("Amplitude")
  plt.savefig('waveplots2.png',transparent=True,bbox_inches='tight', dpi=75)
  st.title("Audio Waveplot")
  st.image('waveplots2.png', caption=' ')
  FIG_SIZE = (10,5)
  mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=2048, hop_length=512, n_mels=256)
  mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)  #power_to_db = amplitude squared to decibel units
  plt.figure(figsize=FIG_SIZE)
  librosa.display.specshow(mel_spect,fmax=8000)
  
  plt.savefig('melspecs.png',transparent=True,bbox_inches='tight',pad_inches=0, dpi=256)
  plt.savefig('mels.png',transparent=True,bbox_inches='tight',pad_inches=0, dpi=60)
  
  
  st.title("Mel-Spectrogram")
  st.image('mels.png', caption=' ')
  
  
  
def data_visual_improved(audiofile):
  sample, srate = librosa.load(audiofile)
  plt.figure()
  librosa.display.waveplot(sample, srate)
  plt.xlabel("Time (seconds)")
  plt.ylabel("Amplitude")
  plt.savefig('waveplots2.png',dpi = 100)
  st.title("Audio Waveplot")
  st.image('waveplots2.png', caption=' ')
  FIG_SIZE = (10,5)
  mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=2048, hop_length=512, n_mels=256)
  mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)  #power_to_db = amplitude squared to decibel units
  plt.figure(figsize=FIG_SIZE)
  librosa.display.specshow(mel_spect,fmax=8000)
  
  plt.savefig('melspecs.png',transparent=True,bbox_inches='tight',pad_inches=0, dpi=256)
  st.title("Mel-Spectrogram")
  st.image('melspecs.png', caption=' ')
  




def classify(img_path):
    #baseline model
    model = load_model("baseline500e.h5")
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    score = tf.nn.softmax(prediction[0])
    result = score
    
    
#     score = tf.nn.softmax(prediction[0])
#     st.write(prediction)
#     st.write(score)
#     print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )

    return result


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

