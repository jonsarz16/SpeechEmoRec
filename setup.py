import streamlit as st
import numpy as np
import librosa, librosa.display
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from keras.models import load_model

st.set_page_config(
     page_title="SER Web App",
     layout="wide",
     initial_sidebar_state="collapsed"
 )
def data_visual_baseline(audiofile):
  sample, srate = librosa.load(audiofile)
  plt.figure()
  librosa.display.waveplot(sample, srate)

  plt.xlabel("Time (seconds)")
  plt.ylabel("Amplitude")
  plt.show()  
  plt.savefig('waveplots1.png',dpi = 70)
  st.title("Audio Waveplot")
  st.image('waveplots1.png', caption=' ')
    
  FIG_SIZE = (10,5)
  hop_length = 512 #stride
  n_fft = 2048 #num. of samples per window
  stft = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)
  spectrogram = np.abs(stft)
  plt.figure(figsize=FIG_SIZE)
  librosa.display.specshow(spectrogram, sr=srate, hop_length=hop_length)
  plt.xlabel("Time")
  plt.ylabel("Frequency")
  plt.title("Spectrogram")
  plt.savefig('specs.png',transparent=True,dpi = 60)
  st.title("Spectrogram")
  st.image('specs.png', caption=' ')


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
  mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=2048, hop_length=128, n_mels=256)
  mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)  #power_to_db = amplitude squared to decibel units
  plt.figure(figsize=FIG_SIZE)
  librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
  
  plt.savefig('melspecs.png',transparent=True,dpi = 80)
  st.title("Mel-Spectrogram")
  st.image('melspecs.png', caption=' ')
   

def pred_baseline(audiofile):
  sample, srate = librosa.load(audiofile)
  mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=2048, hop_length=128, n_mels=256)
  mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max) 
  mel_spect_resize = cv2.resize(mel_spect, (256, 256))
  mel_spect_resize = mel_spect_resize.astype("float")
  mel_spect_resize = np.array(mel_spect1)
  melspecs = np.expand_dims(mel_spect_resize,axis=0) 
  output = melspecs[:,:,:,np.newaxis]
  return output 
     
def pred_improved(audiofile):
  sample, srate = librosa.load(audiofile)
  mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=2048, hop_length=128, n_mels=256)
  mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max) 
  mel_spect_resize = cv2.resize(mel_spect, (256, 256))
  mel_spect_resize = mel_spect_resize.astype("float")
  mel_spect_resize = np.array(mel_spect1)
  melspecs = np.expand_dims(mel_spect_resize,axis=0) 
  output = melspecs[:,:,:,np.newaxis]
  return output




def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

