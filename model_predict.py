import streamlit as st
from preprocess_augment import *
from keras.models import load_model

def run_model(model):
    file_audio = st.file_uploader("", type=['mp3','wav'])
    if file_audio is not None:
        model_predict(file_audio)



def model_predict(audiofile):
  
  sample, srate = librosa.load(audiofile)
  plt.figure()
  librosa.display.waveplot(sample, srate)
  plt.title("Waveplot for audio")
  plt.xlabel("Time (seconds)")
  plt.ylabel("Amplitude")
  plt.show()  
  plt.savefig('waveplots.png',dpi = 70)
  st.image('waveplots.png', caption=' ')
    
  FIG_SIZE = (10,5)
  hop_length = 512 #stride
  n_fft = 2048 #num. of samples per window
  stft = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)
  
  plt.figure(figsize=FIG_SIZE)
  librosa.display.specshow(spectrogram, sr=srate, hop_length=hop_length)
  plt.xlabel("Time")
  plt.ylabel("Frequency")
  # plt.colorbar()
  plt.title("Spectrogram")
  plt.savefig('specs.png',transparent=True,dpi = 50)
  st.image('specs.png', caption=' ')
   
  # mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=n_fft, hop_length=hop_length, n_mels=256)
  # mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)
  # mel_spect1 = cv2.resize(mel_spect, (256, 256))
    
  # mel_spect1 = mel_spect1.astype("float")
  # mel_spect1 = np.array(mel_spect1)
  # input = np.expand_dims(mel_spect1,axis=0)
  # input1 = input[:,:,:,np.newaxis]


