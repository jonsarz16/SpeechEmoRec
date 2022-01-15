import os
import numpy as np
import librosa, librosa.display
import pandas as pd
import IPython.display as ipd
from matplotlib import pyplot as plt

import math
import scipy
import random
from random import randrange, uniform

from tensorflow.python.framework import dtypes
import cv2
import tensorflow as tf
import tensorflow_io as tfio

# Data Directory
dir_list = os.listdir('TESS')
# dir_list = os.listdir('TESS')
dir_list.sort()

# Create DataFrame
speech_df = pd.DataFrame(columns=['file_path', 'actress', 'emotion'])
count = 0
for i in dir_list:
    file_list = os.listdir('TESS/' + i)
    # file_list = os.listdir('TESS/' + i)
    for f in file_list:
        nm = f.split('.')[0].split('-')
        file_path = 'TESS/' + i + '/' + f
        # file_path = 'TESS/' + i + '/' + f
        emotions = ["angry", "sad", "neutral", "happy", "fear", "disgust", "surprise"]

        if "OAF" in file_path:
            actress = "old"
        else:
            actress = "young"

        for x in emotions:   
            if x in file_path:
                emotion = x
            
        speech_df.loc[count] = [file_path, actress, emotion]
        count += 1


print (len(speech_df)) #number of audio files in the dataset
speech_df #print


grouped_data = speech_df.groupby("emotion").count()[['file_path']]
grouped_data.rename(columns={'file_path':'Files'})

for column in speech_df[['file_path']]:
   path = speech_df[column]
   file_name = path.values
   print(file_name)



def createWaveplot(sample, sr, e, a):
  plt.figure()
  librosa.display.waveplot(sample, sr)
  plt.title("Waveplot for audio with {} emotion ({} voice)".format(e,a))
  plt.xlabel("Time (seconds)")
  plt.ylabel("Amplitude")
  plt.show()

emotion = "fear"
actress = "old"
filepath = np.array(speech_df.file_path[np.logical_and(speech_df.emotion==emotion, speech_df.actress==actress)])[0] 
sample, srate = librosa.load(filepath) 
ipd.Audio(sample, rate=srate)


createWaveplot(sample,srate,emotion, actress)


emotion = "angry"
actress = "young"
filepath = np.array(speech_df.file_path[np.logical_and(speech_df.emotion==emotion, speech_df.actress==actress)])[122] 
sample, srate = librosa.load(filepath) 
ipd.Audio(sample, rate=srate)


createWaveplot(sample,srate,emotion,actress)


def time_mask(spec, no_of_mask, mask_size):
  i = 1

  while i <= no_of_mask:
    time_masked = tfio.audio.freq_mask(spec, param=mask_size)
    spec = time_masked
    mask_size = randrange(8, 12)
    i += 1
  return time_masked.numpy()

def freq_mask(spec, no_of_mask, mask_size):
  i = 1
  while i <= no_of_mask:
    freq_masked = tfio.audio.time_mask(spec, param=mask_size)
    spec = freq_masked #set the new modified array to the original variable
    mask_size = randrange(8, 12)
    i += 1
  return freq_masked.numpy()

def tempo(data, rate): #0.8
    """
    Streching the Sound.
    """
    data = librosa.effects.time_stretch(data, rate)
    return data
    
def pitch(data, sample_rate, pitch_pm):
    """
    Pitch Tuning.
    """
    bins_per_octave = 15
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
    return data

ipd.Audio(tempo(sample, 0.3), rate=srate)


# Audio Pre-processing/Visualization


# Convert audio signal to Spectrogram (STFT uses a sliding-frame FFT to produce a 2D matrix) 
#source: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53

FIG_SIZE = (10,5)
hop_length = 512 #stride
n_fft = 2048 #num. of samples per window


stft = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)
stft


# Calculate the Magnitude (absolute values on complex numbers)
spectrogram = np.abs(stft)


# Plot the Spectrogram (standard spectrogram)
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(spectrogram, sr=srate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("Spectrogram")





# Spectrogram to wav
def spsi(msgram, fftsize, hop_length) :
    """
    Takes a 2D spectrogram ([freqs,frames]), the fft legnth (= widnow length) and the hope size (both in units of samples).
    Returns an audio signal.
    """
    
    numBins, numFrames  = msgram.shape
    y_out=np.zeros(numFrames*hop_length+fftsize-hop_length)
        
    m_phase=np.zeros(numBins);      
    m_win=scipy.signal.hanning(fftsize, sym=True)  # assumption here that hann was used to create the frames of the spectrogram
    
    #processes one frame of audio at a time
    for i in range(numFrames) :
            m_mag=msgram[:, i] 
            for j in range(1,numBins-1) : 
                if(m_mag[j]>m_mag[j-1] and m_mag[j]>m_mag[j+1]) : #if j is a peak
                    alpha=m_mag[j-1]
                    beta=m_mag[j]
                    gamma=m_mag[j+1]
                    denom=alpha-2*beta+gamma
                    
                    if(denom!=0) :
                        p=0.5*(alpha-gamma)/denom
                    else :
                        p=0
                        
                    #phaseRate=2*math.pi*(j-1+p)/fftsize;    #adjusted phase rate
                    phaseRate=2*math.pi*(j+p)/fftsize;    #adjusted phase rate
                    m_phase[j]= m_phase[j] + hop_length*phaseRate; #phase accumulator for this peak bin
                    peakPhase=m_phase[j]
                    
                    # If actual peak is to the right of the bin freq
                    if (p>0) :
                        # First bin to right has pi shift
                        bin=j+1
                        m_phase[bin]=peakPhase+math.pi
                        
                        # Bins to left have shift of pi
                        bin=j-1
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until you reach the trough
                            m_phase[bin]=peakPhase+math.pi
                            bin=bin-1
                        
                        #Bins to the right (beyond the first) have 0 shift
                        bin=j+2
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase
                            bin=bin+1
                            
                    #if actual peak is to the left of the bin frequency
                    if(p<0) :
                        # First bin to left has pi shift
                        bin=j-1
                        m_phase[bin]=peakPhase+math.pi

                        # and bins to the right of me - here I am stuck in the middle with you
                        bin=j+1
                        while((bin<(numBins)) and (m_mag[bin]<m_mag[bin-1])) :
                            m_phase[bin]=peakPhase+math.pi
                            bin=bin+1
                        
                        # and further to the left have zero shift
                        bin=j-2
                        while((bin>1) and (m_mag[bin]<m_mag[bin+1])) : # until trough
                            m_phase[bin]=peakPhase
                            bin=bin-1
                            
                #end ops for peaks
            #end loop over fft bins with

            magphase=m_mag*np.exp(1j*m_phase)  #reconstruct with new phase (elementwise mult)
            magphase[0]=0; magphase[numBins-1] = 0 #remove dc and nyquist
            m_recon=np.concatenate([magphase,np.flip(np.conjugate(magphase[1:numBins-1]), 0)]) 
            
            #overlap and add
            m_recon=np.real(np.fft.ifft(m_recon))*m_win
            y_out[i*hop_length:i*hop_length+fftsize]+=m_recon
            
    return y_out



y_out = spsi(spectrogram, fftsize=n_fft, hop_length=hop_length)
ipd.Audio(data=y_out, rate=srate)



# The y-axis is converted to a log scale, and the color dimension is converted to decibels
spec = librosa.amplitude_to_db(spectrogram, ref=np.max)
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(spec, sr=srate, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')

# Create Mel Spectrogram by converting frequencies to mel-scale
def create_melspectrogram(sample, srate, actress, emotion, index):
  mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=2048, hop_length=128, n_mels=256)
  mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)  #power_to_db = amplitude squared to decibel units

  return mel_spect.shape
  # plt.figure(figsize=FIG_SIZE)
  # librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
  # plt.title('Mel Spectrogram');
  # plt.title('Mel Spectrogram '+ str(index) + ": " + lbl_actress + " || " + lbl_emotion)
  # plt.colorbar(format='%+2.0f dB');
  # plt.show()


# Conveting TESS to Mel Spec

#loop inside dataframe (per tuple)
for row in (speech_df[2:3].itertuples()): #row[0] = index number(0 to 2800) row[1] = path  row[2] = actress  row[3] = emotion
  lbl_emotion = row[3]
  lbl_actress = row[2]
  index = row[0]

  sample, srate = librosa.load(row[1])
  p = create_melspectrogram(sample, srate, lbl_actress, lbl_emotion, index)
  print(p)

  ####-------end of data visualization-----------



# import json
# from json import JSONEncoder

# #serialize the NumPy Array into JSON String meaning convert mel_spect 2D array to a json format
# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)


# #Create directory for....
# save_path = '/Mel Spectrogram/'
# # save_path = '/Mel Spectrogram/'
# # Check whether the specified path exists or not
# isExist = os.path.exists(save_path)

# if not isExist:
#   # Create a new directory if does not exist 
#   os.makedirs(save_path)





#image file name
def fn_label(emotion, actress):
  emotions = {
      "angry": "01",
      "disgust": "02",
      "fear": "03",
      "happy": "04",
      "neutral": "05",
      "sad": "06",
      "surprise": "07"
  }

  if actress == 'young':
    act = 'Y'
  else:
    act = 'O'

  return emotions[emotion]+"_"+act


# Training Data



