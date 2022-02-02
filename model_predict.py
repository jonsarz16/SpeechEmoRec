


def model_predict(audiofile):
    sample, srate = librosa.load(audiofile)
    mel_spectrogram = librosa.feature.melspectrogram(sample, sr=srate, n_fft=n_fft, hop_length=hop_length, n_mels=256)
    mel_spect = librosa.power_to_db(mel_spectrogram, ref=np.max)  #power_to_db = amplitude squared to decibel units
    mel_spect1 = cv2.resize(mel_spect, (256, 256))
   
    mel_spect1 = mel_spect1.astype("float")
    mel_spect1 = np.array(mel_spect1)
    input = np.expand_dims(mel_spect1,axis=0) #single photo
    input1 = input[:,:,:,np.newaxis] #single photo
    return input1

