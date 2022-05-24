from preprocess_augment import *

def main(model):
    # load models
    model = load_model(model)

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

