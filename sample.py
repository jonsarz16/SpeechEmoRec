import streamlit as st

st.title("Speech Emotion Recognition")

# import cv2
import numpy as np
import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

# # model = tf.keras.models.load_model("saved_model/mdl_wts.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")



# if uploaded_file is not None:
#     # Convert the file to an opencv image.
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     opencv_image = cv2.imdecode(file_bytes, 1)
#     opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
#     resized = cv2.resize(opencv_image,(224,224))
#     # Now do something with the image! For example, let's display it:
#     st.image(opencv_image, channels="RGB")

#     resized = mobilenet_v2_preprocess_input(resized)
#     img_reshape = resized[np.newaxis,...]
CLASSIFY = st.button("Generate Prediction")    
if CLASSIFY:
#         prediction = model.predict(img_reshape).argmax()
    st.title("the result is")
