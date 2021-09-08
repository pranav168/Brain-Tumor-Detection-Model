from numpy.testing._private.utils import suppress_warnings
import streamlit as st


from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image as SImage
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tempfile



DEMO_IMAGE_YES = tf.keras.preprocessing.image.load_img(
    'Images\itsyes.jpg', grayscale=False, target_size=(64,64)
)
DEMO_IMAGE = 'Images\itsyes.jpg'
demo_image_hd = DEMO_IMAGE
image2 = np.array(Image.open(demo_image_hd))


classifier = load_model('BrainTumorModel.h5')
class_labels=['No','Yes']
@st.cache
def detect_tumor(image):
    # image=tf.keras.preprocessing.image.array_to_img(image)
    image = image.reshape((1,) + image.shape)
    labels=[]
    preds=classifier.predict(image)
    label=class_labels[preds.argmax()]
    return label
                                

st.title('Brain Tumor Detection Application')

st.markdown('''
            * This model is trained to Detect Brain tumor when images of MRI are provided\n
            * Just upload the Image of your MRI report and see the results immediately 
            ''')

img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])


if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    image2=image
    image = cv2.resize(image, (64,64))

else:
    demo_image = DEMO_IMAGE_YES
    image = np.array(demo_image)

st.subheader('Original Image')

st.image(image2, caption=f"Original Image",use_column_width= True)


tumor_detection = detect_tumor(image)
if tumor_detection=='Yes':
    st.subheader('Brain tumor is detected in this image, kindly contact a doctor as soon as possible')
else:
    st.subheader('No Brain tumor detected')
