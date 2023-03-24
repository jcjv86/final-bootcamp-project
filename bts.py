import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import yaml
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import plot_model

bts = tf.keras.models.load_model('./models/bts.model')
trl = tf.keras.models.load_model('./models/trl.model')

st.header('Brain Tumor Scanner')

with st.sidebar:
    st.image('./src/pics/samples/logo.png', width=300)
    st.subheader('*A deep learning application*')
    st.write('''
        ###### *by Juan Jimenez*
This program will check if on a given jpg file of a brain MRI scan there is a tumor.

We trained our models with pictures of 3 different types of tumors and pictures of a healthy brain.

''')

    st.write('''This Deep Learning model is able to identify with a very high accuracy if there is a tumor or not.''')

    st.write('''Specific types of tumor however may be confused among each other,
                specially when they are located in limiting regions or the profile of the picture is confusing or the tumor has not enough contrast.''')
    st.header(''':red[Program developed for studying purposes, not to be used for any other reason!]''')

    st.header(''':red[Please ALWAYS check with a doctor]''')


    st.header('Additional resources')
    with st.expander('Tumor types used by this model.'):

        tab1, tab2, tab3 = st.tabs(['glioma', 'meningioma', 'pituitary'])

        with tab1:
            st.image('./src/pics/samples/glioma.jpg', width=200)
            st.write('A Glioma is a common type of tumor originating in the brain.')
            st.write('About 33 percent of all brain tumors and 80 percent of all malignant tumors are gliomas, which originate in the glial cells that surround and support neurons in the brain, including astrocytes,         oligodendrocytes and ependymal cells.')
            st.write('Gliomas are called intra-axial brain tumors because they grow within the substance of the brain and often mix with normal brain tissue.')
            st.write('Due to the location of these tumors (and since different structures can be involved), the treatment is complicated and the prognosis is not usually favorable, even for the Grade I and Grade II tumors.')
            st.write('More info: [Glioma at Cancer.gov](https://www.cancer.gov/rare-brain-spine-tumor/tumors/gliomatosis-cerebri)')
        with tab2:
            st.image('./src/pics/samples/meningioma.jpg', width=200)
            st.write('A Meningioma is the most common type of primary brain tumor, accounting for approximately 30 percent of all brain tumors.')
            st.write('Overall, meningiomas have the best prognosis, since they originate in the meninges, which are close to the cranium, so surgery is less risky')
            st.write('The relative 5-year survival rate for atypical and anaplastic meningioma is 63.8%')
            st.write('More info: [Meningioma at Cancer.gov](https://www.cancer.gov/rare-brain-spine-tumor/tumors/meningioma)')
        with tab3:
            st.image('./src/pics/samples/pituitary.jpg', width=200)
            st.write('Pituitary tumors originate in the pituitary gland.')
            st.write('This gland is of extreme importance to the human body, since it makes hormornes that regulate the release of other hormones produced by different endocrine system glands.')
            st.write('Since the space where this gland is located is very tight, any abnormal growth can, for example, press on the optic nerves which pass above it, causing blindness.')
            st.write('In addition to this, any alteration in the gland can lead to a increased or reduced hormone release rate, impacting the normal functioning of the body')
            st.write('More info: [Pituitary tumors at Cancer.org](https://www.cancer.org/cancer/pituitary-tumors/about/what-is-pituitary-tumor.html)')


model_load = st.radio(
    "Please select model to be used for the image prediction",
    ('BTS', 'TRL'))


if model_load == 'BTS':
    st.write('You selected BTS model.')
    model = bts
else:
    st.write('You selected TRL model.')
    model = trl


image = st.file_uploader('Please upload a .jpg file', type='jpg')
#import uploader as uploader

def conclusion(number):
    if number==0:
        return 'Tumor - Glioma'
    elif number ==3:
        return 'Tumor - Meningioma'
    elif number == 1:
        return 'Not a tumor'
    elif number == 2:
        return 'Tumor - pituitary'
    else:
        return 'Sorry, not clear'

if image:
    img = Image.open(image)
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    diagnose = (conclusion(classification))
    image = img.resize((500,500))
    st.image(image)
    st.header(diagnose)
