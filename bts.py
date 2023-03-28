import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import yaml
import pickle
import random
from PIL import Image, ImageEnhance
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

with open('./src/lib/pred_dict.pickle', 'rb') as handle:
    pred_dict = pickle.load(handle)

bts = tf.keras.models.load_model('./models/bts.model')
trl = tf.keras.models.load_model('./models/trl.model')

bts._name = 'BTS'
trl._name = 'TRL'

with st.sidebar:
    st.image('./src/pics/samples/logo.png', width=300)
    st.subheader('*A deep learning application for healthcare support*')
    st.write('''##### *by Juan Jimenez*''')
    st.write('This program uses 2 different machine learning models to check if on a MRI brain scan picture there is a tumor growth.')
    st.write('''If the model doesn't arrive to a 99.90% confidence level it will recommend to see a doctor.''')
    st.write('Future implementations could flag the result and send it automatically to a doctor for further review.')
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

    with st.expander('Cancer Associations'):
        st.write('[Asociación Española Contra el Cáncer](https://www.contraelcancer.es/es)')
        st.write('[European Association for Cancer Research](https://www.eacr.org/)')
        st.write('[American Cancer Society](https://cancer.org)')
        st.write('[American Association for Cancer Research](https://www.aacr.org/)')
        st.write('')

    with st.expander('Learning resources'):
        st.write('[Keras](https://keras.io/)')
        st.write('[Google ML Education](https://developers.google.com/machine-learning)')
        st.write('[Brain Anatomy in detail](https://www.physio-pedia.com/Brain_Anatomy)')






model_load = st.radio(
    ":red[**Please select model to be used for the image prediction**]",
    ('BTS', 'TRL'))


if model_load == 'BTS':
    st.write(':red[You selected BTS model.]')
    model = bts
else:
    st.write(':red[You selected TRL model.]')
    model = trl

image = st.file_uploader(':red[Please upload a .jpg file]', type='jpg')

def image_prep(image):
    img = Image.open(image)
    img = Image.fromarray(np.uint8(img))
    img = ImageEnhance.Brightness(img).enhance(1)
    img = ImageEnhance.Contrast(img).enhance(1)
    return img

if image:
    img = image_prep(image)
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)
    x = np.array(x)/255.0
    res = model.predict_on_batch(x)
    score = max(res[0])
    lbl = tf.nn.softmax(res[0])
    if score >= 0.999:
        st.header('{}.\n Confidence level: {:.2f}%. Model used: {}'.format(pred_dict[np.argmax(lbl)], (100 * score), model.name))
        image = img.resize((500,500))
        st.image(image)
    elif score < 0.999:
        if model == bts:
            st.header('{}.\n Confidence level: {:.4f}%. Model used: BTS'.format(pred_dict[np.argmax(lbl)], (100 * score)))
            st.subheader(':red[Running secondary diagnostic with TRL model...]')
            res2 = trl.predict_on_batch(x)
            score2 = max(res2[0])
            lbl2 = tf.nn.softmax(res2[0])
            if score2 >= 0.999:
                st.subheader(':red[Diagnostic confirmed.]')
                st.subheader(pred_dict[np.argmax(lbl2)])
                st.subheader('New confidence level: {:.2f}%.'.format((100 * score2)))
                image = img.resize((300,300))
                st.image(image)
            else:
                st.subheader('New confidence level: {:.3f}%.'.format((100 * score2)))
                st.subheader(pred_dict[np.argmax(lbl2)])
                st.subheader(':red[You may want to consult a doctor as the secondary diagnostic has not reached a 99.90% confidence level.]')
                image = img.resize((300,300))
                st.image(image)
        if model == trl:
            st.header('{}.\n Confidence level: {:.4f}%. Model used: TRL'.format(pred_dict[np.argmax(lbl)], (100 * score)))
            st.subheader(':red[Running secondary diagnostic with BTS model...]')
            res2 = bts.predict_on_batch(x)
            score2 = max(res2[0])
            lbl2 = tf.nn.softmax(res2[0])
            if score2 >= 0.999:
                st.subheader(':red[Diagnostic confirmed.]')
                st.subheader(pred_dict[np.argmax(lbl2)])
                st.subheader('New confidence level: {:.2f}%.'.format((100 * score2)))
                image = img.resize((300,300))
                st.image(image)
            else:
                st.subheader('New confidence level: {:.3f}%.'.format((100 * score2)))
                st.subheader(pred_dict[np.argmax(lbl2)])
                st.subheader(':red[You may want to consult a doctor as the secondary diagnostic has not reached a 99.90% confidence level.]')
                image = img.resize((300,300))
                st.image(image)
