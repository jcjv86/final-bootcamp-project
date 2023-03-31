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
import smtplib
import ssl
sys.path.append('./config')
import email_config as email
from email.message import EmailMessage

#Page config
st.set_page_config(
    page_title='Brain Tumor Scanner',
    page_icon=':brain:',
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None)

#Load prediction dictionary (class labels)
with open('../src/lib/pred_dict.pickle', 'rb') as handle:
    pred_dict = pickle.load(handle)

#Load and rename models
bts = tf.keras.models.load_model('../models/bts.model')
trl = tf.keras.models.load_model('../models/trl.model')
bts._name = 'BTS'
trl._name = 'TRL'

#Sidebar
with st.sidebar:
    #Activate email notifications
    notification = st.checkbox('Activate email notifications', value = True)
    st.subheader('')
    #Select model to load
    model_load = st.radio('**Select model**',('BTS', 'TRL', 'Both'))
    if model_load == 'BTS':
        st.write(':purple[You selected BTS model.]')
        model = bts
    elif model_load == 'TRL':
        st.write(':red[You selected TRL model.]')
        model = trl
    else:
        st.write(':red[You selected both models. Program will run a double diagnostic]')
        model1 = bts
        model2 = trl
    st.header(''':red[Please ALWAYS check with a doctor]''')
#Sidebar ends

#   ///// Program /////

#Image loader and enhancer
image= st.file_uploader(':red[Please upload a .jpg file to start scanning]', type='jpg')
def image_prep(image):
    img = Image.open(image)
    img = Image.fromarray(np.uint8(img))
    img = ImageEnhance.Brightness(img).enhance(1)
    img = ImageEnhance.Contrast(img).enhance(1)
    return img

#Models predicitions
if image and (model_load != 'Both'):
    pic_name = image.name
    #Image loader and predictions
    img = image_prep(image)
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)
    x = np.array(x)/255.0
    res = model.predict_on_batch(x)
    score = max(res[0])
    lbl = tf.nn.softmax(res[0])
    tumor = pred_dict[np.argmax(lbl)]
    if score >= 0.999:
        st.header('{}.\n Confidence level: {:.2f}%. Model used: {}'.format(tumor, (100 * score), model.name))
        image = img.resize((500,500))
        st.image(image)
    elif score < 0.999:
        if model == bts:
            st.header('{}.\n Confidence level: {:.4f}%. Model used: BTS'.format(tumor, (100 * score)))
            st.subheader(':red[Running secondary diagnostic with TRL model...]')
            res2 = trl.predict_on_batch(x)
            score2 = max(res2[0])
            lbl2 = tf.nn.softmax(res2[0])
            tumor2 = pred_dict[np.argmax(lbl2)]
            if ((score2 >= 0.999) and (tumor==tumor2)):
                st.subheader(':red[Diagnose confirmed]')
                st.subheader(tumor)
                st.subheader('New confidence level: {:.2f}%.'.format((100 * score2)))
                image = img.resize((300,300))
                st.image(image)
            elif tumor != tumor2:
                st.subheader('New confidence level: {:.3f}%. {}'.format((100 * score2), tumor2))
                st.subheader(':red[Secondary diagnostic identified a different tumor type. Diagnosis not confirmed.]')
                if notification:
                    st.subheader(':red[Email sent to a Brain Specialist for further review.]')
                image = img.resize((300,300))
                st.image(image)
                if notification:
                    email_sender = email.email_sender
                    email_password = email.email_password
                    email_receiver = email.email_receiver
                    subject = 'Brain MRI scan revision needed - different tumor types identified'
                    greet = 'Dear Brain Specialist,\n\nA revision is needed for MRI scan '
                    conf_lvl = ('\n{}, confidence level: {:.2f}% (BTS model) \n{}, confidence level: {:.3f}% (TRL model)'.format(tumor, (100 * score), tumor2, (100 * score2)))
                    goodbye = '\n\nKind regards,\nBTS'
                    body = greet+pic_name+conf_lvl+goodbye
                    em = EmailMessage()
                    em['From'] = email_sender
                    em['To'] = email_receiver
                    em['Subject'] = subject
                    em.set_content(body)
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(email.smtp_address, email.smtp_port, context=context) as smtp:
                        smtp.login(email_sender, email_password)
                        smtp.sendmail(email_sender, email_receiver, em.as_string())
            else:
                st.subheader('New confidence level: {:.3f}%.'.format((100 * score2)))
                st.subheader(tumor2)
                st.subheader(':red[Secondary diagnostic confidence level lower than 99.90%.]')
                if notification:
                    st.subheader(':red[Email sent to a Brain Specialist for further review.]')
                image = img.resize((300,300))
                st.image(image)
                if notification:
                    email_sender = email.email_sender
                    email_password = email.email_password
                    email_receiver = email.email_receiver
                    subject = 'Brain MRI scan revision needed'
                    greet = 'Dear Brain Specialist,\n\nA revision is needed for MRI scan '
                    conf_lvl = ('\n{}, confidence levels: {:.2f}% (BTS model) and {:.3f}% (TRL model)'.format(tumor, (100 * score), (100 * score2)))
                    goodbye = '\n\nKind regards,\nBTS'
                    body = greet+pic_name+conf_lvl+goodbye
                    em = EmailMessage()
                    em['From'] = email_sender
                    em['To'] = email_receiver
                    em['Subject'] = subject
                    em.set_content(body)
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(email.smtp_address, email.smtp_port, context=context) as smtp:
                        smtp.login(email_sender, email_password)
                        smtp.sendmail(email_sender, email_receiver, em.as_string())
        if model == trl:
            st.header('{}.\n Confidence level: {:.4f}%. Model used: TRL'.format(tumor, (100 * score)))
            st.subheader(':red[Running secondary diagnostic with BTS model...]')
            res2 = bts.predict_on_batch(x)
            score2 = max(res2[0])
            lbl2 = tf.nn.softmax(res2[0])
            tumor2 = pred_dict[np.argmax(lbl2)]
            if ((score2 >= 0.999) and (tumor==tumor2)):
                st.subheader(':red[Diagnose confirmed]')
                st.subheader(tumor2)
                st.subheader('New confidence level: {:.2f}%.'.format((100 * score2)))
                image = img.resize((300,300))
                st.image(image)
            elif tumor != tumor2:
                st.subheader('New confidence level: {:.3f}%. {}'.format((100 * score2), tumor2))
                st.subheader(':red[Secondary diagnostic identified a different tumor type. Diagnosis not confirmed.]')
                if notification:
                    st.subheader(':red[Email sent to a Brain Specialist for further review.]')
                image = img.resize((300,300))
                st.image(image)
                if notification:
                    email_sender = email.email_sender
                    email_password = email.email_password
                    email_receiver = email.email_receiver
                    subject = 'Brain MRI scan revision needed - different tumor types identified'
                    greet = 'Dear Brain Specialist,\n\nA revision is needed for MRI scan '
                    conf_lvl = ('\n{}, confidence level: {:.2f}% (BTS model)\n{}, confidence level: {:.3f}% (TRL model)'.format(tumor, (100 * score), tumor2, (100 * score2)))
                    goodbye = '\n\nKind regards,\nBTS'
                    body = greet+pic_name+conf_lvl+goodbye
                    em = EmailMessage()
                    em['From'] = email_sender
                    em['To'] = email_receiver
                    em['Subject'] = subject
                    em.set_content(body)
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(email.smtp_address, email.smtp_port, context=context) as smtp:
                        smtp.login(email_sender, email_password)
                        smtp.sendmail(email_sender, email_receiver, em.as_string())
            else:
                st.subheader('New confidence level: {:.3f}%.'.format((100 * score2)))
                st.subheader(pred_dict[np.argmax(lbl2)])
                st.subheader(':red[Secondary diagnostic confidence level lower than 99.90%.]')
                if notification:
                    st.subheader(':red[Email sent to a Brain Specialist for further review.]')
                image = img.resize((300,300))
                st.image(image)
                if notification:
                    email_sender = email.email_sender
                    email_password = email.email_password
                    email_receiver = email.email_receiver
                    subject = 'Brain MRI scan revision needed'
                    greet = 'Dear Brain Specialist,\n\nA revision is needed for MRI scan '
                    conf_lvl = ('\n{}, confidence level: {:.2f}% (BTS model), confidence level: {:.3f}% (TRL model)'.format(tumor, (100 * score), (100 * score2)))
                    goodbye = '\n\nKind regards,\nBTS'
                    body = greet+pic_name+conf_lvl+goodbye
                    em = EmailMessage()
                    em['From'] = email_sender
                    em['To'] = email_receiver
                    em['Subject'] = subject
                    em.set_content(body)
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(email.smtp_address, email.smtp_port, context=context) as smtp:
                        smtp.login(email_sender, email_password)
                        smtp.sendmail(email_sender, email_receiver, em.as_string())

elif image and model_load == 'Both':
    pic_name = image.name
    #Image loader and predictions
    img = image_prep(image)
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)
    x = np.array(x)/255.0
    res1 = model1.predict_on_batch(x)
    score1 = max(res1[0])
    lbl1 = tf.nn.softmax(res1[0])
    tumor = pred_dict[np.argmax(lbl1)]
    st.header('{}.\n Confidence level: {:.2f}%. Model used: {}'.format(tumor, (100 * score1), model1.name))
    res2 = model2.predict_on_batch(x)
    score2 = max(res2[0])
    lbl2 = tf.nn.softmax(res2[0])
    tumor2 = pred_dict[np.argmax(lbl2)]
    st.header('{}.\n Confidence level: {:.2f}%. Model used: {}'.format(tumor2, (100 * score2), model2.name))
    if ((score1 < 0.9990) and (score2 < 0.9990)):
        st.subheader(':red[Neither of the models achieved a 99.90% confidence level]')
        if tumor!=tumor2:
            st.subheader(':red[Also, different tumor types were identified.]')
        if notification:
            st.subheader(':red[Email sent to a Brain Specialist for further review.]')
        image = img.resize((300,300))
        st.image(image)
        if notification:
            email_sender = email.email_sender
            email_password = email.email_password
            email_receiver = email.email_receiver
            subject = 'Brain MRI scan revision needed'
            greet = 'Dear Brain Specialist,\n\nA revision is needed for MRI scan '
            conf_lvl = ('\n{}, confidence level: {:.2f}% (BTS model)\n{}, confidence level: {:.3f}% (TRL model)'.format(tumor, (100 * score1), tumor2, (100 * score2)))
            goodbye = '\n\nKind regards,\nBTS'
            body = greet+pic_name+conf_lvl+goodbye
            em = EmailMessage()
            em['From'] = email_sender
            em['To'] = email_receiver
            em['Subject'] = subject
            em.set_content(body)
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(email.smtp_address, email.smtp_port, context=context) as smtp:
                smtp.login(email_sender, email_password)
                smtp.sendmail(email_sender, email_receiver, em.as_string())
    elif tumor != tumor2:
                st.subheader(':red[Secondary diagnostic identified a different tumor type. Diagnosis not confirmed.]')
                if notification:
                    st.subheader(':red[Email sent to a Brain Specialist for further review.]')
                image = img.resize((300,300))
                st.image(image)
                if notification:
                    email_sender = email.email_sender
                    email_password = email.email_password
                    email_receiver = email.email_receiver
                    subject = 'Brain MRI scan revision needed - different tumor types identified'
                    greet = 'Dear Brain Specialist,\n\nA revision is needed for MRI scan '
                    conf_lvl = ('\n{}, confidence level: {:.2f}% (BTS model)\n{}, confidence level: {:.3f}% (TRL model)'.format(tumor, (100 * score1), tumor2, (100 * score2)))
                    goodbye = '\n\nKind regards,\nBTS'
                    body = greet+pic_name+conf_lvl+goodbye
                    em = EmailMessage()
                    em['From'] = email_sender
                    em['To'] = email_receiver
                    em['Subject'] = subject
                    em.set_content(body)
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(email.smtp_address, email.smtp_port, context=context) as smtp:
                        smtp.login(email_sender, email_password)
                        smtp.sendmail(email_sender, email_receiver, em.as_string())
    else:
        image = img.resize((350,350))
        st.image(image)
