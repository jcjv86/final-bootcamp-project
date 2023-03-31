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
import email_config as email
from email.message import EmailMessage
st.set_page_config(page_title='Brain Tumor Scanner', page_icon=':brain:', layout="wide", initial_sidebar_state="auto", menu_items=None)
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
    st.image('../src/pics/samples/logo.png', width=300)
    st.subheader('*A deep learning application for healthcare support*')
    st.write('''##### *by Juan Jimenez*''')
    st.write('This program uses 2 different machine learning models to check if on a MRI brain scan picture there is a tumor growth.')
    st.write('''If none reaches a 99.90% confidence level or they identify different tumor types, it will email a brain specialist automatically.''')
    notification = st.checkbox('''Do you want to activate email notifications?''', value = True)
    st.header(''':red[Please ALWAYS check with a doctor]''')

    st.header('Additional resources')
    with st.expander('Tumor types detected by this model'):

        tab1, tab2, tab3 = st.tabs(['glioma', 'meningioma', 'pituitary'])

        with tab1:
            st.image('../src/pics/samples/glioma.jpg', width=200)
            st.write('A Glioma is a common type of tumor originating in the brain.')
            st.write('About 33 percent of all brain tumors and 80 percent of all malignant tumors are gliomas, which originate in the glial cells that surround and support neurons in the brain, including astrocytes,         oligodendrocytes and ependymal cells.')
            st.write('Gliomas are called intra-axial brain tumors because they grow within the substance of the brain and often mix with normal brain tissue.')
            st.write('Due to the location of these tumors (and since different structures can be involved), the treatment is complicated and the prognosis is not usually favorable, even for the Grade I and Grade II tumors.')
            st.write('More info: [Glioma at Cancer.gov](https://www.cancer.gov/rare-brain-spine-tumor/tumors/gliomatosis-cerebri)')
        with tab2:
            st.image('../src/pics/samples/meningioma.jpg', width=200)
            st.write('A Meningioma is the most common type of primary brain tumor, accounting for approximately 30 percent of all brain tumors.')
            st.write('Overall, meningiomas have the best prognosis, since they originate in the meninges, which are close to the cranium, so surgery is less risky')
            st.write('The relative 5-year survival rate for atypical and anaplastic meningioma is 63.8%')
            st.write('More info: [Meningioma at Cancer.gov](https://www.cancer.gov/rare-brain-spine-tumor/tumors/meningioma)')
        with tab3:
            st.image('../src/pics/samples/pituitary.jpg', width=200)
            st.write('Pituitary tumors originate in the pituitary gland.')
            st.write('This gland is of extreme importance to the human body, since it makes hormornes that regulate the release of other hormones produced by different endocrine system glands.')
            st.write('Since the space where this gland is located is very tight, any abnormal growth can, for example, press on the optic nerves which pass above it, causing blindness.')
            st.write('In addition to this, any alteration in the gland can lead to a increased or reduced hormone release rate, impacting the normal functioning of the body')
            st.write('More info: [Pituitary tumors at Cancer.org](https://www.cancer.org/cancer/pituitary-tumors/about/what-is-pituitary-tumor.html)')

    with st.expander('Model details'):
        st.write('Models were trained with a dataset containing pictures of a healthy brain (notumor) and 3 different tumor classes.')
        st.write('Dataset [source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)')
        st.write('Performance achieved by both models was really good, althought they had slight differences for each tumor type as we can see in the confusion matrices:')
        st.subheader('BTS model confusion matrix:')
        st.image('../src/pics/confusion_matrix_bts.png', width=270)
        st.subheader('TRL model confusion matrix:')
        st.image('../src/pics/confusion_matrix_trl.png', width=270)
        st.write('TRL model was better at diagnosing notumor class, with no false negatives. This is ideal in a critical diagnostic such as detecting a tumor.')
        st.write('''We want to avoid these false negatives at all costs, and although false positives are not ideal (informing the patient they have a tumor and later correcting the diagnostic), it is always better than the opossite (incorrectly informing a patient that they don't have a tumor to later acknowledge they did, when it may be already too late for treatment).''')
        st.write('''However, since the BTS model was better at detecting other tumor types (like meningioma), it was also convenient to take into account its predictions when the confidence levels didn't indicate certainty.''')
        st.write('Eventually, when a 99.90% confidence level is not reached, the program will email a brain specialist for direct review. This email client can be fully configured in the main app, future program versions will separate the program into blocks for an easier configuration.')
        st.write('Even if the model performance detecting tumors is optimal, sometimes it may confuse the tumor types. This happens especially when the perspective of the MRI scan is such that it is hard for it to understand where the tumor is located (if in the brain surface like a meningioma or depper inside like a pituitary tumor).')
        st.write('Also, lack or excessive contrast and/or brightness, picture resolution, etc... may affect performance. This is the main reason for setting a confidence critical level as high as a 99.90%.')

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
#Sidebar ends

#Program
#Model selector - radio button
model_load = st.radio(
    ':red[**Please select model to be used for the image prediction**]',
    ('BTS', 'TRL', 'Both'))


if model_load == 'BTS':
    st.write(':red[You selected BTS model.]')
    model = bts
elif model_load == 'TRL':
    st.write(':red[You selected TRL model.]')
    model = trl
else:
    st.write(':red[You selected both models. Program will run a double diagnostic]')
    model1 = bts
    model2 = trl

#Image loader and enhancer
image= st.file_uploader(':red[Please upload a .jpg file]', type='jpg')
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
