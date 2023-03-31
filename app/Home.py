import streamlit as st
st.set_page_config(page_title='Home - Brain Tumor Scanner', page_icon=':brain:', layout="centered", initial_sidebar_state="auto", menu_items=None)

st.title('')
st.image('../src/pics/samples/logo.png', width=650)
st.subheader('*A deep learning application for healthcare support*')
st.write('''##### *by Juan Jimenez*''')
st.write('This program uses 2 different machine learning models to check if on a MRI brain scan picture there is a tumor growth.')
st.write('''If none reaches a 99.90% confidence level or they identify different tumor types, it will email a brain specialist automatically.''')
st.write('Please select on the sidebar the **model** you want to use and if you want to **enable notifications**.')
st.header(''':red[Please ALWAYS check with a doctor]''')
st.title('')
st.write('Dataset [source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)')
