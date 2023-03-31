import streamlit as st
st.set_page_config(page_title='Brain tumor types - Brain Tumor Scanner', page_icon=':brain:', layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.title('These are the brain tumor types identified by this program.')
st.subheader('Please note that these are generic tumor classes, and each of them has different more specific subtypes.')
st.title('')

tab1, tab2, tab3 = st.tabs(['Glioma - Tumor in the glial cells', 'Meningioma - Tumor in the meninges', 'Pituitary - Tumor in the pituitary gland'])

with tab1:
    st.image('../src/pics/samples/glioma.jpg', width=400)
    st.write('A Glioma is a common type of tumor originating in the brain.')
    st.write('About 33 percent of all brain tumors and 80 percent of all malignant tumors are gliomas, which originate in the glial cells that surround and support neurons in the brain, including astrocytes, oligodendrocytes and ependymal cells.')
    st.write('Gliomas are called intra-axial brain tumors because they grow within the substance of the brain and often mix with normal brain tissue.')
    st.write('Due to the location of these tumors (and since different structures can be involved), the treatment is complicated and the prognosis is not usually favorable, even for the Grade I and Grade II tumors.')
    st.write('More info: [Glioma at Cancer.gov](https://www.cancer.gov/rare-brain-spine-tumor/tumors/gliomatosis-cerebri)')
with tab2:
    st.image('../src/pics/samples/meningioma.jpg', width=400)
    st.write('A Meningioma is the most common type of primary brain tumor, accounting for approximately 30 percent of all brain tumors.')
    st.write('Overall, meningiomas have the best prognosis, since they originate in the meninges, which are close to the cranium, so surgery is less risky')
    st.write('The relative 5-year survival rate for atypical and anaplastic meningioma is 63.8%')
    st.write('More info: [Meningioma at Cancer.gov](https://www.cancer.gov/rare-brain-spine-tumor/tumors/meningioma)')
with tab3:
    st.image('../src/pics/samples/pituitary.jpg', width=400)
    st.write('Pituitary tumors originate in the pituitary gland.')
    st.write('This gland is of extreme importance to the human body, since it makes hormornes that regulate the release of other hormones produced by different endocrine system glands.')
    st.write('Since the space where this gland is located is very tight, any abnormal growth can, for example, press on the optic nerves which pass above it, causing blindness.')
    st.write('In addition to this, any alteration in the gland can lead to a increased or reduced hormone release rate, impacting the normal functioning of the body.')
    st.write('')
    st.write('More info: [Pituitary tumors at Cancer.org](https://www.cancer.org/cancer/pituitary-tumors/about/what-is-pituitary-tumor.html)')
