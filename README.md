## Final Bootcamp Project

![Alt text](src/pics/samples/logo.png?raw=true "Title")

### *A deep learning application for healthcare support*

A tumor growth in an enclosed environment such as the cranium can be very problematic due to the limited space available inside, which normally causes an increase in the internal cranial pressure. Whether they are beningn or malign, they can cause serious health issues.

The following project takes care of identifying if in a given brain MRI scan picture there is a tumor growth using 2 Deep Learning CNN models. If a model doesn't achieve a 99.90% confidence, runs a secondary diagnostic with the other model, and if this doesn't arrive to a 99.90% confidence either, an specialist is contacted automatically via email.

User can also select both models at the same time, so the program will run a double diagnostic even if the first model achieves a 99.90% confidence level. It will also email a specialist if neither of them reaches that threshold.


The models were trained using MRI brain scans of the following types of central nervous system tumors:
glioma, meningioma and pituitary.

##### [Database source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

New models can be generated via models notebook. You can run the data creation cell inside to download and extract the dataset and work from there if you didn't use the setup.py script.

Alternatively, if you just want to use the app with the already trained models, follow the steps below.

## Installation guide:

1- Clone repo into desired folder: <br>
git clone https://github.com/jcjv86/final-bootcamp-project

2- Create venv environment inside the cloned repo folder: <br>
python3 -m venv ./venv

3- Activate environment: <br>
source ./venv/bin/activate

4- Run setup.py to install required libraries, configure venv user and download and unpack the dataset and models (around 1GB download in total): <br>
python setup.py

5- Move into app folder:<br>
cd app

6- Run app: <br>
streamlit run Home.py

## Instructions for the email setup:

Edit 'email_config.py' script on the app/config folder:<br>
- Set the email_sender (email from where the notification is sent)
- Set the email_password (recommended to configure an app password)
- Set the email_receiver (email that will receive the notification)

If you use a different server than gmail you would have to configure the smtp server too:<br>
- Edit the SMTP server address
- Edit the SMTP server port

It is not necessary to set up the email account as the program still works without it, in this case I recommend you to unclick the "Activate email notifications" option so you don't see the error.


#### App in action -work in progress, this section will be updated soon-


## Please **ALWAYS** check with a doctor.


### More resources:

[Asociación Española Contra el Cáncer](https://www.contraelcancer.es/es)

[European Association for Cancer Research](https://www.eacr.org/)

[American Cancer Society](https://cancer.org)

[American Association for Cancer Research](https://www.aacr.org/)

[Brain Anatomy in detail](https://www.physio-pedia.com/Brain_Anatomy)
