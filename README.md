## Final Bootcamp Project

![Alt text](src/pics/samples/logo.png?raw=true "Title")

### *A deep learning application for healthcare support*

A tumor growth in an enclosed environment such as the cranium can be very problematic due to the limited space available inside, which normally causes an increase in the internal cranial pressure. Whether they are beningn or malign, they can cause serious health issues.

The following project takes care of identifying if in a given brain MRI scan picture there is a tumor growth using 2 Deep Learning CNN models. If a model doesn't achieve a 99.90% confidence (threshold), runs a secondary diagnostic with the other model, and if this doesn't reach the threshold either or a different tumor type is identified, an specialist is contacted automatically via email.

User can also select both models at the same time, so the program will run a double diagnostic even if the first model reaches threshold. It will also email a specialist if neither of them reaches that threshold or tthey don't identify the same tumor type.


The models were trained using MRI brain scans of 3 different types of central nervous system tumors:
glioma, meningioma and pituitary.

Models training details can be found in the homonymous page.

##### [Database source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

New models can be generated via models notebook. You can run the data creation cell inside to download and extract the dataset and work from there if you didn't use the setup.py script.

Alternatively, if you just want to use the app with the already trained models, follow the steps below.


### Updates:
- Program is now a multipage app.
- Web browser tab now shows the active page and a brain icon. Default theme set in app/.streamlit/config.toml<br>
![Alt text](src/pics/samples/tab.png?raw=true "Title")
- Additional resources in different pages allow a clearer read.
- BTS model selection and enable email notifications actioned in sidebar.
- If the models don't identify the same tumor type, a brain specialist will be contacted, even if the confidence level is above threshold.
- Email setup can be easily done via the app/config/email_config.py script (no need to modify the main app).
- Email notifications feature the model used and the threshold of each. Also, they don't agree on the tumor type, the email displays the diagnosis of each model.
- Updated app pictures to show the new app at work.


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


#### Multipage Streamlit App demo

1- Home page
![Alt text](src/pics/samples/app_home.png?raw=true "Title")

2- BTS and TRL model successful
![Alt text](src/pics/samples/app_model1_ok.png?raw=true "Title")
![Alt text](src/pics/samples/app_model2_ok.png?raw=true "Title")

3- BTS and TRL model run a second diagnostic
![Alt text](src/pics/samples/app_model1_secondary.png?raw=true "Title")
![Alt text](src/pics/samples/app_model2_secondary.png?raw=true "Title")

4- BTS and TRL model don't reach the threshold: email to specialist
![Alt text](src/pics/samples/app_model1_email.png?raw=true "Title")
![Alt text](src/pics/samples/app_model2_email.png?raw=true "Title")

5- Both models selected, none reaches threshold, email to specialist
![Alt text](src/pics/samples/app_2models_email_same.png?raw=true "Title")

5- Different tumor types identified
![Alt text](src/pics/samples/app_2models_diff2.png?raw=true "Title")

6- Different tumor types identified, threshold also not reached
![Alt text](src/pics/samples/app_2models_diff.png?raw=true "Title")


#### Email notifications

1- Threshold not reached
![Alt text](src/pics/samples/email_threshold.png?raw=true "Title")

2- Different tumor types identified
![Alt text](src/pics/samples/email_diff.png?raw=true "Title")

3- Worst case scenario: threshold not reached, different tumor types identified
![Alt text](src/pics/samples/email_wcs.png?raw=true "Title")


## Please **ALWAYS** check with a doctor.


### More resources:

[Asociación Española Contra el Cáncer](https://www.contraelcancer.es/es)

[European Association for Cancer Research](https://www.eacr.org/)

[American Cancer Society](https://cancer.org)

[American Association for Cancer Research](https://www.aacr.org/)

[Brain Anatomy in detail](https://www.physio-pedia.com/Brain_Anatomy)
