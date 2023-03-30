## Final Bootcamp Project

![Alt text](src/pics/samples/logo.png?raw=true "Title")

### *A Deep Learning application using Convolutional Neural Networks*

A tumor growth in an enclosed environment such as the cranium can be very problematic due to the limited space available inside, which normally causes an increase in the internal cranial pressure. Whether they are beningn or malign, they can cause serious health issues.

The following project takes care of identifying if in a given brain MRI scan picture there is a tumor growth using 2 Deep Learning CNN models. If a model doesn't achieve a 99.90% confidence, runs a secondary diagnostic with the other model, and if this doesn't arrive to a 99.90% confidence either, an specialist is contacted automatically via email.

User can also select both models at the same time, so the program will run a double diagnostic even if the first model achieves a 99.90% confidence level. It will also email a specialist if neither of them reaches that threshold.


The models were trained using MRI brain scans of the following types of central nervous system tumors:

- Meningioma: Overall, they are rarely malign. However, since they can grow slowly until they are very large they can be severely disabling and life-threatening. <br> Meningiomas origin on the meninges, the membranous layers surrounding the brain and spinal cord. Many cases never produce symptoms, although when they do, these include seizures, dementia, trouble talking, vision problems, one sided weakness, or loss of bladder control. When removed via surgery, less that a 20% recur.

- Glioma: Gliomas comprise about 30 percent of all brain tumors and central nervous system tumours, and 80 percent of all malignant brain tumours.<br>They originate in the glial cells that surround and support neurons in the brain, including astrocytes, oligodendrocytes and ependymal cells.  A brain glioma can cause headaches, vomiting, seizures, and cranial nerve disorders as a result of increased intracranial pressure. A glioma of the optic nerve can cause vision loss. Spinal cord gliomas can cause pain, weakness, or numbness in the extremities

- Pituitary: Pituitary tumors are unusual growths that develop in the pituitary gland and account from 10% to 25% of all intracranial neoplasms. <br>This gland is an organ about the size of a pea that is located behind the nose at the base of the brain. Some of these tumors cause the pituitary gland to make too much of certain hormones that control important body functions, while others can cause the gland to make too little. Most pituitary tumors are benign and can be treated with surgery, medications or radiation therapy.


I have used 2 different sequential models: one created from scratch called BTS and another created with transfer learning from model [VGG16](https://keras.io/api/applications/vgg/), called TRL.

They have a similar performance, but the TRL model works better in a critical diagnosis: no false negatives when identifying tumors (but possible false positives between the different tumor types).

Confusion matrix of the BTS model
![Alt text](src/pics/confusion_matrix_bts.png?raw=true "Title")

Confusion matrix of the TRL model
![Alt text](src/pics/confusion_matrix_trl.png?raw=true "Title")

New models can be generated via models notebook. You can run the data creation cell inside to download and extract the dataset and work from there.

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

5- Move into app folder:
cd app

6- Run app: <br>
streamlit run bts.py

## Instructions for the email setup:

Open and edit 'email_credentials.py' script on the app folder. <br>

- Set the email_sender (email from where the notification is sent)
- Set the email_password (recommended to configure an app password)
- Set the email_receiver (email that will receive the notification)

If you use a different server than gmail you would have to configure the smtp servers in the app script.<br>
--In development. Improvements to be included soon--

## App in action -work in progress, new theme set so this will be updated soon-

1- BTS model 100% confidence
![Alt text](src/pics/samples/app1.png?raw=true "Title")

2- TRL model 100% confidence
![Alt text](src/pics/samples/app2.png?raw=true "Title")

3- BTS model confidence low - secondary diagnostic run by TRL model
![Alt text](src/pics/samples/app3.png?raw=true "Title")

4- TRL model confidence low - secondary diagnostic run by BTS model
![Alt text](src/pics/samples/app4.png?raw=true "Title")

5.1- Secondary diagnostic failed to achieve a 99.90% confidence: Specialist contacted via email
![Alt text](src/pics/samples/app5.png?raw=true "Title")

5.2- Email notification sent by BTS program
![Alt text](src/pics/samples/app_mail.png?raw=true "Title")

Extra- Both models can run now even if one reaches 99.90% confidence level, and they will also email specialist if neither reaches it.
![Alt text](src/pics/samples/app6.png?raw=true "Title")
![Alt text](src/pics/samples/app7.png?raw=true "Title")

## Please **ALWAYS** check with a doctor.


### More resources:

[Asociación Española Contra el Cáncer](https://www.contraelcancer.es/es)

[European Association for Cancer Research](https://www.eacr.org/)

[American Cancer Society](https://cancer.org)

[American Association for Cancer Research](https://www.aacr.org/)

[Brain Anatomy in detail](https://www.physio-pedia.com/Brain_Anatomy)




##### [Database source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
