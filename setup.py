import os
import sys
import shutil
import subprocess

#Install required libraries
print('Installing required libraries, please wait')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jupyter-lab'])
print('Jupyter Lab installed')

#Configure user in venv
subprocess.check_call([sys.executable, '-m', 'ipykernel', 'install', '--user', '--name=venv'])
print('venv user configured')

#Import gdown to download data and models
import gdown

#Download and unpack dataset
try:
    os.mkdir('./data')
except:
    print('Error - folders not created. Dataset or folders already present. Delete them and run script again.')

print('\nGetting dataset and extracting it, please wait...\n')
url = 'https://drive.google.com/uc?id=131ospwav2g6KKmG8q3iC1vf4mYk60y9O'
output = './data/brain_tumor_dataset.zip'
unzip = './data'
gdown.download(url, output, quiet=False)
shutil.unpack_archive(output, unzip)
os.remove(output)
print('Dataset successfully created!')

#Download and unpack models
try:
    os.mkdir('./models')
except:
    print('Error - folders not created. Dataset or folders already present. Delete them and run script again.')

print('\n\nGetting models and extracting them, please wait...\n')
url = 'https://drive.google.com/uc?id=13Q8_TId7jObbt3LtZeQMKQgvnaBvZ6Hg'
output = './models/bts_model.zip'
unzip = './models'
gdown.download(url, output, quiet=False)
shutil.unpack_archive(output, unzip)
os.remove(output)
print('BTS model unpacked successfully')

url = 'https://drive.google.com/uc?id=1b9yUJ-x-d4y82Y1jCQ5vcN9usd5xRGEc'
output = './models/trl_model.zip'
unzip = './models'
gdown.download(url, output, quiet=False)
shutil.unpack_archive(output, unzip)
os.remove(output)
print('TRL model unpacked successfully')
