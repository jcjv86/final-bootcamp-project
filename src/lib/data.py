import os
import sys
import shutil
import gdown


try:
    os.mkdir('../data')
except:
    print('Error - folders not created. Dataset or folders already present. Delete them and run script again.')

print('\nGetting dataset and extracting it, please wait...\n')
url = 'https://drive.google.com/uc?id=131ospwav2g6KKmG8q3iC1vf4mYk60y9O'
output = '../data/brain_tumor_dataset.zip'
unzip = '../data'
gdown.download(url, output, quiet=False)
shutil.unpack_archive(output, unzip)
os.remove(output)
print('Dataset successfully created!')
