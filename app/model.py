import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import yaml

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from IPython.core.display import Image
from PIL import Image
from matplotlib.pyplot import imshow


model = tf.keras.models.load_model('../models/model.model')

def conclusion(number):
    if number==2:
        return 'Tumor - Glioma'
    elif number ==0:
        return 'Tumor - Meningioma'
    elif number == 3:
        return 'Not a tumor'
    elif number == 1:
        return 'Tumor - pituitary'
    else:
        return 'Sorry, not clear'

img = Image.open(image)
x = np.array(img.resize((128,128)))
x = x.reshape(1,128,128,3)
res = model.predict_on_batch(x)
classification = np.where(res == np.amax(res))[1][0]
imshow(img)
print(conclusion(classification))
