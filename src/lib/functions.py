import pandas as pd
import numpy as np
import PIL as pil
from PIL import Image as img
import os
import shutil
from random import sample
import sys
import yaml

try:
    with open ("../config.yaml", 'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print('Error reading the config file')

sys.path.insert(0, os.path.abspath(config['lib']))

def image_checker_test(folder):
    src_path = config['data']['raw']+'Testing/'+folder+'/'
    dst_path = config['data']['clean']+folder+'/'
    counter = 0
    try:
        os.mkdir(dst_path)
    except:
        pass
    
    for filename in os.listdir(src_path):
        f = os.path.join(src_path, filename)
        im = img.open(f)
            #Perfect scenario: width = height = 512 pixels.
        if im.width == 512 and im.height == 512:
            try:
                shutil.copy(src_path+filename, dst_path+filename)
            except:
                pass
        #Less ideal scenario: both width and height equal but different to 512 pixels: to resize.
        else:
            try:
                im = im.resize((512, 512))
                im.save(dst_path+filename)
                counter += 1
            except:
                pass
    print(counter, 'images of', folder, 'resized')

def image_checker_train(folder):
    src_path = config['data']['raw']+'Training/'+folder+'/'
    dst_path = config['data']['clean']+folder+'/'
    counter = 0
    try:
        os.mkdir(dst_path)
    except:
        pass
    
    for filename in os.listdir(src_path):
        f = os.path.join(src_path, filename)
        im = img.open(f)
            #Perfect scenario: width = height = 512 pixels.
        if im.width == 512 and im.height == 512:
            try:
                shutil.copy(src_path+filename, dst_path+filename)
            except:
                pass
        #Less ideal scenario: both width and height equal but different to 512 pixels: to resize.
        else:
            try:
                im = im.resize((512, 512))
                im.save(dst_path+filename)
                counter += 1
            except:
                pass
    print(counter, 'images of', folder, 'resized')


def image_solver():
    #Raw data file count
    count_origin = 0
    original_path = config['data']['raw']
    for root_dir, cur_dir, files in os.walk(original_path):
        count_origin += len(files)
    print('Initial number of pictures:', count_origin)
    print('\nChecking and resizing pictures if needed, please wait\n')
    print('Working on the test set...')
    
    #Setting source
    src_test = config['data']['raw']+'Testing/'
    src_train = config['data']['raw']+'Training/'
    
    #Test set iterator
    tst_lst = [folder for folder in os.listdir(src_test)]
    [image_checker_test(i) for i in tst_lst]
    
    print('\nWorking on the train set...')
    #Train set iterator
    trn_lst = [folder for folder in os.listdir(src_train)]
    [image_checker_train(i) for i in trn_lst]
    
    print('\nImage rescaling finished. Pictures stored in data/clean')
    #Final data counter
    count_final = 0
    final_path = config['data']['clean']
    for root_dir, cur_dir, files in os.walk(final_path):
        count_final += len(files)
    
    #All tumor images to tumor folder
    source_1 = config['data']['clean']+'meningioma/'
    source_2 = config['data']['clean']+'pituitary/'
    source_3 = config['data']['clean']+'glioma/'
    destination_folder = config['data']['clean']+'tumor/'
    for file in os.listdir(source_1):
        source = source_1 + file
        destination = destination_folder + file
        shutil.move(source, destination_folder)

    for file in os.listdir(source_2):
        source = source_2 + file
        destination = destination_folder + file
        shutil.move(source, destination_folder)
        
    for file in os.listdir(source_3):
        source = source_3 + file
        destination = destination_folder + file
        shutil.move(source, destination_folder)
    os.rmdir(source_1)
    os.rmdir(source_2)
    os.rmdir(source_3)
    
    print('Number of pictures:', count_final)
    print('Pictures not processed:', count_origin-count_final)
    
    print('\n\nUpsampling the pictures by rotating...')
    
    src_t = config['data']['clean']+'tumor/'
    src_n = config['data']['clean']+'notumor/'
    src_n2 = config['data']['clean']+'notumor/rt/'
    src_n3 = config['data']['clean']+'notumor/rt2/'
    src_n4 = config['data']['clean']+'notumor/rt3/'
    os.mkdir(src_n2)
    os.mkdir(src_n3)
    os.mkdir(src_n4)
    
    for filename in os.listdir(src_t):
        f = os.path.join(src_t, filename)
        im = img.open(f)
        im = im.rotate(115)
        im.save(src_t+'rt115'+filename)

    for filename in os.listdir(src_n):
        try:
            f = os.path.join(src_n, filename)
            im = img.open(f)
            im = im.rotate(210)
            im.save(src_n+'rt210'+filename)
        except:
            pass
    
    for filename in os.listdir(src_n):
        try:
            f = os.path.join(src_n, filename)
            im = img.open(f)
            im = im.rotate(180)
            im.save(src_n2+'rt180'+filename)
        except:
            pass
        
    for filename in os.listdir(src_n):
        try:
            f = os.path.join(src_n, filename)
            im = img.open(f)
            im = im.rotate(15)
            im.save(src_n3+'rt15'+filename)
        except:
            pass

    for filename in os.listdir(src_n):
        try:
            f = os.path.join(src_n, filename)
            im = img.open(f)
            im = im.rotate(60)
            im.save(src_n4+'rt60'+filename)
        except:
            pass
    
    for file in os.listdir(src_n2):
        source = src_n2 + file
        destination = src_n + file
        shutil.move(source, destination)
    
    for file in os.listdir(src_n3):
        source = src_n3 + file
        destination = src_n + file
        shutil.move(source, destination)
    
    for file in os.listdir(src_n4):
        source = src_n4 + file
        destination = src_n + file
        shutil.move(source, destination)
    
    os.rmdir(src_n2)
    os.rmdir(src_n3)
    os.rmdir(src_n4)
    
    #Making sure both classes have the same number of pictures
    count_nt = 0
    count_t = 0
    nt_path = config['data']['clean']+'notumor/'
    t_path = config['data']['clean']+'tumor/'
    for root_dir, cur_dir, files in os.walk(nt_path):
        count_nt += len(files)
    for root_dir, cur_dir, files in os.walk(t_path):
        count_t += len(files)
    remove_cnt = abs(count_t-count_nt)
    
    if count_nt > count_t:
        remove_dir = config['data']['clean']+'notumor/'
        final_no = count_t
    else:
        remove_dir = config['data']['clean']+'tumor/'
        final_no = count_nt
    
    files = os.listdir(remove_dir)
    for file in sample(files,remove_cnt):
        os.remove(remove_dir+file)
    
    test_sample = int(final_no * 0.3)
    
    #Moving all pics to data/clean/train
    train_nt = config['data']['clean']+'train/notumor/'
    train_t = config['data']['clean']+'train/tumor/'
    
    
    for file in os.listdir(nt_path):
        source = nt_path + file
        destination = train_nt + file
        try:
            shutil.move(source, destination)
        except:
            pass
    
    for file in os.listdir(t_path):
        source = t_path + file
        destination = train_t + file
        try:
            shutil.move(source, destination)
        except:
            pass
    
    os.rmdir(nt_path)
    os.rmdir(t_path)
    
    #Sampling into test
    test_nt = config['data']['clean']+'test/notumor/'
    test_t = config['data']['clean']+'test/tumor/'
        
    files = os.listdir(train_nt)
    for file in sample(files,test_sample):
        source = train_nt + file
        destination = test_nt + file
        try:
            shutil.move(source, destination)
        except:
            pass
    
    files = os.listdir(train_t)
    for file in sample(files,test_sample):
        source = train_t + file
        destination = test_t + file
        try:
            shutil.move(source, destination)
        except:
            pass
    
    
    print('Upsampling done!\n')
    print(count_nt, 'pictures on the non-tumor class')
    print(count_t, 'pictures on the tumor class\n')
    print('Number of pictures of both classes adjusted to', final_no)