import glob  
import matplotlib.pyplot as plt
import cv2

import os
import csv


from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import sklearn
import random

correction = 0.2

def process_img(img):
    ret = img[60:140,]
    ret = cv2.resize(ret, (64, 64))
    ret = ret / 255.0 - 0.5
    return ret

def gen_train_sample(img, angle):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        angle = -angle
    img = process_img(img)
    return img, angle


def generator(samples, batch_size=32):
    while 1: # Loop forever so the generator never terminates
        truncated_samples = []
        for line in samples:
            #angle = float(line[3])
            #if np.abs(angle) < 0.1 and random.random() > 0.01:
            #if np.abs(angle) < 0.1:
            #    continue
            truncated_samples.append(line)

        num_samples = len(truncated_samples)
        sklearn.utils.shuffle(truncated_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = truncated_samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                randind = random.randint(0, 2)
                #center
                if randind == 0:
                    name = batch_sample[0]
                    #name = datadir + './IMG/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    img, angle = gen_train_sample(center_image, center_angle)
                elif randind == 1:#left
                    name = batch_sample[1]
                    #name = datadir + './IMG/'+batch_sample[1].split('/')[-1]
                    center_angle = float(batch_sample[3])
                    left_img = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    img, angle = gen_train_sample(left_img, center_angle+correction)
                else:#right
                    name = batch_sample[2]
                    #name = datadir + './IMG/'+batch_sample[2].split('/')[-1]
                    right_img = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    img, angle = gen_train_sample(right_img, center_angle-correction)

                images.append(img)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            #yield sklearn.utils.shuffle(X_train, y_train, w_train)

# compile and train the model using the generator function



from keras.models import Sequential, Model
from keras.layers import Lambda
from keras.layers import *
import keras
from keras import optimizers
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

def build_network():
    model = Sequential()
    
    init_method = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    bias_init = keras.initializers.Constant(0.05)

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), 
        kernel_initializer=init_method, bias_initializer=bias_init))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation='relu',
        kernel_initializer=init_method, bias_initializer=bias_init))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), activation='relu', 
        kernel_initializer=init_method, bias_initializer=bias_init))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu',
        kernel_initializer=init_method, bias_initializer=bias_init))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(200, (3, 3), activation='relu', 
        kernel_initializer=init_method, bias_initializer=bias_init))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', 
        kernel_initializer=init_method, bias_initializer=bias_init))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu', 
        kernel_initializer=init_method, bias_initializer=bias_init))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu', 
        kernel_initializer=init_method, bias_initializer=bias_init))
    model.add(Dropout(0.5))

    model.add(Dense(1, kernel_initializer=init_method, bias_initializer=bias_init))

    return model


if __name__ == "__main__":
    """
    data1: data by myself with high speed
    data2: record data by udacity
    data3: record by myself
    data4: record from the side to middle of the road with high speed
    data5: record on track2 with high speed
    data6: record clockwise with high speed
    data7: track1 speed 9
    data8: track1 speed 9 move from side to middle
    data8: track1 speed 9 clockwise
    """
    dataset=[
            #"./dataset/data1/",
            #"./dataset/data2/",
            #"./dataset/data3/",
            #"./dataset/data4/",
            #"./dataset/data5/", 
            #"./dataset/data6/", 
            "./dataset/data7/", 
            "./dataset/data8/", 
            "./dataset/data9/", 
            ]

    samples = []
    for datadir in dataset:
        with open(datadir+'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                line[0] = datadir + './IMG/'+line[0].split('/')[-1]
                line[1] = datadir + './IMG/'+line[1].split('/')[-1]
                line[2] = datadir + './IMG/'+line[2].split('/')[-1]
                samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    model = build_network()
    
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=adam)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=True, mode='auto')

    filepath="model-improvement-{epoch:02d}-{val_loss:.5f}.h5"
    saver = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, 
            save_best_only=False, save_weights_only=False, mode='auto', period=1)

    model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
            validation_data=validation_generator, validation_steps=len(validation_samples),
            callbacks=[early_stop, saver], epochs=30, verbose = 1)
    
    model.save("./model.h5")
    
