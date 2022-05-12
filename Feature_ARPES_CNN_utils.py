import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import datasets, layers, models,activations
from tensorflow.keras.layers import BatchNormalization, Conv2D, PReLU, Dropout
from keras.initializers import Constant
import cv2 as cv

'''This file contains functions used by Feature_ARPES_CNN_main.py'''


def load_data_ARPES_exp(load_path, file_names, pixels):
    '''This loads experimental ARPES spectra. It interpolate the spectra to match the number of pixels the 
    model has been trained with'''
    #Load the simulated data for segmentation
    print ("loading data...")
    print (file_names)
    #number of images should be the same for X and Y
    num_im = len(file_names)
    #Then I load the other images in the array train_X
    images = np.zeros((num_im, pixels, pixels,1))
    r = list(range(num_im))
    #random.shuffle(r)
    label = 0
    for i in r:
        #first I read the header to know what type of defect it is to make the Y_vector
        file_name = load_path+file_names[i]
        temp = np.loadtxt(file_name, skiprows=1)
        inter = cv.resize(temp, dsize=(pixels,pixels), interpolation=cv.INTER_CUBIC)
        inter = normalize(inter,1)
        inter = np.rot90(inter, k=3)
        images[label,:,:,0] = inter
        label += 1
    return images

def load_data_ARPES(load_path, file_names, pixels):
    '''This load the simulated ARPES spectra and their labels (the bare bands)'''
    print ("loading data...")
    #number of images should be the same for X and Y
    num_im = len(file_names)
    #Then I load the other images in the array train_X
    images = np.zeros((num_im, pixels, pixels,1))
    r = list(range(num_im))
    #random.shuffle(r)
    label = 0
    for i in r:
        #first I read the header to know what type of defect it is to make the Y_vector
        file_name = load_path+file_names[i]
        f = open(file_name)
        #header = f.readline()
        #label = int(header.split('=')[-1])
        #label = int(re.findall(r'\d+', file_name)[0])
        label_str_regex = re.compile(r'label_(\d+)')
        label_temp = label_str_regex.search(file_name).group()
        label = int(re.findall(r'\d+', label_temp)[0])
        images[label,:,:,0] = np.loadtxt(file_name, max_rows = pixels)
    return images

def normalize(array, new_max):
    '''normalizes an array'''
    max_ = np.max(array)
    min_ = np.min(array)
    array = new_max*((array - min_)/(max_ - min_))
    return array

def scale_(STM_image):
    '''same, minus the average'''
    max_ = np.max(STM_image)
    min_ = np.min(STM_image)
    scaled_image = ((STM_image - min_)/(max_ - min_))
    avg = np.average(scaled_image)
    scaled_image -= avg
    return scaled_image


def get_uncompiled_model_filter(model_number, dropout_rate, pixels, num_filters, kernel_size):
    '''Here are the models
    There is a more elegant way to do that.
    Model 3, 4, and 5 are UNet types with repectively 3, 4, 5 down/up blocks
    models 6, 7, and 8 are with skip connections. They were working less well then 3 and 4.
    '''
    if model_number == 1:
        model = models.Sequential()
        #put some layers
        #model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size = kernel_size, input_shape=(pixels, pixels, 1), padding='same'))
        model.add(layers.Activation(activations.relu))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(layers.Activation(activations.relu))
        model.add(BatchNormalization())
        model.add(Conv2D(1, (3, 3), padding='same', activation='sigmoid'))
    if model_number == 2:
        model = models.Sequential()
        #put some layers
        model.add(Conv2D(num_filters, (3, 3), input_shape=(pixels, pixels, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(num_filters, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(1, (3, 3), padding='same', activation='tanh'))
    if model_number == 3:
        model = models.Sequential()
        #put some layers
        #block1 down
        model.add(Conv2D(num_filters, kernel_size = kernel_size, input_shape=(pixels, pixels, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        #block2 down
        model.add(Conv2D(2*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        #block3 down
        model.add(Conv2D(4*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        #block1 up
        model.add(Conv2D(4*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #block2 up
        model.add(Conv2D(2*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #block3 up
        model.add(Conv2D(1*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #final
        model.add(Conv2D(1, kernel_size = kernel_size, padding='same'))
        model.add(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))

    if model_number == 4:
        model = models.Sequential()
        #put some layers
        #block1 down
        model.add(Conv2D(num_filters, kernel_size = kernel_size, input_shape=(pixels, pixels, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        #block2 down
        model.add(Conv2D(2*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        #block3 down
        model.add(Conv2D(4*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        # #block4 down
        model.add(Conv2D(8*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        # #block1 up
        model.add(Conv2D(8*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #block2 up
        model.add(Conv2D(4*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #block3 up
        model.add(Conv2D(2*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #block4 up
        model.add(Conv2D(1*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #final
        model.add(Conv2D(1, kernel_size = kernel_size, padding='same'))
        model.add(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))

    if model_number == 5:
        model = models.Sequential()
        #put some layers
        #block1 down
        model.add(Conv2D(num_filters, kernel_size = kernel_size, input_shape=(pixels, pixels, 1), padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        #model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        #block2 down
        model.add(Conv2D(2*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        #block3 down
        model.add(Conv2D(4*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        # #block4 down
        model.add(Conv2D(8*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        # #block5 down
        model.add(Conv2D(16*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        # #block0 up
        model.add(Conv2D(16*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        # #block1 up
        model.add(Conv2D(8*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #block2 up
        model.add(Conv2D(4*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #block3 up
        model.add(Conv2D(2*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #block4 up
        model.add(Conv2D(1*num_filters, kernel_size = kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(layers.Activation(activations.relu))
        model.add(layers.UpSampling2D())
        #final
        model.add(Conv2D(1, kernel_size = kernel_size, padding='same'))
        model.add(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))

    if model_number == 6:#one block down and up
        inputs = layers.Input(shape=(pixels, pixels, 1))
        c0 = layers.Conv2D(num_filters, activation='relu', kernel_size = kernel_size, padding='same')(inputs)
        c0bn = layers.BatchNormalization()(c0)
        c1 = layers.Conv2D(num_filters, activation='relu', kernel_size = kernel_size, padding='same')(c0bn)  # This layer for concatenating in the expansive part
        c1bn = layers.BatchNormalization()(c1)
        c2 = layers.MaxPool2D((2,2))(c1bn)
        c3 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(c2)
        c3bn = layers.BatchNormalization()(c3)
        c4 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(c3bn)  # This layer for concatenating in the expansive part
        c4bn = layers.BatchNormalization()(c4)
        t1 = layers.UpSampling2D()(c4bn)
        concat01 = layers.concatenate([t1, c1bn], axis=-1)
        t2 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(concat01)
        t2bn = layers.BatchNormalization()(t2)
        t3 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(t2bn)
        t3bn = layers.BatchNormalization()(t3)
        outputs = layers.Conv2D(1, activation='relu', kernel_size=3, padding='same')(t3bn)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-netmodel")
        
    if model_number == 7:#two blocks down and up
        inputs = layers.Input(shape=(pixels, pixels, 1))
        c0 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(inputs)
        c0bn = layers.BatchNormalization()(c0)
        c1 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(c0bn)  # This layer for concatenating in the expansive part
        c1bn = layers.BatchNormalization()(c1)
        c2 = layers.MaxPool2D((2,2))(c1bn)
        c3 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(c2)
        c3bn = layers.BatchNormalization()(c3)
        c4 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(c3bn)  # This layer for concatenating in the expansive part
        c4bn = layers.BatchNormalization()(c4)
        c5 = layers.MaxPool2D((2,2))(c4bn)
        c6 = layers.Conv2D(4*num_filters, activation='relu', kernel_size=3, padding='same')(c5)
        c6bn = layers.BatchNormalization()(c6)
        c7 = layers.Conv2D(4*num_filters, activation='relu', kernel_size=3, padding='same')(c6bn)  # This layer for concatenating in the expansive part
        c7bn = layers.BatchNormalization()(c7)
        #start going up
        t1 = layers.UpSampling2D()(c7bn)
        concat01 = layers.concatenate([t1, c4bn], axis=-1)
        t2 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(concat01)
        t2bn = layers.BatchNormalization()(t2)
        t3 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(t2bn)
        t3bn = layers.BatchNormalization()(t3)
        t4 = layers.UpSampling2D()(t3bn)
        concat02 = layers.concatenate([t4, c1bn], axis=-1)
        t5 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(concat02)
        t5bn = layers.BatchNormalization()(t5)
        t6 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(t5bn)
        t6bn = layers.BatchNormalization()(t6)
        outputs = layers.Conv2D(1, activation='relu', kernel_size=3, padding='same')(t6bn)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-netmodel")

    if model_number == 8:#three blocks down and up
        #first block
        inputs = layers.Input(shape=(pixels, pixels, 1))
        c0 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(inputs)
        c0bn = layers.BatchNormalization()(c0)
        c1 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(c0bn)  # This layer for concatenating in the expansive part
        c1bn = layers.BatchNormalization()(c1)
        c2 = layers.MaxPool2D((2,2))(c1bn)
        #second block
        c3 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(c2)
        c3bn = layers.BatchNormalization()(c3)
        c4 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(c3bn)  # This layer for concatenating in the expansive part
        c4bn = layers.BatchNormalization()(c4)
        c5 = layers.MaxPool2D((2,2))(c4bn)
        #third block
        c6 = layers.Conv2D(4*num_filters, activation='relu', kernel_size=3, padding='same')(c5)
        c6bn = layers.BatchNormalization()(c6)
        c7 = layers.Conv2D(4*num_filters, activation='relu', kernel_size=3, padding='same')(c6bn)  # This layer for concatenating in the expansive part
        c7bn = layers.BatchNormalization()(c7)
        c8 = layers.MaxPool2D((2,2))(c7bn)
        #fourth block
        c9 = layers.Conv2D(8*num_filters, activation='relu', kernel_size=3, padding='same')(c8)
        c9bn = layers.BatchNormalization()(c9)
        c10 = layers.Conv2D(8*num_filters, activation='relu', kernel_size=3, padding='same')(c9bn)  # This layer for concatenating in the expansive part
        c10bn = layers.BatchNormalization()(c10)
        #start going up
        #first block up
        t1 = layers.UpSampling2D()(c10bn)
        concat01 = layers.concatenate([t1, c7bn], axis=-1)
        t2 = layers.Conv2D(4*num_filters, activation='relu', kernel_size=3, padding='same')(concat01)
        t2bn = layers.BatchNormalization()(t2)
        t3 = layers.Conv2D(4*num_filters, activation='relu', kernel_size=3, padding='same')(t2bn)
        t3bn = layers.BatchNormalization()(t3)
        #2d block up
        t4 = layers.UpSampling2D()(t3bn)
        concat02 = layers.concatenate([t4, c4bn], axis=-1)
        t5 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(concat02)
        t5bn = layers.BatchNormalization()(t5)
        t6 = layers.Conv2D(2*num_filters, activation='relu', kernel_size=3, padding='same')(t5bn)
        t6bn = layers.BatchNormalization()(t6)
        #third block up
        t7 = layers.UpSampling2D()(t6bn)
        concat03 = layers.concatenate([t7, c1bn], axis=-1)
        t8 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(concat03)
        t8bn = layers.BatchNormalization()(t8)
        t9 = layers.Conv2D(num_filters, activation='relu', kernel_size=3, padding='same')(t8bn)
        t9bn = layers.BatchNormalization()(t9)
        outputs = layers.Conv2D(1, activation='relu', kernel_size=3, padding='same')(t9bn)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="u-netmodel")
    return model


def get_compiled_model(model_number, loss, dropout_rate, pixels, learning_rate, num_filters, kernel_size):
    '''Return the compiled model'''
    model = get_uncompiled_model_filter(model_number, dropout_rate, pixels, num_filters, kernel_size)
    print("Fred is compiling the model...")
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if loss == "MAE":
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=opt, metrics=['mean_absolute_error'])
    if loss == "MSE":
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer= opt, metrics=['mean_squared_error'])
    if loss == "BCE":
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer= opt, metrics = ['mean_squared_error'])
    if loss == "dice":
        model.compile(loss=dice_loss, optimizer= opt, metrics = ['mean_squared_error'])
    return model

def dice_loss(y_true, y_pred):
    '''dice Loss; not used in this project'''
    y_true = tf.cast(y_true, tf.float32)
    #y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def save_data(save_path, data, file_names):
    '''A function to save data'''
    print("saving data...")
    n = data.shape[0]
    for i in range(n):
        file_name = save_path+file_names[i][:-4]+"_extracted.txt"
        data_to_be_saved = data[i,:,:,0]
        np.savetxt(file_name, data_to_be_saved)