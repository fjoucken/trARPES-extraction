import tensorflow as tf
import winsound
import matplotlib.pyplot as plt
import datetime
import os.path
import glob
from Feature_ARPES_CNN_utils import  load_data_ARPES, get_compiled_model

'''This code creates and trains a CNN model for extracting bare bands from ARPES spectra.
It is similar to what was reported in https://aip.scitation.org/doi/full/10.1063/1.5132586 but is adapted to 
time-resolved spectra, which are very noisy.
It uses functions from Feature_ARPES_CNN_utils.py'''

#Checking you have GPUs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

x = datetime.datetime.now()

#Here are set all the parameters
#pixels is the number of pixels. 128 is working fine.
pixels = 128
#num of filters in the CNN
num_filters = 2
#size of the kernel in the CNN
kernel_size = 3
#Choose the model
model_num = 3
#Number of epochs
no_epochs = 10
#batch size
batch_size = 20
#drop out rate
drop_out_rate = 0.0
#learning rate
learning_rate = 0.0001
#Loss should be MAE. BCE works as well but less well
loss = "MAE"   #BCE is binary cross entropy, "dice" for dice
#I set the model name
model_name = x.strftime("%m_%d_%y_%H_%M_%S_")+"ARPES_features_thicker_excited_"+loss+"_LR_"+str(learning_rate)+"_filter_"+str(num_filters)+"_DO_"+str(drop_out_rate)+"_epochs_"+str(no_epochs)+"_model_"+str(model_num)
#This is to load a pre-trained model.
#if "Y", you load the model with the name loaded_model_name
load_weight = "N"
loaded_model_name = "04_22_22_15_36_21_ARPES_features_thicker_excited_MAE_LR_0.0005_filter_16_DO_0.0_epochs_200_model_3.ckpt"
print ('TF version is:')
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#This is the folder where you get your train and test data
folder_name = 'all_14'

#First I define the paths for training X
base_name_train_X = 'D:/Machine_learning/Generated_training_data/ARPES/Feature_extraction/'+folder_name+'/train/X/txt/'
base_name_train_Y = 'D:/Machine_learning/Generated_training_data/ARPES/Feature_extraction/'+folder_name+'/train/Y/txt/'

base_name_test_X = 'D:/Machine_learning/Generated_training_data/ARPES/Feature_extraction/'+folder_name+'/test/X/txt/'
base_name_test_Y = 'D:/Machine_learning/Generated_training_data/ARPES/Feature_extraction/'+folder_name+'/test/Y/txt/'

#You get the names of the files in the folders
open_path = base_name_train_X+'*.txt'
file_names = [os.path.basename(x) for x in glob.glob(open_path)]
#loading the training data X
print ("loading train_X")
train_X = load_data_ARPES(base_name_train_X, file_names, pixels)

#Then I define the paths for training Y and load the files' names
open_path = base_name_train_Y+'*.txt'
file_names = [os.path.basename(x) for x in glob.glob(open_path)]
#loading the training data X
print ("loading train_Y")
train_Y = load_data_ARPES(base_name_train_Y, file_names, pixels)
print ("train_X shape is:", train_X.shape)
print ("train_Y shape is:", train_Y.shape)

#same for testing data
open_path = base_name_test_X+'*.txt'
file_names = [os.path.basename(x) for x in glob.glob(open_path)]
#loading the testing data X
print ("loading test_X")
test_X = load_data_ARPES(base_name_test_X, file_names, pixels)
#Then I define the paths for testing Y
open_path = base_name_test_Y+'*.txt'
file_names = [os.path.basename(x) for x in glob.glob(open_path)]
#loading the testing data Y
print ("loading test_Y")
test_Y = load_data_ARPES(base_name_test_Y, file_names, pixels)
print ("test_Y shape is:", test_Y.shape)


#create model
model = get_compiled_model(model_number = model_num, 
                            loss = loss,
                            dropout_rate = drop_out_rate, 
                            pixels = pixels,
                            learning_rate = learning_rate,
                            num_filters = num_filters,
                            kernel_size = kernel_size)

model.summary()

# Loads the weights if you wanted to
if load_weight == "Y":
    #Here I change the name for saving
    checkpoint_path = "D:/Machine_learning/My_models/ARPES/"+loaded_model_name
    model.load_weights(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)


#This is where you save the model
checkpoint_path = "D:/Machine_learning/My_models/ARPES/"+model_name+".ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#Training the model
print("Fred is fitting the model:")
history = model.fit(train_X, 
                    train_Y, 
                    epochs=no_epochs, 
                    batch_size=batch_size, 
                    shuffle=True,
                    validation_data=(test_X, test_Y),
                    callbacks=[cp_callback])

#Make some noise!
winsound.Beep(800, 1000)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
#plt.ylim([0, 1.5])
plt.legend(loc='upper right')
plt.show()

test_loss, test_acc = model.evaluate(test_X,  test_Y, verbose=2)
print ("test_loss and test_accuracy are:")
print (test_loss, test_acc)
