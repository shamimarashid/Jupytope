#!/usr/bin/env python
# coding: utf-8

#Written by Shamima Rashid, Nanyang Technological University, Singapore. June 2022
## Adapted from original code provided by: 
## © MIT 6.S191: Introduction to Deep Learning
# http://introtodeeplearning.com

import tensorflow as tf
#Use Dynamic GPU Memory. Comment out 6-8 for running on CPU only.
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#Uncomment for running on CPU only.
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import RobustScaler
import seaborn as sns

from keras import backend as K
import gc

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

import mitdeeplearning as mdl

import time
import functools

from IPython import display as ipythondisplay
from tqdm import tqdm

from Bio import SeqIO
import random


fn1 = 'Sites_final_Influenza.csv' #19992 rows of cleaned data
fields_1 = ['Hydropathy', 'SS', 'RSA', 'PHI', 'PSI', 'main_chain_rel', 'all_polar_rel', 'CA_Up', 'CA_down', 'CA_PCB_Angle',
            'CA_Up.1','CA_down.1', 'CA_Count_r12',  'Residue_Depth','CA_Depth', 'B_Norm', 'Target'] #16 features
fields_2 = ['Hydropathy', 'SS', 'RSA', 'PHI', 'PSI', 'main_chain_rel', 'CA_Up', 'CA_down', 'CA_PCB_Angle', 
            'Residue_Depth','B_Norm', 'Target'] #11 features

df = pd.read_csv(fn1, usecols=fields_2) 
Subtype = {"H1N1":0, "H3N2":1}
SS_Num = {"H":1, "E":2, "C":3}

df['Target']= df['Target'].apply(lambda x: Subtype[x])
df['SS'] = df['SS'].apply(lambda x: SS_Num[x])

df.head(6)

F_names = fields_2[:-1]

X = df[F_names]
Y = df['Target']
X = X.to_numpy()
Y = Y.to_numpy()
print("Input Shape (Total samples, num_features): %s Target shape: %s" %(X.shape, Y.shape))
print("X type: ",type(X), "Y type: ", type(Y))



#Generate random target
#Fixed seed for reproducibility
np.random.seed(1)

## Prepare vector of randomly generated labels
n = 2
np.random.seed(1)
R = np.random.randint(n, size=(len(Y),))
#R = R + 1
R=R.tolist()


R = np.asarray(R)

#print(R.shape)

#For debugging
#X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.25, random_state = 1)
#XR_train, XR_test, R_train, R_test= train_test_split(X, R, test_size=0.25, random_state = 1)

#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, "\n")
#print(XR_train.shape, XR_test.shape, R_train.shape, R_test.shape)



#function to prepare data batches
def get_batch_seq(X_samples, Y_samples, batch_size, time_step, num_features):
    #np.random.seed(0) #will cause the identical batches per draw
    n = X_samples.shape[0]
    assert (batch_size <= n), "batchsize should be smaller than n"
    #print("N is %d and batch_size is %d"%(n, batch_size))     
    rand_ind = np.random.choice(n, batch_size, replace=False)
    input_batch = X_samples[rand_ind, :]
    #print("ip batch shape: ", input_batch.shape)
    output_batch = Y_samples[rand_ind]
    #print("op batch shape: ", output_batch.shape)
   
    
    #time_step is the 'sliding window' size.

    #x_batch, y_batch provide the inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, time_step, num_features])
    y_batch = np.reshape(output_batch, [batch_size, 1]) #one column of labels
   
    
    return x_batch, y_batch  



def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        #return_sequences = True,
        #batch_input_shape = (batch_size, time_step, num_features),
        recurrent_initializer = 'glorot_uniform',
        recurrent_activation = 'sigmoid',
        stateful = False 
    )




def build_model(batch_size, time_step, num_features, rnn_units):
    #embedding_dim = 256
    b = batch_size

    bias_initializer = tf.keras.initializers.HeNormal()
    
    model = tf.keras.Sequential([
    
    tf.keras.Input(batch_size = b, shape = (time_step, num_features)),
    
        
    LSTM(rnn_units),

       
    tf.keras.layers.Dense(4, activation='softmax') #, bias_initializer=bias_initializer) 
                                     
    
    ])

    return model


def compute_loss(y, y_hat):
    
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = scce(y, y_hat)
        
    return loss

def reset_weights(model):
  for layer in model.layers: 
    if isinstance(layer, tf.keras.Model):
      reset_weights(layer)
      continue
    for k, initializer in layer.__dict__.items():
      if "initializer" not in k:
        continue
      # find the corresponding variable
      var = getattr(layer, k.replace("_initializer", ""))
      var.assign(initializer(var.shape, var.dtype))



# Build a simple model with default hyperparameters

# model = build_model(batch_size=32, time_step=1, num_features=23, rnn_units=1024)
# model.summary()
#[x, y] = get_batch_seq(X_train, Y_train, 32, 1, 23)

### Hyperparameter setting and optimization ###
#Optimization parameters:
num_training_iterations = 4000   #100 #12000 gives ~0.87 accu for HA subtype
batch_size = 128
learning_rate = 5e-4 #1e-3   #Experiment between 1e-5(0.00001) and 1e-1 (0.1) 
#5e-3
## For Adam optimizer
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07
#print('{:10f}'.format(delta))

#model parameters
time_step = 1
num_features = 11
rnn_units = 1024 

 
trials = 2 #100
All_Test_Accuracy = []

for i in range(trials):
    
    np.random.seed(0)
    model = build_model(batch_size, time_step, num_features, rnn_units)   
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1, beta_2, epsilon)

    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.25, random_state = 1)
    #XR_train, XR_test, R_train, R_test= train_test_split(X, R, test_size=0.25, random_state = 1)   

    ##################
    # Begin training!#
    ##################
    
    history = []    

    plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
    
    

    for iter in tqdm(range(num_training_iterations)):
        #Grab a batch and propagate it through the network:   
   
        
        x_batch, y_batch = get_batch_seq(X_train, Y_train, batch_size, time_step, num_features) #time_step = 1
        #x_batch, y_batch = get_batch_seq(XR_train, R_train, batch_size, time_step, num_features)        
        


        with tf.GradientTape() as tape:      
            
            y_hat = model(x_batch)  
            loss = compute_loss(y_batch, y_hat)        
        
    
        grads = tape.gradient(loss, model.trainable_variables)
        # Apply the gradients to the optimizer so it can update the model accordingly
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
          
    
        # Update the progress bar
        history.append(loss.numpy().mean())    
        #plotter.plot(history) 


    #test the model on test sequences
    #[x, y] = get_batch_seq(X_test, Y_test, 4981, 1, 11)
    #[x, r] = get_batch_seq(XR_test, R_test, 4981, 1, 11)
    [x, y] = get_batch_seq(X_test, Y_test, X_test.shape[0], time_step, num_features)
    #[x, r] = get_batch_seq(XR_test, R_test, XR_test.shape[0], time_step, num_features)      
    y_hat = model(x)
    #R_hat = model(x)
    #Evaluate predictions on test
    m = tf.keras.metrics.SparseCategoricalAccuracy()
    m.update_state(y, y_hat)
    #m.update_state(r, R_hat)
    accuracy = m.result().numpy()
    All_Test_Accuracy.append(accuracy)
    m.reset_states()  
    
    

    print("Trial %d Accuracy %f Actual shape: %s Prediction Shape: %s \n" %(i, accuracy, y.shape, y_hat.shape))
    #print("Trial %d Accuracy %f Actual shape: %s Prediction Shape: %s \n" %(i, accuracy, r.shape, R_hat.shape))
    
    
    reset_weights(model)
    del model     
    K.clear_session()         
    gc.collect()



#Change filename as needed
# fn = 'LSTM_Perf_HATrueLabels.txt'
# op = open(fn, 'w')
# for t in All_Test_Accuracy:
#     print(t, file=op)
# op.close()