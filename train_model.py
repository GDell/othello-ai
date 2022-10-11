import imp
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
from model import fetch_model
import tensorflow as tf

from sklearn.model_selection import train_test_split



def sqrt_int(x):
    return int(math.sqrt(x))


# Load training and testing data.
def load_train_data():
    num_test_trials = len(next(os.walk('data'))[1])
    
    board_data = []
    move_data = []
    for i in range(num_test_trials):
        board_files = [name for name in os.listdir(f'./data/trial_{i+1}/') if "board" in name]
        move_files = [name for name in os.listdir(f'./data/trial_{i+1}/') if "selected_move" in name]

        for board_file in board_files:
            with open(f'./data/trial_{i+1}/{board_file}') as f:
                # Load and add the board. 
                board_data += [[json.loads(line) for line in f.readlines() if "[" in line]]

        for move_file in move_files:     
            with open(f'./data/trial_{i+1}/{move_file}') as f: 
                move_data += [[json.loads(line) for line in f.readlines()]]
    
    train_x, test_x, train_y, test_y = train_test_split(board_data, move_data, test_size=0.25) #  random_state=42

    return (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))


(train_X, train_Y), (test_X, test_Y) = load_train_data()


train_X = train_X.reshape(-1, 8, 8, 1)
test_X = test_X.reshape(-1, 8, 8, 1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# Normalize to a value between 0.0 and 1.0
train_X, test_X = train_X / 3., test_X / 3.


# print(train_X)
# train_Y = train_Y.reshape(-1, 8, 8, 1)
# test_Y = test_Y.reshape(-1, 8, 8, 1)

train_Y = train_Y.astype('float32')
test_Y = test_Y.astype('float32')
train_Y = train_Y.reshape(-1, 64)
test_Y = test_Y.reshape(-1, 64)


# print("HERE IS THE SHAPE")
# print(test_Y.shape)
# print(test_Y[0])
# print("Length")
# print(len(train_Y))
# print(len(train_X))

# print(train_X.shape,train_Y.shape)

model = fetch_model()
import json
def loss_function(y_true, y_pred):
    # print("This is y_true")
    # # print(y_true)
    # print(tf.print(y_true))
    
    # print(f"This is y_pred")
    # print(y_pred)
    # print(tf.print(y_pred))
    # print(tf.print(y_pred[0))
    # print(json.loads(y_pred[0]))

    loss = tf.reduce_mean(tf.pow(y_pred - y_true, 2)) # axis=1
    return loss


model.compile(
    optimizer='adam',
    loss=loss_function, 
    metrics=['accuracy']
)


history = model.fit(train_X, train_Y, batch_size=32, epochs=100, verbose=1, validation_data=(test_X, test_Y))
 
