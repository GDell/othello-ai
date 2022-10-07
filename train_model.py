import imp
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import math
import os
import json
from model import fetch_model

from sklearn.model_selection import train_test_split


BOARD_SIZE = 64

from keras.datasets import fashion_mnist

def sqrt_int(x):
    return int(math.sqrt(x))

# Load training and testing data here.

# Example:
# Play 100 games
# Each board and move choice is stored and labeled as having won or lost at the end of the game. 
# Boards and move choices from 80 games are used to train, the rest is used to test accuracy.


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
    
    # Load train and test data for X (board data) and Y (Move choice)
    # train_x = np.array(board_data[0:int(len(board_data)*0.8)])
    # test_x = np.array(board_data[int(len(board_data)*0.8):])

    # train_y = np.array(move_data[0:int(len(move_data)*0.8)])
    # test_y = np.array(move_data[int(len(move_data)*0.8):])
    
    train_x, test_x, train_y, test_y = train_test_split(board_data, move_data, test_size=0.25, random_state=42)

    return (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))


(train_X, train_Y), (test_X, test_Y) = load_train_data()


print("HERE ARE THE TRAINING VALUES")
print(train_X)
print(train_Y)
print(len(train_X))
print(len(train_Y))

#Normalize data between values 0 and 1 



train_X = train_X.reshape(-1, 8, 8, 1)
test_X = test_X.reshape(-1, 8, 8, 1)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X, test_X = train_X / 3., test_X / 3.


# print(train_X)

train_Y = train_Y.reshape(-1, 8, 8, 1)
test_Y = test_Y.reshape(-1, 8, 8, 1)

train_Y = train_X.astype('float32')
test_Y = test_X.astype('float32')

print("HERE IS THE SHAPE")
print(test_Y.shape)

# print(train_X.shape,train_Y.shape)

model = fetch_model()

def my_loss_fn(y_true, y_pred):
        print("Y true")
        print(y_true[0])
        print("Y pred")
        print(y_pred[0])
        squared_difference = y_true - y_pred
        
        return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

model.compile(optimizer='adam',
            loss=my_loss_fn,
            metrics=['accuracy'])
history = model.fit(train_X, train_Y, batch_size=2,epochs=20, verbose=1,validation_data=(test_X, test_Y))

# history = model.fit(train_X, train_Y, epochs=10, 
#                     validation_data=(test_X, test_Y))


# (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

# print(train_X)

### Data pre-preprocessing

# Convert training data to matrix
# train_X = train_X.reshape(-1, sqrt_int(BOARD_SIZE), sqrt_int(BOARD_SIZE), 3)
# test_X = test_X.reshape(-1, sqrt_int(BOARD_SIZE), sqrt_int(BOARD_SIZE), 3)


# # Data needs to be fed to the network as type float32
# train_X = train_X.astype('float32')
# test_X = test_X.astype('float32')

# # Diving by 3 to reduce our 3 possible board values to a number between 0.-1.
# train_X = train_X / 3.
# test_X = test_X / 3.


# print("Training data shape")
# print(train_X[0])
# print(train_X.shape)
# print(train_Y.shape)
# print("Here is Train X")
# print(train_X)
# print(train_Y)
# print(test_X.shape)
# print(test_Y.shape)

