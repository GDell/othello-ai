import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import math
import os

BOARD_SIZE = 64

# from keras.datasets import fashion_mnist

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
    game_result = False # Did the AI win?
    for i in range(num_test_trials):
        board_files = [name for name in os.listdir(f'./data/trial_{i+1}/') if "board" in name]
        move_files = [name for name in os.listdir(f'./data/trial_{i+1}/') if "selected_move" in name]
        game_result = [name for name in os.listdir(f'./data/trial_{i+1}/') if "game_result" in name][0]
        print("here are the board files ")
        print(board_files)
        # board_data = open("./data/trial/")


load_train_data()


# (train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()



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

