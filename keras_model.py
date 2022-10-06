import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import math
import os

BOARD_SIZE = os.getenv('BOARD_SIZE')

from keras.datasets import fashion_mnist



# Load training and testing data here.

# Example:
# Play 100 games
# Each board and move choice is stored and labeled as having won or lost at the end of the game. 
# Boards and move choices from 80 games are used to train, the rest is used to test accuracy.

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

print("Training data shape")
print(train_X.shape)
print(train_Y.shape)
print(train_Y)
print(test_X.shape)
print(test_Y.shape)



### Data pre-preprocessing

# Convert training data to matrix
train_X = train_X.reshape(-1, math.sqrt(BOARD_SIZE), math.sqrt(BOARD_SIZE), 2)
test_X = test_X.reshape(-1, math.sqrt(BOARD_SIZE), math.sqrt(BOARD_SIZE), 2)


# Data needs to be fed to the network as type float32
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# Diving by 3 to reduce our 3 possible board values to a number between 0.-1.
train_X = train_X / 3.
test_X = test_X / 3.