from tensorflow.keras.layers import Dense, Input # Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, Concatenate # MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import activations

import numpy as np
import os

batch_size = 64
epochs = 20
num_output_nodes = 64
print("THIS IS THE BOARD SIZE")
print(num_output_nodes)

potential_moves = [[1,1],[2,2]]
board = [8*8]


# The board
input1_board = Input(shape=(8,8,3))
# The possible moves
input2_move_choices = Input(shape=(8,8,1))

# Concatenate these inputs
input = input1_board
# input = Concatenate()([input1_board, input2_move_choices])
input_layer = Conv2D(32, kernel_size=(3,3),activation='linear',input_shape=(8,8,3),padding='same')(input)
middle_layer = Dense(num_output_nodes*2)(input_layer)
middle_layer_1 = Dense(num_output_nodes*4)(middle_layer)

# Output the board (8x8)
output_layer = Dense(num_output_nodes)(middle_layer_1) # activation=activations.sigmoid
model = Model(inputs=[input1_board, input2_move_choices], outputs=output_layer)


model.summary()


# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(8,8,2),padding='same'))
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D((2, 2),padding='same'))
# model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
# model.add(LeakyReLU(alpha=0.1))                  
# model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# model.add(Flatten())
# model.add(Dense(128, activation='linear'))
# model.add(LeakyReLU(alpha=0.1))                  
# model.add(Dense(num_output_nodes, activation='softmax'))