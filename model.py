from tensorflow.keras.layers import Dense, Input # Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import activations
from tensorflow.keras import losses
import tensorflow as tf
# Reference https://www.tensorflow.org/tutorials/images/cnn

import numpy as np
import os

batch_size = 64
epochs = 20
num_output_nodes = 64


def fetch_model():
    # The board
    input1_board = Input(shape=(8,8,1))
    # The possible moves
    # input2_move_choices = Input(shape=(8,8,1))

    # Concatenate these inputs
    input = input1_board
    # input = Concatenate()([input1_board, input2_move_choices])
    input_layer = Conv2D(8, kernel_size=(1,1),activation='relu',input_shape=(8,8), padding = "same")(input)
    pooling_layer = MaxPooling2D((2, 2))(input_layer)
    middle_layer = Conv2D(64, (3, 3), activation='relu', padding = "same")(pooling_layer)
    # pooling_layer_1 = MaxPooling2D((2, 2))(middle_layer)

    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Output the board (8x8)
    output_layer = Dense(64, activation='sigmoid')(middle_layer) # activation=activations.sigmoid
    output_layer = tf.reshape(output_layer, [-1, 8, 8, 1])
    model = Model(inputs=input1_board, outputs=output_layer) # [input1_board, input2_move_choices]


    model.summary()

    return model




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

if __name__ == '__main__': 
    compile_model()