from tensorflow.keras.layers import Dense, Input # Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
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


# def fetch_model():
#     # The board

#     input1_board = Input(shape=(8,8,1))
#     # The possible moves
#     # input2_move_choices = Input(shape=(8,8,1))

#     # Concatenate these inputs
#     input = input1_board
#     # input = Concatenate()([input1_board, input2_move_choices])
#     input_layer = Conv2D(8, kernel_size=(2,2),activation='relu',input_shape=(8,8), padding = "same")(input)
#     # pooling_layer = MaxPooling2D((2, 2))(input_layer)
#     middle_layer = Conv2D(64, kernel_size=(2, 2), activation='relu', padding = "same")(input_layer)
#     # pooling_layer_1 = MaxPooling2D((2, 2))(middle_layer)

#     # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     # model.add(layers.MaxPooling2D((2, 2)))
#     # model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#     # Output the board (8x8)
#     # output_layer = Dense(num_output_nodes)(middle_layer) # activation=activations.sigmoid
#     output_layer = tf.sigmoid(tf.reshape(middle_layer, [-1, 64])) # 8, 8
#     model = Model(inputs=input1_board, outputs=output_layer)
#     model.summary()

#     return model


def fetch_model():
    go_model = Sequential()
    go_model.add(Conv2D(32, kernel_size=(2, 2),activation='linear',input_shape=(8,8,1),padding='same'))
    # go_model.add(LeakyReLU(alpha=0.1))
    go_model.add(MaxPooling2D((2, 2),padding='same'))
    go_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    # go_model.add(LeakyReLU(alpha=0.1))
    go_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    go_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    # go_model.add(LeakyReLU(alpha=0.1))                  
    go_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    # go_model.add(Flatten())
    # go_model.add(Dense(128, activation='linear'))                 
    go_model.add(Dense(64, activation='sigmoid'))
    return go_model

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