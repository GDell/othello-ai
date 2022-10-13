from tensorflow.keras.layers import Dense, Input # Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import activations
from tensorflow.keras import losses
from tensorflow.keras.activations import sigmoid
import tensorflow as tf


num_output_nodes = 64



def model() -> Model:
    go_model = Sequential()
    go_model.add(Conv2D(32, kernel_size=(2, 2),activation='linear',input_shape=(8,8,1),padding='same'))
    go_model.add(LeakyReLU(alpha=0.1))
    go_model.add(MaxPooling2D((2, 2),padding='same'))
    go_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    go_model.add(LeakyReLU(alpha=0.1))
    go_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    go_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    go_model.add(LeakyReLU(alpha=0.1))                  
    go_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    go_model.add(Flatten())              
    go_model.add(Dense(num_output_nodes, activation=sigmoid))
    go_model.summary()
    return go_model


if __name__ == '__main__': 
    model()

