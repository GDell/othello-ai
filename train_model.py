import numpy as np
import matplotlib.pyplot as plt
import os
import json
from model import fetch_model
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, model_from_json


# Load training and testing data.
def load_train_data():
    num_test_trials = len(next(os.walk('data/epoch_1'))[1])
    
    board_data = []
    move_data = []
    for i in range(num_test_trials):
        board_files = [name for name in os.listdir(f'./data/epoch_1/trial_{i+1}/') if "board" in name]
        move_files = [name for name in os.listdir(f'./data/epoch_1/trial_{i+1}/') if "selected_move" in name]

        for board_file in board_files:
            with open(f'./data/epoch_1/trial_{i+1}/{board_file}') as f:
                # Load and add the board. 
                board_data += [[json.loads(line) for line in f.readlines() if "[" in line]]

        for move_file in move_files:     
            with open(f'./data/epoch_1/trial_{i+1}/{move_file}') as f: 
                move_data += [[json.loads(line) for line in f.readlines()]]
    
    train_x, test_x, train_y, test_y = train_test_split(board_data, move_data, test_size=0.25) #  random_state=42

    return (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))


def prep_training_data():
    (train_X, train_Y), (test_X, test_Y) = load_train_data()

    print("HERE IS THE INPUT")
    print(train_X[0])

    train_X = train_X.reshape(-1, 8, 8)
    test_X = test_X.reshape(-1, 8, 8)

    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    print("HERE IS THE INPUT")
    print(train_X[0])

    # Normalize to a value between 0.0 and 1.0
    train_X, test_X = train_X / 3., test_X / 3.

    train_Y = train_Y.astype('float32')
    test_Y = test_Y.astype('float32')
    train_Y = train_Y.reshape(-1, 64)
    test_Y = test_Y.reshape(-1, 64)

    return (train_X, train_Y), (test_X, test_Y)


def loss_function(y_true, y_pred):
    # print("This is y_true")
    # # print(y_true)
    # print(tf.print(y_true))
    # print(f"This is y_pred")
    # print(y_pred)
    # print(tf.print(y_pred))
    loss = tf.reduce_mean(tf.pow(y_pred - y_true, 2)) # axis=1
    return loss


def train_model():

    (train_X, train_Y), (test_X, test_Y) = prep_training_data()

    model = fetch_model()
    model.summary()

    model.compile(
        optimizer='adam',
        loss=loss_function, 
        metrics=['accuracy']
    )

    history = model.fit(train_X, train_Y, batch_size=32, epochs=100, verbose=1, validation_data=(test_X, test_Y))
    
    model_json = model.to_json()
    with open("./models/test_model/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('./models/test_model/model.h5')
    print("Done training model")


def test_model():
    # Load the model
    model = model_from_json(open('./models/test_model/model.json').read())
    model.load_weights('./models/test_model/model.h5')

    (train_X, train_Y), (test_X, test_Y) = prep_training_data()

    print("This is the first")
    print(train_X[0].shape)
    model_test_input = train_X[0].reshape(-1, 8, 8)
    print(model_test_input.shape)
    print(model_test_input)

    # one_board = train_X[0]

    prediction = model.predict(model_test_input)
    # print(len(prediction))
    twod_pred = prediction.reshape(-1,8,8)
    # print(prediction.reshape(-1,8,8))
    # print(type(prediction))


    # print(twod_pred[0])
    print("\n INPUT")
    print(model_test_input * 3)
    print("\n PREDICTION")
    for var in range(0,len(twod_pred[0])):
        row = np.ndarray.tolist(twod_pred[0][var])
        count = 0 
        for item in row:
            twod_pred[0][var][count] = round(item, 2)
            count += 1

    print(twod_pred)

# train_model()
test_model()


