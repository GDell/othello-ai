from array import array
import numpy as np
import json
import os
from re import A
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json



# Load training and testing data.
def load_train_data(epoch: str) -> \
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    
    # Calculate the number of trial epochs to load.
    num_test_trials = len(next(os.walk(f'data/'))[1]) - 2
    
    board_data = []
    move_data = []

    # Load training data from eac epoch.
    for i in range(num_test_trials):
        board_files = [name for name in os.listdir(f'./data/{epoch}_{i+1}/') if "board" in name]
        move_files = [name for name in os.listdir(f'./data/{epoch}_{i+1}/') if "selected_move" in name]

        # Load Board Training Data
        for board_file in board_files:
            with open(f'./data/{epoch}_{i+1}/{board_file}') as f:
                # Load and add the board. 
                board_data += [[json.loads(line) for line in f.readlines() if "[" in line]]

        # Load Labels (Chosen Moves)
        for move_file in move_files:     
            with open(f'./data/{epoch}_{i+1}/{move_file}') as f: 
                move_data += [[json.loads(line) for line in f.readlines()]]
    
    # Obtain a test split (train 75% / test 25%)
    train_x, test_x, train_y, test_y = train_test_split(board_data, move_data, test_size=0.25) #  random_state=42

    return (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))


def prep_training_data(epoch: str ="starting_training_set") -> \
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    
    # Load the training data.
    (train_X, train_Y), (test_X, test_Y) = load_train_data(epoch)
    
    # Reshape and perform type conversion to prep the input data for the network.
    train_X = train_X.reshape(-1, 8, 8)
    test_X = test_X.reshape(-1, 8, 8)
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    # Normalize to a value between 0.0 and 1.0 so the network can consume the input.
    train_X, test_X = train_X / 3., test_X / 3.

    # Reshape and perform type conversion to prep the labels for calculating loss.
    train_Y = train_Y.astype('float32')
    test_Y = test_Y.astype('float32')
    train_Y = train_Y.reshape(-1, 64)
    test_Y = test_Y.reshape(-1, 64)

    return (train_X, train_Y), (test_X, test_Y)


def prep_board_for_network(board: array) -> np.array:
    board = board.reshape(-1, 8, 8)
    board = board.astype('float32')
    board = board / 3 
    return board


def load_model(model_name: str = "test_model") -> Model:
    model = model_from_json(open(f'./models/{model_name}/model.json').read())
    model.load_weights(f'./models/{model_name}/model.h5')
    return model


