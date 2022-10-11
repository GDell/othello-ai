from sklearn.model_selection import train_test_split
import numpy as np
import json
import os


# Load training and testing data.
def load_train_data():
    num_test_trials = len(next(os.walk('data/epoch_1'))[1])
    
    board_data = []
    move_data = []
    for i in range(num_test_trials):
        board_files = [name for name in os.listdir(f'./data/epoch_1/trial_{i+1}/') if "board" in name]
        move_files = [name for name in os.listdir(f'./data/epoch_1/trial_{i+1}/') if "selected_move" in name]

        # Load Board Training Data
        for board_file in board_files:
            with open(f'./data/epoch_1/trial_{i+1}/{board_file}') as f:
                # Load and add the board. 
                board_data += [[json.loads(line) for line in f.readlines() if "[" in line]]

        # Load Labels (Chosen Moves)
        for move_file in move_files:     
            with open(f'./data/epoch_1/trial_{i+1}/{move_file}') as f: 
                move_data += [[json.loads(line) for line in f.readlines()]]
    
    # Obtain a test split (train 75% / test 25%)
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

