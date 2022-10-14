from array import array
from utils import prep_training_data
from utils import prep_board_for_network
from utils import load_model
from tensorflow.keras.models import Model
import numpy as np
import operator


def predict_move(model: Model, board: array) -> tuple[np.ndarray, array]:
    '''
        Takes a model and a board as input, and returns a tuple with a prediction of move values 
        as a 8x8 2D np.ndarray and an array of predicted moves in (row, col) format ranked from 
        highest to lowest predicted value. 
    '''
    board = prep_board_for_network(np.asarray(board))
    print("\n INPUT")
    print(board * 3)
    prediction = model.predict(board)
    prediction, predicted_moves = process_prediction(prediction)
    print("\n PREDICTION")
    print(prediction)
    print(predicted_moves)
    return prediction, predicted_moves


def process_prediction(prediction: np.ndarray) -> tuple[np.ndarray, array]:
    '''
        Takes in a np.ndarray prediction from model.predict(), reshapes the prediction
        to 8x8 2D np.ndarray and returns a tuple with the reshaped prediction and an array 
        of possible moves (row, col).
    '''
    possible_moves = []
    prediction = prediction.reshape(-1,8,8)
    for row in range(0,len(prediction[0])):
        row_data = np.ndarray.tolist(prediction[0][row])
        for column in range(0, len(row_data)):
            rounded_prediction = row_data[column] # round(row_data[column], 3)
            prediction[0][row][column] = rounded_prediction
            # if rounded_prediction > 0.0:
            possible_moves.append({
                'move': (row, column), 
                'value': rounded_prediction
            })
    possible_moves.sort(key=operator.itemgetter('value'), reverse=True)
    return prediction, possible_moves


def test_model():
    ''' 
        Takes no input and obtains a move prediction using the most recent model and a random 
        board in the training data set. 
    '''
    model = load_model()
    (train_X, train_Y), (test_X, test_Y) = prep_training_data()
    model_test_input = train_X[0].reshape(-1,8,8)
    prediction = model.predict(model_test_input)
    print("\n INPUT")
    print(model_test_input * 3)
    twod_pred, possible_moves = process_prediction(prediction)
    print("\n PREDICTION")
    print(twod_pred)
    print(possible_moves)



if __name__ == "__main__":
    test_model()

