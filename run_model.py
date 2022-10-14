from array import array
from utils import prep_training_data
from utils import prep_board_for_network
from utils import load_model
from tensorflow.keras.models import Model
import numpy as np
import operator


def predict_move(model: Model, board: array) -> tuple[np.ndarray, array]:
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
    possible_moves = []
    prediction = prediction.reshape(-1,8,8)
    for row in range(0,len(prediction[0])):
        row_data = np.ndarray.tolist(prediction[0][row])
        for column in range(0, len(row_data)):
            rounded_prediction = round(row_data[column], 2)
            prediction[0][row][column] = rounded_prediction
            if rounded_prediction > 0.0:
                possible_moves.append({
                    'move': (row, column), 
                    'value': rounded_prediction
                })
    possible_moves.sort(key=operator.itemgetter('value'), reverse=True)
    return prediction, possible_moves


def test_model():
    ''' 
        Takes no input and obtains a move prediction from the most recent model using a random board in the training data set. 
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

