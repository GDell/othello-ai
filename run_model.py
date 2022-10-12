from array import array
from utils import prep_training_data
from utils import prep_board_for_network
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
import numpy as np


def load_model(model_name: str = "test_model") -> Model:
    model = model_from_json(open(f'./models/{model_name}/model.json').read())
    model.load_weights(f'./models/{model_name}/model.h5')
    return model


def predict_move(board: array, model: Model, possible_moves: array):
    board = prep_board_for_network(board)
    print("\n INPUT")
    print(board * 3)
    prediction = model.predict(board)
    prediction, predicted_moves = process_prediction(prediction)
    print("\n PREDICTION")
    print(prediction)
    print(predicted_moves)


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
    print("this is the prediction type: ")
    print(type(prediction))
    return prediction, sorted(possible_moves, reverse=True,  key=lambda d: d['value'])


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

