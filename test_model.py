from array import array
from utils import prep_training_data
from utils import prep_board_for_network
from tensorflow.keras.models import model_from_json
import numpy as np


def load_model():
    model = model_from_json(open('./models/test_model/model.json').read())
    model.load_weights('./models/test_model/model.h5')
    return model


def predict_move(board: array) -> tuple:
    board = prep_board_for_network(board)


def process_prediction(prediction):
    prediction = prediction.reshape(-1,8,8)
    for var in range(0,len(prediction[0])):
        row = np.ndarray.tolist(prediction[0][var])
        count = 0 
        for item in row:
            prediction[0][var][count] = round(item, 2)
            count += 1
    return prediction


def test_model():
    # Load the model
    model = load_model()
    (train_X, train_Y), (test_X, test_Y) = prep_training_data()
    model_test_input = train_X[0].reshape(-1,8,8)
    prediction = model.predict(model_test_input)
    print("\n INPUT")
    print(model_test_input * 3)
    twod_pred = process_prediction(prediction)
    print("\n PREDICTION")
    print(twod_pred)



if __name__ == "__main__":
    test_model()

