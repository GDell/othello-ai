from utils import prep_training_data
from tensorflow.keras.models import model_from_json
import numpy as np
import os

def load_model():
    model = model_from_json(open('./models/test_model/model.json').read())
    model.load_weights('./models/test_model/model.h5')
    return model


def test_model():
    # Load the model
    model = load_model()
    (train_X, train_Y), (test_X, test_Y) = prep_training_data()

    print("This is the first")
    print(train_X[0].shape)
    model_test_input = train_X[0].reshape(-1, 8, 8)
    print(model_test_input.shape)
    print(model_test_input)

    prediction = model.predict(model_test_input)
    # print(len(prediction))
    twod_pred = prediction.reshape(-1,8,8)
    # print(prediction.reshape(-1,8,8))
    # print(type(prediction))

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



if __name__ == "__main__":
    test_model()

