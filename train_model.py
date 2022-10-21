import matplotlib.pyplot as plt
import tensorflow as tf
from utils import prep_training_data
from model import model as othello_model



def loss_function(y_true, y_pred):
    # print(tf.print(y_true))
    # print(tf.print(y_pred))
    loss = tf.reduce_mean(tf.pow(y_pred - y_true, 2)) # axis=1
    return loss



def save_model(model, model_name):
    model_json = model.to_json()
    with open(f"./models/{model_name}/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f'./models/{model_name}/model.h5')
    return model_json


def train_model(model_gen: str):
    # Load all training epochs
    (train_X, train_Y), (test_X, test_Y) = prep_training_data(model_gen)

    # Load, compile, and train the model with the training data.
    model = othello_model()
    model.summary()
    model.compile(
        optimizer='adam',
        loss=loss_function, 
        metrics=['accuracy']
    )
    history = model.fit(train_X, train_Y, batch_size=32, epochs=1000, verbose=1, validation_data=(test_X, test_Y))
    
    # Save the trained model & weights as json.
    model_json = save_model(model, model_gen)
    print("Done training model")


if __name__ == "__main__":
    train_model('model_gen_1')

