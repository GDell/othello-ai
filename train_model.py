import matplotlib.pyplot as plt
import tensorflow as tf
from utils import prep_training_data
from model import model as othello_model



def loss_function(y_true, y_pred):
    # print("This is y_true")
    # print(y_true)
    # print(tf.print(y_true))
    # print(f"This is y_pred")
    # print(y_pred)
    # print(tf.print(y_pred))
    loss = tf.reduce_mean(tf.pow(y_pred - y_true, 2)) # axis=1
    return loss


def train_model():
    (train_X, train_Y), (test_X, test_Y) = prep_training_data("epoch")

    model = othello_model()
    model.summary()
    model.compile(
        optimizer='adam',
        loss=loss_function, 
        metrics=['accuracy']
    )

    history = model.fit(train_X, train_Y, batch_size=32, epochs=1000, verbose=1, validation_data=(test_X, test_Y))
    
    model_json = model.to_json()
    with open("./models/test_model/model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights('./models/test_model/model.h5')
    print("Done training model")


if __name__ == "__main__":
    train_model()

