import pandas as pd
import numpy as np
import sklearn.metrics as sk_metrics
from functools import partial
import matplotlib.pyplot as plt
import json

from keras.engine.saving import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.vis_utils import plot_model


def load_data(path):
    df = pd.read_csv(path, sep=";")
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    return X, y


def reshape_features(X):
    X = X.tolist()
    for idx, x in enumerate(X):
        x.extend([0, 0])
        X[idx] = x
    X = np.asarray(X)
    X = X.reshape(-1, 6, 4, 1)
    return X


def create_model():
    DefaultConv2D = partial(Conv2D, kernel_size=3, activation='relu', padding="SAME")  # configurazioni di default
    model = Sequential([
        DefaultConv2D(filters=8, input_shape=[6, 4, 1]),
        DefaultConv2D(filters=8),
        MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=16),
        Flatten(),
        Dense(units=64, activation='relu'),
        Dropout(0.5),
        Dense(units=32, activation='relu'),
        Dropout(0.5),
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    model.summary()
    return model


def train_and_test_model(create_modelPlot=False, call_resultPlot=False, metrics_all=False, path="dataset/train_set.csv",
                         pathT="dataset/test_set.csv", validation_split=0.2, epochs=20, verbose=2, batch_size=100,
                         nTest_X=-100000, nTest_y=-100000):

    train_X, train_y = load_data(path=path)
    test_X, test_y = load_data(path=pathT)
    train_X = reshape_features(train_X)
    test_X = reshape_features(test_X)
    model = create_model()
    model.fit(train_X, train_y, validation_split=validation_split, epochs=epochs, verbose=verbose,batch_size=batch_size)

    if create_modelPlot:
        plot_model(model, show_shapes=True, to_file='modelPlot.png')

    if metrics_all:
        predictions = model.predict_classes(test_X[nTest_X:], verbose=1)
        print(compute_metrics(test_y[nTest_y:], predictions))
    else:
        # loss, accuracy = model.evaluate(test_X[nTest_X:-100000,:], test_y[nTest_y:-300000])
        loss, accuracy = model.evaluate(test_X[nTest_X:], test_y[nTest_y:])
        print('accuratezza del modello: ' + str(accuracy))
        print('loss : ' + str(loss))

    if (call_resultPlot):
        create_resultPlot(model)

    return model


def compute_metrics(label_array, predictions):
    accuracy_score = sk_metrics.accuracy_score(y_true=label_array, y_pred=predictions)
    precision = sk_metrics.precision_score(y_true=label_array, y_pred=predictions)
    recall = sk_metrics.recall_score(y_true=label_array, y_pred=predictions)
    f_measure = sk_metrics.f1_score(y_true=label_array, y_pred=predictions)
    matthews_corrcoef = sk_metrics.matthews_corrcoef(y_true=label_array, y_pred=predictions)

    return accuracy_score, precision, recall, f_measure, matthews_corrcoef

def create_resultPlot(model):
    # summarize history for accuracy
    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def train_and_save_model(validation_split=0.2, epochs=20, verbose=2, batch_size=100, path="dataset/train_set.csv"):
    train_X, train_y = load_data(path=path)
    train_X = reshape_features(train_X)
    model = create_model()
    model.fit(train_X, train_y, validation_split=validation_split, epochs=epochs, verbose=verbose,
              batch_size=batch_size)
    export_model(model)


def export_model(model, path="model.json", path_weights="model.h5", path_history="history.json"):
    model_json = model.to_json()
    with open(path, "w") as json_file:
        json_file.write(model_json)
    with open(path_history, "w") as json_file:
        json_file.write(model.history.to_json)
    # serialize weights to HDF5
    model.save_weights(path_weights)

    print("Saved model to disk")


def load_and_test_model(nTest_X=-100000, nTest_y=-100000, path="dataset/test_set.csv"):
    model = import_model()
    test_X, test_y = load_data(path=path)
    test_X = reshape_features(test_X)
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    loss, accuracy = model.evaluate(test_X[nTest_X:], test_y[nTest_y:])
    print('accuratezza del modello: ' + str(accuracy))
    print('loss : ' + str(loss))


def import_model(path="model.json", path_weights="model.h5", path_history="history.json"):
    # load json and create model
    json_file = open(path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_weights)
    loaded_model_history = json.load(open(path_history, 'r'))
    print("Loaded model from disk")
    return loaded_model, loaded_model_history


# MAIN
# train_and_test_model(create_modelPlot=False, call_resultPlot=False, epochs=1, nTest_X=-100000, nTest_y=-100000)

train_and_test_model(create_modelPlot=False, call_resultPlot=False, metrics_all=True, epochs=20, nTest_X=-100000, nTest_y=-100000)
#train_and_test_model(create_modelPlot=False, call_resultPlot=False, metrics_all=True, epochs=1, nTest_X=-10, nTest_y=-10)
