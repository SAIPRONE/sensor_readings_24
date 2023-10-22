# -*- coding: utf-8 -*-
"""
@author: saibrone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    input_vars = [f'Sensor{i}' for i in range(1, 25)]
    x = df[input_vars].values
    y = pd.get_dummies(df['Command']).values

    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    return train_test_split(x_scaled, y, test_size=0.2)


def build_and_train_model(x_train, y_train, x_test, y_test, epochs=30, batch_size=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax'),
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

    return model


def print_accuracy(model, x_train, y_train, x_test, y_test):
    print("Training accuracy: ", accuracy_score(np.argmax(y_train, axis=1), np.argmax(model.predict(x_train), axis=1)))
    print("Test accuracy: ", accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1)))


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_and_preprocess_data('sensor_readings_24.csv')
    model = build_and_train_model(x_train, y_train, x_test, y_test)
    print_accuracy(model, x_train, y_train, x_test, y_test)
