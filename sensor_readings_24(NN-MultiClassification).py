# -*- coding: utf-8 -*-
"""
@author: saibrone
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

def load_and_preprocess_data(file_path, input_columns):
    df = pd.read_csv(file_path)
    x = df[input_columns].values
    y = pd.get_dummies(df['Command']).values

    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return train_test_split(x_scaled, y, test_size=0.2, random_state=42)

def build_and_evaluate_model(x_train, y_train, x_test, y_test):
    model_nn = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(y_train.shape[1], activation='softmax'),
    ])

    model_nn.compile(loss='categorical_crossentropy', 
                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     metrics=['accuracy'])

    model_nn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=100)

    print("The accuracy on the training data is: ", 
          accuracy_score(np.argmax(y_train, axis=1), np.argmax(model_nn.predict(x_train), axis=1)))

    print("The accuracy on the test data is: ", 
          accuracy_score(np.argmax(y_test, axis=1), np.argmax(model_nn.predict(x_test), axis=1)))

if __name__ == "__main__":
    input_vars = ['Sensor'+str(i) for i in range(1, 25)]
    x_train, x_test, y_train, y_test = load_and_preprocess_data('sensor_readings_24.csv', input_vars)
    build_and_evaluate_model(x_train, y_train, x_test, y_test)
