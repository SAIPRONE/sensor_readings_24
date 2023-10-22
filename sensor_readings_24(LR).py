# -*- coding: utf-8 -*-
"""
@author: saibrone
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path, input_columns):
    df = pd.read_csv(file_path)
    x = df[input_columns].values
    y = pd.get_dummies(df['Command']).values
    
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    return train_test_split(x_scaled, y, test_size = 0.2, random_state=42)

def build_and_evaluate_model(x_train, y_train, x_test, y_test):
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    model.fit(x_train, y_train)
    
    print("The accuracy on the training data is: ", accuracy_score(np.argmax(y_train, axis=1), model.predict(x_train)))
    print("The accuracy on the test data is: ", accuracy_score(np.argmax(y_test, axis=1), model.predict(x_test)))

    # Confusion Matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), model.predict(x_test))
    plt.figure()
    ax = plt.axes()
    sns.heatmap(cm, cmap = plt.cm.Blues, annot = True, fmt = '.0f', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation = 0)
    plt.show()

if __name__ == "__main__":
    input_vars = ['Sensor'+str(i) for i in range(1, 25)]
    x_train, x_test, y_train, y_test = load_and_preprocess_data('sensor_readings_24.csv', input_vars)
    build_and_evaluate_model(x_train, y_train, x_test, y_test)
