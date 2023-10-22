# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 11:16:45 2023

@author: saibr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    input_vars = ['Sensor'+str(i) for i in range(1, 60)]
    x = df[input_vars].values
    y = df['Command'].values

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(y)
    
    scaler = preprocessing.StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    return train_test_split(x_scaled, y, test_size=0.2), enc


def train_model(x_train, y_train, model):
    model.fit(x_train, y_train)
    return model


def print_accuracy(model, x_train, y_train, x_test, y_test):
    print("Training accuracy: ", accuracy_score(y_train, model.predict(x_train)))
    print("Test accuracy: ", accuracy_score(y_test, model.predict(x_test)))


def plot_confusion_matrix(y_test, model_pred, enc):
    cm = confusion_matrix(y_test, model_pred)
    plt.figure()
    ax = plt.axes()
    labels = enc.inverse_transform(np.arange(len(set(y_test))))
    sn.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='.0f', ax=ax, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=0)
    plt.show()


if __name__ == "__main__":
    (x_train, x_test, y_train, y_test), enc = load_and_preprocess_data('sensor_readings_24.csv')

    modelSVM = SVC()
    modelSVM = train_model(x_train, y_train, modelSVM)
    print("Support Vector Machine Model")
    print_accuracy(modelSVM, x_train, y_train, x_test, y_test)
    plot_confusion_matrix(y_test, modelSVM.predict(x_test), enc)

    modelLR = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    modelLR = train_model(x_train, y_train, modelLR)
    print("\nLogistic Regression Model")
    print_accuracy(modelLR, x_train, y_train, x_test, y_test)
    plot_confusion_matrix(y_test, modelLR.predict(x_test), enc)
