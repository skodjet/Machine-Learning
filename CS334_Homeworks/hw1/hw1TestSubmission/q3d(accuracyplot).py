"""
THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje

I collaborated with the following classmates for this homework:
None
"""

import knn
import pandas as pd
import numpy as np
import seaborn as sb
import argparse
from matplotlib import pyplot as plt


# Setup (copied from knn.py)
# load the train and test data
xTrain = pd.read_csv("q3xTrain.csv")
yTrain = pd.read_csv("q3yTrain.csv")
xTest = pd.read_csv("q3xTest.csv")
yTest = pd.read_csv("q3yTest.csv")

# Get the training accuracy for k values between 1 and sqrt n, where n is the number of rows
n = np.array(xTrain).shape[0]
training_accuracy = []
for j in range(1, int(pow(n, 0.5) + 1)):
    knn_object = knn.Knn(j)
    knn_object.train(xTrain, yTrain['label'])
    y_hat_train = knn_object.predict(xTrain)
    train_acc = knn.accuracy(y_hat_train, yTrain['label'])
    training_accuracy.append(train_acc)

# Get the test accuracy for the same k values
n = np.array(xTest).shape[0]
test_accuracy = []
for j in range(1, int(pow(n, 0.5) + 1)):
    knn_object = knn.Knn(j)
    knn_object.train(xTrain, yTrain['label'])
    y_hat_test = knn_object.predict(xTest)
    test_acc = knn.accuracy(y_hat_test, yTest['label'])
    test_accuracy.append(test_acc)

# Plot the training accuracy against the test accuracy
k_values = []
for i in range(1, int(pow(n, 0.5) + 1)):
    k_values.append(i)

# Plot the figure
plt.figure()
plt.plot(k_values, training_accuracy, label="training")
plt.plot(k_values, test_accuracy, label="test")
plt.legend()
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K-Nearest Neighbors Accuracy For Different K Values")
plt.show()
