"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
   WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
   Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

# Helper file for plots for question 3

import argparse
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import sgdLR
from lr import LinearRegression, file_to_numpy

"""
3a. For a batch size of 1 and a random subset of 40% of the training data, 
try various learning rates and plot the MSE on the training data as a function of 
the epoch.
"""

# set up the program to take in arguments from the command line (copied from sgdLR.py)
parser = argparse.ArgumentParser()
parser.add_argument("xTrain",
                    help="filename for features of the training data")
parser.add_argument("yTrain",
                    help="filename for labels associated with training data")
parser.add_argument("xTest",
                    help="filename for features of the test data")
parser.add_argument("yTest",
                    help="filename for labels associated with the test data")
parser.add_argument("--seed", default=334,
                    type=int, help="default seed number")

args = parser.parse_args()
# load the train and test data
xTrain = file_to_numpy(args.xTrain)
yTrain = file_to_numpy(args.yTrain)
xTest = file_to_numpy(args.xTest)
yTest = file_to_numpy(args.yTest)

# setting the seed for deterministic behavior
np.random.seed(args.seed)

# Get a random subset of 40% of the training data.
train_combined = np.hstack((xTrain, yTrain))
train_subset_rows = np.random.choice(train_combined.shape[0],
                                size=int(np.shape(train_combined)[0] * 0.40),
                                replace=False)
train_subset = train_combined[train_subset_rows]
train_subset_x = train_subset[:, :-1]
train_subset_y = train_subset[:, -1]
train_subset_y = np.reshape(train_subset_y, (np.shape(train_subset_y)[0], 1))



# # Try learning rates 1, 0.1, 0.01, 0.001, 0.0001, 0.00001
model_one = sgdLR.SgdLR(1, 1, 10)
model_two = sgdLR.SgdLR(0.1, 1, 10)
model_three = sgdLR.SgdLR(0.01, 1, 10)
model_four = sgdLR.SgdLR(0.001, 1, 10)
model_five = sgdLR.SgdLR(0.0001, 1, 10)
model_six = sgdLR.SgdLR(0.00001, 1, 10)

trainStats_one = model_one.train_predict(train_subset_x, train_subset_y, xTest, yTest)
trainStats_two = model_two.train_predict(train_subset_x, train_subset_y, xTest, yTest)
trainStats_three = model_three.train_predict(train_subset_x, train_subset_y, xTest, yTest)
trainStats_four = model_four.train_predict(train_subset_x, train_subset_y, xTest, yTest)
trainStats_five = model_five.train_predict(train_subset_x, train_subset_y, xTest, yTest)
trainStats_six = model_six.train_predict(train_subset_x, train_subset_y, xTest, yTest)

batch_size = np.shape(train_subset_x)[0]

trainStats_one_yaxis = []
trainStats_two_yaxis = []
trainStats_three_yaxis = []
trainStats_four_yaxis = []
trainStats_five_yaxis = []
trainStats_six_yaxis = []

for entry in trainStats_one:
    if entry % batch_size == 0:
        val_to_append_one = trainStats_one[entry]["train-mse"]
        trainStats_one_yaxis.append(val_to_append_one)

for entry in trainStats_two:
    if entry % batch_size == 0:
        val_to_append_two = trainStats_two[entry]["train-mse"]
        trainStats_two_yaxis.append(val_to_append_two)

for entry in trainStats_three:
    if entry % batch_size == 0:
        val_to_append_three = trainStats_three[entry]["train-mse"]
        trainStats_three_yaxis.append(val_to_append_three)

for entry in trainStats_four:
    if entry % batch_size == 0:
        val_to_append_four = trainStats_four[entry]["train-mse"]
        trainStats_four_yaxis.append(val_to_append_four)

for entry in trainStats_five:
    if entry % batch_size == 0:
        val_to_append_five = trainStats_five[entry]["train-mse"]
        trainStats_five_yaxis.append(val_to_append_five)

for entry in trainStats_six:
    if entry % batch_size == 0:
        val_to_append_six = trainStats_six[entry]["train-mse"]
        trainStats_six_yaxis.append(val_to_append_six)

x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.figure()
plt.plot(x_axis, trainStats_one_yaxis, label="Learning Rate 1")
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Training MSE for Learning Rate 1")
plt.legend()
plt.show()

plt.figure()
plt.plot(x_axis, trainStats_two_yaxis, label="Learning Rate 0.1")
plt.plot(x_axis, trainStats_three_yaxis, label="Learning Rate 0.01")
plt.plot(x_axis, trainStats_four_yaxis, label="Learning Rate 0.001")
plt.plot(x_axis, trainStats_five_yaxis, label="Learning Rate 0.0001")
plt.plot(x_axis, trainStats_six_yaxis, label="Learning Rate 0.00001")
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Training MSE for Learning Rates Under 1")
plt.ylim(0, 30)
plt.legend()
plt.show()

"""3b. Train the model on the entire dataset at the optimal learning rate"""
ideal_model = sgdLR.SgdLR(0.001, 1, 10)
ideal_trainStats = ideal_model.train_predict(xTrain, yTrain, xTest, yTest)

batch_size = np.shape(xTrain)[0]
x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ideal_trainStats_yaxis = []
ideal_testStats_yaxis = []
for entry in ideal_trainStats:
    if entry % batch_size == 0:
        val_to_append_one = ideal_trainStats[entry]["train-mse"]
        ideal_trainStats_yaxis.append(val_to_append_one)
        val_to_append_two = ideal_trainStats[entry]["test-mse"]
        ideal_testStats_yaxis.append(val_to_append_two)

plt.figure()
plt.plot(x_axis, ideal_trainStats_yaxis, label="Training MSE")
plt.plot(x_axis, ideal_testStats_yaxis, label="Test MSE")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Training and Test MSE for the Optimal Learning Rate (0.001)")
plt.legend()
plt.show()

