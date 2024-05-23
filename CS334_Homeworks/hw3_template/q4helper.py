"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
   WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
   Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

# Helper file for question 4.

import argparse
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import sgdLR
from lr import LinearRegression, file_to_numpy

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

batch_size = np.shape(xTrain)[0]

"""
4a. Choose a variety of batch sizes (include 1, n, and a few more, where the training size is divisible by the number). 
For each batch size, find a reasonable learning rate then run SGD using that batch size. Plot the MSE of the training 
data and the test data as a function of the total time for different batch sizes.
"""


def goodRate(size):
    """Helper method to find a good learning rate"""
    test1 = sgdLR.SgdLR(0.1, size, 10)
    test2 = sgdLR.SgdLR(0.01, size, 10)
    test3 = sgdLR.SgdLR(0.001, size, 10)
    test4 = sgdLR.SgdLR(0.0001, size, 10)

    one = test1.train_predict(xTrain, yTrain, xTest, yTest)
    two = test2.train_predict(xTrain, yTrain, xTest, yTest)
    three = test3.train_predict(xTrain, yTrain, xTest, yTest)
    four = test4.train_predict(xTrain, yTrain, xTest, yTest)

    x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    one_mse = []
    two_mse = []
    three_mse = []
    four_mse = []

    end_of_epoch = np.shape(xTrain)[0] / size

    for ent in one:
        if ent % end_of_epoch == 0:
            val = one[ent]["train-mse"]
            one_mse.append(val)

    for ent in two:
        if ent % end_of_epoch == 0:
            val = two[ent]["train-mse"]
            two_mse.append(val)

    for ent in three:
        if ent % end_of_epoch == 0:
            val = three[ent]["train-mse"]
            three_mse.append(val)

    for ent in four:
        if ent % end_of_epoch == 0:
            val = four[ent]["train-mse"]
            four_mse.append(val)

    # Plot the train MSEs
    plt.figure()
    plt.plot(x_axis, one_mse, label="Rate 0.1")
    plt.plot(x_axis, two_mse, label="Rate 0.01")
    plt.plot(x_axis, three_mse, label="Rate 0.001")
    plt.plot(x_axis, four_mse, label="Rate 0.0001")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.ylim(10, 30)
    plt.legend()
    plt.show()


# Start with batch size 1. From question 3a, a good learning rate was 0.001.
one_model = sgdLR.SgdLR(0.001, 1, 20)

trainStats_one = one_model.train_predict(xTrain, yTrain, xTest, yTest)

one_train_yaxis = []
one_test_yaxis = []
one_time = []

# Get the train MSE, test MSE, and time taken for each epoch
for entry in trainStats_one:
    if entry % batch_size == 0:
        val_to_append_one = trainStats_one[entry]["train-mse"]
        one_train_yaxis.append(val_to_append_one)
        val_to_append_two = trainStats_one[entry]["test-mse"]
        one_test_yaxis.append(val_to_append_two)
        val_to_append_three = trainStats_one[entry]["time"]
        one_time.append(val_to_append_three)


# Train with batch size 10 since it divides the training size
# Find a good learning rate for batch size 10
# goodRate(10)
# From the above function call, a good learning rate seems to be 0.001

ten_model = sgdLR.SgdLR(0.001, 10, 10)

trainStats_ten = ten_model.train_predict(xTrain, yTrain, xTest, yTest)

ten_train_yaxis = []
ten_test_yaxis = []
ten_time = []

# Get the train MSE, test MSE, and time taken for each epoch
end_of_epoch = np.shape(xTrain)[0] / 10
for entry in trainStats_ten:
    if entry % end_of_epoch == 0:
        val_to_append_one = trainStats_ten[entry]["train-mse"]
        ten_train_yaxis.append(val_to_append_one)
        val_to_append_two = trainStats_ten[entry]["test-mse"]
        ten_test_yaxis.append(val_to_append_two)
        val_to_append_three = trainStats_ten[entry]["time"]
        ten_time.append(val_to_append_three)


# Try a batch size of 78, which also divides the training size
# Find a good learning rate for batch size 78
# goodRate(78)
# From the above function call, 0.01 seems to be a good learning rate
seventy_eight_model = sgdLR.SgdLR(0.01, 78, 10)

trainStats_seventy_eight = seventy_eight_model.train_predict(xTrain, yTrain, xTest, yTest)

seventy_eight_train_yaxis = []
seventy_eight_test_yaxis = []
seventy_eight_time = []

# Get the train MSE, test MSE, and time taken for each epoch
end_of_epoch = np.shape(xTrain)[0] / 78
for entry in trainStats_seventy_eight:
    if entry % end_of_epoch == 0:
        val_to_append_one = trainStats_seventy_eight[entry]["train-mse"]
        seventy_eight_train_yaxis.append(val_to_append_one)
        val_to_append_two = trainStats_seventy_eight[entry]["test-mse"]
        seventy_eight_test_yaxis.append(val_to_append_two)
        val_to_append_three = trainStats_seventy_eight[entry]["time"]
        seventy_eight_time.append(val_to_append_three)


# Try a batch size of 559, which also divides the training data
# Find a good learning rate for batch size 559
# goodRate(559)
# From the above function call, 0.001 seems to be a good learning rate
five_five_nine_model = sgdLR.SgdLR(0.001, 78, 10)

trainStats_five_five_nine = five_five_nine_model.train_predict(xTrain, yTrain, xTest, yTest)

five_five_nine_train_yaxis = []
five_five_nine_test_yaxis = []
five_five_nine_time = []

# Get the train MSE, test MSE, and time taken for each epoch
end_of_epoch = np.shape(xTrain)[0] / 559
for entry in trainStats_five_five_nine:
    if entry % end_of_epoch == 0:
        val_to_append_one = trainStats_five_five_nine[entry]["train-mse"]
        five_five_nine_train_yaxis.append(val_to_append_one)
        val_to_append_two = trainStats_five_five_nine[entry]["test-mse"]
        five_five_nine_test_yaxis.append(val_to_append_two)
        val_to_append_three = trainStats_five_five_nine[entry]["time"]
        five_five_nine_time.append(val_to_append_three)


# # Try a batch size of 16770, which is n
# # Find a good learning rate for batch size 16770
# goodRate(16770)
# # From the above function call, 0.0001 seems to be a good learning rate
n_model = sgdLR.SgdLR(0.001, 78, 10)

trainStats_n = n_model.train_predict(xTrain, yTrain, xTest, yTest)

n_train_yaxis = []
n_test_yaxis = []
n_time = []

# Get the train MSE, test MSE, and time taken for each epoch
end_of_epoch = np.shape(xTrain)[0] / 16770
for entry in trainStats_n:
    if entry % end_of_epoch == 0:
        val_to_append_one = trainStats_n[entry]["train-mse"]
        n_train_yaxis.append(val_to_append_one)
        val_to_append_two = trainStats_n[entry]["test-mse"]
        n_test_yaxis.append(val_to_append_two)
        val_to_append_three = trainStats_n[entry]["time"]
        n_time.append(val_to_append_three)


# Plot the resulting graphs.
# From question 2b, the closed form training MSE is 18.82209990233228
# and the test MSE is 16.60635159992399 with a time of 0.00999307632446289.
# Plot these points on the graphs as well.

# Plot the training MSE
plt.figure()
plt.plot(one_time, one_train_yaxis, label="Batch Size 1 Training MSE")
plt.plot(ten_time, ten_train_yaxis, label="Batch Size 10 Training MSE")
plt.plot(seventy_eight_time, seventy_eight_train_yaxis, label="Batch Size 78 Training MSE")
plt.plot(five_five_nine_time, five_five_nine_train_yaxis, label="Batch Size 559 Training MSE")
plt.plot(n_time, n_train_yaxis, label="Batch size n Training MSE")
plt.plot([0.00999307632446289, 1, 2, 3, 4, 5], [18.82209990233228, 18.82209990233228, 18.82209990233228,
                                                18.82209990233228, 18.82209990233228, 18.82209990233228],
         label="Closed Form Solution Training MSE (time 0.009)")
plt.xlabel("Time (seconds)")
plt.ylabel("Training Mean Squared Error (MSE)")
plt.title("Training MSE as a Function of Time for Various Batch Sizes")
plt.xlim(0, 2)
plt.legend()
plt.show()

# Plot the test MSE
plt.figure()
plt.plot(one_time, one_test_yaxis, label="Batch Size 1 Test MSE")
plt.plot(ten_time, ten_test_yaxis, label="Batch Size 10 Test MSE")
plt.plot(seventy_eight_time, seventy_eight_test_yaxis, label="Batch Size 78 Test MSE")
plt.plot(five_five_nine_time, five_five_nine_test_yaxis, label="Batch Size 559 Test MSE")
plt.plot(n_time, n_test_yaxis, label="Batch size n Test MSE")
plt.plot([0.00999307632446289, 1, 2, 3, 4, 5], [16.60635159992399, 16.60635159992399, 16.60635159992399,
                                                16.60635159992399, 16.60635159992399, 16.60635159992399],
         label="Closed Form Solution Test MSE (time 0.009)")
plt.xlabel("Time (seconds)")
plt.ylabel("Test Mean Squared Error (MSE)")
plt.title("Test MSE as a Function of Time for Various Batch Sizes")
plt.xlim(0, 2)
plt.legend()
plt.show()
