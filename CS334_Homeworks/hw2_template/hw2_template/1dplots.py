"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

"""
Helper file to create plots for question 1d in homework 2
"""

import dt
import pandas as pd
import numpy as np
import argparse
from matplotlib import pyplot as plt

# setup (copied from dt.py)
# set up the program to take in arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--xTrain", default="q4xTrain.csv", help="filename for features of the training data")
parser.add_argument("--yTrain",
                    default="q4yTrain.csv",
                    help="filename for labels associated with training data")
parser.add_argument("--xTest",
                    default="q4xTest.csv",
                    help="filename for features of the test data")
parser.add_argument("--yTest",
                    default="q4yTest.csv",
                    help="filename for labels associated with the test data")

args = parser.parse_args()
# load the train and test data
xTrain = pd.read_csv(args.xTrain).to_numpy()
yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
xTest = pd.read_csv(args.xTest).to_numpy()
yTest = pd.read_csv(args.yTest).to_numpy().flatten()

# Create a plot for train and test as a function of max depth (holding min leaf samples constant at 1)
# Get the training and test accuracy for max depth 1-20
train_accuracy_depth = []
test_accuracy_depth = []
for depth in range(1, 21):

    # Taken from dt_train_test() in dt.py
    train_acc_tester = dt.DecisionTree('entropy', maxDepth=depth, minLeafSample=1)
    train_acc_tester.train(xTrain, yTrain)

    # Predict labels for both xTrain and xTest
    yhat_training = train_acc_tester.predict(xTrain)
    yhat_test = train_acc_tester.predict(xTest)

    # Get the accuracy for xTrain and xTest
    train_accuracy = dt.accuracy_score(yTrain, yhat_training)
    test_accuracy = dt.accuracy_score(yTest, yhat_test)

    train_accuracy_depth.append(train_accuracy)
    test_accuracy_depth.append(test_accuracy)


# Create an array for the x-axis of the plot
x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20]


print("Training accuracy for varying depth: ", train_accuracy_depth)
print("Test accuracy for varying depth : ", test_accuracy_depth)

# Plot the depth plot
plt.figure()
plt.plot(x_axis, train_accuracy_depth, label="training accuracy", linewidth=2)
plt.plot(x_axis, test_accuracy_depth, label="test accuracy", linewidth=2)
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy (Entropy) for Various Depth (Min Leaf Samples Fixed at 1)")
plt.legend()
plt.show()


# Create a plot for train and test as a function of minimum leaf samples (holding max depth constant at 5)
train_accuracy_min_leaf = []
test_accuracy_min_leaf = []

# Get the training and test accuracy for min leaf samples 1-20
for min_leaf in range(1, 21):
    # Taken from dt_train_test() in dt.py
    train_acc_tester = dt.DecisionTree('entropy', maxDepth=5, minLeafSample=min_leaf)
    train_acc_tester.train(xTrain, yTrain)

    # Predict labels for xTrain and xTest
    yhat_training = train_acc_tester.predict(xTrain)
    yhat_test = train_acc_tester.predict(xTest)

    # Get the accuracy for xTrain and xTest
    train_accuracy = dt.accuracy_score(yTrain, yhat_training)
    test_accuracy = dt.accuracy_score(yTest, yhat_test)

    train_accuracy_min_leaf.append(train_accuracy)
    test_accuracy_min_leaf.append(test_accuracy)

print("Training accuracy for varying min leaf samples: ", train_accuracy_min_leaf)
print("Test accuracy for varying min leaf samples: ", test_accuracy_min_leaf)

# Plot the min leaf plot
plt.figure()
plt.plot(x_axis, train_accuracy_min_leaf, label="training accuracy", linewidth=2)
plt.plot(x_axis, test_accuracy_min_leaf, label="test accuracy", linewidth=2)
plt.xlabel("Minimum Leaf Samples")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy (Entropy) for Various Min Leaf Samples (Depth Fixed at 5)")
plt.legend()
plt.show()


# Create depth and min leaf sample plots using gini index as well
# Get the training and test accuracy for max depth 1-20
train_accuracy_depth = []
test_accuracy_depth = []
for depth in range(1, 21):

    # Taken from dt_train_test() in dt.py
    train_acc_tester = dt.DecisionTree('gini', maxDepth=depth, minLeafSample=1)
    train_acc_tester.train(xTrain, yTrain)

    # Predict labels for both xTrain and xTest
    yhat_training = train_acc_tester.predict(xTrain)
    yhat_test = train_acc_tester.predict(xTest)

    # Get the accuracy for xTrain and xTest
    train_accuracy = dt.accuracy_score(yTrain, yhat_training)
    test_accuracy = dt.accuracy_score(yTest, yhat_test)

    train_accuracy_depth.append(train_accuracy)
    test_accuracy_depth.append(test_accuracy)


# Create an array for the x-axis of the plot
x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20]


print("Training accuracy for varying depth: ", train_accuracy_depth)
print("Test accuracy for varying depth : ", test_accuracy_depth)

# Plot the depth plot
plt.figure()
plt.plot(x_axis, train_accuracy_depth, label="training accuracy", linewidth=2)
plt.plot(x_axis, test_accuracy_depth, label="test accuracy", linewidth=2)
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy (gini) for Various Depth (Min Leaf Samples Fixed at 1)")
plt.legend()
plt.show()


# Create a plot for train and test as a function of minimum leaf samples (holding max depth constant at 5)
train_accuracy_min_leaf = []
test_accuracy_min_leaf = []

# Get the training and test accuracy for min leaf samples 1-20
for min_leaf in range(1, 21):
    # Taken from dt_train_test() in dt.py
    train_acc_tester = dt.DecisionTree('gini', maxDepth=5, minLeafSample=min_leaf)
    train_acc_tester.train(xTrain, yTrain)

    # Predict labels for xTrain and xTest
    yhat_training = train_acc_tester.predict(xTrain)
    yhat_test = train_acc_tester.predict(xTest)

    # Get the accuracy for xTrain and xTest
    train_accuracy = dt.accuracy_score(yTrain, yhat_training)
    test_accuracy = dt.accuracy_score(yTest, yhat_test)

    train_accuracy_min_leaf.append(train_accuracy)
    test_accuracy_min_leaf.append(test_accuracy)

print("Training accuracy for varying min leaf samples: ", train_accuracy_min_leaf)
print("Test accuracy for varying min leaf samples: ", test_accuracy_min_leaf)

# Plot the min leaf plot
plt.figure()
plt.plot(x_axis, train_accuracy_min_leaf, label="training accuracy", linewidth=2)
plt.plot(x_axis, test_accuracy_min_leaf, label="test accuracy", linewidth=2)
plt.xlabel("Minimum Leaf Samples")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy (gini) for Various Min Leaf Samples (Depth Fixed at 5)")
plt.legend()
plt.show()

