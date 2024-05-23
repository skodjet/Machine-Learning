"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

import argparse
import numpy as np
from sklearn.model_selection import KFold
import perceptron
from matplotlib import pyplot as plt


def tune_perceptron(trainx, trainy, epochList):
    """
    This is a modified version of the tune_perceptron function in perceptron.py.
    This function also returns the errors for each epoch (for graphing purposes).
    """

    # Set k as 5.
    k = 5

    # Perform k-fold cross validation using sklearn
    kfold_cv = KFold(n_splits=k, shuffle=True)

    trainy = np.reshape(trainy, (np.shape(trainy)[0], 1))
    combined = np.hstack((trainx, trainy))
    epoch = 0
    min_errors = 1000000000

    epoch_errors = [0 for val in epochList]

    for trn_indices, tst_indices in kfold_cv.split(combined):

        training_data = combined[trn_indices]
        test_data = combined[tst_indices]
        train_x = training_data[:, :-1]
        train_y = training_data[:, -1]
        test_x = test_data[:, :-1]
        test_y = test_data[:, -1]
        train_y = np.reshape(train_y, (np.shape(train_y)[0]))
        test_y = np.reshape(test_y, (np.shape(test_y)[0]))

        error_sum = 0

        for epoch_index, current_epoch in enumerate(epochList):
            new_perceptron = perceptron.Perceptron(epoch=current_epoch)
            stats = new_perceptron.train(xFeat=train_x, y=train_y)
            y_hat = new_perceptron.predict(test_x)
            errors = perceptron.calc_mistakes(y_hat, test_y)
            epoch_errors[epoch_index] += errors

    for index, errors in enumerate(epoch_errors):
        if errors < min_errors:
            min_errors = errors
            epoch = epochList[index]

    return epoch, epoch_errors


# set up the program to take in arguments from the command line (copied from perceptron.py)
parser = argparse.ArgumentParser()
parser.add_argument("xTrain",
                    help="filename for features of the training data (binary)")
parser.add_argument("yTrain",
                    help="filename for labels associated with training data (binary)")
parser.add_argument("xTest",
                    help="filename for features of the test data (binary)")
parser.add_argument("yTest",
                    help="filename for labels associated with the test data (binary)")
parser.add_argument("xTrainCount",
                    help="filename for features of the training data (count)")
parser.add_argument("yTrainCount",
                    help="filename for labels associated with training data (count)")
parser.add_argument("xTestCount",
                    help="filename for features of the test data (count)")
parser.add_argument("yTestCount",
                    help="filename for labels associated with the test data (count)")
parser.add_argument("--seed", default=334,
                    type=int, help="default seed number")

args = parser.parse_args()
# load the train and test data assumes you'll use numpy
xTrain_bin = perceptron.file_to_numpy(args.xTrain)
yTrain_bin = perceptron.file_to_numpy(args.yTrain)
xTest_bin = perceptron.file_to_numpy(args.xTest)
yTest_bin = perceptron.file_to_numpy(args.yTest)
xTrain_ct = perceptron.file_to_numpy(args.xTrainCount)
yTrain_ct = perceptron.file_to_numpy(args.yTrainCount)
xTest_ct = perceptron.file_to_numpy(args.xTestCount)
yTest_ct = perceptron.file_to_numpy(args.yTestCount)

# Transform yTrain and yTest to 1D numpy arrays.
yTrain_bin_rows = np.shape(yTrain_bin)[0]
yTrain_bin = np.reshape(yTrain_bin, yTrain_bin_rows)
yTest_bin_rows = np.shape(yTest_bin)[0]
yTest_bin = np.reshape(yTest_bin, yTest_bin_rows)
yTrain_ct_rows = np.shape(yTrain_ct)[0]
yTrain_ct = np.reshape(yTrain_ct, yTrain_ct_rows)
yTest_ct_rows = np.shape(yTest_ct)[0]
yTest_ct = np.reshape(yTest_ct, yTest_ct_rows)

# transform to -1 and 1
yTrain_bin = perceptron.transform_y(yTrain_bin)
yTest_bin = perceptron.transform_y(yTest_bin)
yTrain_ct = perceptron.transform_y(yTrain_ct)
yTest_ct = perceptron.transform_y(yTest_ct)

# Perform padding on xTrain and xTest.
bin_train_ones = np.ones((np.shape(xTrain_bin)[0], 1))
bin_test_ones = np.ones((np.shape(xTest_bin)[0], 1))
xTrain_bin = np.hstack((xTrain_bin, bin_train_ones))
xTest_bin = np.hstack((xTest_bin, bin_test_ones))
ct_train_ones = np.ones((np.shape(xTrain_ct)[0], 1))
ct_test_ones = np.ones((np.shape(xTest_ct)[0], 1))
xTrain_ct = np.hstack((xTrain_ct, ct_train_ones))
xTest_ct = np.hstack((xTest_ct, ct_test_ones))

"""2c. Use tune_perceptron to find the optimal number of epochs."""
# 2c. Find the optimal number of epochs for both binary and count datasets.
epoch_list = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
optimal_epochs_bin = tune_perceptron(xTrain_bin, yTrain_bin, epoch_list)
optimal_epochs_ct = tune_perceptron(xTrain_ct, yTrain_ct, epoch_list)
print(f"optimal epochs for binary: {optimal_epochs_bin}\noptimal epochs for count: {optimal_epochs_ct}")
y_axis_bin = optimal_epochs_bin[1]
y_axis_ct = optimal_epochs_ct[1]
# Plot the results
plt.figure()
plt.plot(epoch_list, y_axis_bin, label="binary errors")
plt.plot(epoch_list, y_axis_ct, label="count errors")
plt.xlabel("Number of epochs")
plt.ylabel("Errors")
plt.title("Errors as a Function of Epochs for Binary and Count Datasets")
plt.legend()
plt.show()

# Train a perceptron model using the optimal epoch numbers for binary and count datasets.
# Train binary with 500 epochs.
optimal_bin_model = perceptron.Perceptron(500)
bin_stats = optimal_bin_model.train(xTrain_bin, yTrain_bin)
bin_train_yhat = optimal_bin_model.predict(xTrain_bin)
bin_test_yhat = optimal_bin_model.predict(xTest_bin)
bin_train_mistakes = perceptron.calc_mistakes(bin_train_yhat, yTrain_bin)
bin_test_mistakes = perceptron.calc_mistakes(bin_test_yhat, yTest_bin)
# Train count with 250 epochs.
optimal_ct_model = perceptron.Perceptron(250)
ct_stats = optimal_ct_model.train(xTrain_ct, yTrain_ct)
ct_train_yhat = optimal_ct_model.predict(xTrain_ct)
ct_test_yhat = optimal_ct_model.predict(xTest_ct)
ct_train_mistakes = perceptron.calc_mistakes(ct_train_yhat, yTrain_ct)
ct_test_mistakes = perceptron.calc_mistakes(ct_test_yhat, yTest_ct)

print(f"binary training mistakes: {bin_train_mistakes}\nbinary test mistakes: {bin_test_mistakes}\n"
      f"count training mistakes: {ct_train_mistakes}\ncount test mistakes: {ct_test_mistakes}")

"""
2d. Output the 15 words with the most positive weights, and the 15 words with the most negative weights for the 
binary and count datasets.
"""

# Find the 15 highest and lowest weights and their corresponding indices from the binary and count datasets.
binary_w = optimal_bin_model.w
binary_graph_vals = []
binary_vals = []
for index, w_val in enumerate(binary_w):
    binary_graph_vals.append(w_val)
    binary_vals.append((w_val, index))

binary_graph_vals = binary_graph_vals[1:]
binary_vals = binary_vals[1:]
binary_vals.sort()
binary_bottom_fifteen = binary_vals[:15]
binary_top_fifteen = binary_vals[-15:]
binary_top_fifteen.reverse()

count_w = optimal_ct_model.w
count_graph_vals = []
count_vals = []
for index, w_val in enumerate(count_w):
    count_graph_vals.append(w_val)
    count_vals.append((w_val, index))

count_graph_vals = count_graph_vals[1:]
count_vals = count_vals[1:]
count_vals.sort()
count_bottom_fifteen = count_vals[:15]
count_top_fifteen = count_vals[-15:]
count_top_fifteen.reverse()

print(f"highest binary weights: {binary_top_fifteen}\nlowest binary weights: {binary_bottom_fifteen}\n"
      f"highest count weights: {count_top_fifteen}\nlowest count weights: {count_bottom_fifteen}")

# Plot the results.
plt.figure()
x_axis = ['url', 'onc', 'dear', 'thi', 'on', 'nbsp', 'number', 'i', 'exmh', 'the', 'us', 'at', 'a',
          'hello', 'httpaddr', 'from', 'nextpart', 'hi', 'in', 'begin']
plt.plot(x_axis, binary_graph_vals, label="binary word frequency")
plt.xlabel("Word")
plt.ylabel("Weight")
plt.title("Weights of Each Word in the Binary Dataset")
plt.legend()
plt.show()

plt.figure()
plt.plot(x_axis, count_graph_vals, label="count word frequency")
plt.xlabel("Word")
plt.ylabel("Weight")
plt.title("Weights of Each Word in the Count Dataset")
plt.legend()
plt.show()
