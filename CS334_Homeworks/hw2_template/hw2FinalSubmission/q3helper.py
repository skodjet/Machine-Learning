"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

"""
A helper file for question 3
"""

import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import q2

# set up the program to take in arguments from the command line (copied from dt.py)
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


# 3. a. Find the optimal hyperparameters for decision tree and knn.

# Find optimal number of neighbors for K-Nearest Neighbors. Create an sklearn K-Nearest Neighbors classifier.

# Plot the area under the curve as a function of number of neighbors (taken from 1dplots.py)
knn_aucVals = []
for neighbors in range(1, 41):
    new_knn_classifier = KNeighborsClassifier(n_neighbors=neighbors)
    new_aucTrain, new_aucVal, new_time = q2.kfold_cv(new_knn_classifier, xTrain, yTrain, 10)
    knn_aucVals.append(new_aucVal)

x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

plt.figure()
plt.plot(x_axis, knn_aucVals, label="AUC")
plt.xlabel("Neighbors")
plt.ylabel("AUC")
plt.title("AUC for Varying Neighbors")
plt.legend()
plt.show()

# Find the optimal max depth and min leaf samples for decision tree.
decision_tree_aucs = []
for depth in range(1, 21):
    new_decision_tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1)
    aucTrain, aucVal, time = q2.kfold_cv(new_decision_tree, xTrain, yTrain, 10)
    decision_tree_aucs.append(aucVal)

x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20]



plt.figure()
plt.plot(x_axis, decision_tree_aucs, label="AUC")
plt.xlabel("Depth")
plt.ylabel("AUC")
plt.title("AUC for Varying Depth (min leaf samples held constant at 1)")
plt.legend()
plt.show()

decision_tree_aucs_two = []
for leafs in range(1, 21):
    new_decision_tree = DecisionTreeClassifier(max_depth=18, min_samples_leaf=leafs)
    aucTrain, aucVal, time = q2.kfold_cv(new_decision_tree, xTrain, yTrain, 10)
    decision_tree_aucs_two.append(aucVal)

plt.figure()
plt.plot(x_axis, decision_tree_aucs_two, label="AUC")
plt.xlabel("Min Leaf Samples")
plt.ylabel("AUC")
plt.title("AUC for Varying Min Leaf Samples (depth held constant at 18)")
plt.legend()
plt.show()


# 3. b.
# Train k-nn on the entire training dataset.
knn_entire_dataset = KNeighborsClassifier(n_neighbors=1)

# Train knn and predict the test dataset (taken from q2.py)
knn_entire_dataset = knn_entire_dataset.fit(xTrain, yTrain)
predicted_test_values = knn_entire_dataset.predict(xTest)

# Create the ROC curve and get the AUC (taken from q2.py)
test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(yTest, predicted_test_values)
testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
accuracy = metrics.accuracy_score(yTest, predicted_test_values)

# print(f"\ntestAuc: {testAuc}\naccuracy: {accuracy}")

# Create a combined numpy array of xFeat and y
y_reshaped = np.reshape(yTrain, (np.shape(yTrain)[0], 1))
combined = np.hstack((xTrain, y_reshaped))

# Randomly remove 5% of the training data by taking a random 95% of the training data (taken from q2.py)
ninety_five_percent = int(np.shape(combined)[0] * 0.95)
random_rows = np.random.choice(combined.shape[0], size=ninety_five_percent, replace=False)
holdout_rows = combined[random_rows]
holdout_xFeat = holdout_rows[:, :-1]
holdout_y = holdout_rows[:, -1]

# Train knn on the smaller training dataset and predict the test dataset
knn_five_off = KNeighborsClassifier(n_neighbors=1)
knn_five_off = knn_five_off.fit(holdout_xFeat, holdout_y)
predicted_test_values = knn_five_off.predict(xTest)

# Create the ROC curve and get the AUC
test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(yTest, predicted_test_values)
testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
accuracy = metrics.accuracy_score(yTest, predicted_test_values)

# print(f"\n5% removed testAuc: {testAuc}\n5% removed accuracy: {accuracy}")

# Randomly remove 10% of the training data by taking a random 90% of the training data (taken from q2.py)
ninety_percent = int(np.shape(combined)[0] * 0.90)
random_rows = np.random.choice(combined.shape[0], size=ninety_percent, replace=False)
holdout_rows = combined[random_rows]
holdout_xFeat = holdout_rows[:, :-1]
holdout_y = holdout_rows[:, -1]

# Train knn on the smaller training dataset and predict the test dataset
knn_ten_off = KNeighborsClassifier(n_neighbors=1)
knn_ten_off = knn_ten_off.fit(holdout_xFeat, holdout_y)
predicted_test_values = knn_ten_off.predict(xTest)

# Create the ROC curve and get the AUC
test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(yTest, predicted_test_values)
testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
accuracy = metrics.accuracy_score(yTest, predicted_test_values)

# print(f"\n10% removed testAuc: {testAuc}\n10% removed accuracy: {accuracy}")

# Randomly remove 20% of the training data by taking a random 80% of the training data (taken from q2.py)
eighty_percent = int(np.shape(combined)[0] * 0.80)
random_rows = np.random.choice(combined.shape[0], size=eighty_percent, replace=False)
holdout_rows = combined[random_rows]
holdout_xFeat = holdout_rows[:, :-1]
holdout_y = holdout_rows[:, -1]

# Train knn on the smaller training dataset and predict the test dataset
knn_twenty_off = KNeighborsClassifier(n_neighbors=1)
knn_twenty_off = knn_twenty_off.fit(holdout_xFeat, holdout_y)
predicted_test_values = knn_twenty_off.predict(xTest)

# Create the ROC curve and get the AUC
test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(yTest, predicted_test_values)
testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
accuracy = metrics.accuracy_score(yTest, predicted_test_values)

# print(f"\n20% removed testAuc: {testAuc}\n20% removed accuracy: {accuracy}")


# 3. c.
# Train a decision tree on the entire dataset.
dtree_entire_dataset = DecisionTreeClassifier(max_depth=18, min_samples_leaf=1)

# Train the decision tree and predict the test dataset (taken from q2.py)
dtree_entire_dataset = dtree_entire_dataset.fit(xTrain, yTrain)
predicted_test_values = dtree_entire_dataset.predict(xTest)

# Create the ROC curve and get the AUC (taken from q2.py)
test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(yTest, predicted_test_values)
testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
accuracy = metrics.accuracy_score(yTest, predicted_test_values)

print(f"\ntestAuc: {testAuc}\naccuracy: {accuracy}")

# Create a combined numpy array of xFeat and y
y_reshaped = np.reshape(yTrain, (np.shape(yTrain)[0], 1))
combined = np.hstack((xTrain, y_reshaped))

# Randomly remove 5% of the training data by taking a random 95% of the training data (taken from q2.py)
ninety_five_percent = int(np.shape(combined)[0] * 0.95)
random_rows = np.random.choice(combined.shape[0], size=ninety_five_percent, replace=False)
holdout_rows = combined[random_rows]
holdout_xFeat = holdout_rows[:, :-1]
holdout_y = holdout_rows[:, -1]

# Train decision tree on the smaller training dataset and predict the test dataset
dtree_five_off = DecisionTreeClassifier(max_depth=18, min_samples_leaf=1)
dtree_five_off = dtree_five_off.fit(holdout_xFeat, holdout_y)
predicted_test_values = dtree_five_off.predict(xTest)

# Create the ROC curve and get the AUC
test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(yTest, predicted_test_values)
testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
accuracy = metrics.accuracy_score(yTest, predicted_test_values)

# print(f"\n5% removed testAuc: {testAuc}\n5% removed accuracy: {accuracy}")

# Randomly remove 10% of the training data by taking a random 90% of the training data (taken from q2.py)
ninety_percent = int(np.shape(combined)[0] * 0.90)
random_rows = np.random.choice(combined.shape[0], size=ninety_percent, replace=False)
holdout_rows = combined[random_rows]
holdout_xFeat = holdout_rows[:, :-1]
holdout_y = holdout_rows[:, -1]

# Train knn on the smaller training dataset and predict the test dataset
dtree_ten_off = DecisionTreeClassifier(max_depth=18, min_samples_leaf=1)
dtree_ten_off = dtree_ten_off.fit(holdout_xFeat, holdout_y)
predicted_test_values = dtree_ten_off.predict(xTest)

# Create the ROC curve and get the AUC
test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(yTest, predicted_test_values)
testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
accuracy = metrics.accuracy_score(yTest, predicted_test_values)

# print(f"\n10% removed testAuc: {testAuc}\n10% removed accuracy: {accuracy}")

# Randomly remove 20% of the training data by taking a random 80% of the training data (taken from q2.py)
eighty_percent = int(np.shape(combined)[0] * 0.80)
random_rows = np.random.choice(combined.shape[0], size=eighty_percent, replace=False)
holdout_rows = combined[random_rows]
holdout_xFeat = holdout_rows[:, :-1]
holdout_y = holdout_rows[:, -1]

# Train knn on the smaller training dataset and predict the test dataset
dtree_twenty_off = DecisionTreeClassifier(max_depth=18, min_samples_leaf=1)
dtree_twenty_off = dtree_twenty_off.fit(holdout_xFeat, holdout_y)
predicted_test_values = dtree_twenty_off.predict(xTest)

# Create the ROC curve and get the AUC
test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(yTest, predicted_test_values)
testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
accuracy = metrics.accuracy_score(yTest, predicted_test_values)

# print(f"\n20% removed testAuc: {testAuc}\n20% removed accuracy: {accuracy}")
