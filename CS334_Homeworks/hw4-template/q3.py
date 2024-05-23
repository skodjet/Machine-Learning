"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

import argparse
import numpy as np
import perceptron
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression

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


"""3a. Train the appropriate Naive Bayes algorithm against each of the two datasets and assess performance."""
# Train, predict, and calculate the mistakes for multinomial naive bayes.
mnnb = MultinomialNB()
mnnb.fit(xTrain_ct, yTrain_ct)
mnnb_prediction = mnnb.predict(xTest_ct)
mnnb_mistakes = perceptron.calc_mistakes(mnnb_prediction, yTest_ct)

# Train, predict, and calculate the mistakes for Bernoulli naive bayes.
bnnb = BernoulliNB()
bnnb.fit(xTrain_bin, yTrain_bin)
bnnb_prediction = bnnb.predict(xTest_bin)
bnnb_mistakes = perceptron.calc_mistakes(bnnb_prediction, yTest_bin)

print(f"Multinomial Naive Bayes Mistakes: {mnnb_mistakes}\nBernoulli Naive Bayes Mistakes {bnnb_mistakes}")

plt.figure()
x_axis_mn = ["Multinomial Naive Bayes"]
x_axis_bn = ["Bernoulli Naive Bayes"]
plt.scatter(x_axis_mn, mnnb_mistakes, label="Multinomial")
plt.scatter(x_axis_bn, bnnb_mistakes, label="Bernoulli")
plt.xlabel("Classifier Type")
plt.ylabel("Mistakes")
plt.title("Number of Mistakes for Multinomial and Bernoulli Naive Bayes")
plt.ylim(250, 500)
plt.legend()
plt.show()

"""3b. Train a logistic regression model against each of the two constructed datasets and assess performance."""
logreg_ct = LogisticRegression()
logreg_ct.fit(xTrain_ct, yTrain_ct)
logreg_count_prediction = logreg_ct.predict(xTest_ct)
logreg_count_mistakes = perceptron.calc_mistakes(logreg_count_prediction, yTest_ct)

logreg_bin = LogisticRegression()
logreg_bin.fit(xTrain_bin, yTrain_bin)
logreg_bin_prediction = logreg_bin.predict(xTest_bin)
logreg_bin_mistakes = perceptron.calc_mistakes(logreg_bin_prediction, yTest_bin)

print(f"Logistic Regression Count Mistakes: {logreg_count_mistakes}\n"
      f"Logistic Regression Binary Mistakes: {logreg_bin_mistakes}")

plt.figure()
x_axis_ct = ["Count"]
x_axis_bin = ["Binary"]
plt.scatter(x_axis_ct, logreg_count_mistakes, label="Count")
plt.scatter(x_axis_bin, logreg_bin_mistakes, label="Binary")
plt.xlabel("Dataset Type")
plt.ylabel("Mistakes")
plt.title("Number of Mistakes Made by Logistic Regression for Count and Binary Datasets")
plt.ylim(250, 500)
plt.legend()
plt.show()

