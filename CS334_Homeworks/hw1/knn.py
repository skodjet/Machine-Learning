"""
THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje

I collaborated with the following classmates for this homework:
None
"""

import argparse
import numpy as np
import pandas as pd


class Knn(object):
    k = 0    # number of neighbors to use
    training_data = None
    training_labels = None
    training_rows = None
    training_columns = None
    test_data = None
    test_rows = None
    test_columns = None

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # Memorize the training set (KNN doesn't do any work in the training phase)
        self.training_data = np.array(xFeat)
        self.training_labels = np.array(y)
        # Memorize number of columns to help with testing phase
        self.training_rows = self.training_data.shape[0]
        self.training_columns = self.training_data.shape[1]

        return self

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label

        self.test_data = np.array(xFeat)
        self.test_rows = self.test_data.shape[0]
        self.test_columns = self.test_data.shape[1]

        # For each test point, find the Euclidean Distance between the test point
        # and every training point.
        for test_row in self.test_data:
            distances = []
            for training_row in self.training_data:
                # Find the Euclidean distance for each data point
                sum_of_squared_differences = np.sum((training_row - test_row) ** 2)
                euc_distance = np.sqrt(sum_of_squared_differences)
                distances.append(euc_distance)

            # Find the k-nearest neighbors for the test point
            sorted_indices = np.argsort(distances)
            k_nearest_indices = sorted_indices[:self.k]
            # Find the labels for the k-nearest neighbors
            k_nearest_labels = []
            for j in range(self.k):
                k_nearest_labels.append(self.training_labels[k_nearest_indices[j]])

            # Find which label appears the most often within k_nearest_labels
            # seen_labels has format label:count to keep track of labels already seen previously.
            seen_labels = {}
            for j in range(len(k_nearest_labels)):
                if k_nearest_labels[j] in seen_labels:
                    seen_labels[k_nearest_labels[j]] = seen_labels[k_nearest_labels[j]] + 1
                else:
                    seen_labels[k_nearest_labels[j]] = 1

            prediction = max(seen_labels, key=seen_labels.get)
            yHat.append(prediction)

        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    true_count = 0
    for i in range(len(yHat)):
        if yHat[i] == yTrue[i]:
            true_count = true_count + 1

    acc = true_count / len(yHat)
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
