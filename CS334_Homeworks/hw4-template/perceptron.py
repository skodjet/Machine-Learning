"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

import argparse
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}

        # Initialize w
        w_length = np.shape(xFeat)[1]
        self.w = np.random.rand(w_length)

        for epoch in range(1, self.mEpoch + 1):
            # Perform the update for each point
            mistakes = 0
            for index, x_vector in enumerate(xFeat):
                update_val = self.sample_update(x_vector, y[index])
                if update_val[1] == 1:
                    self.w = update_val[0]
                    mistakes += 1

            # Shuffle the data
            y = np.reshape(y, (np.shape(y)[0], 1))
            combined = np.hstack((xFeat, y))
            np.random.shuffle(combined)
            xFeat = combined[:, :-1]
            y = combined[:, -1]
            y = np.reshape(y, (np.shape(y)[0]))


            # Update stats at the end of each epoch.
            new_entry_key = epoch
            new_entry_value = mistakes
            stats[new_entry_key] = new_entry_value

            # Exit if there are no mistakes
            if mistakes == 0:
                break

        return stats

    def sample_update(self, xi, yi):
        """
        Given a single sample, give the resulting update to the weights

        Parameters
        ----------
        xi : numpy array of shape 1 x d
            Training sample 
        yi : single value (-1, +1)
            Training label

        Returns
        -------
            wnew: numpy 1d array
                Updated weight value
            mistake: 0/1 
                Was there a mistake made 
        """

        # Predict xFeat
        prediction = None
        x_dot_w = np.dot(self.w, xi)
        if x_dot_w >= 0:
            prediction = 1
        else:
            prediction = -1

        # Don't update weights if prediction is correct.
        if prediction == yi:
            wnew = xi
            mistake = 0
            return wnew, mistake
        # Update the weights if the prediction is incorrect.
        else:
            wnew = None
            mistake = 1
            # Mistake on positive (yi = 1)
            if yi == 1:
                wnew = np.add(self.w, xi)

            # Mistake on negative (yi = -1)
            else:
                wnew = np.subtract(self.w, xi)

        return wnew, mistake

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
            Predicted response per sample
        """
        yHat = []

        # Perform the dot product of x and w.
        x_dot_w = np.matmul(xFeat, self.w)

        # Loop through x_dot_w and check the sign for each of its values.
        for value in x_dot_w:
            if value >= 0:
                yHat.append(1)
            elif value < 0:
                yHat.append(-1)

        return yHat


def transform_y(y):
    """
    Given a numpy 1D array with 0 and 1, transform the y 
    label to be -1 and 1

    Parameters
    ----------
    y : numpy 1-d array with labels of 0 and 1
        The true label.      

    Returns
    -------
    y : numpy 1-d array with labels of -1 and 1
        The true label but 0->1
    """
    output_array = np.zeros(np.shape(y)[0], dtype=int)

    # Update the values in output_array based on the values in y.
    for index in range(0, (np.shape(y)[0])):
        if y[index] == 0:
            output_array[index] = -1
        elif y[index] == 1:
            output_array[index] = 1

    return output_array

def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    err = 0

    # Loop through yHat and check how many values are different form yTrue.
    for index in range(0, np.shape(yHat)[0]):
        if yHat[index] != yTrue[index]:
            err += 1

    return err


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()



def tune_perceptron(trainx, trainy, epochList):
    """
    Tune the preceptron to find the optimal number of epochs

    Parameters
    ----------
    trainx : a nxd numpy array
        The input from either binary / count matrix
    trainy : numpy 1d array of shape n
        The true label.    
    epochList: a list of positive integers
        The epoch list to search over  

    Returns
    -------
    epoch : int
        The optimal number of epochs
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
            new_perceptron = Perceptron(epoch=current_epoch)
            stats = new_perceptron.train(xFeat=train_x, y=train_y)
            y_hat = new_perceptron.predict(test_x)
            errors = calc_mistakes(y_hat, test_y)
            epoch_errors[epoch_index] += errors

    for index, errors in enumerate(epoch_errors):
        if errors < min_errors:
            min_errors = errors
            epoch = epochList[index]

    return epoch


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # Transform yTrain and yTest to 1D numpy arrays.
    yTrain_rows = np.shape(yTrain)[0]
    yTrain = np.reshape(yTrain, yTrain_rows)
    yTest_rows = np.shape(yTest)[0]
    yTest = np.reshape(yTest, yTest_rows)

    # transform to -1 and 1
    yTrain = transform_y(yTrain)
    yTest = transform_y(yTest)

    np.random.seed(args.seed)
    model = Perceptron(args.epoch)

    # Perform padding on xTrain and xTest.
    train_ones = np.ones((np.shape(xTrain)[0], 1))
    test_ones = np.ones((np.shape(xTest)[0], 1))
    xTrain = np.hstack((xTrain, train_ones))
    xTest = np.hstack((xTest, test_ones))

    trainStats = model.train(xTrain, yTrain)
    print(trainStats)

    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))


if __name__ == "__main__":
    main()
