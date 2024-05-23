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

from lr import LinearRegression, file_to_numpy


def grad_pt(beta, xi, yi):
    """
    Calculate the gradient for a mini-batch sample.

    Parameters
    ----------
    beta : 1d array with shape d
    xi : 2d numpy array with shape b x d
        Batch training data
    yi: 2d array with shape bx1
        Array of responses associated with training data.

    Returns
    -------
        grad : 1d array with shape d
    """

    # The negative gradient with respect to a signle sample is:
    # (xi)^T(yi - xibeta)
    xi_beta = np.matmul(xi, beta)
    step_two = np.subtract(yi, xi_beta)
    negative_gradient = np.matmul(xi.T, step_two)

    return negative_gradient


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """

        # Keep track of time
        start_time = time.time()

        trainStats = {}

        train_combined = np.hstack((xTrain, yTrain))

        # For some reason, unless I use 1D arrays I can't get the autograder to run...
        yTrain = np.reshape(yTrain, (np.shape(yTrain)[0]))
        yTest = np.reshape(yTest, (np.shape(yTest)[0]))

        # Keep track of the current key number for train stats.
        key_number = 0

        # Randomly initialize the beta variables
        d_val = np.shape(xTrain)[1]
        self.beta = np.random.rand(d_val)

        for epoch in range(self.mEpoch):
            # Randomly shuffle the training data
            np.random.shuffle(train_combined)

            # Calculate the number of batches.
            n_val = np.shape(train_combined)[0]
            batches = int(n_val / self.bs)

            start_index = 0
            end_index = self.bs
            for batch in range(batches):

                # For each batch, compute the gradients for the samples in the current batch
                if end_index < n_val:
                    current_batch = train_combined[start_index:end_index]
                else:
                    current_batch = train_combined[start_index:n_val - 1]

                xi_vals = current_batch[:, :-1]
                yi_vals = current_batch[:, -1]

                gradient = grad_pt(self.beta, xi_vals, yi_vals)

                # Take the average of the gradients
                avg_gradient = np.average(gradient)

                # Calculate the updated values using the given learning rate
                step_size_gradient = self.lr * avg_gradient
                self.beta = self.beta + step_size_gradient

                # Find the predictions for train and test
                train_prediction = self.predict(xTrain)
                test_prediciton = self.predict(xTest)

                # Calculate the residuals for train and test
                train_residual = np.subtract(yTrain, train_prediction)
                test_residual = np.subtract(yTest, test_prediciton)

                # Calculate the train and test MSE
                train_n_val = np.shape(xTrain)[0]
                test_n_val = np.shape(xTest)[0]

                train_mse = np.mean(train_residual ** 2)
                test_mse = np.mean(test_residual ** 2)

                # Get the time for this batch
                end_time = time.time()
                total_time = end_time - start_time

                # Update the dictionary

                value_information = {'time': total_time, 'train-mse': train_mse, 'test-mse': test_mse}
                trainStats[key_number] = value_information
                key_number += 1

                # Update the indices for the next batch
                start_index = end_index
                end_index += self.bs

        return trainStats


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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
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
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

