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


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        # Keep track of time
        start_time = time.time()

        trainStats = {}

        # Train the model on xTrain
        # Generate beta by calculating (x^tx)^-1x^ty
        x_transpose_x = np.matmul(xTrain.T, xTrain)
        inverse = np.linalg.inv(x_transpose_x)
        intermediate_step = np.matmul(inverse, xTrain.T)
        beta_hat = np.matmul(intermediate_step, yTrain)
        self.beta = beta_hat

        # Calculate the residual for both the training and test dataset
        train_prediction = self.predict(xTrain)
        test_prediciton = self.predict(xTest)
        train_residual = np.subtract(yTrain, train_prediction)
        test_residual = np.subtract(yTest, test_prediciton)

        # Calculate the MSE from the residuals
        train_n_val = np.shape(xTrain)[0]
        test_n_val = np.shape(xTest)[0]

        train_in_between_step = np.matmul(train_residual.T, train_residual)
        test_in_between_step = np.matmul(test_residual.T, test_residual)
        train_mse = ((1/train_n_val) * train_in_between_step)[0][0]
        test_mse = ((1/test_n_val) * test_in_between_step)[0][0]

        # Calculate the total time taken
        end_time = time.time()
        total_time = end_time - start_time

        # Fill in trainStats with iteration number: time, train MSE, and test MSE
        iteration_number = 0
        temp_dictionary = {'time': total_time, 'train-mse': train_mse, 'test-mse': test_mse}
        trainStats = {iteration_number: temp_dictionary}

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

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()

    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
