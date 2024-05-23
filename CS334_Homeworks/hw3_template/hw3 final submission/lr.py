"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
   WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
   Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


class LinearRegression(ABC):
    """
    Base Linear Regression class from which all 
    linear regression algorithm implementations are
    subclasses. Can not be instantiated.
    """
    beta = None      # Coefficients

    @abstractmethod
    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        Train the linear regression and predict the values

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : dictionary
            key refers to the batch number
            value is another dictionary with time elapsed and mse
        """
        pass

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

        # yHat = Xbeta (yHat has a size of n x 1)
        # X has a size of n x d
        # beta has a size of d x 1
        n_val = np.shape(xFeat)[0]
        d_val = np.shape(xFeat)[1]

        # Multiply xFeat and beta to get the prediction vector yhat
        yhat_vals = np.matmul(xFeat, self.beta)

        yHat = yhat_vals

        return yHat

    def mse(self, xFeat, y):
        """
        """
        yHat = self.predict(xFeat)
        return mean_squared_error(y, yHat)


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()
