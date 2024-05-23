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
from sklearn.tree import DecisionTreeClassifier


def generate_bootstrap(xTrain, yTrain):
    """
    Helper function to generate a bootstrap sample from the data. Each
    call should generate a different random bootstrap sample!

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of responses associated with training data.

    Returns
    -------
    xBoot : nd-array with shape n x d
        Bootstrap sample from xTrain
    yBoot : 1d array with shape n
        Array of responses associated with xBoot
    oobIdx : 1d array with shape k (which can be 0-(n-1))
        Array containing the out-of-bag sample indices from xTrain 
        such that using this array on xTrain will yield a matrix 
        with only the out-of-bag samples (i.e., xTrain[oobIdx, :]).
    """

    # Combine the data together
    reshaped_yTrain = np.reshape(yTrain, (np.shape(yTrain)[0], 1))
    combined_data = np.hstack((xTrain, reshaped_yTrain))

    # Use numpy random choice to generate n random samples.
    sample_space = np.shape(combined_data)[0]
    bootstrap_samples_indices = np.random.choice(sample_space, sample_space, replace=True)
    random_samples = combined_data[bootstrap_samples_indices]
    xBoot = random_samples[:, :-1]
    yBoot = random_samples[:, -1]

    # Find which samples were not used (out of bag samples)
    oob_rows = []
    n = np.shape(xTrain)[0]

    for row in range(n):
        if row not in bootstrap_samples_indices:
            oob_rows.append(row)

    oobIdx = np.array(oob_rows)

    return xBoot, yBoot, oobIdx


def generate_subfeat(xTrain, maxFeat):
    """
    Helper function to generate a subset of the features from the data. Each
    call is likely to yield different columns (assuming maxFeat is less than
    the original dimension)

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data 
    maxFeat : int
        Maximum number of features to consider in each tree

    Returns
    -------
    xSubfeat : nd-array with shape n x maxFeat
        Subsampled features from xTrain
    featIdx: 1d array with shape maxFeat
        Array containing the subsample indices of features from xTrain
    """

    # Choose maxFeat random features.
    features = np.shape(xTrain)[1]
    random_attributes = np.random.choice(features, maxFeat, replace=False)

    xSubfeat = xTrain[:, random_attributes]
    featIdx = np.array(random_attributes)

    return xSubfeat, featIdx


class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    model = {}         # keeping track of all the models developed, where
                       # the key is the bootstrap sample. The value should be a dictionary
                       # and have 2 keys: "tree" to store the tree built
                       # "feat" to store the corresponding featIdx used in the tree


    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

        self.maxFeat = maxFeat
        self.oob_dict = {}

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """

        stats = {}

        self.model = {}

        for epoch in range(1, self.nest + 1):
            # Draw a bootstrap sample of size n from the training data.
            x_boot, y_boot, oob_indices = generate_bootstrap(xFeat, y)

            # Grow a random-forest decision tree on a random subspace of features.

            x_subfeat, feature_indices = generate_subfeat(x_boot, self.maxFeat)
            new_tree = DecisionTreeClassifier(criterion=self.criterion, min_samples_leaf=self.minLeafSample,
                                              max_depth=self.maxDepth)

            new_tree.fit(x_subfeat, y_boot)

            # Keep track of the tree and bootstrap samples in the "model" variable.
            bootstrap_number = epoch - 1
            sub_dictionary = {"tree": new_tree, "feat": feature_indices}
            self.model[bootstrap_number] = sub_dictionary
            self.oob_dict[bootstrap_number] = oob_indices

            # Keep track of the labels for each row
            y_reshape = np.reshape(y, (np.shape(y)[0], 1))
            combined_data = np.hstack((xFeat, y_reshape))

            # Keep track of the out of bag errors
            errors = 0

            # Predict the responses
            for row in combined_data:
                row_label = row[-1]

                row_predictions = []
                for bootstrap_num, sub_dict in self.model.items():
                    tree = sub_dict.get("tree")
                    feat = np.asarray(sub_dict.get("feat"))
                    oob_indices = self.oob_dict[bootstrap_num]
                    oob_rows = xFeat[oob_indices]
                    row_vals = row[feat]
                    oob_vals = oob_rows[:, feat]

                    # If the row is out of bag for this tree, predict the response.
                    if row_vals in oob_vals:
                        prediction_shape = np.reshape(row_vals, (1, np.shape(row_vals)[0]))
                        prediction = tree.predict(prediction_shape)[0]
                        row_predictions.append(prediction)

                # Perform a majority vote
                prediction_dict = {}
                for prediction_val in row_predictions:

                    if prediction_val not in prediction_dict.keys():
                        prediction_dict[prediction_val] = 1
                    else:
                        prediction_dict[prediction_val] = prediction_dict.get(prediction_val) + 1

                majority_class = None
                max_val = float('-inf')
                for key, value in prediction_dict.items():
                    if value > max_val:
                        max_val = value
                        majority_class = key

                # Determine if the majority vote is the correct prediction
                correct_prediction = row_label
                if majority_class != correct_prediction:
                    errors += 1

            stats[epoch] = errors

        return stats

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

        votes = {}

        # This is a classification problem, so the prediction is the
        # majority vote of all trees built
        for row in xFeat:
            row_predictions = []

            for boostrap_num, sub_dict in self.model.items():
                tree = sub_dict.get("tree")
                feat = np.asarray(sub_dict.get("feat"))
                column_subset = row[feat]

                prediction_shape = np.reshape(column_subset, (1, np.shape(column_subset)[0]))
                prediction = tree.predict(prediction_shape)[0]

                row_predictions.append(prediction)

            # Perform a majority vote
            prediction_dict = {}
            for prediction_val in row_predictions:
                if prediction_val not in prediction_dict.keys():
                    prediction_dict[prediction_val] = 1
                else:
                    prediction_dict[prediction_val] = prediction_dict.get(prediction_val) + 1

            majority_class = None
            max_val = float('-inf')
            for key, value in prediction_dict.items():
                if value > max_val:
                    max_val = value
                    majority_class = key

            yHat.append(majority_class)

        return yHat


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


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

    np.random.seed(args.seed)

    model = RandomForest(nest=args.epoch, maxFeat=3, criterion='entropy', maxDepth=5, minLeafSample=10)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)



if __name__ == "__main__":
    main()