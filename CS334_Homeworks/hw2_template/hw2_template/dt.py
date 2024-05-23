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
from sklearn.metrics import accuracy_score


def calculate_split_score(y, criterion):
    """
    Given a numpy array of labels associated with a node, y,
    calculate the score based on the criterion specified.

    Parameters
    ----------
    y : numpy.1d array with shape n
        Array of labels associated with a node
    criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
    Returns
    -------
    score : float
        The gini or entropy associated with a node
    """

    # Count all the different classes in y
    counts = {}
    rows = y.shape[0]
    for val in y:
        if val not in counts:
            counts[val] = 1
        else:
            counts[val] = counts.get(val) + 1

    if criterion == "entropy":
        entropies = []
        # Calculate p(X=k)log2p(X=k) for each category
        for category in counts.keys():
            current_count = counts.get(category)
            prob = current_count / rows
            current_entropy = prob * np.log2(prob)
            entropies.append(current_entropy)

        # Sum the entropies and return
        score = 0
        for entropy in entropies:
            score += entropy

        score = score * -1

        if score == -0.0:
            score = float(0.0)

        return score

    elif criterion == "gini":
        ginis = []
        # Calculate p(X=k)*(1-p(X=k)) for each category
        for category in counts.keys():
            current_count = counts.get(category)
            prob = current_count / rows
            current_gini = prob * (1 - prob)
            ginis.append(current_gini)

        # Calculate and return the total gini index
        score = 0
        for gini in ginis:
            score += gini

        return score

    return 0


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    decision_tree = {} # The dictionary that holds the current decision tree node
    root = None        # The root node

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int
            Maximum depth of the decision tree
        minLeafSample : int
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape n x d
            Training data
        y : numpy.1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """

        # If xFeat is pure, max_depth has been reached, if there is a leaf with fewer nodes than
        # minLeafSample, or if it is impossible to split xFeat with the minLeafSample constraint, stop.
        entropy = calculate_split_score(y, "entropy")
        rows = np.shape(xFeat)[0]

        root = self

        if entropy == 0.0 or self.maxDepth == 0 or rows <= self.minLeafSample\
                or (rows - self.minLeafSample) < self.minLeafSample:

            counts = {}
            for value in y:
                if value not in counts:
                    counts[value] = 1
                else:
                    counts[value] = counts.get(value) + 1

            majority_class = max(counts, key=counts.get)

            self.decision_tree = {"lchild": None, "rchild": None, "svar": None, "sval": None, "class": majority_class}

            return self

        # Call a recursive method to build the tree
        self.CreateTree(attributes=xFeat, target=y, depth=0, root=self)

        # Return the root node
        return self

    def CreateTree(self, attributes, target, depth, root):
        """
        Recursive method to build the decision tree

        Parameters
            ----------
            attributes: The non-target variables for the current subset of data (initially xFeat)
            target: The target variable for the current subset of data (initially y)
            depth: The depth of the current tree node

            Returns
            -------
            node_dict: A dictionary containing the left and right child nodes,
            the split variable, and the split value for the current node.
            {"lchild": <left child here>, "rchild": <right child here>, "svar": <split variable here>,
             "sval": <split value here>, "class": <majority class here (only for leaf nodes)>}

        """
        node_dict = {}

        # Check the stopping criteria (purity, max depth, and min leaf samples).
        # If the stopping criteria is met, return the majority class of target.
        entropy = calculate_split_score(target, 'entropy')
        samples = np.shape(attributes)[0]
        if entropy == 0.0 or depth >= self.maxDepth or samples <= self.minLeafSample:
            counts = {}
            for value in target:
                if value not in counts:
                    counts[value] = 1
                else:
                    counts[value] = counts.get(value) + 1

            majority_class = max(counts, key=counts.get)

            node_dict = {"lchild": None, "rchild": None, "svar": None, "sval": None, "class": majority_class}
            self.decision_tree = node_dict
            return node_dict

        # Set the minimum and maximum splitting points
        leftmost_split = self.minLeafSample
        rightmost_split = samples - self.minLeafSample

        # If it is impossible to split the array any further without the resulting arrays having less than
        # the minimum leaf samples, return the current node
        if rightmost_split < self.minLeafSample:
            counts = {}
            for value in target:
                if value not in counts:
                    counts[value] = 1
                else:
                    counts[value] = counts.get(value) + 1

            majority_class = max(counts, key=counts.get)

            node_dict = {"lchild": None, "rchild": None, "svar": None, "sval": None, "class": majority_class}
            self.decision_tree = node_dict
            return node_dict

        # Holds the best splitting point for all columns in the format
        overall_lowest_score = float("inf")
        overall_best_split_col = None
        overall_best_split_val = None
        overall_best_split_index = None
        lchild_attr = None
        rchild_attr = None
        lchild_targets = None
        rchild_targets = None

        # Check all possible splitting points for each attribute
        column_counter = 0
        for col in attributes.T:
            col_target = np.stack((col, target), axis=1)

            # Sort in ascending order by the left column
            cols_argsort = col_target[:, 0].argsort()
            col_target = col_target[cols_argsort]

            # Find the best splitting point for the column
            lowest_score = float("inf")
            best_split_val = None
            best_split_index = None
            best_split_attr_l = None
            best_split_attr_r = None
            best_split_y_l = None
            best_split_y_r = None

            for split in range(leftmost_split, rightmost_split + 1):
                left_samples = col_target[:split, 0]
                right_samples = col_target[split:, 0]
                left_targets = col_target[:split, 1]
                right_targets = col_target[split:, 1]

                # Calculate the information gain for the split (if entropy is chosen)
                if self.criterion == 'entropy':
                    left_split_entropy = calculate_split_score(left_targets, 'entropy')
                    right_split_entropy = calculate_split_score(right_targets, 'entropy')
                    left_proportion = np.shape(left_samples)[0] / samples
                    right_proportion = np.shape(right_samples)[0] / samples
                    total_entropy = (left_proportion * left_split_entropy) + \
                                    (right_proportion * right_split_entropy)
                    if total_entropy < lowest_score:
                        lowest_score = total_entropy
                        best_split_val = col_target[split, 0]
                        best_split_index = split
                        best_split_attr_l = left_samples
                        best_split_attr_r = right_samples
                        best_split_y_l = left_targets
                        best_split_y_r = right_targets

                # Calculate the gini index for the split (if gini is chosen)
                elif self.criterion == 'gini':
                    left_split_gini = calculate_split_score(left_targets, 'gini')
                    right_split_gini = calculate_split_score(right_targets, 'gini')
                    left_prop = np.shape(left_samples)[0] / samples
                    right_prop = np.shape(right_samples)[0] / samples
                    total_gini = (left_prop * left_split_gini) + (right_prop * right_split_gini)
                    if total_gini < lowest_score:
                        lowest_score = total_gini
                        best_split_val = col_target[split, 0]
                        best_split_index = split
                        best_split_attr_l = left_samples
                        best_split_attr_r = right_samples
                        best_split_y_l = left_targets
                        best_split_y_r = right_targets

            # Update the overall best split if necessary.
            if lowest_score < overall_lowest_score:
                overall_lowest_score = lowest_score
                overall_best_split_col = column_counter
                overall_best_split_val = best_split_val
                overall_best_split_index = best_split_index

            column_counter = column_counter + 1

        # After finding the best split column and value, sort both attributes and target by the best column
        target_reshape = np.reshape(target, (samples, 1))
        attr_target = np.hstack((attributes, target_reshape))
        attr_target_argsort = attr_target[:, overall_best_split_col].argsort()
        attr_target = attr_target[attr_target_argsort]
        attributes = attr_target[:, :-1]
        target = attr_target[:, -1]

        # Split the dataset into left and right sides
        lchild_attr = attributes[:overall_best_split_index]
        rchild_attr = attributes[overall_best_split_index:]
        lchild_targets = target[:overall_best_split_index]
        rchild_targets = target[overall_best_split_index:]

        # Recurse on the left and right children of the decision tree
        lchild = DecisionTree(self.criterion, self.maxDepth, self.minLeafSample)
        rchild = DecisionTree(self.criterion, self.maxDepth, self.minLeafSample)

        self.decision_tree = {"lchild": lchild, "rchild": rchild, "svar": overall_best_split_col,
                              "sval": overall_best_split_val}

        lchild.CreateTree(lchild_attr, lchild_targets, depth + 1, root)
        rchild.CreateTree(rchild_attr, rchild_targets, depth + 1, root)


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict
        what class the values will have.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape m x d
            The data to predict.

        Returns
        -------
        yHat : numpy.1d array with shape m
            Predicted class label per sample
        """
        yHat = np.array([]) # variable to store the estimated class label
        # For each value in xFeat, loop through the decision tree and predict its class

        predicted_vals = []
        for row in xFeat:
            current_node = self

            while ((current_node.decision_tree.get("lchild") is not None) and
                   (current_node.decision_tree.get("rchild") is not None)):
                current_node_var = current_node.decision_tree.get("svar")
                current_node_val = current_node.decision_tree.get("sval")

                # Go left
                if row[current_node_var] < current_node_val:
                    current_node = current_node.decision_tree.get("lchild")
                # Go right
                elif row[current_node_var] >= current_node_val:
                    current_node = current_node.decision_tree.get("rchild")
                else:
                    break

            # Append the predicted label
            predicted_vals.append(current_node.decision_tree.get("class"))

        output = np.array(predicted_vals)
        yHat = output

        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : numpy.nd-array with shape n x d
        Training data
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data.
    xTest : numpy.nd-array with shape m x d
        Test data
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain, yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest, yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
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

    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
