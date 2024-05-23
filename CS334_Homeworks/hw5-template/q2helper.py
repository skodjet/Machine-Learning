"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
   WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
   Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

# A helper file for questions 2b and 2c.
import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import rf
from matplotlib import pyplot as plt


def main():
    """
    Main file to run from the command line. Taken from rf.py
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
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = rf.file_to_numpy(args.xTrain)
    yTrain = rf.file_to_numpy(args.yTrain)
    xTest = rf.file_to_numpy(args.xTest)
    yTest = rf.file_to_numpy(args.yTest)

    np.random.seed(args.seed)

    # 2b. Find the best parameters on the wine quality training dataset
    # based on classification error.
    tree_values = [1, 2, 5, 10, 20, 50, 100]

    for trees in tree_values:
        current_tree_test_errors = 0
        current_tree_training_errors = 0
        model = rf.RandomForest(nest=trees, maxFeat=3, criterion='entropy', maxDepth=5, minLeafSample=10)
        trainStats = model.train(xTrain, yTrain)
        print(trainStats)
        yHatTraining = model.predict(xTrain)
        yHatTest = model.predict(xTest)

        # Calculate the training error for the model
        for index, val in enumerate(yHatTraining):
            if val != yTrain[index]:
                current_tree_training_errors += 1

        # Calculate the test error for the model
        for index, val in enumerate(yHatTest):
            if val != yTest[index]:
                current_tree_test_errors += 1

        print(f"training error for {trees} trees = {current_tree_training_errors}")
        print(f"test error for {trees} trees = {current_tree_test_errors}")

    # Results for the tree values
    training_errors = [141, 133, 128, 139, 146, 148, 147]
    test_errors = [68, 60, 65, 61, 64, 64, 65]


    # Plot the results.
    plt.figure()
    plt.plot(tree_values, training_errors, label="training error")
    plt.xlabel("Number of Trees")
    plt.ylabel("Training Error")
    plt.title("Training Error for Various Numbers of Trees in Random Forest")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(tree_values, test_errors, label="test error")
    plt.xlabel("Number of Trees")
    plt.ylabel("Test Error")
    plt.title("Test Error for Various Numbers of Trees in Random Forest")
    plt.legend()
    plt.show()


    # Next, find the optimal number of features with the optimal number of trees (20).
    feature_values = [1, 2, 4, 6, 8, 10, 11]

    for feature_num in feature_values:
        current_tree_test_errors = 0
        current_tree_training_errors = 0
        model = rf.RandomForest(nest=20, maxFeat=feature_num, criterion='entropy', maxDepth=5, minLeafSample=10)
        trainStats = model.train(xTrain, yTrain)
        print(trainStats)
        yHatTraining = model.predict(xTrain)
        yHatTest = model.predict(xTest)

        # Calculate the training error for the model
        for index, val in enumerate(yHatTraining):
            if val != yTrain[index]:
                current_tree_training_errors += 1

            # Calculate the test error for the model
        for index, val in enumerate(yHatTest):
            if val != yTest[index]:
                current_tree_test_errors += 1

        print(f"training error for {feature_num} features = {current_tree_training_errors}")
        print(f"test error for {feature_num} features = {current_tree_test_errors}")

    # Values obtained from the above loop
    training_errors = [152, 152, 134, 132, 103, 104, 91]
    test_errors = [65, 65, 60, 62, 63, 56, 56]

    # Plot the results.
    plt.figure()
    plt.plot(feature_values, training_errors, label="training error")
    plt.xlabel("Number of Features")
    plt.ylabel("Training Error")
    plt.title("Training Error for Various Numbers of Features With 20 Trees")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(feature_values, test_errors, label="test error")
    plt.xlabel("Number of Features")
    plt.ylabel("Test Error")
    plt.title("Test Error for Various Numbers of Features With 20 Trees")
    plt.legend()
    plt.show()


    # Find the optimal max depth for each tree with the optimal number of trees (20)
    # and the optimal number of features (4)
    depth_values = [1, 2, 3, 4, 5, 6, 7]

    for depth_val in depth_values:
        current_tree_test_errors = 0
        current_tree_training_errors = 0
        model = rf.RandomForest(nest=20, maxFeat=4, criterion='entropy', maxDepth=depth_val, minLeafSample=10)
        trainStats = model.train(xTrain, yTrain)
        print(trainStats)
        yHatTraining = model.predict(xTrain)
        yHatTest = model.predict(xTest)

        # Calculate the training error for the model
        for index, val in enumerate(yHatTraining):
            if val != yTrain[index]:
                current_tree_training_errors += 1

        # Calculate the test error for the model
        for index, val in enumerate(yHatTest):
            if val != yTest[index]:
                current_tree_test_errors += 1

        print(f"training error for {depth_val} depth = {current_tree_training_errors}")
        print(f"test error for {depth_val} depth = {current_tree_test_errors}")

    # Values obtained from the above
    training_errors = [152, 152, 143, 141, 135, 125, 120]
    test_errors = [65, 65, 64, 65, 60, 59, 62]

    # Plot the results.
    plt.figure()
    plt.plot(depth_values, training_errors, label="training error")
    plt.xlabel("Depth")
    plt.ylabel("Training Error")
    plt.title("Training Error for Various Depth Trees in Random Forest (20 Trees and 4 Features)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(depth_values, test_errors, label="test error")
    plt.xlabel("Depth")
    plt.ylabel("Test Error")
    plt.title("Test Error for Various Depth Trees in Random Forest (20 Trees and 4 Features)")
    plt.legend()
    plt.show()

    # Find the optimal min leaf sample for each tree with the optimal number of trees (20),
    # the optimal number of features (4), and the optimal depth
    min_leaf_values = [1, 5, 10, 20, 50, 100, 200]

    for min_leaf_val in min_leaf_values:
        current_tree_test_errors = 0
        current_tree_training_errors = 0
        model = rf.RandomForest(nest=20, maxFeat=4, criterion='entropy', maxDepth=6, minLeafSample=min_leaf_val)
        trainStats = model.train(xTrain, yTrain)
        print(trainStats)
        yHatTraining = model.predict(xTrain)
        yHatTest = model.predict(xTest)

        # Calculate the training error for the model
        for index, val in enumerate(yHatTraining):
            if val != yTrain[index]:
                current_tree_training_errors += 1

        # Calculate the test error for the model
        for index, val in enumerate(yHatTest):
            if val != yTest[index]:
                current_tree_test_errors += 1

        print(f"training error for {min_leaf_val} minimum leaf samples = {current_tree_training_errors}")
        print(f"test error for {min_leaf_val} minimum leaf samples = {current_tree_test_errors}")

    training_errors = [111, 124, 118, 125, 133, 152, 152]
    test_errors = [59, 60, 61, 62, 61, 65, 65]

    # Plot the results.
    plt.figure()
    plt.plot(min_leaf_values, training_errors, label="training error")
    plt.xlabel("Min Leaves")
    plt.ylabel("Training Error")
    plt.title("Training Error for Various Min Leaf Values in Random Forest (20 Trees, 4 Features, and 6 Max Depth)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(min_leaf_values, test_errors, label="test error")
    plt.xlabel("Min Leaves")
    plt.ylabel("Test Error")
    plt.title("Test Error for Various Min Leaf Values in Random Forest (20 Trees, 4 Features, and 6 Max Depth")
    plt.legend()
    plt.show()

    # 2c. Using the optimal parameters, how well does random forest perform on
    # the test data?
    optimal_model = rf.RandomForest(nest=20, maxFeat=4, criterion='entropy', maxDepth=6, minLeafSample=50)
    trainStats = optimal_model.train(xTrain, yTrain)
    print(trainStats)
    yHatTest = optimal_model.predict(xTest)

    optimal_model_test_errors = 0
    # Calculate the test error for the model
    for index, val in enumerate(yHatTest):
        if val != yTest[index]:
            optimal_model_test_errors += 1

    print(f"test error for optimal model = {optimal_model_test_errors}")



if __name__ == "__main__":
    main()