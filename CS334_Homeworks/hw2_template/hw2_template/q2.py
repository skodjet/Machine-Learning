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
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import time

 
def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set. 

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy.nd-array with shape n x d
        Features of the dataset 
    y : numpy.1d-array with shape n 
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """

    start_time = time.time()

    trainAuc = 0
    testAuc = 0
    timeElapsed = 0

    # Create a combined numpy array of xFeat and y
    y_reshaped = np.reshape(y, (np.shape(y)[0], 1))
    combined = np.hstack((xFeat, y_reshaped))

    # Find the number of samples that need to be held out.
    holdout_samples = int(np.shape(xFeat)[0] * testSize)

    # Take holdout_samples number of random rows from xFeat
    random_rows = np.random.choice(combined.shape[0], size=holdout_samples, replace=False)
    holdout_rows = combined[random_rows]
    holdout_xFeat = holdout_rows[:, :-1]
    holdout_y = holdout_rows[:, -1]

    # Remove the rows from combined that appear in holdout_rows.
    new_rows = []
    for row_index in range(combined.shape[0]):
        if row_index not in random_rows:
            new_rows.append(row_index)

    training_rows = combined[new_rows]

    # Break training_rows into xFeat and y
    training_xFeat = training_rows[:, :-1]
    training_y = training_rows[:, -1]

    # Train and test the decision tree.
    model = model.fit(training_xFeat, training_y)
    predicted_training_values = model.predict(training_xFeat)
    predicted_test_values = model.predict(holdout_xFeat)

    # Create the ROC curve for both training and test data
    train_false_pos_rate, train_true_pos_rate, train_threshholds = \
        metrics.roc_curve(training_y, predicted_training_values)

    test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(holdout_y, predicted_test_values)

    # Get the area under the curve for training and test data
    trainAuc = metrics.auc(train_false_pos_rate, train_true_pos_rate)
    testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)

    # Calculate the time taken
    end_time = time.time()
    timeElapsed = end_time - start_time

    return trainAuc, testAuc, timeElapsed


def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. Return the model performance on the training and
    validation (test) set. 


    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy.nd-array with shape n x d
        Features of the dataset 
    y : numpy.1d-array with shape n
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    start_time = time.time()

    trainAuc = 0
    testAuc = 0
    timeElapsed = 0

    # Generate the k-folds using sklearn
    k_fold_cv = KFold(n_splits=k)

    # Generate the AUC for each of the k folds
    train_aucs = []
    test_aucs = []
    for training_indices, test_indices in k_fold_cv.split(xFeat, y):
        # Get the training and test data for this fold
        training_xFeat = xFeat[training_indices]
        training_y = y[training_indices]
        test_xFeat = xFeat[test_indices]
        test_y = y[test_indices]

        # Train the decision tree and predict the training and test data
        model = model.fit(training_xFeat, training_y)
        predicted_training_values = model.predict(training_xFeat)
        predicted_test_values = model.predict(test_xFeat)

        # Create the ROC curve for both training and test data of this fold
        train_false_pos_rate, train_true_pos_rate, train_threshholds = \
            metrics.roc_curve(training_y, predicted_training_values)
        test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(test_y, predicted_test_values)

        # Get the auc for train and test data for this fold
        current_fold_trainAuc = metrics.auc(train_false_pos_rate, train_true_pos_rate)
        current_fold_testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
        train_aucs.append(current_fold_trainAuc)
        test_aucs.append(current_fold_testAuc)

    # Get the average of the train and test aucs
    train_auc_sum = 0
    test_auc_sum = 0
    for auc in range(k):
        train_auc_sum = train_auc_sum + train_aucs[auc]
        test_auc_sum= test_auc_sum + test_aucs[auc]

    trainAuc = train_auc_sum / k
    testAuc = test_auc_sum / k

    # Calculate the time taken
    end_time = time.time()
    timeElapsed = end_time - start_time

    return trainAuc, testAuc, timeElapsed


def mc_cv(model, xFeat, y, testSize, s):
    """
    Evaluate the model using s samples from the
    Monte Carlo cross validation approach where
    for each sample you split xFeat into
    random train and test based on the testSize.
    Returns the model performance on the training and
    test datasets.

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy.nd-array with shape n x d
        Features of the dataset 
    y : numpy.1d-array with shape n
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    start_time = time.time()
    trainAuc = 0
    testAuc = 0
    timeElapsed = 0

    # Use sklearn to create the Monte Carlo cross-validator
    trainSize = 1.0 - testSize
    mc_cross_validator = ShuffleSplit(n_splits=s, test_size=testSize, train_size=trainSize)

    # Generate the AUC for each of the s samples
    train_aucs = []
    test_aucs = []
    for training_indices, test_indices in mc_cross_validator.split(xFeat, y):
        # Get the training and test data for this sample
        training_xFeat = xFeat[training_indices]
        training_y = y[training_indices]
        test_xFeat = xFeat[test_indices]
        test_y = y[test_indices]

        # Train the decision tree and predict the training and test data
        model = model.fit(training_xFeat, training_y)
        predicted_training_values = model.predict(training_xFeat)
        predicted_test_values = model.predict(test_xFeat)

        # Create the ROC curve for both training and test data of this fold
        train_false_pos_rate, train_true_pos_rate, train_threshholds = \
            metrics.roc_curve(training_y, predicted_training_values)
        test_false_pos_rate, test_true_pos_rate, test_threshholds = metrics.roc_curve(test_y, predicted_test_values)

        # Get the auc for train and test data for this fold
        current_fold_trainAuc = metrics.auc(train_false_pos_rate, train_true_pos_rate)
        current_fold_testAuc = metrics.auc(test_false_pos_rate, test_true_pos_rate)
        train_aucs.append(current_fold_trainAuc)
        test_aucs.append(current_fold_testAuc)

    # Get the average of the train and test aucs
    train_auc_sum = 0
    test_auc_sum = 0
    for auc in range(s):
        train_auc_sum = train_auc_sum + train_aucs[auc]
        test_auc_sum = test_auc_sum + test_aucs[auc]

    trainAuc = train_auc_sum / s
    testAuc = test_auc_sum / s

    # Calculate the time taken
    end_time = time.time()
    timeElapsed = end_time - start_time

    return trainAuc, testAuc, timeElapsed


def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : numpy.nd-array with shape nxd
        Training data
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data
    xTest : numpy.nd-array with shape mxd
        Test data
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain,
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest,
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
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
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)

    # use the holdout set with a validation size of 30 of training
    aucTrain1, aucVal1, time1 = holdout(dtClass, xTrain, yTrain, 0.30)
    # use 2-fold validation
    aucTrain2, aucVal2, time2 = kfold_cv(dtClass, xTrain, yTrain, 2)
    # use 5-fold validation
    aucTrain3, aucVal3, time3 = kfold_cv(dtClass, xTrain, yTrain, 5)
    # use 10-fold validation
    aucTrain4, aucVal4, time4 = kfold_cv(dtClass, xTrain, yTrain, 10)
    # use MCCV with 5 samples
    aucTrain5, aucVal5, time5 = mc_cv(dtClass, xTrain, yTrain, 0.30, 5)
    # use MCCV with 10 samples
    aucTrain6, aucVal6, time6 = mc_cv(dtClass, xTrain, yTrain, 0.30, 10)
    # train it using all the data and assess the true value
    trainAuc, testAuc = sktree_train_test(dtClass, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1],
                           ['2-fold', aucTrain2, aucVal2, time2],
                           ['5-fold', aucTrain3, aucVal3, time3],
                           ['10-fold', aucTrain4, aucVal4, time4],
                           ['MCCV w/ 5', aucTrain5, aucVal5, time5],
                           ['MCCV w/ 10', aucTrain6, aucVal6, time6],
                           ['True Test', trainAuc, testAuc, 0]],
                           columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)


if __name__ == "__main__":
    main()
