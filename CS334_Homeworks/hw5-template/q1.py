"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
   WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
   Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def normalize_feat(xTrain, xTest):
    """
    Normalize the featyres of the wine quality dataset (where applicable).

    Parameters
    ----------
    xTrain
    xTest

    Returns
    -------
    normalized training and test data
    """

    normalizer = StandardScaler()
    xTrain_transformed = normalizer.fit_transform(xTrain)
    xTest_transformed = normalizer.transform(xTest)

    return xTrain_transformed, xTest_transformed


def unreg_log(xTrain, yTrain, xTest, yTest):
    """
    Train an unregularized logistic regression model on the wine dataset.
    Then, predict the probabilities on the test data and calculate ROC.

    Parameters
    ----------
    xTrain
    yTrain
    xTest
    yTest

    Returns
    -------
    False positive rates and true positive rates, along with the auc value.
    """

    # Create an unregularized logistic regression model.
    log_reg = LogisticRegression(penalty=None)

    # Train the model.
    log_reg.fit(xTrain, yTrain)

    # Predict the probabilities of the test data.
    y_predictions = log_reg.predict(xTest)

    # Calculate the ROC.
    false_positive, true_positive, thresholds = roc_curve(yTest, y_predictions)

    # Calculate the AUC
    auc_val = auc(false_positive, true_positive)

    return false_positive, true_positive, auc_val


def run_pca(xTrain, xTest):
    """
    Runs PCA on the normalized training dataset.

    Parameters
    ----------
    xTrain
    xTest

    Returns
    -------
    Transformed xTrain, transformed xTest, and PCA model components.
    """

    p_component_analysis = PCA()
    p_component_analysis.fit_transform(xTrain)

    # Get which original features are important, and how much of the variance they explain.
    important_components = p_component_analysis.components_
    explained_variance = p_component_analysis.explained_variance_ratio_

    # Calculate how many principal components are needed to capture at least
    # 95% of the variance in the original data.
    total_explained_variance = 0
    components_needed = 0
    for variance_val in explained_variance:
        if total_explained_variance >= 0.95:
            break
        else:
            total_explained_variance += variance_val
            components_needed += 1

    p_components_transform = PCA(n_components=components_needed)
    xTrain_transformed = p_components_transform.fit_transform(xTrain)
    xTest_transformed = p_components_transform.transform(xTest)

    new_important_components = important_components[:components_needed]

    return xTrain_transformed, xTest_transformed, new_important_components


def main():
    # Set up the program to take in datasets.
    # (Loosely taken from homework 1 question 4).
    parse = argparse.ArgumentParser()
    parse.add_argument("--xTrain",
                       default="q1xTrain.csv")
    parse.add_argument("--yTrain",
                      default="q1yTrain.csv")
    parse.add_argument("--xTest",
                       default="q1xTest.csv")
    parse.add_argument("--yTest",
                       default="q1yTest.csv")

    arguments = parse.parse_args()

    # Load the wine dataset
    xTrain = pd.read_csv(arguments.xTrain)
    xTest = pd.read_csv(arguments.xTest)
    yTrain = pd.read_csv(arguments.yTrain)
    yTest = pd.read_csv(arguments.yTest)

    xTrain = xTrain.to_numpy()
    xTest = xTest.to_numpy()
    yTrain = yTrain.to_numpy()
    yTest = yTest.to_numpy()

    # Reshape yTrain and yTest (sklearn complains unless I do this).
    yTrain = np.reshape(yTrain, (np.shape(yTrain)[0]))
    yTest = np.reshape(yTest, (np.shape(yTest)[0]))

    # 1a. Normalize the data.
    xTrain, xTest = normalize_feat(xTrain, xTest)

    # 1b. Return the true positive rates, false positive rates, and AUC value
    # of a logistic regression model.
    true_pos, false_pos, auc_val = unreg_log(xTrain, yTrain, xTest, yTest)

    # 1c. Run PCA on the normalized training dataset.
    transformed_xTrain, transformed_xTest, pca_components = run_pca(xTrain, xTest)

    # 1d. Train logistic regression models on both the normalized and PCA datasets.
    #     Plot the ROC curves for both models.
    normalized_fpr, normalized_tpr, normalized_auc = unreg_log(xTrain, yTrain, xTest, yTest)
    pca_fpr, pca_tpr, pca_auc = unreg_log(transformed_xTrain, yTrain, transformed_xTest, yTest)

    plt.figure()
    plt.plot(normalized_fpr, normalized_tpr, label="normalized dataset")
    plt.plot(pca_fpr, pca_tpr, label="PCA dataset")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Normalized and PCA Datasets")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
