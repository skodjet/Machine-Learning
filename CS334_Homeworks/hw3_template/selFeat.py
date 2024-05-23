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
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from seaborn import heatmap
from seaborn import set
from matplotlib import pyplot as plt
from matplotlib import rcParams


def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """

    """
    Drop the "date" column and add "weekend" and "morning" columns, 
    which show if the data point occurred on a weekend, and whether or 
    not the data was taken during the morning (12:00 AM - 11:59 AM)
    
    1 = yes
    0 = no
    """

    # Extract the day of the week from the "date" column
    date = df['date']
    df['date'] = pd.to_datetime(date)
    day_of_week = df['date'].dt.dayofweek
    df['weekend'] = day_of_week

    # Determine if each observation is on a weekday or a weekend and adjust its value accordingly
    for i, observation in df.iterrows():
        if observation['weekend'] == 5 or observation['weekend'] == 6:
            df.at[i, 'weekend'] = 1
        else:
            df.at[i, 'weekend'] = 0

    # Extract the time of day from the "date" column
    hours = df['date'].dt.hour
    df['morning'] = hours

    # Determine if each observation is in the morning or not, and adjust its value accordingly
    for i, observation in df.iterrows():
        if observation['morning'] < 12:
            df.at[i, 'morning'] = 1
        else:
            df.at[i, 'morning'] = 0

    df = df.drop(columns=['date'])

    return df


def cal_corr(df):
    """
    Given a pandas dataframe (include the target variable at the last column), 
    calculate the correlation matrix (compute pairwise correlation of columns)

    Parameters
    ----------
    df : pandas dataframe
        Training or test data (with target variable)
    Returns
    -------
    corrMat : pandas dataframe
        Correlation matrix
    """
    # calculate the correlation matrix and perform the heatmap
    corrMat = pd.DataFrame.corr(df, method='pearson')

    return corrMat


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """

    # Drop unneeded attributes
    df = df.drop(columns=['RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'RH_6', 'T7', 'RH_7', 'T8',
                          'RH_8', 'T9', 'RH_9', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint',
                          'weekend', 'morning'])

    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """

    # Save the columns of trainDF and testDF
    trainDF_cols = trainDF.columns
    testDF_cols = testDF.columns

    # Perform standardization of the data
    standard_scaler = StandardScaler()

    # Fit the standard scaler to the training dataset, then fit both datasets using those parameters.
    standard_scaler.fit(trainDF)
    trainDF_numpy = standard_scaler.transform(trainDF)
    testDF_numpy = standard_scaler.transform(testDF)

    # Copy the scaled data to trainDF and testDF
    trainDF = pd.DataFrame(trainDF_numpy, columns=trainDF_cols)
    testDF = pd.DataFrame(testDF_numpy, columns=testDF_cols)

    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)

    # Import the target variable and concatenate
    yTrain = pd.read_csv("eng_yTrain.csv")
    train_copy = pd.DataFrame.copy(xNewTrain)
    train_copy['label'] = yTrain['label']

    # Calculate the Pearson coefficient for xTrain
    train_pearson = cal_corr(train_copy)

    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)

    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)

    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
