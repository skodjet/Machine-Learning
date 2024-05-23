"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje */

I collaborated with the following classmates for this homework:
Clay Winder
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def preprocess(xFeat, y):
    """
    :param xFeat:
    :param y:
    :return: The preprocessed data
    """

    # Remove player name, player ID, and year.
    xFeat = xFeat.drop(columns=["last_name, first_name", "player_id", "year"])
    y = y.drop(columns=["last_name, first_name", "player_id"])

    # Perform min max scaling
    scaler = MinMaxScaler()
    data = scaler.fit_transform(xFeat)

    # Combine the data
    ynp = y.to_numpy()
    combined = np.concatenate((data, ynp), axis=1)
    df = pd.DataFrame(combined)



    # TODO: Test. Remove.
    print(df)
    print("y:\n", y)

    # TODO: This was ineffective. Uncomment before submitting
    # # Code for heatmap. This proved too confusing, so we decided to do PCA instead.
    # corrMat = df.corr(method='pearson')
    # heatmap = sns.heatmap(corrMat)
    # plt.show()

    # Convert x to numpy


    p_component_analysis = PCA()
    p_component_analysis.fit_transform(xFeat)

    # Get which original features are important, and how much of the variance they explain.
    important_components = p_component_analysis.components_
    explained_variance = p_component_analysis.explained_variance_ratio_

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




def train(xTrain, yTrain):
    pass

def predict(xTest):
    pass

def evaluate(yHat, yTest):
    pass
def main():
    data = pd.read_csv("StatcastData.csv")
    X = pd.read_csv("StatcastX.csv")
    Y = pd.read_csv("StatcastY.csv")
    Xnp = X.to_numpy()
    Ynp = Y.to_numpy()

    # TODO: Test. Remove.
    # print(X)

    scaledX = preprocess(X, Y)






if __name__ == "__main__":
    main()


