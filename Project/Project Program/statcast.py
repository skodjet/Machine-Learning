import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess(xFeat, y):
    #TODO: do data preprocessing
    #1. Split data into xFeat and y labels
    #2. Scale data
    #3. remove unnecessary features (Pearson corrrelation, PCA)
    #4. Decide on model
    #5. Evaluate with k-fold cross validation / OOB error
    #6. Profit

    # print("xFeat: \n", xFeat)

    # Find the first year that every player played.
    firstYear = {}
    rows_to_remove = []
    for i in range(len(xFeat['player_id'])):
        id = xFeat['player_id'][i]
        year = xFeat['year'][i]
        if id not in firstYear:
            firstYear[id] = year
            rows_to_remove.append(i)

    # TODO: Test. Remove.
    # print("firstYear: ", firstYear)
    # print(len(firstYear))

    wobas = {}
    for i in range(len(y['player_id'])):
        id = y['player_id'][i]
        woba = y['woba'][i]
        year = y['year'][i]

        # Player id already in wobas. Just add new year.
        if id in wobas.keys():
            wobas[id][year] = [woba, i]

        # Player id not in wobas. Create a new sub dictionary and add the year.
        else:
            sub_dict = {}
            sub_dict[year] = [woba, i]
            wobas[id] = sub_dict

    # TODO: Test. Remove.
    print("wobas: ", wobas)






    # Calculate the difference in wOBA for every year but the player's first year in the dataset.
    # For each player:
    # for id, year in firstYear.items():
    #     previous_year = year
    #     target_year = year + 1
    #     for i in range(len(xFeat['player_id'])):
    #         dataset_id = xFeat['player_id'][i]
    #         dataset_year = xFeat['year'][i]
    #         if id == dataset_id and dataset_year == target_year:
    #             pass

    # Loop through the wobas dictionary calculating the differences.
    woba_diff = np.zeros((xFeat.to_numpy().shape[0]))
    for id, sub_dict in wobas.items():
        first_item = True
        previous_year = None
        for year, pair in sub_dict.items():

            # TODO: Test. Remove.
            # print(f"year: {year}, pair: {pair}")

            if first_item:
                row = pair[1]
                woba_diff[row] = -1
                first_item = False
                previous_year = year
            else:
                woba_val = pair[0]
                row = pair[1]
                previous_woba_val = sub_dict[previous_year][0]
                year_difference = int(year) - int(previous_year)
                difference = woba_val / previous_woba_val

                woba_diff[row] = difference

                previous_year = year

    # TODO: Test. Remove.
    print("woba diff: ", list(woba_diff))

    # Get a list of the rows that need to be removed.
    indices_to_remove = []
    for index, i in enumerate(woba_diff):
        if i == -1.0:
            indices_to_remove.append(index)

    xFeat = xFeat.drop(indices_to_remove)
    y = y.drop(indices_to_remove)

    # TODO: Test. Remove.
    # print("indices to remove: ", indices_to_remove)
    # print(f"xFeat: \n{xFeat}\ny: \n{y}\n")


    
    playerKey = xFeat[['last_name, first_name', 'year']].copy()
    xFeat = xFeat.drop(columns=["last_name, first_name"])
    xFeat = xFeat.drop(columns=["player_id"])
    xFeat = xFeat.drop(columns=["year"])
    # xFeat = xFeat.drop(columns=["xba", "xslg", "xobp", "xiso", "xwobacon", "xbacon", "xbadiff", "xslgdiff", "wobacon", "bacon",
    #                             "single", "double", "triple", "home_run", "b_sac_fly", "ab", "walk", "pa", "babip", "batting_avg", "bb_percent", "slg_percent", "on_base_percent"])
    y = y.drop(columns=["last_name, first_name"])
    y = y.drop(columns=["player_id"])
    # print("playerKey: ", playerKey)
    ynp = y.to_numpy()
    
    # print("xFeat: \n", xFeat)
    # print('y:\n', y)

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(xFeat))

    
    
    combined = np.concatenate((data, ynp), axis=1)
    df = pd.DataFrame(combined)
    
    removeInd1 = []
    
    corrMat = df.corr(method='pearson')
    npCorrMat = corrMat.to_numpy()
    for i in range(npCorrMat.shape[0]-1):
        if abs(npCorrMat[i][-1]) < 0.25:
            # print("just bad: ", df.columns[i])
            removeInd1.append(xFeat.columns[i-1])
    
    
    corrMat = df.corr(method='pearson')
    npCorrMat = corrMat.to_numpy()
    for i in range(npCorrMat.shape[0]-1):
        for j in range(npCorrMat.shape[1]-1):
            if abs(npCorrMat[i][j]) > 0.9 and i != j:
                # print("just similar: ", df.columns[i], ", ", df.columns[j])
                if xFeat.columns[i-1] not in removeInd1 and xFeat.columns[j-1] not in removeInd1:
                    removeInd1.append(xFeat.columns[j-1])

    print('removeInd1: ', len(removeInd1))
    print('xFeat columns: ', len(list(xFeat.columns)))
    
    # print("xFeat 3 ", xFeat)
    
    for i in removeInd1:
        xFeat = xFeat.drop(columns=i)

    # print("columns: ", xFeat.columns)
        

    # print("xFeat 4 ", xFeat)
    
    # for col in xFeat.columns:
    #     print("col: ", col)
        
        
        
            
    # plot = sns.heatmap(corrMat, cmap="crest", annot=True)
    # plt.show()
    
    # print(data.isna().any(axis=1))
    
    # data = data.dropna()
    
    # print("data ", data)
    
    # pca = PCA(n_components=0.90, svd_solver='full')
    # pca.fit_transform(data)
    # r_xTrain = pca.fit_transform(data)
    # variance = pca.explained_variance_ratio_
    # print("Im gonna try something walter: ", pca.components_)
    # featureInd = []
    # for i in range(15):
    #     featureInd.append(pca.components_[i].argmax())
    # for i in range(len(featureInd)):
    #     print("Feature: ", xFeat.columns[featureInd[i]], " ", pca.components_[i][featureInd[i]])
    # print("Im gonna try something walter: ", featureInd)
    # print("components #: ", pca.n_components_)
    scaler1 = StandardScaler()
    xFeat = scaler1.fit_transform(xFeat)

    y = y.to_numpy()
    
    return xFeat, y, playerKey

def train(xTrain, yTrain):
    return

def predict(xTest):
    return

def evaluate(yHat, yTest):
    return

def main():
    data = pd.read_csv("StatcastData.csv")
    X = pd.read_csv("StatcastX.csv")
    Y = pd.read_csv("StatcastY.csv")
    # StatY = pd.read_csv("stats (4).csv")
    Xnp = X.to_numpy()
    Ynp = Y.to_numpy()





    scaledX, scaledY, playerKey = preprocess(X, Y)
    scaledY = scaledY
    
    # scaledX = scaledX.to_numpy()
    
    keptData = np.empty((0,len(scaledX[0]))) 
    keptLabels = np.empty((0,1))
    testData = np.empty((0,len(scaledX[0])))
    testLabels = np.empty((0,1))
    
    predictorData = np.empty((0,len(scaledX[0])))
    predictorLables = np.empty((0,1))
    
    for i in range(len(scaledX)):
        if playerKey['year'][i] == 2023:
            testData = np.vstack((testData, scaledX[i]))
            testLabels = np.vstack((testLabels, scaledY[i]))
        else:
            keptData = np.vstack((keptData, scaledX[i]))
            keptLabels = np.vstack((keptLabels, scaledY[i]))

    x_train, x_test, y_train, y_test = keptData, testData, keptLabels, testLabels
    # x_train, x_test, y_train, y_test = train_test_split(scaledX, scaledY, test_size=.30)
    
    print("testData: ", testData)
    print("testLabels: ", testLabels)
    print("keptData: ", keptData)
    print("keptLabels: ", keptLabels)
    
    # print("scaledX: ", scaledX)
    # print("scaledY: ", scaledY)


    knn = KNeighborsRegressor(n_neighbors=31)
    knn.fit(x_train, y_train)
    yHat = knn.predict(x_test)
    
    y_test = y_test
    yHat = yHat
    
    error = mean_squared_error(y_test, yHat, squared=False)
    print("RMSE: ", error)
    
    r2 = r2_score(y_test, yHat)
    print("R2:",r2)

    # print("Statcast: ", r2_score(scaledY, StatY['xwoba']))
    
    print("Gradient Boost time")
    
    gb = GradientBoostingRegressor()
    gb.fit(x_train, y_train)
    
    yHat = gb.predict(x_test)
    yHat = yHat
    print("Score: ", r2_score(y_test, yHat))
    
if __name__ == "__main__":
    main()
    
    
    