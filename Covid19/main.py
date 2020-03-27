import warnings
import datetime

import pandas as pd
import numpy as np
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocessHelper import translateCountryNames,createBorderDict,addDate, addBorderColumns,\
    addPopulationColumn,addVirusDay,addGdpPerCapitaColumn
warnings.filterwarnings(action="ignore")

models = [RandomForestRegressor(n_estimators=500, random_state=16),
        AdaBoostRegressor(random_state=0, n_estimators=100), svm.SVR(), LogisticRegression(random_state=0),
        linear_model.Ridge(alpha=.5), LinearRegression()]
droppers = ['new_cases', 'new_deaths', 'total_cases', 'total_deaths']

def readData(str):
    data = pd.read_csv(str)
    data.fillna(0, inplace=True)
    return data

def preprocessData(data):
    indexNames = data[(data['total_cases'] == 0)].index
    data.drop(indexNames, inplace=True)
    #start_time = datetime.time().second
    #countrySet = set(data['Country'])
    data = addGdpPerCapitaColumn(data)
    print(data)
    data = addDate(data)
    #data = addBorderColumns(data, countrySet)
    data = addPopulationColumn(data)
    data = addVirusDay(data)
    data = pd.concat([data, pd.get_dummies(data['Country'], prefix='Country')], axis=1)
    data = data.drop("Country", axis=1)
    data.to_csv("processedData.csv", index=False)
    #processing_time = datetime.time.second - start_time
    #print("Preprocessing completed in {} seconds!".format(processing_time))

def seperateQuantities(data):
    new_cases = data['new_cases']
    new_deaths = data['new_deaths']
    total_cases = data['total_cases']
    total_deaths = data['total_deaths']
    return new_cases, new_deaths, total_cases, total_deaths

def applyModel(model, X_train, X_test, y_train, y_test, quantity):
    model = model
    model.fit(X_train, y_train)
    print(model.__str__().split("(")[0] + "for " +quantity)
    print("--------------------")
    y_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    print("RMSE for test: {}".format(np.sqrt(mse_test)))

def main():
    data = readData("full_data.csv")
    preprocessData(data)
    data = readData("processedData.csv")
    new_cases, new_deaths, total_cases, total_deaths = seperateQuantities(data)
    quantities = [new_cases, new_deaths, total_cases, total_deaths]

    for m in models:
        for i in range (4):
            x = data.drop(droppers[i], axis=1)
            X_train, X_test, y_train, y_test = \
                train_test_split(x, quantities[i], test_size=0.1, random_state=61)
            applyModel(m, X_train, X_test, y_train, y_test, droppers[i])
        print("\n")

if __name__ == '__main__':
    main()