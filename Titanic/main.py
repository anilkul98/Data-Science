import pandas as pd
from processHelper import preprocessData, applyModels

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
Y_test = pd.read_csv("gender_submission.csv")['Survived']
X_train,Y_train, X_test = preprocessData(train, test)
applyModels(X_train, Y_train, X_test, Y_test)



