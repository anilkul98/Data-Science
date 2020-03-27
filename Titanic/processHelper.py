import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

models = [KNeighborsClassifier(n_neighbors=5),
          SVC(gamma='auto'),
          DecisionTreeClassifier(random_state=0),
          RandomForestClassifier(max_depth=20, random_state=0),
          AdaBoostClassifier(n_estimators=100, random_state=0),
          GradientBoostingClassifier(),
          GaussianNB(),
          LinearDiscriminantAnalysis()
          ]

luckies = ["Mrs", "Miss", "Master", "Sir", "Lady", "Ms", "Mle", "Counthess"]
unluckies = ["Mr", "Don", "Rev", "Dr", "Jonkheer"]

def preprocessData(train,test):
    Y_train = train['Survived']
    train['Train'] = train.apply(lambda row: 1, axis=1)
    test['Train'] = test.apply(lambda row: 0, axis=1)
    data = pd.concat([train, test], ignore_index=True, axis=0)
    data['Title'] = data.apply(lambda row: getTitle(row['Name']), axis=1)
    data['Lucky'] = data.apply(lambda row: 1 if row['Title'] in luckies else 0, axis=1)
    data['UnLucky'] = data.apply(lambda row: 1 if row['Title'] in unluckies else 0, axis=1)
    data = handleCategoricalData(data, 'Embarked')
    data = handleCategoricalData(data, 'Sex')
    data = handleCategoricalData(data, 'Pclass')
    data['child'] = isChild(data)
    dropColumn(data, 'Title')
    dropColumn(data,'Name')
    dropColumn(data, 'Cabin')
    dropColumn(data, 'Ticket')
    dropColumn(data, 'PassengerId')
    changeNanWithMean(data['Age'])
    changeNanWithMean(data['Fare'])
    data.drop(columns='Survived', axis=1, inplace=True)
    X_train = getTrainData(data)
    X_test = getTestData(data)
    dropColumn(train,'Train')
    dropColumn(test, 'Train')
    return X_train,Y_train, X_test

def handleTicket(str):
    if (str.isalpha()):
        return 0
    else:
        arr = str.split()
        l = len(arr)
        return int(arr[l-1])

def handleCategoricalData(data,str):
    data = pd.concat([data, pd.get_dummies(data[str], prefix=str)], axis=1)
    dropColumn(data, str)
    return data

def dropColumn(data,str):
    data.drop(columns=str, axis=1, inplace=True)

def changeNanWithMean(data):
    data.fillna(int(data.mean()), inplace=True)

def getTrainData(data):
    return data.loc[data['Train'] == 1]

def getTestData(data):
    return data.loc[data['Train'] == 0]

def isChild(data):
    return data.apply(lambda row: 1 if 18 < row['Age'] else 0, axis=1)

def getTitle(str):
    arr = str.split(".")
    s = arr[0]
    a = s.split()
    l = len(a)
    return a[l-1]

def applyModels(X_train, Y_train, X_test, Y_test):
    for model in models:
        result = ""
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        result += model.__str__().split("(")[0] + "\n"
        result += "----------------------------" + "\n"
        result += "Confusion Matrix:" + "\n"
        result += "{}".format(confusion_matrix(Y_test, y_pred)) + "\n"
        result += "AUROC: {}".format(roc_auc_score(Y_test, y_pred)) + "\n"
        result += "##################################\n"
        file = open("output.txt", "a+")
        file.write(result)