import numpy as np
import math
import pandas as pd
import csv
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, explained_variance_score
import sklearn
from sklearn import svm, ensemble
import neptune.new as neptune
from neptune.new.types import File
from sklearn import svm
from  sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score



######### parameters for each ML method to be used with grid search


param_grid_randomforest = { 
            "n_estimators"      : [10,20,100],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False],
            }
param_grid_svr= {'C': [0.1, 1, 10 ], 'kernel': ['rbf', 'poly', 'sigmoid'], 'epsilon':[0.001,0.1,1],'degree':[2,3,4], 'gamma': ['auto', 'scale']}

param_grid_mlp = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,1)],
          'activation': ['relu','tanh','logistic'],
          'alpha': [0.0001, 0.05],
          'learning_rate': ['constant','adaptive'],
          'solver': ['adam']}


########## uncomment the ML model you want to test

#ns = svm.SVR()
#ns = ensemble.RandomForestRegressor()
ns =  MLPRegressor()
ns = GridSearchCV(ns, param_grid = param_grid_svr ,error_score=np.nan, cv=3,scoring='neg_root_mean_squared_error' )


def get_rar_dataset(filename, n=None):


    with open(filename) as data_file:
        reader = csv.reader(data_file)
        names = np.array(list(next(reader)))

    data = pd.read_csv(filename, dtype=object)
    data = data.to_numpy()

    n = len(names) - 1

    # ## Extract data names, membership values and Gram matrix

    names = names[1:n+1]
    mu = np.array([float(row[0]) for row in data[0:n+1]])
    gram = np.array([[float(k.replace('NA', '0')) for k in row[1:n+1]]
                     for row in data[0:n+1]])

    assert(len(names.shape) == 1)
    assert(len(mu.shape) == 1)
    assert(len(gram.shape) == 2)

    assert(names.shape[0] == gram.shape[0] == gram.shape[1] == mu.shape[0])

    X = np.array([[x] for x in np.arange(n)])

    return X, gram, mu, names




file_name='subclassOf-old-sim'
X, gram, mu, names = get_rar_dataset(file_name+".csv")
print('done extracting matrix')

print(file_name)
evs = 0
rmse = 0
for i in range(1):#set to any number of folds you want, not recommended if running grid search
     ########### uncomment if you do not want to use grid search
     # X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=0.3, random_state= i)
     # train_test = gram[X_train.flatten()][:, X_train.flatten()]
     # test_test = gram[X_test.flatten()][:, X_train.flatten()]
     # test_names = names[X_test.flatten()]
     #ns.fit(train_test, mu_train)
     # predicted_test = ns.predict(test_test)
     # predict_train= ns.predict(train_test)
     # evs = evs + explained_variance_score(mu_test,predicted_test)
     # rmse= rmse+ math.sqrt(mean_squared_error(mu_test,predicted_test))
     #print("evs: ",evs)   
     #print("rmse: ",rmse) 
     
     
     ######## uncomment if you want to use grid search
     ns.fit(gram, mu)
     print(ns.best_score_, ns.best_params_, ns.best_estimator_)
     



file_name='disjointmatrix'
X, gram, mu, names = get_rar_dataset(file_name+".csv")
print('done extracting matrix')

print(file_name)
evs = 0
rmse = 0
for i in range(1):#set to any number of folds you want, not recommended if running grid search
     ########### uncomment if you do not want to use grid search
     # X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=0.3, random_state= i)
     # train_test = gram[X_train.flatten()][:, X_train.flatten()]
     # test_test = gram[X_test.flatten()][:, X_train.flatten()]
     # test_names = names[X_test.flatten()]
     #ns.fit(train_test, mu_train)
     # predicted_test = ns.predict(test_test)
     # predict_train= ns.predict(train_test)
     # evs = evs + explained_variance_score(mu_test,predicted_test)
     # rmse= rmse+ math.sqrt(mean_squared_error(mu_test,predicted_test))
     #print("evs: ",evs)   
     #print("rmse: ",rmse) 
     
     
     ######## uncomment if you want to use grid search
     ns.fit(gram, mu)
     print(ns.best_score_, ns.best_params_, ns.best_estimator_)
     
     
     
     
     
file_name='subclassOf-722-matrix-new-sim'
X, gram, mu, names = get_rar_dataset(file_name+".csv")
print('done extracting matrix')

print(file_name)
evs = 0
rmse = 0
for i in range(1):#set to any number of folds you want, not recommended if running grid search
     ########### uncomment if you do not want to use grid search
     # X_train, X_test, mu_train, mu_test = train_test_split(X, mu, test_size=0.3, random_state= i)
     # train_test = gram[X_train.flatten()][:, X_train.flatten()]
     # test_test = gram[X_test.flatten()][:, X_train.flatten()]
     # test_names = names[X_test.flatten()]
     #ns.fit(train_test, mu_train)
     # predicted_test = ns.predict(test_test)
     # predict_train= ns.predict(train_test)
     # evs = evs + explained_variance_score(mu_test,predicted_test)
     # rmse= rmse+ math.sqrt(mean_squared_error(mu_test,predicted_test))
     #print("evs: ",evs)   
     #print("rmse: ",rmse) 
     
     
     ######## uncomment if you want to use grid search
     ns.fit(gram, mu)
     print(ns.best_score_, ns.best_params_, ns.best_estimator_)