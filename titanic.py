# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:20:47 2020

@author: diggee
"""

#%% importing libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

#%% reading data

def get_data():
    full_train_data = pd.read_csv('train.csv', index_col = 'PassengerId')
    full_test_data = pd.read_csv('test.csv', index_col = 'PassengerId')
    return full_train_data, full_test_data

#%% preprocessing data
    
def clean_data(full_train_data, full_test_data):
    # dropping columns with almost all NaN values and columns that won't have an impact on model
    cols_to_drop = ['Name','Ticket']
    full_train_data.drop(cols_to_drop, axis = 1, inplace = True)
    full_test_data.drop(cols_to_drop, axis = 1, inplace = True)
    
    # extracting cabin information
    full_train_data['Cabin'].fillna(0, inplace = True)
    full_train_data['Cabin'] = full_train_data['Cabin'].apply(lambda x: 0 if x == 0 else 1)
    full_test_data['Cabin'].fillna(0, inplace = True)
    full_test_data['Cabin'] = full_test_data['Cabin'].apply(lambda x: 0 if x == 0 else 1)
    
    # dealing with training missing values
    total_elements = full_train_data.shape[0]
    for column in full_train_data.columns[full_train_data.isnull().any()]:
        if full_train_data[column].isnull().sum()/total_elements < 0.05:
            full_train_data.dropna(subset = [column], inplace = True)
        elif full_train_data[column].isnull().sum()/total_elements > 0.05 and full_train_data[column].dtype == int:
            full_train_data[column].fillna(full_train_data[column].mean(), inplace = True)
        else:
            full_train_data[column].fillna(full_train_data[column].value_counts().index[0], inplace = True)
    
    # dealing with test missing values
    total_elements = full_test_data.shape[0]
    for column in full_test_data.columns[full_test_data.isnull().any()]:
        if full_test_data[column].dtype == int:
            full_test_data[column].fillna(full_test_data[column].mean(), inplace = True)
        else:
            full_test_data[column].fillna(full_test_data[column].value_counts().index[0], inplace = True)
    
    # creating the feature children
    full_train_data['Child'] = full_train_data['Age'].apply(lambda x: 1 if x <= 18 else 0)
    full_test_data['Child'] = full_test_data['Age'].apply(lambda x: 1 if x <= 18 else 0)
    
    # dealing with price outliers
    full_train_data['Fare'] = full_train_data['Fare'].map(lambda x: 300 if x>=300 else x)
    full_test_data['Fare'] = full_test_data['Fare'].map(lambda x: 300 if x>=300 else x)      
            
    # dealing with categorical values
    categorical_cols = full_train_data.columns[full_train_data.dtypes == object]
    cols_to_transform = []
    for column in categorical_cols:
        if full_train_data[column].value_counts()[0]/full_train_data[column].value_counts().sum() > 0.9:
            full_train_data.drop(column, axis = 1, inplace = True)
            full_test_data.drop(column, axis = 1, inplace = True)          
        else:
            cols_to_transform.append(column)

    classes_sex = {'male':0, 'female':1}    
    classes_embarked = {'S':1, 'C':2, 'Q':1}
    full_train_data.replace(classes_sex, inplace = True)
    full_train_data.replace(classes_embarked, inplace = True)
    full_test_data.replace(classes_sex, inplace = True)
    full_test_data.replace(classes_embarked, inplace = True)
    
    X = full_train_data.drop('Survived', axis = 1)
    y = full_train_data['Survived']
    X_valid = full_test_data
    return X, y, X_valid

#%% data scaling
    
def scaled_data(X, X_valid):
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    X_valid = scaler_X.transform(X_valid)
    return X, X_valid, scaler_X  

#%% regressor functions
    
def regressor_fn_optimised(X, y, X_valid, choice):      
    from bayes_opt import BayesianOptimization
    
    if choice == 1:           
        def regressor_fn(alpha):            
            regressor = RidgeClassifier(alpha = alpha)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)
            return cval.mean()
        pbounds = {'alpha': (0, 1000)}
        
    elif choice == 2:           
        def regressor_fn(n_neighbors):     
            n_neighbors = int(n_neighbors)
            regressor = KNeighborsClassifier(n_neighbors = n_neighbors)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)
            return cval.mean()
        pbounds = {'n_neighbors': (2,10)}
        
    elif choice == 3:          
        def regressor_fn(n_estimators, max_depth):     
            max_depth, n_estimators = int(max_depth), int(n_estimators)
            regressor = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)
            return cval.mean()
        pbounds = {'n_estimators': (10, 500), 'max_depth': (2,20)}
        
    elif choice == 4: 
        X, X_valid, scaler_X = scaled_data(X, X_valid)       
        def regressor_fn(C, gamma):            
            regressor = SVC(C = C, kernel = 'rbf', gamma = gamma)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 5)
            return cval.mean()
        pbounds = {'C': (0.1, 100), 'gamma': (0.01, 100)}
        
    elif choice == 5:
        def regressor_fn(learning_rate, max_depth, n_estimators):            
            max_depth, n_estimators = int(max_depth), int(n_estimators)
            regressor = LGBMClassifier(learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 3)
            return cval.mean()
        pbounds = {'learning_rate': (0.01, 1), 'max_depth': (2,40), 'n_estimators': (10, 100)}        
        
    else:
        def regressor_fn(learning_rate, max_depth, n_estimators):            
            max_depth, n_estimators = int(max_depth), int(n_estimators)
            regressor = XGBClassifier(learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators)        
            cval = cross_val_score(regressor, X, y, scoring = 'balanced_accuracy', cv = 3)
            return cval.mean()
        pbounds = {'learning_rate': (0.01, 10), 'max_depth': (2,500), 'n_estimators': (10, 500)}
    
    optimizer = BayesianOptimization(regressor_fn, pbounds, verbose = 2)
    optimizer.probe(params = {'learning_rate':0.1, 'max_depth':10, 'n_estimators':20}, lazy = True)
    optimizer.maximize(init_points = 20, n_iter = 500)    
    # change next line in accordance with choice of regressor made
    y_valid_pred = XGBClassifier(learning_rate = optimizer.max['params']['learning_rate'], max_depth = int(optimizer.max['params']['max_depth']), n_estimators = int(optimizer.max['params']['max_depth'])).fit(X, y).predict(X_valid)
    return y_valid_pred, optimizer.max

#%% main
    
if __name__ == '__main__':
    full_train_data, full_test_data = get_data()
    X, y, X_valid = clean_data(full_train_data, full_test_data)
    y_valid_pred, optimal_params = regressor_fn_optimised(X, y, X_valid, choice = 6)
    # y_valid_pred = XGBClassifier.fit(X, y).predict(X_valid)      
    df = pd.DataFrame({'PassengerId':full_test_data.index, 'Survived':y_valid_pred})
    df.to_csv('prediction.csv', index = False)
    
    