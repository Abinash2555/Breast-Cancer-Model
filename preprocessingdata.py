# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:57:05 2022

@author: Abinash
"""

# import libraries
from sklearn.model_selection import train_test_split # for split dataset in train and test
from sklearn.preprocessing import StandardScaler # for making data in Standard range
class Preprocessing:
    def __init__(self, datasets):
        self._dataset = datasets
        
    def dataPreprocessing(self):
        # create second DataFrame by droping target
        X = self._dataset.drop(['target'], axis = 1)
        print("The shape of 'cancer_df2' is : ", self._dataset.shape)
        
        y = self._dataset['target']
        
        # split dataset into train and test

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5) 
        return X_train, X_test, y_train, y_test
        
    def featureScaling(self, X_train, X_test):
        # Feature scaling

        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc = sc.transform(X_test)
        
        return X_train_sc, X_test_sc