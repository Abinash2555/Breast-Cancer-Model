# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 09:06:56 2022

@author: Abinash
"""
# import libraries
import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
from sklearn.datasets import load_breast_cancer # for dataset
from plots import DataVisulization
from preprocessingdata import Preprocessing
from algorithm import Algorithm


class Model:
    def __init__(self, dataset):
        self._dataset = dataset
        
    def datasetdetails(self):        
        cancer_df = pd.DataFrame(np.c_[self._dataset['data'],self._dataset['target']],
             columns = np.append(self._dataset['feature_names'], ['target']))
        return cancer_df


        
if __name__ == "main":
    #Load dataset from Sklearn
    cancer_dataset = load_breast_cancer()
    
    # Instance of Model Class
    mod = Model(cancer_dataset)
    df = mod.datasetdetails()
    
    # Instance of Preprocessing Class
    dp = Preprocessing(df)
    
    # Instance of DataVisulization
    dv = DataVisulization(df)
    
    # Instance of Algorithm
    algo = Algorithm()
    
    # Countplot
    dv.countplot()
    
    # Pairplot
    dv.pairplot()
    
    # Heatmap
    dv.heatmap()
    
    # Heatmap of correlation matrix
    dv.correlation_matrix()
    
    # Barplot
    dv.barplot()
    
    #Splitting into Train and Test
    X_train, X_test, y_train, y_test = dp.dataPreprocessing()
    
    # Data standardization
    X_train_sc, X_test_sc = dp.featureScaling(X_train, X_test)
    
    # Naive Bayes Classifier
    nb_model, y_pred_nb = algo.train_classifier( X_train, X_test, y_train, y_test, "NB")
    # Naive Bayes Classifier with Standard scaled Data 
    nb_model_sc, y_pred_nb_sc = algo.train_classifier( X_train_sc, X_test_sc, y_train, y_test, "NB")
    
    # K – Nearest Neighbor Classifier
    knn_model, y_pred_knn = algo.train_classifier( X_train, X_test, y_train, y_test, "KNN")
    # Naive Bayes Classifier with Standard scaled Data 
    knn_model_sc, y_pred_knn_sc = algo.train_classifier( X_train_sc, X_test_sc, y_train, y_test, "KNN")
    
    # Random Forest Classifier
    rf_model, y_pred_rf = algo.train_classifier( X_train, X_test, y_train, y_test, "RF")
    # Random Forest Classifier with Standard scaled Data 
    rf_model_sc, y_pred_rf_sc = algo.train_classifier( X_train_sc, X_test_sc, y_train, y_test, "RF")
    
    # Decision Tree Classifier
    dt_model, y_pred_dt = algo.train_classifier( X_train, X_test, y_train, y_test, "DT")
    # Decision Tree Classifier with Standard scaled Data 
    dt_model_sc, y_pred_dt_sc = algo.train_classifier( X_train_sc, X_test_sc, y_train, y_test, "DT")
    
    # Adaboost Classifier
    adboost_model, y_pred_adboost = algo.train_classifier( X_train, X_test, y_train, y_test, "AdaBoost")
    # Adaboost Classifier with Standard scaled Data 
    adboost_model_sc,y_pred_adboost_sc = algo.train_classifier( X_train_sc, X_test_sc, y_train, y_test, "AdaBoost")
    
    # XGBoost Classifier
    xgboost_model, y_pred_xgboost = algo.train_classifier( X_train, X_test, y_train, y_test, "XGBoost")
    # XGBoost Classifier with Standard scaled Data 
    xgboost_model_sc, y_pred_xgboost_sc = algo.train_classifier( X_train_sc, X_test_sc, y_train, y_test, "XGBoost")
    # Comparing algorithm with 
    model_compare = algo.models_evaluation(X_train, y_train, knn_model, adboost_model, dt_model, rf_model, nb_model, xgboost_model)
    
    # Confusion Matrix
    confusion_matrix_xgb = algo.ResultEvalution(y_test, y_pred_xgboost)
    # Save Model
    algo.save_model(xgboost_model)