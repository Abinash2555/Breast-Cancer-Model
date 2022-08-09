# import libraries
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import pickle
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

class Algorithm:
    # Define dictionary with performance metrics
    
    def train_classifier(self,X_train, X_test, y_train, y_test, name):
      classifier = {
          'NB': GaussianNB(),
          'KNN': KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
          'DT': DecisionTreeClassifier(criterion = 'entropy', random_state = 51),
          'RF': RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51),
          'AdaBoost': AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', 
                                          random_state = 200),
                                        n_estimators=2000,
                                        learning_rate=0.1,algorithm='SAMME.R',random_state=1,),
          'XGBoost': XGBClassifier()
                  
      }
      algo = classifier.get(name, "Invalid Input")
      algo = algo.fit(X_train, y_train)
      y_pred = algo.predict(X_test)
      accuracy = accuracy_score(y_test, y_pred)
      print(accuracy)
      return algo, y_pred
  
    def ResultEvalution(self, y_test, y_pred):
        
        """# Confusion Matrix"""
        
        cm = confusion_matrix(y_test, y_pred)
        plt.title('Heatmap of Confusion Matrix', fontsize = 15)
        sns.heatmap(cm, annot = True)
        
    
        
        """# Cross-validation of the ML model"""
        
    def Cross_validation(self,X_train, y_train, classifier):
        # Cross validation
        cross_validation = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        print("Cross validation of XGBoost model = ",cross_validation)
        print("Cross validation of XGBoost model (in mean) = ",cross_validation.mean())
        
        
    def models_evaluation(self,X_train, y_train, knn_model, adboost_model,dtr_model,rfc_model,gnb_model, xgboost_model,folds=10,):
    
        '''
        X : data set features
        y : data set target
        folds : number of cross-validation folds
        
        '''
        scoring = {'accuracy':make_scorer(accuracy_score), 
               'precision':make_scorer(precision_score),
               'recall':make_scorer(recall_score), 
               'f1_score':make_scorer(f1_score)}
        # Perform cross-validation to each machine learning classifier
        knn = cross_validate(knn_model, X_train, y_train, cv=folds, scoring = scoring)
        adboost = cross_validate(adboost_model, X_train, y_train, cv=folds, scoring = scoring)
        dtr = cross_validate(dtr_model,  X_train, y_train, cv=folds, scoring= scoring)
        rfc = cross_validate(rfc_model,  X_train, y_train, cv=folds, scoring= scoring)
        gnb = cross_validate(gnb_model,  X_train, y_train, cv=folds, scoring= scoring)
        xgboost = cross_validate(xgboost_model,  X_train, y_train, cv=folds, scoring= scoring)
    
        # Create a data frame with the models perfoamnce metrics scores
        models_scores_table = pd.DataFrame({'KNN':[knn['test_accuracy'].mean(),
                                                                   knn['test_precision'].mean(),
                                                                   knn['test_recall'].mean(),
                                                                   knn['test_f1_score'].mean()],
                                           
                                          'AdaBoost':[adboost['test_accuracy'].mean(),
                                                                       adboost['test_precision'].mean(),
                                                                       adboost['test_recall'].mean(),
                                                                       adboost['test_f1_score'].mean()],
                                           
                                          'Decision Tree':[dtr['test_accuracy'].mean(),
                                                           dtr['test_precision'].mean(),
                                                           dtr['test_recall'].mean(),
                                                           dtr['test_f1_score'].mean()],
                                           
                                          'Random Forest':[rfc['test_accuracy'].mean(),
                                                           rfc['test_precision'].mean(),
                                                           rfc['test_recall'].mean(),
                                                           rfc['test_f1_score'].mean()],
                                           
                                          'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                                  gnb['test_precision'].mean(),
                                                                  gnb['test_recall'].mean(),
                                                                  gnb['test_f1_score'].mean()],
                                           
                                           'XGBoost':[xgboost['test_accuracy'].mean(),
                                                                  xgboost['test_precision'].mean(),
                                                                  xgboost['test_recall'].mean(),
                                                                  xgboost['test_f1_score'].mean()]},
                                          
                                          index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
        # Add 'Best Score' column
        models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
        
        # Return models performance metrics scores data frame
        return(models_scores_table)

    """# Save Model"""
    def save_model(self,classifier):
        # save model
        pickle.dump(classifier, open('model.pickle', 'wb'))
      