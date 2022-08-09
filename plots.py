# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:57:22 2022

@author: Abinash
"""

# import libraries
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization

class DataVisulization:
    def __init__(self, datasets):
        self._datasets = datasets

    def pairplot(self):
        sns.pairplot(self._datasets, hue='target',
                     vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])

    def countplot(self):
         sns.countplot(self._datasets['target'])

    def heatmap(self):
        plt.figure(figsize=(16,9))
        sns.heatmap(self._datasets)

    def correlation_matrix(self):
        plt.figure(figsize=(20,20))
        sns.heatmap(self._datasets.corr(), annot = True, cmap ='coolwarm', linewidths=2)

    def barplot(self):
        plt.figure(figsize = (16,5))
        ax = sns.barplot(self._datasets.corrwith(self._datasets.target).index, self._datasets.corrwith(self._datasets.target))
        ax.tick_params(labelrotation = 90)