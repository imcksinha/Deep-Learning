# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 05:07:12 2019

@author: test2
"""
# Creating SOM

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Importing the data
os.chdir('C:/Chandan/Deep Learning/16_page_p0s1_file_1/Deep_Learning_A_Z/Volume 2 - Unsupervised Deep Learning/Part 4 - Self Organizing Maps (SOM)/Section 16 - Building a SOM')
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(data = X)
som.train_random(data = X, num_iteration = 100)

from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = None,
         markersize = 10,
         markeredgewidth = 2)
show()

# Identify the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,6)], mappings[(7,6)]), axis = 0)
frauds = sc.inverse_transform(frauds)