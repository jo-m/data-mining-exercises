#!/usr/bin/python

import numpy as np

X = np.loadtxt('data/Xtrain.csv', delimiter=',')
y = np.loadtxt('data/Ytrain.csv')

print X.shape
print y.shape
