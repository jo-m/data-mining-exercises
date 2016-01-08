#!/usr/bin/python

import numpy as np


class OcpSvm(object):
	def __init__(self, la):
		self.la = la

	def proj_lambda(self, w):
		return w * min(1,
			1 / (np.sqrt(self.la) * np.linalg.norm(w, 2)))

	def fit(self, X, y):
		n, d = X.shape
		w = np.zeros(d)
		for t in range(n):
			X_, y_ = X[t, :], y[t]
			if y_ * np.dot(w, X_) < 1:
				eta = 1 / np.sqrt(t + 1)
				w += eta * y_ * X_
				w = self.proj_lambda(w)
		self.w = w

	def predict(self, X):
		pred = np.sign(np.dot(X, self.w))
		pred[pred == 0] = -1
		return pred

def load_data():
	names = ['Xtrain', 'Ytrain', 'Xtest', 'Ytest']
	return (np.loadtxt('data/%s.csv' % n, delimiter=',') for n in names)

def permute(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm, :], y[perm]

def evaluate():
	Xtrain, Ytrain, Xtest, Ytest = load_data()
	Xtrain, Ytrain = permute(Xtrain, Ytrain)
	for la in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
		clf = OcpSvm(la)
		clf.fit(Xtrain, Ytrain)
		Ypred = clf.predict(Xtest)
		score = np.sum((Ypred == Ytest)) / float(Ypred.size)
		print 'SVM lambda = %f; Score: %f' % (la, score)

if __name__ == '__main__':
	evaluate()
