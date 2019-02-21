from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, LinearSVC
import numpy as np
from sklearn.preprocessing import scale
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.neural_network import MLPClassifier
import logging
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level = logging.INFO, format = '%(levelname)-10s  %(message)s')

class SupervisedLearning:

	def getOptimalParams(self, X, Y):

		tuned_parameters = [{'activation': ['logistic','relu','tanh','identity'], 'solver': ['lbfgs','sgd','adam'],'learning_rate':['constant','invscaling','adaptive'],'alpha':[1e-5,1e-3,1e-7,1e-9]}]
		clf = GridSearchCV(MLPClassifier(), tuned_parameters, cv=10,
		   scoring='accuracy')
		clf.fit(X, Y)

		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		
		'''
		tuned_parameters = [{'loss': ['hinge','squared_hinge'], 'C': [1, 10, 100, 1000]}]
		clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=10,
		   scoring='accuracy')
		clf.fit(X, Y)

		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		'''
		

	def cross_validation(self, X , Y, folds = 10):
		clf = LinearSVC(C=1, loss='hinge',max_iter=1000, multi_class='ovr', random_state=None,penalty='l2', tol=0.0001)
		#clf = MLPClassifier(solver='sgd', alpha=1e-9, activation="logistic", learning_rate="adaptive")
		metrics = ["accuracy","f1_weighted"]
		for metric in metrics:
			out = cross_val_score(clf, X=X, y=Y, scoring=metric, cv=folds, n_jobs=-1, verbose=0, fit_params=None)
			self.show_results(out, metric)
	

	def show_results(self, out, metric):
		print("Results using the following metric: " + metric + "\n")
		print("Fold Results" + "\n")
		print(out)
		print("Mean " + str(np.mean(out)) + "\n")
		print("Median " + str(np.median(out)) +"\n")
		print("Std " + str(np.std(out)) + "\n")