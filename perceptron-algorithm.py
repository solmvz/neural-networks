import pandas as pd
import numpy as np
import sklearn as sk
from numpy import mean
from numpy import std
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

#dataset
x, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
#print(x.shape, y.shape)

model = Perceptron()

#model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


#new example prediction
model.fit(x, y)
row = [0.12777556,-3.64400522,-2.23268854,-1.82114386,1.75466361,0.1243966,1.03397657,2.35822076,1.01001752,0.56768485]
yhat = model.predict([row])
print('Predicted Class: %d' % yhat)
