import numpy as np
import pandas as pd
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from mlutils.features import shuffle_feature
from sklearn.ensemble.forest import RandomForestRegressor

def train_regressor_cv(model, X_train, y_train, k_folds=5):
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)
    print ("R2 Score on training set: ", model.score(X_train, y_train))   
    print ("Training RMSE score using %i-fold crossvalidation: " %k_folds, np.mean(scores))

def train_regressor(model, X_train, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    rmse_score = mean_squared_error(y_train, y_pred)
    print ("  R2 (Training):", model.score(X_train, y_train))   
    print ("RMSE (Training):" , rmse_score)
    if hasattr(model, 'oob_score_'):
        print("      OOB Score:", model.oob_score_)

def predict_and_evaluate_regressor(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print ("             R2: %.5f" % r2)
    print ("           RMSE: %.5f" % rmse)
    return y_pred, r2, rmse
