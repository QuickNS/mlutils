from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def train_classifier_cv(model, X_train, y_train, k_folds=5):
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring='accuracy')
    print ("Accuracy (Training): ", model.score(X_train, y_train))   
    print ("Average accuracy score using %i-fold crossvalidation: " %k_folds, np.mean(scores))
    return scores

def train_classifier(model, X_train, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    score = accuracy_score(y_train, y_pred)
    print ("Accuracy (Training):" , score)
    if hasattr(model, 'oob_score_'):
        print("          OOB Score:", model.oob_score_)

def predict_and_evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print ("     Accuracy score: %.5f" % accuracy_score(y_test, y_pred))
    return y_pred


