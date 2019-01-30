import IPython, graphviz, re
from sklearn.tree import export_graphviz
from sklearn.ensemble import forest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from treeinterpreter import treeinterpreter as ti
from mlutils.plot import plot_hist, plot_line
from scipy import stats
from sklearn.metrics import accuracy_score, r2_score

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.

    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))

def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))

def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)

def parallel_trees(m, fn, n_jobs=8):
   return list(ProcessPoolExecutor(n_jobs).map(fn, m.estimators_))

def rf_feat_importance(m, df):
   return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)
   
def select_features(df, fi, threshold=None):
   """Returns a DataFrame with only the top important features.
   Can receive a threshold parameter so that all features with importance above it will be kept.
   By default, the mean value of importance is used. Median can also be specified
   
   Parameters:
    -----------
    df: The original dataframe
    fi: feature importance information (obtained from rf_feat_importance)
    threshold: float or 'median'|'mean'

   """
   if not threshold or threshold == 'mean':
      threshold = fi.imp.mean()
   elif threshold == 'median':
      threshold = fi.imp.median()
   
   to_keep = fi[fi.imp>threshold].cols
   return df[to_keep].copy()

def plot_feat_importance(fi, top=None):
   if top is not None:
      r = fi[:top]
   else:
      r = fi
   
   plot_line(r, 'cols', 'imp', sort=False, xticks_rotation=90)
   plt.show()
   
   r.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
   plt.show()


def tree_interpret_prediction(m, df, row):
   prediction, bias, contributions = ti.predict(m, row)
   idxs = np.argsort(contributions[0])
   for i in [o for o in zip(df.columns[idxs], df.iloc[0][idxs], contributions[0][idxs])]:
      print(i)
   return prediction[0], bias[0], contributions[0].sum()

def analyze_predictions_in_group(df, c, y_fld, preds):
   """ Analyzes the predictions of a forest for variance
   and plots the relationship across a categorical feature
   
    Parameters:
    -----------
    df: The original dataset (before pre-processing categorical columns)
    c: The column you want to analyze variance across (categorical)
    y_fld: The target variable
    preds: The array of predictions across all forest estimators

    Usage:
    ------------
    def get_preds(t): return t.predict(X_val)
    preds = np.stack(parallel_trees(model, get_preds))

    analyze_predictions_in_group(raw_valid, 'Size', 'UnitsSold', preds)
    """
   # plot histogram for categorical column
   plot_hist(df, c, title=c)
   
   x = df.copy()
   x['pred_std'] = np.std(preds, axis=0)
   x['pred'] = np.mean(preds, axis=0)

   flds = [c, y_fld, 'pred', 'pred_std']
   # group by column and average target, pred and pred_std
   summ = x[flds].groupby(c, as_index=False).mean()
   print(summ)
   
   summ = summ[~pd.isnull(summ[y_fld])]
   f = plt.figure(figsize=(20,6))
   ax = f.add_subplot(121)
   summ.plot(c, y_fld, 'barh', ax=ax)
   ax.set_title('Actual Values')
   
   ax2 = f.add_subplot(122)
   summ.plot(c, 'pred', 'barh', xerr='pred_std', alpha=0.6,ax=ax2)
   ax2.set_title('Predicted Values')
   
   plt.show()
   
   print("Weighted Standard Deviation")
   summ['weighed_std'] = summ.pred_std/summ.pred
   print(summ[[c,'weighed_std']].sort_values(by='weighed_std', ascending=False))

def analyze_estimators_growth_accuracy(m, x_val, y_val):
   preds = np.stack([t.predict(x_val) for t in m.estimators_])
   plt.plot([accuracy_score(y_val, stats.mode(preds[:i+1], axis=0)[0][0]) for i in range(len(m.estimators_))])
   plt.show()

def analyze_estimators_growth_r2(m, x_val, y_val):
   preds = np.stack([t.predict(x_val) for t in m.estimators_])
   plt.plot([r2_score(y_val, np.mean(preds[:i+1], axis=0)) for i in range(len(m.estimators_))])
   plt.show()

def analyze_single_feature_removal(m, x, y, f_list, metric='oob', val_data=None):
   """
   Uses a RandomForestRegressor to analyze the impact of removing a feature on a score metric.
   You can pass a list of features and it will iterate removing one feature at a time.
   """
   def score_model(x, y, x_val=None, y_val=None):
      m.fit(x,y)
      if (metric == 'oob'):
         return m.oob_score_
      elif metric == 'accuracy':
         y_pred = m.predict(x_val)        
         return accuracy_score(y_val, y_pred)
      else:   
         return m.score(x_val, y_val)

   if metric == 'oob':
      m.oob_score = True
   elif metric == 'accuracy' and val_data:
      m.oob_score = False
   elif metric == 'r2' and val_data:
      m.oob_score = False
   else:
      raise ValueError("metric must be 'oob' or 'accuracy'. If 'accuracy' val_data must be supplied")

   # let's get a baseline score to compare to
   if val_data:            
      score = score_model(x, y, val_data[0], val_data[1])
   else:
      score = score_model(x, y)
   print("Baseline -", score)
   print("-"*30)

   for c in f_list:
      if val_data:
         n_score = score_model(x.drop(c, axis=1), y, val_data[0].drop(c, axis=1), val_data[1])
      else:
         n_score = score_model(x.drop(c, axis=1), y)
      print(f"{c} - {n_score} ({np.round(n_score - score,5)})")

def analyze_multiple_feature_removal(m, x, y, f_list, metric='oob', val_data=None):
   """
   Uses a RandomForestRegressor to analyze the impact of removing a set of features on a score metric.
   """
   def score_model(x, y, x_val=None, y_val=None):
      m.fit(x,y)
      if (metric == 'oob'):
         return m.oob_score_
      else:
         return m.score(x_val, y_val)
   
   if metric == 'oob':
      m.oob_score = True
      
   elif metric == 'accuracy' and val_data:
      m.oob_score = False
   
   elif metric == 'r2' and val_data:
      m.oob_score = False
   else:
      raise ValueError("metric must be 'oob' or 'accuracy'. If 'accuracy' val_data must be supplied")

   # let's get a baseline score to compare to
   if val_data:            
      score = score_model(x, y, val_data[0], val_data[1])
   else:
      score = score_model(x, y)
   print("Baseline -", score)
   print("-"*30)

   if val_data:
      n_score = score_model(x.drop(f_list, axis=1), y, val_data[0].drop(f_list, axis=1), val_data[1])
   else:
      n_score = score_model(x.drop(f_list, axis=1), y)

   print(f"Result - {n_score} ({np.round(n_score - score,5)})")
    