import sklearn
import matplotlib.pyplot as plt
import re, random, math, scipy
from scipy.cluster import hierarchy as hc
from mlutils.plot import plot_hist
from IPython.display import display
import warnings

def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    """ Get a list of index values for Validation set from a dataset
    
    Arguments:
        n : int, Total number of elements in the data set.
        cv_idx : int, starting index [idx_start = cv_idx*int(val_pct*n)] 
        val_pct : (int, float), validation set percentage 
        seed : seed value for RandomState
        
    Returns:
        list of indexes 
    """
    np.random.seed(seed)
    n_val = int(val_pct*n)
    idx_start = cv_idx*n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start+n_val]


def shuffle_feature(df, c):
    df_shuffled = df.copy()
    values = list(df[c].values)
    shuffled = random.sample(values, len(values))
    df_shuffled[c] = shuffled
    return df_shuffled

def convert_vector_to_matrix(v):
    """
    Utility function that converts a 1-D numpy array into a 2-D matrix. This allows passing to scikit-learn models
    """
    return v[:,None]
