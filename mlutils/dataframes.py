import numpy as np 
import pandas as pd
from pandas_summary import DataFrameSummary
import random, scipy
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as hc

def display_all(df:pd.DataFrame, transpose:bool = False):
    """
    Displays a DataFrame up to 1000x1000. Allows transposing
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        if transpose:
            display(df.T)
        else:
            display(df)

def summarize(df:pd.DataFrame, transpose:bool = False):
    """
    Displays extended summary statistics of a pandas dataframe object
    """
    r = DataFrameSummary(df)
    display_all(r.summary(), transpose=transpose)

def display_mem_usage(df):
    """
    Displays memory utilization of a pandas dataframe object
    """
    print(df.info(memory_usage='deep'))
    print()
    for dtype in ['float','int','object','bool','category']:
        selected_dtype = df.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).sum()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Total memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

    print()

    for dtype in ['float','int','object','bool','category']:
        selected_dtype = df.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

def detect_missing(data:pd.DataFrame):
    """
    Prints number of missing values on each column. Alternatively, can display percentages
    """
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    return missing_data[missing_data['Total'] > 0]

def detect_outliers(x:pd.Series, limits:tuple=(None)):
    """ 
    Returns indexes of outlier values in a numeric series. Can pass in lower and upper bounds using a tuple. If missing, the lower and upper bounds are set using interquantile range.

    Example:
        detect_outliers(df[column])
        detect_outliers(df[column], (0, 5000))
    """
    if limits is None:
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)
    else:
        lower_bound = limits[0]
        upper_bound = limits[1]

    indices = x[(x > upper_bound) | (x < lower_bound)].index
    print(f"Number of outliers found: {len(indices)}, using interval ({lower_bound},{upper_bound})")
    return indices

def split_dataset(df:pd.DataFrame,n:int=None, shuffle=False, random_seed=None):
    """
    Splits a DataFrame in a length n training dataset and a validation set with the remainder.
    If n is None, automatically uses a 80/20 split.
    By default keeps the dataset in order, but shuffle can be set to True to target a random sample for each dataset (without repetition).
    A seed can also be supplied for consistency
    """
    if n is None:
        n = int(np.round(0.8 * len(df)))

    if shuffle:      
        if random_seed:
            random.seed(random_seed)

        indexes = list(df.index.values)
        random.shuffle(indexes)
        shuffled = df.iloc[indexes]
        return shuffled[:n].copy(), shuffled[n:].copy()
    else:
        return df[:n].copy(), df[n:].copy()

def analyze_redundant_features(df, threshold=0.1, no_plot=False):
    """
    This function computes groups of correlated features and displays a dendrogram with distance information.
    It returns a tuple containing a DataFrame with all the computed distances beneath the specified threshold and a list with groups of features that should be evaluated, ideally
    
    """
    corr = np.round(scipy.stats.spearmanr(df).correlation, 4)
    # draw dedrogram
    corr_condensed = hc.distance.squareform(1-corr)
    z = hc.linkage(corr_condensed, method='average')
    if not no_plot:
        fig = plt.figure(figsize=(16,10))
        dendrogram = hc.dendrogram(z, labels=df.columns, orientation='left', leaf_font_size=16)
        plt.show()
    
    pair_list = []
    n = len(df.columns)
    Z = np.asarray(z, order='c')

    def c_name(i):
        return df.columns[int(i)]

    for i,p in enumerate(Z):
    # this means original samples for leaf nodes
        cluster_name = f'cluster_{i}'
        if (p[0]<n or p[1] < n):
            if p[0] >= n:
                idx = int(p[0])-n
                pair_list.append([cluster_name, c_name(p[1]), f'cluster{idx}', p[2]])
            elif p[1] >= n:
                idx = int(p[1])-n
                pair_list.append([cluster_name, c_name(p[0]), f'cluster_{idx}', p[2]])
            else:
                pair_list.append([cluster_name, c_name(p[0]), c_name(p[1]), p[2]])
        else:
            idx1 = int(p[0])-n
            idx2 = int(p[1])-n
            pair_list.append([cluster_name, f'cluster_{idx1}', f'cluster_{idx2}', p[2]])

    p_df = pd.DataFrame(pair_list, columns=['cluster','a','b','distance'])
    p_df.distance = p_df.distance.astype('float')
    p_df.set_index('cluster', inplace=True, drop=True)
    
    # create list of clustered features whose distance to another feature or cluster is less than threshold
    p_df = p_df[p_df.distance < threshold]
    # couple of healper functions

    def is_leaf(s):
        return not s.startswith('cluster_')
    
    def cluster_index(s):
        return int(s.split('_')[1])

    remove_list = []

    def process_pair(row):
        l = []

        if is_leaf(row.a):
            l = l + [row.a]
        else:
            l = process_pair(p_df.loc[row.a,:]) + a
            remove_list.append(cluster_index(row.a))
        if is_leaf(row.b):
            l = l + [row.b]
        else:
            l = process_pair(p_df.loc[row.b,:]) + l
            remove_list.append(cluster_index(row.b))
        return l
        
    f_list = []
    
    for i, c in p_df.iterrows():
        f_list.append(process_pair(c))
    
    result = np.array( f_list)
    remote_list = np.unique(np.array(remove_list))
    result = np.delete(result, remote_list, 0)
    return p_df, result