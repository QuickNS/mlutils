import numpy as np 
import pandas as pd
from pandas_summary import DataFrameSummary
from sklearn_pandas import DataFrameMapper
from pandas.api.types import is_string_dtype, is_numeric_dtype
import sklearn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re, random, math, scipy
from scipy.cluster import hierarchy as hc
from mlutils.plot import plot_hist
from IPython.display import display
import warnings

def summarize(df:pd.DataFrame):
    display(DataFrameSummary(df).summary())

def display_all(df:pd.DataFrame, transpose:bool =False):
    """
    Displays a DataFrame up to 1000x1000. Allows transposing
    """
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        if transpose:
            display(df.T)
        else:
            display(df)

def detect_missing(data:pd.DataFrame):
    """
    Prints number of missing values on each column. Alternatively, can display percentages
    """
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    return missing_data[missing_data['Total'] > 0]

def one_hot_encode(x):
    categorical = x.select_dtypes(exclude=[np.number])
    return pd.get_dummies(x, columns=categorical.columns)

def make_categorical(df, feature, value_list):
    return df[feature].astype("category", categories = value_list)

def detect_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    indices = x[(x > upper_bound) | (x < lower_bound)].index
    return indices

def split_vals(df,n):
    return df[:n].copy(), df[n:].copy()

def add_datepart(df, fldname, drop=True, time=False):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.

    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.
    """

    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

def create_categories(df):
    """
     Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.

    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values.
    """
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()

def create_categories_for_columns(df, columns):
    """
     Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.

    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values.
    """
    for c in columns:
        df[c] = df[c].astype('category').cat.as_ordered()

def apply_categories(df, trn):
    """
    Changes any columns of strings in df into categorical variables using trn as
    a template for the category codes.

    Parameters:
    -----------
    df: A pandas dataframe. Any columns of strings will be changed to
        categorical values. The category codes are determined by trn.

    trn: A pandas dataframe. When creating a category for df, it looks up the
        what the category's code were in trn and makes those the category codes
        for df.
    """
    for n,c in df.items():
        if (n in trn.columns) and (trn[n].dtype.name=='category'):
            df[n] = pd.Categorical(c, categories=trn[n].cat.categories, ordered=True)

def get_cardinality(df):
    r = [(c, len(df[c].cat.categories)) for c in df.select_dtypes(include='category').columns]
    return pd.DataFrame(r,columns=['column','cardinality']).sort_values(by='cardinality', ascending=False)

def numericalize(df, col, name, max_n_cat):
    """ Changes the column col from a categorical type to it's integer codes.

    Parameters:
    -----------
    df: A pandas dataframe. df[name] will be filled with the integer codes from
        col.

    col: The column you wish to change into the categories.
    name: The column name you wish to insert into df. This column will hold the
        integer codes.

    max_n_cat: If col has more categories than max_n_cat it will change the
        it to its integer codes. If max_n_cat is None, then col will always be
        converted.

    """
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = col.cat.codes+1

def scale_vars(df, mapper):
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def process_dataframe(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None, preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res

def fix_missing(df, col, name, na_dict):
    """ Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def get_sample(df, n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

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

def display_mem_usage(df):
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

def shuffle_feature(df, c):
    df_shuffled = df.copy()
    values = list(df[c].values)
    shuffled = random.sample(values, len(values))
    df_shuffled[c] = shuffled
    return df_shuffled

def analyze_redundant_features(df, threshold=0.1, no_plot=False):
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

def convert_vector_to_matrix(v):
    """
    Utility function that converts a 1-D numpy array into a 2-D matrix. This allows passing to scikit-learn models
    """
    return v[:,None]