import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import re, warnings, sklearn


def create_categories(df:pd.DataFrame, columns=None):
    """
    Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace. Optionally, we can pass a list of columns to be changed.

    Parameters:
    -----------
    df: A pandas dataframe to apply changes to.

    columns: a list of columns. If none specified all string columns will be changed to categorical values
    """
    if columns is None:
        for n,c in df.items():
            if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()
    else:
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

def add_date_features(df, fldname, drop=True, time=False):
    """add_date_features converts a column of df from a datetime64 to many columns containing
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

def get_cardinality(df):
    """
    Returns the number of categories in each categorical feature. Very high cardinality might indicate it's better to use other alternatives like mean encodings
    """
    r = [(c, len(df[c].cat.categories)) for c in df.select_dtypes(include='category').columns]
    return pd.DataFrame(r,columns=['column','cardinality']).sort_values(by='cardinality', ascending=False)

def numericalize(df, max_n_cat=None):
    """ Changes the column col from a categorical type to it's integer codes.

    Parameters:
    -----------
    df: A pandas dataframe. df[name] will be filled with the integer codes from
        col.

    max_n_cat: If number of distinct values on a categorical column is higher than max_n_cat it will convert it to its integer codes. If max_n_cat is None, then all categorical columns will be converted
    """
    for name, col in df.items():
        if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
            df[name] = col.cat.codes+1

def one_hot_encode(df:pd.DataFrame):
    """
    Returns a new dataframe with all categorical columns transformed using one-hot-encoding
    """
    categorical = df.select_dtypes(include='category').columns
    return pd.get_dummies(df, columns=categorical)

def scale_vars(df, mapper=None):
    """
    Applies standard scaling in all numerical columns optionally using an existing mapper.
    Changed are applied in-place. Returns a mapper for future use.
    """
    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)
    if mapper is None:
        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return mapper

def fix_missing(df, na_dict=None):
    """ Fill missing data in all numerical columns of df with the median or a previously stored value through the na_dict, and add a {name}_na column
    which specifies if the data was missing. Returns the na_dict for future use.
    """
    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    
    na_dict_initial = na_dict.copy()
    for name, col in df.items():
        if is_numeric_dtype(col):
            if pd.isnull(col).sum() or (name in na_dict):
                df[name+'_na'] = pd.isnull(col)
                filler = na_dict[name] if name in na_dict else col.median()
                df[name] = col.fillna(filler)
                na_dict[name] = filler

    # if there is missing data in test data that was not present initially when the dictionary was created, the dataset will have extra columns. Let's remove them
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
        return na_dict_initial

    return na_dict

def get_sample(df, n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()

def process_dataframe(df, y_fld=None, drop_flds=None, ignore_flds=None, do_scale=False, mapper=None, na_dict=None, max_n_cat=None, subset=None):
    """

    All-in-one method to prepare a dataset for modeling. The following steps are taken:

    if subset value is set, applies all operations on a randomly chosen sample of size=subset
    
    Fixes missing values (by calling the fix_missing function) and optionally passing a na_dict created before

    Applies standard scaling to all numerical values (by using the scale_vars function), optionally using a mapper

    Numericalizes all categorical columns that have cardinality higher than max_n_cat (or all if not specified)

    One-Hot encodes all remaining non-numerical features

    Parameters:
    -----------
    df: A pandas dataframe.

    y_fld: The label column (if any, leave None if not in df)

    drop_flds: a collection of columns to be dropped from the dataset

    ignore_flds: a collection of columns to remain unchanged during processing

    do_scale: if the function should perform standard scaling on numeric features

    na_dict: a dictionary of na replacements, created by this function or fix_missing function

    max_n_cat: the maximum cardinality of features for one-hot-encoding. All above will be replace by int values

    subset: the size of a randomly chosen subset of the original dataset to perform operations

    mapper: a mapping of scaling operations, created by this function or the scale_vars function

    
    """
    if not ignore_flds: ignore_flds=[]
    if not drop_flds: drop_flds=[]

    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        drop_flds += [y_fld]
    
    # drop fields
    df.drop(drop_flds, axis=1, inplace=True)

    # fix missing
    if na_dict is None: na_dict = {}
    
    na_dict = fix_missing(df, na_dict)
    
    # scale numerical values
    if do_scale: mapper = scale_vars(df, mapper)
    
    # turn categories to number codes
    numericalize(df, max_n_cat)

    # one-hot-encode remaining categories under threshold of cardinality
    df = pd.get_dummies(df, dummy_na=True)
    
    # add ignored fields back in
    df = pd.concat([ignored_flds, df], axis=1)
    
    # prepare result
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    
    return res

