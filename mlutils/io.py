import gzip
from urllib.request import urlretrieve
from tqdm import tqdm
import os
import feather
import pandas as pd

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)

def get_data(url, filename):
    if not os.path.exists(filename):

        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urlretrieve(url, filename, reporthook=t.update_to)

def save_feather(df, filename):
    """
    Saves a pandas dataframa in feather format. Useful after large processing.
    """
    df.save_feather(filename)
    print(f"Saved file {filename}")

def load_feather(filename):
    """
    Loads a feather file in pandas format
    """
    return feather.read_dataframe(filename)
    