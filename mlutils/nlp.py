from sklearn.feature_extraction.text import CountVectorizer
import re

re_tok = re.compile('([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()
