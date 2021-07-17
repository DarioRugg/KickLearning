import unicodedata
import numpy as np

from sklearn.exceptions import NotFittedError

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA

import pickle as pk


nltk.download('stopwords', quiet=True)


class TextEncoder:
    def __init__(self, pca_var_explained=0.8, data=None):

        self.data = data

        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'[0-9]*[a-zA-Z]+[0-9]*|20[0-9]{2}|1[0-9]{3}')
        self.stammer = SnowballStemmer(language="english")

        self.tfidf_transformer = TfidfVectorizer(max_df=0.8, min_df=0.05)

        self.pca_transformer = PCA(n_components=0.8, random_state=1234)

    def _preprocessing(self, text):
        line = text.lower()

        words = self.tokenizer.tokenize(text)
        # removing stop words
        words = filter(lambda w: w not in self.stop_words, words)
        # removing stem
        stemmed = list(map(self.stammer.stem, words))

        return " ".join(stemmed)

    def preprocess(self):
        self.data = list(map(self._preprocessing, self.data))

    def tfidf_fit(self, data=None):
        self.tfidf_transformer.fit(data if data is not None else self.data)

    def pca_fit(self, data=None):
        self.pca_transformer.fit(data if data is not None else self.data)

    def tfidf_transform(self):
        try:
            self.data = self.tfidf_transformer.transform(self.data)
        except NotFittedError as e:
            print("You must first fit on the data")
            raise e

    def pca_transform(self):
        try:
            self.data = self.pca_transformer.transform(self.data)
        except NotFittedError as e:
            print("You must first fit on the data")
            raise e

    def fit_pipeline(self, data=None):

        # getting the new data if it's passed to the function
        if data is not None: self.data = data

        self.preprocess()
        self.tfidf_fit()
        self.tfidf_transform()

        # sample_idx = np.random.choice(self.data.shape[0], size=round(self.data.shape[0]*0.25), replace=False)
        # self.data = self.data[sample_idx, :]
        
        self.data = self.data.toarray()

        # normalizing TfIdf
        self.data = normalize(self.data)

        self.pca_fit()

    def transform_pipeline(self, data=None):

        # getting the new data if it's passed to the function
        if data is not None: self.data = data

        self.preprocess()

        # transform TfIdf
        self.tfidf_transform()
        
        self.data = self.data.toarray()

        # normalizing TfIdf
        self.data = normalize(self.data)

        self.pca_transform()

        return self.data
    
    def save_object(self, path):
        pk.dump(self, open(path,"wb"))
        print(f"saved object in pikle file")