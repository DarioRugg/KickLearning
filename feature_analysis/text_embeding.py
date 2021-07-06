import unicodedata

from sklearn.exceptions import NotFittedError

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import KernelPCA

nltk.download('stopwords', quiet=True)


class TextEncoder:
    def __init__(self, kpca_n_components=7, data=None):

        self.data = data

        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stammer = SnowballStemmer(language="english")

        self.tfidf_transformer = TfidfVectorizer(stop_words=stopwords.words('english'))

        self.pca_transformer = KernelPCA(n_components=kpca_n_components, kernel="rbf")

    def _preprocessing(self, text):
        line = text.lower()

        word = line.split()
        # removing stop words
        words = filter(lambda w: w not in self.stop_words, word)
        # removing punctuation from each word
        punct_remove = list(map(lambda w: " ".join(self.tokenizer.tokenize(w)), words))
        # removing stem
        stemmed = list(map(self.stammer.stem, punct_remove))
        # removing Accent
        accents = list(
            map(lambda w: unicodedata.normalize(u'NFKD', w).encode('ascii', 'ignore').decode('utf8'), stemmed))
        line = " ".join(accents)

        return line.strip()

    def new_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

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

    def fit_tfidf_pipeline(self, data=None):

        # getting the new data if it's passed to the function
        if data is not None: self.new_data(data)

        self.preprocess()
        self.tfidf_fit()

    def transform_pipeline(self, data=None, fit_on_data=False):

        # getting the new data if it's passed to the function
        if data is not None: self.new_data(data)

        self.preprocess()

        if fit_on_data: self.tfidf_fit()

        try:
            self.tfidf_transform()
        except NotFittedError as e:
            print("You can also set 'fit_on_data=True'")
            raise e

        self.pca_fit()
        self.pca_transform()

        return self.get_data()
