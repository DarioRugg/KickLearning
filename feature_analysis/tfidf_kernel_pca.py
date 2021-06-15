import numpy as np
import re
import string

# to be done:
# legmatization/stamming removing stopwords and punctation

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.corpus import

from sklearn.decomposition import KernelPCA

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

def tokenizer(text):
    text = text.translate(str.maketrans('', '', string.punctuation))

symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
for i in symbols:
    data = np.char.replace(data, i, ' ')
re.sub(symbols, )
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
features_dataset = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())

print(features_dataset.shape)


# for text embeddings information

transformer = KernelPCA(n_components=2, kernel="rbf")
features_dataset_transformed = transformer.fit_transform(features_dataset)

print(features_dataset_transformed.shape)

def ciao():
    pass