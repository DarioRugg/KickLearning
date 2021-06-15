import numpy as np
import re, unicodedata

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import KernelPCA

# to be done:
# legmatization/stamming removing stopwords and punctation
# commenst

corpus = [
    'This is the first document.',
    'This documenting is the second document.',
    'And this is the third one.',
    'Is this the first documented?'
    'Smaller than Ben, \n ok',
    ' @gmail, people are studying',
    'Small, studied, study!!',
]

def preprocessor(text):
    stop_words = set(stopwords.words('english'))
    line = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()

    word = line.split()
    # removing stop words
    words = [w for w in word if not w in stop_words]
    # removing punctuation from each word
    punct_remove = [" ".join(tokenizer.tokenize(w)) for w in words]
    # removing stem
    stemmed = [ps.stem(w) for w in punct_remove]
    # removing Accent
    accents = [unicodedata.normalize(u'NFKD', w).encode('ascii', 'ignore').decode('utf8') for w in stemmed]
    line = " ".join(accents)
    
    return line.strip().split()
    
'''
Printed output: it works with "one.", "documenting", "documented", "@gmail", "smaller and small"

['first', 'document']
['document', 'second', 'document']
['third', 'one']
['first', 'documented', 'smal', 'ben', 'ok']
['gmail', 'peopl', 'studi']
['small', 'studi', 'studi']
'''    
   

vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
features_dataset = vectorizer.fit_transform(corpus)


# print(vectorizer.get_feature_names())
# print(features_dataset.shape)
# for text embeddings information

transformer = KernelPCA(n_components=2, kernel="rbf")
features_dataset_transformed = transformer.fit_transform(features_dataset)

print(features_dataset_transformed.shape)

'''
def tokenizer(text):
    text = text.translate(str.maketrans('', '', string.punctuation))

symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
for i in symbols:
    data = np.char.replace(data, i, ' ')
re.sub(symbols, )
'''