import unicodedata

import numpy as np

from sklearn.exceptions import NotFittedError

import torch
from transformers import BertModel, BertTokenizer

from sklearn.decomposition import KernelPCA


class TextEncoder:
    def __init__(self, data=None, model_name="bert-base-cased", kpca_n_components=7):

        self.data = data

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.model = BertModel.from_pretrained(model_name)

        self.pca_transformer = KernelPCA(n_components=kpca_n_components, kernel="rbf")

    def new_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def pca_fit(self, data=None):
        self.pca_transformer.fit(data if data is not None else self.data)

    def pca_transform(self):
        try:
            self.data = self.pca_transformer.transform(self.data)
        except NotFittedError as e:
            print("You must first fit on the data")
            raise e

    def complete_pipeline(self, data=None, fit_on_data=False):

        # getting the new data if it's passed to the function
        if data is not None: self.new_data(data)

        # tokenizing
        tokenized = self.tokenizer(self.data)

        # padding
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

        # attention masks
        attention_mask = np.where(padded != 0, 1, 0)

        # to tensors
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        # embeddings
        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        #getting the CLS token
        self.data = last_hidden_states[0][:,0,:].numpy()

        # dimensionality reduction
        try:
            if fit_on_data: self.pca_fit()
            self.pca_transform()
        except NotFittedError as e:
            print("You can also set 'fit_on_data=True'")
            raise e

        return self.get_data()
