# for historic information
from sklearn.decomposition import TruncatedSVD

dataset = None

svd_transform = TruncatedSVD(n_components=4)
new_dataset = svd_transform.fit_transform(dataset)