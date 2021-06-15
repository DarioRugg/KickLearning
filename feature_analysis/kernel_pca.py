from sklearn.decomposition import KernelPCA

features_dataset = None

transformer = KernelPCA(n_components=7, kernel="rbf")
features_dataset_transformed = transformer.fit_transform(features_dataset)