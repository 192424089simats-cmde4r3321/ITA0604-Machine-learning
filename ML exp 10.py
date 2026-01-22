from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture

X = load_iris().data

em = GaussianMixture(n_components=3)
em.fit(X)

labels = em.predict(X)
print("Cluster Labels:", labels)
