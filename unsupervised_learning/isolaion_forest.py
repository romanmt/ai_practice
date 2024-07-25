import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load the iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# add cluster labels to the dataset
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Add cluster labels to the dataset
X['Cluster'] = kmeans.labels_

# Fit isolation forest to model
isoforest = IsolationForest(contamination=0.1, random_state=42)
X['Anomaly'] = isoforest.fit_predict(X.drop('Cluster', axis=1))

# Anomalous data points
anomalies = X[X['Anomaly'] == -1]
print(anomalies)

# plotting
plt.figure(figsize=(8, 6))
plt.scatter(X['petal length (cm)'], X['petal width (cm)'], c=X['Anomaly'], cmap='coolwarm')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Anomaly Detection in Iris Dataset using Isolation Forest')
plt.show()
