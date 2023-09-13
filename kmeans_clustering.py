import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('participants.csv')

selected_features = df[['trophies', 'finals', 'appearances', 'win_percent', 'years_participated', 'trophy_frequency', 'wins', 'semifinals', 'consecutive_trophies', 'finals_percent']]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(selected_features)

inertia = []
k_range = range(1,10)

for k in k_range:
  k_means = KMeans(n_clusters = k, random_state = 42)
  k_means.fit(scaled_features)
  inertia.append(k_means.inertia_)

plt.figure(figsize = (8,6))
plt.plot(k_range, inertia, marker = 'o', alpha = 0.8)
plt.xlabel('Number of Clusters(k)')
plt.ylabel('Within Cluster Sum of Squares')
plt.title('Elbow')

plt.grid(True)
plt.show()

optimal_k = 6;
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

cluster_table = df.loc[:,['name', 'cluster']]
cluster_table_ordered = cluster_table.sort_values(by = 'cluster')
print(cluster_table_ordered)





