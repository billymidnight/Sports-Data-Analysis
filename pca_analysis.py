
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('participants.csv')

# Select numeric features
numeric_features = ['trophies', 'finals', 'appearances', 'win_percent', 'years_participated', 'trophy_frequency', 'wins', 'semifinals', 'consecutive_trophies', 'finals_percent']

# Extract the numeric features
X = df[numeric_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
principal_components = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()



# Project data onto the first two principal components
X_projected = principal_components[:, :2]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_projected[:, 0], X_projected[:, 1], c=df['rating'], cmap='viridis', edgecolor='k', alpha=0.6)

# Add participant names to the plot
for i, name in enumerate(df['name']):
    plt.annotate(name, (X_projected[i, 0], X_projected[i, 1]), textcoords='offset points', xytext=(5,5), ha='left')

plt.colorbar(label='Rating')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Data Projection onto First Two Principal Components')
plt.show()
