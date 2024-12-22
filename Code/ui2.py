import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
import models as md
warnings.filterwarnings("ignore")

# Read the dataset
df = pd.read_csv('Files/dataset.csv')
st.write("### Dataset Shape:", df.shape)
st.write("### First 10 rows of the dataset:", df.head(10))

# Plotting the country transaction distribution
plt.figure(figsize=(9, 10))
sns.countplot(y='Country', data=df)
plt.title("Number of Transactions by Country")
st.pyplot(plt)
numeric_columns = [col for col in ['Quantity', 'UnitPrice', 'Sales'] if col in df.columns]

if numeric_columns:
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(1, len(numeric_columns), i)
        sns.boxplot(y=df[column])
        plt.title(f'Box Plot of {column}')
        plt.ylabel(column)

    plt.tight_layout()
    st.pyplot(plt)
else:
    st.write("No valid numeric columns found for box plots.")


# Show missing values
st.write("### Missing Values:", df.isnull().sum())

# Clean data
df = df[df.CustomerID.notnull()]
df['CustomerID'] = df.CustomerID.astype(int)
st.write("### First 5 Customer IDs:", df.CustomerID.head())

df['Sales'] = df.Quantity * df.UnitPrice
st.write("### First 5 Sales Values:", df.Sales.head())

df.to_csv('Files/cleaned_transactions.csv', index=None)

# Aggregation of data
invoice_data = df.groupby('CustomerID').agg(total_transactions=('InvoiceNo', 'nunique'))
product_data = df.groupby('CustomerID').agg(total_products=('StockCode', 'count'), total_unique_products=('StockCode', 'nunique'))
sales_data = df.groupby('CustomerID').agg(total_sales=('Sales', 'sum'), avg_product_value=('Sales', 'mean'))

# Plot Top 10 Customers by Total Transactions
plt.figure(figsize=(9, 6))
sns.barplot(x=invoice_data.index[:10], y=invoice_data['total_transactions'][:10])
plt.title("Top 10 Customers by Total Transactions")
plt.xlabel("CustomerID")
plt.ylabel("Total Transactions")
plt.xticks(rotation=45)
st.pyplot(plt)

# Further aggregation
cart_data = df.groupby(['CustomerID', 'InvoiceNo']).agg(cart_value=('Sales', 'sum'))
cart_data.reset_index(inplace=True)
agg_cart_data = cart_data.groupby('CustomerID').agg(avg_cart_value=('cart_value', 'mean'), min_cart_value=('cart_value', 'min'), max_cart_value=('cart_value', 'max'))

# Combine customer data
customer_df = invoice_data.join([product_data, sales_data, agg_cart_data])

# Write customer data to CSV
customer_df.to_csv('Files/analytical_base_table.csv')

# More aggregation and transformation
item_dummies = pd.get_dummies(df.StockCode)
item_dummies['CustomerID'] = df.CustomerID
item_data = item_dummies.groupby('CustomerID').sum()

top_120_items = item_data.sum().sort_values().tail(120).index
top_120_item_data = item_data[top_120_items]
top_120_item_data.to_csv('Files/threshold_item_data.csv')

# Scaling and PCA
scaler = StandardScaler()
item_data_scaled = scaler.fit_transform(item_data)
pca = PCA()
pca.fit(item_data_scaled)
PC_items = pca.transform(item_data_scaled)

# Cumulative explained variance plot
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 6))
plt.grid()
plt.plot(range(len(cumulative_explained_variance)), cumulative_explained_variance, marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
st.pyplot(plt)

# Reduced PCA components
n_components = min(300, item_data_scaled.shape[1])
pca = PCA(n_components=n_components)
PC_items = pca.fit_transform(item_data_scaled)
items_pca = pd.DataFrame(PC_items)
items_pca.columns = ['PC{}'.format(i + 1) for i in range(PC_items.shape[1])]
items_pca.index = item_data.index

# Write PCA data to CSV
items_pca.to_csv('Files/pca_item_data.csv')

# Join data and apply KMeans clustering
base_df = pd.read_csv('Files/analytical_base_table.csv', index_col=0)
threshold_item_data = pd.read_csv('Files/threshold_item_data.csv', index_col=0)
pca_item_data = pd.read_csv('Files/pca_item_data.csv', index_col=0)

threshold_df = base_df.join(threshold_item_data)
pca_df = base_df.join(pca_item_data)

# Scaling data for KMeans clustering
scaler = StandardScaler()
threshold_df_scaled = scaler.fit_transform(threshold_df)
pca_df_scaled = scaler.fit_transform(pca_df)

# KMeans clustering and visualization
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=123)
kmeans.fit(threshold_df_scaled)
threshold_df['cluster'] = kmeans.predict(threshold_df_scaled)
sns.lmplot(x='total_sales', y='avg_cart_value', hue='cluster', data=threshold_df, fit_reg=False)
plt.title('Cluster Analysis: Total Sales vs Average Cart Value')
st.pyplot(plt)

# Display clustering metrics
kmeans_ari = md.kmeans_ari
kmeans_mse = md.kmeans_mse
kmeans_accuracy = md.kmeans_accuracy

# Calculate Silhouette Score
kmeans_silhouette = silhouette_score(threshold_df_scaled, threshold_df['cluster'])

# Calculate Davies-Bouldin Index
kmeans_db_index = davies_bouldin_score(threshold_df_scaled, threshold_df['cluster'])

# Calculate Dunn Index
# The Dunn Index requires custom implementation
dunn_index = md.calculate_dunn_index(threshold_df_scaled, threshold_df['cluster'])

# Calculate Calinski-Harabasz Index
kmeans_ch_index = calinski_harabasz_score(threshold_df_scaled, threshold_df['cluster'])

st.write(f"KMeans ARI: {kmeans_ari}")
st.write(f"KMeans MSE: {kmeans_mse}")
st.write(f"KMeans Accuracy (%): {kmeans_accuracy}")
st.write(f"KMeans Silhouette Score: {kmeans_silhouette:.2f}")
st.write(f"KMeans Davies-Bouldin Index: {kmeans_db_index:.2f}")
st.write(f"KMeans Dunn Index: {dunn_index:.2f}")
st.write(f"KMeans Calinski-Harabasz Index: {kmeans_ch_index:.2f}")

# --- DBSCAN Clustering ---
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(threshold_df_scaled)
threshold_df['dbscan_cluster'] = dbscan_labels

# Calculate Adjusted Rand Index and Mean Squared Error for DBSCAN
dbscan_ari = adjusted_rand_score(threshold_df['cluster'], threshold_df['dbscan_cluster'])
dbscan_mse = mean_squared_error(threshold_df['cluster'], threshold_df['dbscan_cluster'])

# Calculate accuracy percentage for DBSCAN
dbscan_accuracy = (dbscan_ari + 1) / 2 * 100

# Calculate Silhouette Score for DBSCAN
dbscan_silhouette = silhouette_score(threshold_df_scaled, threshold_df['dbscan_cluster'])

# Calculate Davies-Bouldin Index for DBSCAN
dbscan_db_index = davies_bouldin_score(threshold_df_scaled, threshold_df['dbscan_cluster'])

# Calculate Dunn Index for DBSCAN
dbscan_dunn_index = md.calculate_dunn_index(threshold_df_scaled, threshold_df['dbscan_cluster'])

# Calculate Calinski-Harabasz Index for DBSCAN
dbscan_ch_index = calinski_harabasz_score(threshold_df_scaled, threshold_df['dbscan_cluster'])

# --- Agglomerative Clustering ---
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(threshold_df_scaled)
threshold_df['agg_cluster'] = agg_labels

# Calculate Adjusted Rand Index and Mean Squared Error for Agglomerative Clustering
agg_ari = adjusted_rand_score(threshold_df['cluster'], threshold_df['agg_cluster'])
agg_mse = mean_squared_error(threshold_df['cluster'], threshold_df['agg_cluster'])

# Calculate accuracy percentage for Agglomerative Clustering
agg_accuracy = (agg_ari + 1) / 2 * 100

# Calculate Silhouette Score for Agglomerative Clustering
agg_silhouette = silhouette_score(threshold_df_scaled, threshold_df['agg_cluster'])

# Calculate Davies-Bouldin Index for Agglomerative Clustering
agg_db_index = davies_bouldin_score(threshold_df_scaled, threshold_df['agg_cluster'])

# Calculate Dunn Index for Agglomerative Clustering
agg_dunn_index = md.calculate_dunn_index(threshold_df_scaled, threshold_df['agg_cluster'])

# Calculate Calinski-Harabasz Index for Agglomerative Clustering
agg_ch_index = calinski_harabasz_score(threshold_df_scaled, threshold_df['agg_cluster'])

# --- Gaussian Mixture Model (GMM) ---
gmm = GaussianMixture(n_components=3, random_state=123)
gmm_labels = gmm.fit_predict(threshold_df_scaled)
threshold_df['gmm_cluster'] = gmm_labels

# Calculate Adjusted Rand Index and Mean Squared Error for GMM
gmm_ari = md.gmm_ari
gmm_mse = md.gmm_mse

# Calculate accuracy percentage for GMM
gmm_accuracy = md.gmm_accuracy

# Calculate Silhouette Score for GMM
gmm_silhouette = silhouette_score(threshold_df_scaled, threshold_df['gmm_cluster'])

# Calculate Davies-Bouldin Index for GMM
gmm_db_index = davies_bouldin_score(threshold_df_scaled, threshold_df['gmm_cluster'])

# Calculate Dunn Index for GMM
gmm_dunn_index = md.calculate_dunn_index(threshold_df_scaled, threshold_df['gmm_cluster'])

# Calculate Calinski-Harabasz Index for GMM
gmm_ch_index = calinski_harabasz_score(threshold_df_scaled, threshold_df['gmm_cluster'])

# --- Display All Metrics ---
st.write(f"DBSCAN ARI: {dbscan_ari}")
# st.write(f"DBSCAN MSE: {dbscan_mse}")
# st.write(f"DBSCAN Accuracy (%): {dbscan_accuracy}")
st.write(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")
st.write(f"DBSCAN Davies-Bouldin Index: {dbscan_db_index:.2f}")
st.write(f"DBSCAN Dunn Index: {dbscan_dunn_index:.2f}")
st.write(f"DBSCAN Calinski-Harabasz Index: {dbscan_ch_index:.2f}")

st.write(f"Agglomerative ARI: {agg_ari}")
# st.write(f"Agglomerative MSE: {agg_mse}")
# st.write(f"Agglomerative Accuracy (%): {agg_accuracy}")
st.write(f"Agglomerative Silhouette Score: {agg_silhouette:.2f}")
st.write(f"Agglomerative Davies-Bouldin Index: {agg_db_index:.2f}")
st.write(f"Agglomerative Dunn Index: {agg_dunn_index:.2f}")
st.write(f"Agglomerative Calinski-Harabasz Index: {agg_ch_index:.2f}")

st.write(f"GMM ARI: {gmm_ari}")
# st.write(f"GMM MSE: {gmm_mse}")
# st.write(f"GMM Accuracy (%): {gmm_accuracy}")
st.write(f"GMM Silhouette Score: {gmm_silhouette:.2f}")
st.write(f"GMM Davies-Bouldin Index: {gmm_db_index:.2f}")
st.write(f"GMM Dunn Index: {gmm_dunn_index:.2f}")
st.write(f"GMM Calinski-Harabasz Index: {gmm_ch_index:.2f}")
