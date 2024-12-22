import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# Set pandas options for better display
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 20)  
pd.set_option('display.float_format', '{:.2f}'.format)  

# Function to load data and handle errors
def load_data(file):
    try:
        df = pd.read_csv(file)
        st.write("Dataset Shape:", df.shape)
        st.write("First 10 rows of the dataset:", df.head(10))
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path.")
        st.stop()  # Stops execution if the dataset is not found
    except pd.errors.EmptyDataError:
        st.error("The dataset file is empty. Please check the file content.")
        st.stop()  # Stops execution if the dataset is empty
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        st.stop()  # Stops execution if another error occurs

# Function for Data Cleaning and Preprocessing page
def data_cleaning_page(df):
    # Page Background Color (Light Blue)
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f8ff;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.title("Data Cleaning and Preprocessing")

    # Ensure the dataframe is loaded properly before proceeding
    if not df.empty:
        st.write("Dataset Shape:", df.shape)
        st.write("First 10 rows of the dataset:", df.head(10))

        # Countplot for Transactions by Country
        plt.figure(figsize=(9, 10))
        sns.countplot(y='Country', data=df)
        plt.title("Number of Transactions by Country")
        st.pyplot()

        # Missing values
        st.write("Missing Values:\n", df.isnull().sum())

        # Remove rows with missing CustomerID
        df = df[df.CustomerID.notnull()]
        df['CustomerID'] = df.CustomerID.astype(int)

        st.write("First 5 Customer IDs:\n", df.CustomerID.head())

        # Calculate 'Sales' column
        df['Sales'] = df.Quantity * df.UnitPrice
        st.write("First 5 Sales Values:\n", df.Sales.head())

        # Save cleaned data to CSV
        cleaned_file_path = 'Files/cleaned_transactions.csv'
        df.to_csv(cleaned_file_path, index=None)

        # Aggregation of invoice, product, and sales data
        invoice_data = df.groupby('CustomerID').agg(total_transactions=('InvoiceNo', 'nunique'))
        product_data = df.groupby('CustomerID').agg(
            total_products=('StockCode', 'count'),
            total_unique_products=('StockCode', 'nunique')
        )
        sales_data = df.groupby('CustomerID').agg(
            total_sales=('Sales', 'sum'),
            avg_product_value=('Sales', 'mean')
        )

        # Display aggregated data
        st.write("Invoice Data:\n", invoice_data.head())
        st.write("Product Data:\n", product_data.head())
        st.write("Sales Data:\n", sales_data.head())

        # Visualizing total transactions by customer
        plt.figure(figsize=(9, 6))
        sns.barplot(x=invoice_data.index[:10], y=invoice_data['total_transactions'][:10])
        plt.title("Top 10 Customers by Total Transactions")
        plt.xlabel("CustomerID")
        plt.ylabel("Total Transactions")
        plt.xticks(rotation=45)
        st.pyplot()
    else:
        st.error("Failed to load the dataset properly.")
        st.stop()

# Function for Clustering and Visualization page
def clustering_page():
    # Page Background Color (Red)
    st.markdown(
        """
        <style>
        body {
            background-color: #ff0000;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    st.title("Clustering and Visualization")

    # Load cleaned and transformed data
    cleaned_file_path = 'Files/cleaned_transactions.csv'
    df = pd.read_csv(cleaned_file_path)
    item_dummies = pd.get_dummies(df.StockCode)
    item_dummies['CustomerID'] = df.CustomerID
    item_data = item_dummies.groupby('CustomerID').sum()

    # Standardize and apply PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    item_data_scaled = scaler.fit_transform(item_data)
    pca = PCA()
    pca.fit(item_data_scaled)
    PC_items = pca.transform(item_data_scaled)

    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot Cumulative Explained Variance
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(range(len(cumulative_explained_variance)), cumulative_explained_variance, marker='o')
    plt.title('Cumulative Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    st.pyplot()

    # Reduce dimensions to 300 (or fewer)
    n_components = min(300, item_data_scaled.shape[1])
    pca = PCA(n_components=n_components)
    PC_items = pca.fit_transform(item_data_scaled)

    items_pca = pd.DataFrame(PC_items)
    items_pca.columns = ['PC{}'.format(i + 1) for i in range(PC_items.shape[1])]
    items_pca.index = item_data.index

    # Save PCA transformed data
    pca_file_path = 'Files/pca_item_data.csv'
    items_pca.to_csv(pca_file_path)

    # Aggregate sales data for joining with PCA results
    sales_data = df.groupby('CustomerID').agg(
        total_sales=('Sales', 'sum'),
        avg_product_value=('Sales', 'mean')
    )
    
    # Merge the PCA data with the aggregated sales data
    pca_df = pd.read_csv(pca_file_path, index_col=0)
    pca_df = pca_df.join(sales_data)

    # Clustering on PCA-transformed data
    from sklearn.cluster import KMeans

    pca_scaler = StandardScaler()
    pca_df_scaled = pca_scaler.fit_transform(pca_df.dropna())
    p_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=123)
    p_kmeans.fit(pca_df_scaled)
    pca_df['cluster'] = p_kmeans.predict(pca_df_scaled)

    # Visualize clustering result
    sns.lmplot(x='total_sales', y='avg_product_value', hue='cluster', data=pca_df, fit_reg=False)
    plt.title('Cluster Analysis: Total Sales vs Average Cart Value')
    plt.xlabel('Total Sales')
    plt.ylabel('Average Cart Value')
    st.pyplot()

    # Adjusted Rand Score
    threshold_df = pd.read_csv('Files/analytical_base_table.csv')
    threshold_scaler = StandardScaler()
    threshold_df_scaled = threshold_scaler.fit_transform(threshold_df)
    t_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=123)
    t_kmeans.fit(threshold_df_scaled)
    threshold_df['cluster'] = t_kmeans.predict(threshold_df_scaled)

    # Adjusted Rand Score comparison
    from sklearn.metrics import adjusted_rand_score
    adjusted_rand = adjusted_rand_score(pca_df['cluster'], threshold_df['cluster'])
    st.write(f"Adjusted Rand Score between PCA and Threshold Clustering: {adjusted_rand:.2f}")

# Main function to run the app
def main():
    # Load dataset
    df = load_data('Files/dataset.csv')

    # Sidebar page selection with unique keys
    page = st.sidebar.radio("Select Page", ["Data Cleaning and Preprocessing", "Clustering and Visualization"], key="page_selector")

    # Page navigation logic
    if page == "Data Cleaning and Preprocessing":
        data_cleaning_page(df)
    elif page == "Clustering and Visualization":
        clustering_page()

if __name__ == "__main__":
    main() 