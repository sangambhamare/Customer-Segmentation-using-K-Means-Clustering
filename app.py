import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import kagglehub

# Set Streamlit App Title
st.title("Customer Segmentation using K-Means Clustering")

# Download and Load Dataset
@st.cache_data
def load_data():
    try:
        path = kagglehub.dataset_download("vjchoudhary7/customer-segmentation-tutorial-in-python")
        df = pd.read_csv(f"{path}/Mall_Customers.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

# Check if dataset is loaded
if df is not None:
    # Display dataset preview
    st.subheader("Customer Data Preview")
    st.write(df.head())

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    df = df.drop(["CustomerID", "Gender"], axis=1)  # Drop non-numeric columns
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Determine Optimal K using Elbow Method
    def optimal_k(X):
        distortions = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
        
        fig, ax = plt.subplots()
        ax.plot(K, distortions, marker='o', linestyle='-')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Distortion")
        ax.set_title("Elbow Method for Optimal K")
        return fig

    st.subheader("Elbow Method")
    st.pyplot(optimal_k(df_scaled))

    # Select number of clusters
    k = st.slider("Select Number of Clusters (K)", 2, 10, 5)

    # Apply K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)

    df["Cluster"] = clusters

    # Display Clustered Data
    st.subheader("Clustered Data")
    st.write(df)

    # Visualizing Clusters
    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots()
    scatter = ax.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster"], cmap="viridis", alpha=0.7)
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Clusters")
    st.pyplot(fig)

    # Download segmented customer data
    st.subheader("Download Clustered Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=csv, file_name="customer_segments.csv", mime="text/csv")

    st.success("Customer segmentation completed successfully!")
else:
    st.warning("Unable to load the dataset. Please check KaggleHub access.")
