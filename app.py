from re import X
# from scipy import cluster
import streamlit as st
import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, k_means
from sklearn.metrics import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

# @st.cache
def load_data():
    # Load dataset
    df = pd.read_csv(r'C:\\Users\\Hello\Desktop\\Html Tutorial\Document\\KAIM Courses\Week-02 Streamlit\data\Week1_challenge_data_source(CSV).csv')
    return df

# Page 1: User Overview Analysis
def user_overview_analysis(df):
    st.title("User Overview Analysis")
    
    # Display basic statistics of the dataset
    st.write("Dataset Overview:")
    st.write(df.describe())

    # Plot: Distribution of Throughput by Handset Type
    fig, ax = plt.subplots()
    sns.boxplot(x='Handset Type', y='Avg Bearer TP DL (kbps)', data=df, ax=ax)
    plt.xticks(rotation=90)
    plt.title('Throughput Distribution by Handset Type')
    st.pyplot(fig)

# Page 2: User Engagement Analysis
def user_engagement_analysis(df):
    st.title("User Engagement Analysis")
    
    # Engagement score calculation
    df = df.iloc[:150000]
    X = df[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['Engagement Cluster'] = clusters
    df = df.iloc[:len(clusters)]
    st.write(X)

    # Plot: Engagement clusters
    fig, ax = plt.subplots()
    plt.scatter(X['Avg Bearer TP DL (kbps)'], X['Avg Bearer TP UL (kbps)'], c=clusters, cmap='viridis')
    plt.title('User Engagement Clusters')
    st.pyplot(fig)
    
# Page 3: Satisfaction Analysis
def satisfaction_analysis(df):
    st.title("Satisfaction Analysis")
    features = df.select_dtypes(include=[float, int])
    features_filled = features.fillna(features.mean())


    # Initialize KMeans with the number of clusters
    k_means = KMeans(n_clusters=3)

    # Fit KMeans to your data (replace 'data' with your dataset)
    k_means.fit(features_filled)

    # Now you can access cluster centers
    cluster_centers = k_means.cluster_centers_

    k_means = KMeans(n_clusters=3, random_state=42)

    # Identify the cluster with lower throughput and retransmission (less engaged cluster)
    low_throughput_value = cluster_centers[:, 0].min()  # Low throughput value
    low_tcp_retrans_value = cluster_centers[:, 1].min()  # Low TCP retransmission value

    # Identify the cluster with higher throughput and retransmission (highly engaged cluster)
    high_throughput_value = cluster_centers[:, 0].max()  # high throughput value
    high_tcp_retrans_value = cluster_centers[:, 1].max()  # high TCP retransmission value

    # Example cluster center values (replace with your actual cluster centers)
    less_engaged_cluster_center = [low_throughput_value, low_tcp_retrans_value]
    worst_experience_cluster_center = [high_tcp_retrans_value, low_throughput_value]


    df['Avg_Throughput'] = (df['Avg Bearer TP DL (kbps)'] + df['Avg Bearer TP UL (kbps)']) / 2

    df['Avg_TCP_Retrans'] = (df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']) / 2

    df['Avg_Throughput'].fillna(df['Avg_Throughput'].mean(), inplace=True)
    df['Avg_TCP_Retrans'].fillna(df['Avg_TCP_Retrans'].mean(), inplace=True)

    df['Engagement_Score'] = df.apply(lambda row: euclidean(
    [row['Avg_Throughput'], row['Avg_TCP_Retrans']], less_engaged_cluster_center), axis=1)

    df['Experience_Score'] = df.apply(lambda row: euclidean(
    [row['Avg_Throughput'], row['Avg_TCP_Retrans']], worst_experience_cluster_center), axis=1)

    df['Satisfaction Score'] = (df['Experience_Score'] + df['Engagement_Score']) / 2
    
    # Top 10 Satisfied Customers
    top_customers = df[['Bearer Id', 'Satisfaction Score']].sort_values(by='Satisfaction Score', ascending=False).head(10)
    st.write("Top 10 Satisfied Customers:")
    st.write(top_customers)

    # Satisfaction Score Distribution
    fig, ax = plt.subplots()
    sns.histplot(df['Satisfaction Score'], kde=True, ax=ax)
    plt.title('Satisfaction Score Distribution')
    st.pyplot(fig)

# Main App
def main():
    st.sidebar.title("Telecom Dashboard")
    
    # Sidebar for Navigation
    page = st.sidebar.selectbox("Select a Page", ["User Overview Analysis", "User Engagement Analysis", "Satisfaction Analysis"])
    
    # Load data
    df = load_data()
    
    # Display selected page
    if page == "User Overview Analysis":
        user_overview_analysis(df)
    elif page == "User Engagement Analysis":
        user_engagement_analysis(df)
    elif page == "Satisfaction Analysis":
        satisfaction_analysis(df)

if __name__ == "__main__":
    main()