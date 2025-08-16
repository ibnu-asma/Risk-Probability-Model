
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """Loads data from a CSV file."""
    logging.info(f"Loading data from {filepath}")
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"File not found at {filepath}")
        return None

def calculate_rfm(df):
    """Calculates RFM metrics for each customer."""
    logging.info("Calculating RFM metrics")
    # Ensure TransactionStartTime is in datetime format
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime']).dt.tz_localize(None)
    
    # Set a snapshot date for recency calculation
    snapshot_date = pd.to_datetime('2025-06-30')
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda date: (snapshot_date - date.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    })

    rfm.rename(columns={'TransactionStartTime': 'Recency',
                        'TransactionId': 'Frequency',
                        'Amount': 'Monetary'}, inplace=True)
    
    return rfm

def identify_high_risk_cluster(rfm):
    """Identifies the high-risk cluster based on RFM metrics."""
    logging.info("Identifying high-risk cluster")
    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster (low frequency, low monetary)
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })
    
    high_risk_cluster = cluster_summary.sort_values(by=['Frequency', 'Monetary'], ascending=True).index[0]
    logging.info(f"High-risk cluster identified as: {high_risk_cluster}")
    
    return rfm, high_risk_cluster

def main():
    """Main function to run the RFM clustering and high-risk identification."""
    processed_path = "C:/Users/Cyber Defense/Desktop/week5/credit-risk-model/data/processed/data_processed.csv"
    output_path = "C:/Users/Cyber Defense/Desktop/week5/credit-risk-model/data/processed/data_processed_with_risk.csv"

    df = load_data(processed_path)
    if df is None:
        return

    rfm_df = calculate_rfm(df)
    rfm_with_clusters, high_risk_cluster_id = identify_high_risk_cluster(rfm_df)

    # Create the is_high_risk column
    rfm_with_clusters['is_high_risk'] = (rfm_with_clusters['Cluster'] == high_risk_cluster_id).astype(int)

    # Merge the is_high_risk column back to the original dataframe
    df = df.merge(rfm_with_clusters[['is_high_risk']], on='CustomerId', how='left')

    df.to_csv(output_path, index=False)
    logging.info(f"Data with is_high_risk column saved to {output_path}")

if __name__ == "__main__":
    main()
