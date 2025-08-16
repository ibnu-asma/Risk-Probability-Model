
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import logging
from xverse.transformer import WOE

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

def create_aggregate_features(df):
    """Creates aggregate features for each customer."""
    logging.info("Creating aggregate features")
    aggregates = {
        'Amount': ['sum', 'mean', 'std', 'count'],
        'TransactionId': ['count']
    }
    agg_df = df.groupby('CustomerId').agg(aggregates)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    return df.merge(agg_df, on='CustomerId', how='left')

def extract_datetime_features(df):
    """Extracts datetime features from the 'TransactionStartTime' column."""
    logging.info("Extracting datetime features")
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['hour'] = df['TransactionStartTime'].dt.hour
    df['day'] = df['TransactionStartTime'].dt.day
    df['month'] = df['TransactionStartTime'].dt.month
    df['year'] = df['TransactionStartTime'].dt.year
    return df

def get_preprocessor():
    """Returns a ColumnTransformer for preprocessing numerical and categorical features."""
    numerical_features = ['Amount', 'Amount_sum', 'Amount_mean', 'Amount_std', 'Amount_count', 'TransactionId_count']
    categorical_features = ['ProductCategory', 'ChannelId']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def get_day_of_week(df):
    """Extracts day of the week from TransactionStartTime."""
    df['day_of_week'] = pd.to_datetime(df['TransactionStartTime']).dt.dayofweek
    return df

def main():
    """Main function to run the data processing pipeline."""
    data_path = "C:/Users/Cyber Defense/Desktop/week5/credit-risk-model/data/raw/data.csv"
    processed_path = "C:/Users/Cyber Defense/Desktop/week5/credit-risk-model/data/processed/data_processed.csv"

    df = load_data(data_path)
    if df is None:
        return

    df = create_aggregate_features(df)
    df = extract_datetime_features(df)

    # For WoE, we need a binary target. We'll assume 'is_high_risk' is created in a later step.
    # For now, we'll just demonstrate the pipeline without WoE fitting.
    # If 'is_high_risk' column existed, you would fit the WoE transformer:
    # woe = WOE()
    # woe.fit(df[categorical_features], df['is_high_risk'])
    # df_woe = woe.transform(df[categorical_features])

    preprocessor = get_preprocessor()
    
    logging.info("Applying preprocessing pipeline")
    processed_data = preprocessor.fit_transform(df)
    
    # Note: The output of the preprocessor is a numpy array. 
    # For saving to CSV with meaningful column names, you would need to reconstruct the DataFrame.
    # This is left as an exercise as the structure depends on the final feature set.
    
    # For now, we save the dataframe with the new features before the final preprocessing step
    # to be used in the next stages of the pipeline.
    df.to_csv(processed_path, index=False)
    logging.info(f"Processed data saved to {processed_path}")


if __name__ == "__main__":
    main()
