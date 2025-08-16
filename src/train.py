
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import logging
from data_processing import get_preprocessor # We can reuse the preprocessor

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

def train_and_evaluate(df):
    """Trains, evaluates, and logs models using MLflow."""
    logging.info("Starting model training and evaluation")

    # Define features (X) and target (y)
    # Drop non-feature columns
    X = df.drop(["is_high_risk", "TransactionId", "BatchId", "AccountId", "SubscriptionId", "CustomerId", "CurrencyCode", "CountryCode", "ProviderId", "ProductId", "TransactionStartTime"], axis=1)
    y = df["is_high_risk"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Get the preprocessor from our data_processing script
    preprocessor = get_preprocessor()

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    # Define hyperparameter grids
    param_grids = {
        "LogisticRegression": {
            'classifier__C': [0.1, 1, 10]
        },
        "GradientBoosting": {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1]
        }
    }

    mlflow.set_experiment("Credit_Risk_Modeling")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            logging.info(f"Training {model_name}...")
            
            # Create pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', model)])

            # GridSearchCV
            grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=3, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Log params and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            logging.info(f"{model_name} ROC-AUC: {roc_auc:.4f}")

    logging.info("Model training and evaluation complete.")

def main():
    """Main function to run the training pipeline."""
    data_path = "C:/Users/Cyber Defense/Desktop/week5/credit-risk-model/data/processed/data_processed_with_risk.csv"
    df = load_data(data_path)
    if df is not None:
        # For the purpose of this script, fill NA in target column if any
        df['is_high_risk'].fillna(0, inplace=True)
        train_and_evaluate(df)

if __name__ == "__main__":
    main()
