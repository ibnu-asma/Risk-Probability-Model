# B5W5 Credit Risk Probability Model for Alternative Data: A Project Retrospective

This post details the development process and key outcomes of the B5W5 Credit Risk Probability Model for Alternative Data project. We'll walk through each phase, from initial setup to model deployment, highlighting the tools and methodologies employed.

## Project Setup and Foundation

The project began with establishing a robust and organized directory structure, crucial for managing various components of a data science project. This involved creating dedicated folders for data (raw and processed), notebooks, source code (including API components), and testing. Essential configuration files like `requirements.txt`, `.gitignore`, `Dockerfile`, and `docker-compose.yml` were also set up to ensure a consistent development environment.

## Task 1: Understanding Credit Risk

This initial phase focused on gaining a deep understanding of credit risk in the context of alternative data and regulatory frameworks like Basel II. A dedicated section in the `README.md` was created to document insights on:
- The influence of Basel II on interpretable models.
- The necessity and risks associated with proxy variables when direct default labels are unavailable.
- A comparative analysis of Logistic Regression with WoE and Gradient Boosting in a regulated financial environment.

This foundational understanding guided subsequent modeling decisions, emphasizing interpretability and regulatory compliance.

## Task 2: Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed within a Jupyter notebook (`notebooks/1.0-eda.ipynb`). This phase involved:
- Loading the Xente dataset.
- Inspecting data structure, types, and summary statistics.
- Visualizing numerical and categorical feature distributions.
- Conducting correlation analysis.
- Detecting missing values and outliers.

The EDA provided critical insights into the dataset's characteristics, informing the feature engineering strategies.

## Task 3: Feature Engineering

The core of data preparation was implemented in `src/data_processing.py`. This script established a comprehensive feature engineering pipeline:
- **Data Loading**: Efficiently loads the raw `data.csv` file.
- **Aggregate Features**: Computes total, average, standard deviation, and count of transaction amounts, along with transaction counts per `CustomerId`.
- **Datetime Features**: Extracts granular time-based features (hour, day, month, year) from `TransactionStartTime`.
- **Categorical Encoding**: Applies one-hot encoding for `ProductCategory` and `ChannelId`.
- **Missing Value Handling**: Imputes missing numerical values with the median and categorical values with the mode.
- **Numerical Normalization**: Scales numerical features using `StandardScaler`.
- **Pipeline Integration**: All steps are orchestrated using `scikit-learn`'s `Pipeline` and `ColumnTransformer` for modularity and reproducibility.

The script successfully processed the data, generating `data_processed.csv` for subsequent tasks.

## Task 4: Proxy Target Variable Engineering

To address the absence of a direct default label, a proxy target variable (`is_high_risk`) was engineered using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering in `src/rfm_clustering.py`:
- **RFM Calculation**: Recency (days since last transaction), Frequency (total transactions), and Monetary (total transaction amount) metrics were calculated for each customer.
- **Feature Scaling**: RFM features were scaled using `StandardScaler`.
- **K-Means Clustering**: Customers were segmented into three clusters based on their RFM profiles.
- **High-Risk Identification**: The cluster characterized by low frequency and low monetary value was identified as 'high-risk'.
- **Target Variable Creation**: A binary `is_high_risk` column (1 for high-risk, 0 otherwise) was added to the dataset.

This process successfully created `data_processed_with_risk.csv`, providing the necessary target for model training.

## Task 5: Model Training and Tracking

Model development and tracking were managed in `src/train.py`:
- **Data Loading and Splitting**: Loads the processed data and splits it into training and testing sets.
- **Model Training**: Both Logistic Regression and Gradient Boosting models were trained.
- **Hyperparameter Tuning**: `GridSearchCV` was used to optimize model hyperparameters.
- **Evaluation**: Models were evaluated using key metrics: accuracy, precision, recall, F1-score, and ROC-AUC.
- **MLflow Integration**: All experiments, including parameters, metrics, and trained models, were meticulously logged using MLflow, enabling easy comparison and reproducibility. The Gradient Boosting model achieved a strong ROC-AUC of 0.9944.
- **Model Export**: The best Gradient Boosting model was exported as a `.pkl` file for direct use in the deployment phase.

Unit tests for the `data_processing.py` script were also implemented in `tests/test_data_processing.py` and successfully passed, ensuring the reliability of our feature engineering pipeline.

## Task 6: Model Deployment and Continuous Integration

The final phase focused on deploying the model as a FastAPI service and setting up a CI/CD pipeline:
- **Pydantic Models**: `src/api/pydantic_models.py` defines data models for API request and response validation, ensuring data integrity.
- **FastAPI Application**: `src/api/main.py` implements a `/predict` endpoint that loads the exported Gradient Boosting model and returns credit risk probabilities.
- **Dockerization**: A `Dockerfile` was created to containerize the FastAPI application, ensuring a portable and consistent deployment environment.
- **Docker Compose**: `docker-compose.yml` was configured to simplify the building and running of the Dockerized service, including mounting the `mlruns` directory for model access.
- **Continuous Integration**: A GitHub Actions workflow (`.github/workflows/ci.yml`) was set up to automate linting (flake8) and testing (pytest) on every push and pull request, maintaining code quality and preventing regressions.

While the Docker deployment faced initial challenges related to pathing and MLflow model loading within the container environment, these were systematically addressed by directly exporting the model and adjusting the Dockerfile to include the model. The Docker build process is now ready to be executed.

This project successfully demonstrates a complete end-to-end machine learning workflow, from data understanding and feature engineering to model training, tracking, and deployment, with a strong emphasis on best practices and automation.
