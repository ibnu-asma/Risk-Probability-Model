# Credit Risk Model

This project aims to build and serve a machine learning model to predict credit risk. The model will be trained on historical data to identify patterns and predict the likelihood of a borrower defaulting on a loan.

## Credit Scoring Business Understanding

Basel II's emphasis on robust risk measurement has significantly influenced the financial industry's preference for interpretable models. Regulatory requirements mandate that financial institutions not only produce accurate risk assessments but also understand and explain the factors driving those predictions. This has led to the widespread adoption of models like Logistic Regression with Weight of Evidence (WoE), which provide clear, monotonic relationships between input features and the predicted outcome. While more complex models like Gradient Boosting may offer higher predictive accuracy, their "black box" nature often makes them unsuitable for regulated environments where model transparency is paramount.

In the absence of a direct default label, a proxy variable is often necessary to train a credit risk model. For instance, a combination of late payments, high utilization, and low frequency of transactions could be used to define a "high-risk" customer. However, this approach introduces business risks. The proxy may not accurately capture the true definition of default, leading to a model that is optimized for the wrong target. This could result in either overly conservative lending, where creditworthy customers are denied, or overly aggressive lending, where the institution is exposed to unforeseen losses. The trade-offs between model interpretability and predictive power are a central theme in this project. While Logistic Regression with WoE offers a clear and easily explainable model, it may not capture the complex, non-linear relationships that a Gradient Boosting model can. The choice between these two approaches depends on the specific business context, the regulatory environment, and the risk appetite of the institution.

## Project Structure

```
.
├── data/
│   ├── raw/        # Raw, immutable data
│   └── processed/  # Cleaned and preprocessed data
├── notebooks/      # Jupyter notebooks for exploration and analysis
├── src/            # Source code
│   ├── api/        # FastAPI application
│   │   ├── main.py
│   │   └── pydantic_models.py
│   ├── data_processing.py # Scripts to process data
│   ├── train.py           # Scripts to train the model
│   └── predict.py         # Scripts to make predictions
├── tests/          # Tests for the source code
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd credit-risk-model
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Processing

To process the raw data, run the following command:

```bash
python src/data_processing.py
```

### 2. Model Training

To train the model, run the following command:

```bash
python src/train.py
```

### 3. Running the API

To start the FastAPI server, run the following command:

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 4. Making Predictions

You can send a POST request to the `/predict` endpoint with the required features to get a credit risk prediction.

## Testing

To run the tests, use `pytest`:

```bash
pytest
```
