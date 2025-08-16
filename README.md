# Credit Risk Model

## Project Objective

This project aims to develop a robust credit risk model using alternative data from the Xente platform. 

The primary goal is to predict the likelihood of a customer being high-risk, enabling more informed lending decisions. 
The project will explore the trade-offs between traditional interpretable models, such as Logistic Regression with Weight of Evidence (WoE),
and more complex, high-performance models like Gradient Boosting. By leveraging machine learning, this project seeks to improve financial 
inclusion by providing a more accurate and equitable way to assess credit risk for individuals who may not have a traditional credit history.

## Dataset

This project uses the Xente Customer Transaction Data, which can be downloaded from Kaggle. 
You can find the dataset at the following URL: [https://www.kaggle.com/datasets/infinix-mobility-uganda/xente-fraud-detection-challenge]
(https://www.kaggle.com/datasets/infinix-mobility-uganda/xente-fraud-detection-challenge)

Please download the dataset and place the `xente.csv` file in the `data/raw` directory.


## Credit Scoring Business Understanding

The Basel II accord requires financial institutions to maintain minimum capital reserves to cover credit risk, encouraging the use of internal models for risk assessment [1]. 
This regulatory framework emphasizes the need for interpretable models, as banks must be able to explain the factors driving their risk predictions to regulators [5]. 
Consequently, models like Logistic Regression with Weight of Evidence (WoE), which offer transparency and clear relationships between features and outcomes, 
have been widely adopted in the financial industry [7].

In situations where a direct default label is unavailable, a proxy variable must be engineered from alternative data sources [1]. 
For example, a "high-risk" customer could be identified by a combination of factors such as payment history, transaction data, 
and even digital footprint [3, 4]. However, using a proxy variable introduces business risks. If the proxy does not accurately represent true default behavior, 
the resulting model may be flawed, leading to either overly restrictive lending or excessive risk exposure.

This project explores the trade-off between model interpretability and predictive accuracy. While Logistic Regression with WoE provides a transparent and 
easily explainable model, it may not capture the complex, non-linear patterns that a Gradient Boosting model can [10].

 The choice between these models is a critical business decision that depends on regulatory requirements, 
 the institution's risk tolerance, and the availability of data. Recent advancements in Explainable AI (XAI), 
 such as SHAP and LIME, are helping to bridge this gap by providing methods to interpret "black-box" models like Gradient Boosting, 
 potentially allowing for both high accuracy and transparency [9, 11].

### References

[1] Basel Committee on Banking Supervision. (2006). *International Convergence of Capital Measurement and Capital Standards: A Revised Framework*. Bank for International Settlements.

[2] BIS (2005). *An Explanatory Note on the Basel II IRB Risk Weight Functions*. Available at: https://www.bis.org/bcbs/irbriskweight.htm

[3] Plaid (2023). *Alternative data for lending*. Available at: https://plaid.com/resources/lending/alternative-data-for-lending/

[4] FICO (2022). *The Future of Credit Scoring: AI, Alternative Data, and the Future of Fair Lending*. Available at: https://www.fico.com/blogs/future-credit-scoring-ai-alternative-data-and-future-fair-lending

[5] Finalyse (2022). *Model risk management under the new EBA guidelines*. Available at: https://www.finalyse.com/model-risk-management-under-the-new-eba-guidelines

[6] Siddiqi, N. (2017). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. John Wiley & Sons.

[7] Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications*. SIAM.

[8] Finlay, S. (2012). *Credit Scoring, Response Modeling, and Insurance Rating: A Practical Guide to Forecasting Consumer Behavior*. Palgrave Macmillan.

[9] Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. In Advances in Neural Information Processing Systems 30 (pp. 4765–4774).

[10] Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794).

[11] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 1135–1144).

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