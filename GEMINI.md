
GEMINI.md: Automating Project Structure and Scripts for B5W5 Credit Risk Project
This file provides Gemini CLI prompts to automate the creation of the project structure and scripts for the B5W5: Credit Risk Probability Model for Alternative Data project. The prompts generate the directory structure (credit-risk-model/) and required files (e.g., README.md, data_processing.py, Dockerfile) for all six tasks, assuming the repository is cloned at C:\Users\Cyber Defense\credit-risk-model. Each task section includes a prompt to ask whether to proceed to the next task.
Creating Project Structure
Prompt: Generate a batch script to create the project directory structure and initial files:
gemini -p "Generate a Windows batch script to create the directory structure: credit-risk-model/ with subdirectories .github/workflows, data/raw, data/processed, notebooks, src/api, tests, and files .github/workflows/ci.yml, notebooks/1.0-eda.ipynb, src/__init__.py, src/data_processing.py, src/train.py, src/predict.py, src/api/main.py, src/api/pydantic_models.py, tests/test_data_processing.py, Dockerfile, docker-compose.yml, requirements.txt, .gitignore, README.md, GEMINI.md." > setup_project.bat

Output Handling:

Run the script:setup_project.bat



Prompt for .gitignore:
gemini -p "Generate a .gitignore file for a Python project, excluding: data/, .env, *.pyc, __pycache__/, venv/." > .gitignore

Prompt for requirements.txt:
gemini -p "Generate a requirements.txt file with Python packages: pandas, numpy, scikit-learn, mlflow, pytest, fastapi, uvicorn, flake8, xverse, woe." > requirements.txt

Prompt to Proceed:
gemini -p "Do you want to proceed to Task 1: Understanding Credit Risk? (Type 'yes' to continue or 'no' to stop)."

Task 1: Understanding Credit Risk
Goal: Create the “Credit Scoring Business Understanding” section in README.md.Prompt:
gemini -p "Generate a markdown file for README.md with a section titled 'Credit Scoring Business Understanding' (300 words) answering: 1) How does Basel II’s emphasis on risk measurement influence interpretable models? 2) Why is a proxy variable necessary without a default label, and what are the business risks? 3) Trade-offs between Logistic Regression with WoE and Gradient Boosting in a regulated financial context. Use references: https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf, https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf, https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf, https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03, https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/, https://www.risk-officer.com/Credit_Risk.htm. Include a project structure section listing all directories and files." > README.md

Output Handling:

Save and review README.md for accuracy.
Commit to GitHub:git add README.md
git commit -m "Add Credit Scoring Business Understanding section"
git push origin main



Prompt to Proceed:
gemini -p "Do you want to proceed to Task 2: Exploratory Data Analysis? (Type 'yes' to continue or 'no' to stop)."

Task 2: Exploratory Data Analysis (EDA)
Goal: Generate EDA code for notebooks/1.0-eda.ipynb.Prompt:
gemini -p "Generate Python code for a Jupyter notebook (notebooks/1.0-eda.ipynb) to perform EDA on the Xente dataset from Kaggle. Include: 1) Load data with pandas from data/raw/xente.csv, 2) Show data structure (rows, columns, types), 3) Summary statistics, 4) Numerical feature distributions (histograms with seaborn), 5) Categorical feature distributions (bar plots), 6) Correlation analysis (heatmap), 7) Missing value detection, 8) Outlier detection (box plots). Format as Jupyter notebook code blocks with markdown explanations." > notebooks/1.0-eda.ipynb

Output Handling:

Save and open 1.0-eda.ipynb in Jupyter to validate.
Commit:git add notebooks/1.0-eda.ipynb
git commit -m "Add EDA notebook"
git push origin main



Prompt to Proceed:
gemini -p "Do you want to proceed to Task 3: Feature Engineering? (Type 'yes' to continue or 'no' to stop)."

Task 3: Feature Engineering
Goal: Create src/data_processing.py with a feature engineering pipeline.Prompt:
gemini -p "Generate a Python script for src/data_processing.py to process the Xente dataset. Include: 1) Load data from data/raw/xente.csv, 2) Create aggregate features (total/average transaction amount, transaction count, std of amounts per CustomerId), 3) Extract datetime features (hour, day, month, year from TransactionStartTime), 4) Encode categorical variables (one-hot for ProductCategory, label encoding for ChannelId), 5) Handle missing values (median imputation for numerical, mode for categorical), 6) Normalize numerical features. Use sklearn.pipeline.Pipeline, xverse, and woe for WoE/IV transformations. Add docstrings and logging." > src/data_processing.py

Output Handling:

Save and test:python src/data_processing.py


Commit:git add src/data_processing.py
git commit -m "Add feature engineering script"
git push origin main



Prompt to Proceed:
gemini -p "Do you want to proceed to Task 4: Proxy Target Variable Engineering? (Type 'yes' to continue or 'no' to stop)."

Task 4: Proxy Target Variable Engineering
Goal: Create a script to engineer the is_high_risk column.Prompt:
gemini -p "Generate a Python script for src/rfm_clustering.py to: 1) Load the Xente dataset from data/raw/xente.csv, 2) Calculate RFM metrics (Recency, Frequency, Monetary) per CustomerId with snapshot date 2025-06-30, 3) Scale RFM features with StandardScaler, 4) Apply K-Means clustering (3 clusters, random_state=42), 5) Identify high-risk cluster (low frequency, low monetary), 6) Create binary is_high_risk column (1 for high-risk, 0 otherwise), 7) Save dataset with is_high_risk to data/processed/xente_processed.csv. Include docstrings." > src/rfm_clustering.py

Output Handling:

Save and test:python src/rfm_clustering.py


Commit:git add src/rfm_clustering.py data/processed/xente_processed.csv
git commit -m "Add RFM clustering script"
git push origin main



Prompt to Proceed:
gemini -p "Do you want to proceed to Task 5: Model Training and Tracking? (Type 'yes' to continue or 'no' to stop)."

Task 5: Model Training and Tracking
Goal: Create src/train.py and unit tests for tests/test_data_processing.py.Prompt for train.py:
gemini -p "Generate a Python script for src/train.py to: 1) Load processed data from data/processed/xente_processed.csv, 2) Split into train/test sets, 3) Train Logistic Regression and Gradient Boosting models, 4) Tune hyperparameters with GridSearchCV, 5) Evaluate with accuracy, precision, recall, F1, ROC-AUC, 6) Log to MLflow and register the best model. Use scikit-learn and mlflow. Include docstrings." > src/train.py

Prompt for Unit Tests:
gemini -p "Generate a Python script for tests/test_data_processing.py with at least two unit tests for a helper function in src/data_processing.py (e.g., a function to calculate aggregate features). Use pytest and include docstrings." > tests/test_data_processing.py

Output Handling:

Save and test:python src/train.py
pytest tests/test_data_processing.py


Commit:git add src/train.py tests/test_data_processing.py
git commit -m "Add model training and unit tests"
git push origin main



Prompt to Proceed:
gemini -p "Do you want to proceed to Task 6: Model Deployment and Continuous Integration? (Type 'yes' to continue or 'no' to stop)."

Task 6: Model Deployment and Continuous Integration
Goal: Create FastAPI, Docker, and CI/CD files.Prompt for main.py:
gemini -p "Generate a Python script for src/api/main.py to create a FastAPI application with a /predict endpoint that: 1) Loads an MLflow model, 2) Accepts customer data matching the model’s features, 3) Returns risk probability. Use fastapi and mlflow. Include docstrings." > src/api/main.py

Prompt for pydantic_models.py:
gemini -p "Generate a Python script for src/api/pydantic_models.py with Pydantic models for FastAPI request/response validation for the /predict endpoint. Match features from src/data_processing.py. Include docstrings." > src/api/pydantic_models.py

Prompt for Dockerfile:
gemini -p "Generate a Dockerfile to set up a Python environment, install requirements.txt, and run a FastAPI application with uvicorn from src/api/main.py." > Dockerfile

Prompt for docker-compose.yml:
gemini -p "Generate a docker-compose.yml file to build and run the FastAPI service from Dockerfile, exposing port 8000." > docker-compose.yml

Prompt for ci.yml:
gemini -p "Generate a GitHub Actions workflow file for .github/workflows/ci.yml to: 1) Run flake8 linting on src/ and tests/, 2) Run pytest on tests/. Fail the build if either step fails." > .github/workflows/ci.yml

Output Handling:

Save and test:docker-compose up --build


Commit:git add src/api/main.py src/api/pydantic_models.py Dockerfile docker-compose.yml .github/workflows/ci.yml
git commit -m "Add FastAPI, Docker, and CI/CD files"
git push origin main



Notes

Validation: Test each generated script (e.g., python src/data_processing.py) and validate outputs in notebooks/1.0-eda.ipynb.
Dataset Path: Assumes Xente dataset at data/raw/xente.csv; adjust prompts if different.
Submission: Include all files in your GitHub repository for interim (29 June 2025) and final (1 July 2025) submissions.
Interactive Flow: Run the “Prompt to Proceed” commands after each task and respond with “yes” or “no” to continue or pause.
