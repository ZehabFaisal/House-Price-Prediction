# Project - House Price Prediction

## Goal
Train a regression model that predicts a sample house price from area, bedrooms, bathrooms, and age.

## Folder structure
```text
02_house_price_prediction/
├── data/
│   └── houses.csv
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Setup and run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## How it works
- `houses.csv` contains small synthetic example data.
- `train_model.py` separates data into training and test subsets.
- A Random Forest Regressor learns relationships between features and target prices.
- MAE, RMSE, and R² are printed for a basic evaluation.
- The trained model is saved with joblib.
- `app.py` accepts inputs and displays a prediction.

## Critical limitation
The sample values are invented for coding practice and have no reliable relationship to real housing markets. Do not describe this model as accurate or use it to estimate actual properties. For a portfolio-quality project, use a credible public dataset with clear licensing, document its geography and currency, compare against a simple baseline, and evaluate on a separate test set.

## Suggested improvements
- Compare Linear Regression, Random Forest, and Gradient Boosting.
- Plot actual vs. predicted prices and residuals.
- Add cross-validation and feature importance.
- Explain dataset bias and why a test score may not generalize.

Developed a Python regression demo using pandas and scikit-learn to predict sample house prices; evaluated a Random Forest model using MAE, RMSE, and R² on held-out data.