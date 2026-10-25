from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "houses.csv"
MODEL_PATH = ROOT / "house_price_model.joblib"
FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
TARGET = "price"

def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + [TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.25, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    print(f"MAE:  {mean_absolute_error(y_test, predictions):,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²:   {r2_score(y_test, predictions):.3f}")
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    print("Reminder: sample data is synthetic and not suitable for real valuations.")

if __name__ == "__main__":
    main()