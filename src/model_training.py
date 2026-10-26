
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_splitting import X_train, X_test, y_train, y_test
from src.data_loader import df


# Create models directory in the project root
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def model(df):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    linear_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    forest_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42))
    ])

    linear_params = {
        "model__fit_intercept": [True, False]
    }

    forest_params = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5]
    }

    linear_search = GridSearchCV(
        estimator=linear_pipeline,
        param_grid=linear_params,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    forest_search = GridSearchCV(
        estimator=forest_pipeline,
        param_grid=forest_params,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    searches = {
        "Linear Regression": linear_search,
        "Random Forest Regressor": forest_search
    }

    results = []

    for name, search in searches.items():
        print(f"\nTraining {name}...")

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Save the fitted pipeline
        filename = (
            "linear_regression.pkl"
            if name == "Linear Regression"
            else "random_forest.pkl"
        )

        model_path = MODEL_DIR / filename
        joblib.dump(best_model, model_path)

        print(f"Saved model to: {model_path}")

        predictions = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "Best Parameters": search.best_params_,
            "CV MAE": -search.best_score_,
            "Test MAE": mae,
            "Test RMSE": rmse,
            "Test R2": r2
        })

        print("Best parameters:", search.best_params_)
        print(f"Cross-validation MAE: {-search.best_score_:.2f}")
        print(f"Test MAE: {mae:.2f}")
        print(f"Test RMSE: {rmse:.2f}")
        print(f"Test R²: {r2:.3f}")

    results_df = pd.DataFrame(results)

    print("\nModel Comparison:")
    print(
        results_df[
            ["Model", "CV MAE", "Test MAE", "Test RMSE", "Test R2"]
        ].to_string(index=False)
    )

    return searches, results_df


trained_models, results_df = model(df)
