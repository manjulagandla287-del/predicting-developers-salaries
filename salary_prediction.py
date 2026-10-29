"""
Predicting Developers' Salaries
Beginner-friendly Machine Learning project.

How to run:
1. Install dependencies: pip install pandas scikit-learn
2. Run: python salary_prediction.py

This demo uses a small example dataset created inside the script.
For a real project, replace the sample data with a properly licensed dataset.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    # Example data for learning only (not real salary survey results).
    data = {
        "country": [
            "India", "India", "India", "USA", "USA", "USA",
            "UK", "UK", "Germany", "Germany", "India", "USA",
            "UK", "Germany", "India", "USA", "UK", "Germany",
            "India", "USA", "UK", "Germany", "India", "USA",
        ],
        "years_experience": [
            0, 1, 3, 2, 5, 8, 1, 4, 3, 6, 5, 10,
            7, 9, 2, 4, 6, 2, 8, 12, 3, 5, 10, 7,
        ],
        # Example annual salary values in USD; illustrative only.
        "annual_salary_usd": [
            5000, 7000, 12000, 65000, 85000, 120000,
            35000, 55000, 50000, 75000, 18000, 150000,
            80000, 95000, 10000, 75000, 65000, 45000,
            22000, 135000, 48000, 70000, 25000, 110000,
        ],
    }

    df = pd.DataFrame(data)

    features = ["country", "years_experience"]
    target = "annual_salary_usd"

    X = df[features]
    y = df[target]

    # Encode country names and keep years of experience numeric.
    preprocessor = ColumnTransformer(
        transformers=[
            ("country", OneHotEncoder(handle_unknown="ignore"), ["country"]),
            ("experience", "passthrough", ["years_experience"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # A small illustrative dataset is used; metrics may vary by split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n=== Developers' Salary Prediction Demo ===")
    print(f"R-squared (R2): {r2_score(y_test, predictions):.2f}")
    print(f"Mean Absolute Error (MAE): ${mean_absolute_error(y_test, predictions):,.2f}")
    print("\nNote: This tiny example dataset is artificial and is not suitable")
    print("for real-world salary predictions or job-market conclusions.")

    # Try one example prediction.
    example = pd.DataFrame(
        {"country": ["India"], "years_experience": [4]}
    )
    predicted_salary = model.predict(example)[0]
    print(
        "\nIllustrative prediction for 4 years' experience in India: "
        f"${predicted_salary:,.0f} per year"
    )


if __name__ == "__main__":
    main()
