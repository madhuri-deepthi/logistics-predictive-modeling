# YuvaIntern Week 4
# Predictive Modeling and Optimization in Logistics Systems

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------
# 1. CREATE PROJECT FOLDERS
# -------------------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("generated", exist_ok=True)


# -------------------------------------------------
# 2. GENERATE SIMULATED LOGISTICS DATASET
# -------------------------------------------------

np.random.seed(42)

n = 10000

data = pd.DataFrame({
    "Distance_Miles": np.random.randint(50, 1500, n),
    "Congestion_Index": np.random.uniform(1, 10, n),
    "Carrier_Tier": np.random.choice(
        ["Economy", "Standard", "Premium"], n
    ),
    "Weather_Condition": np.random.choice(
        ["Clear", "Rainy", "Stormy"], n
    ),
    "Dispatch_Shift": np.random.choice(
        ["Morning", "Afternoon", "Night"], n
    )
})

# Create numerical effects for simulated target
carrier_effect = data["Carrier_Tier"].map({
    "Economy": 5,
    "Standard": 2,
    "Premium": -2
})

weather_effect = data["Weather_Condition"].map({
    "Clear": 0,
    "Rainy": 4,
    "Stormy": 10
})

shift_effect = data["Dispatch_Shift"].map({
    "Morning": 1,
    "Afternoon": 2,
    "Night": 0
})

# Calculate simulated transit time
data["Final_Transit_Time"] = (
    3
    + data["Distance_Miles"] / 55
    + data["Congestion_Index"] * 0.8
    + carrier_effect
    + weather_effect
    + shift_effect
    + np.random.normal(0, 2, n)
)

# Save dataset
data.to_csv("data/logistics_data.csv", index=False)

print("\nDataset created successfully!")
print(data.head())


# -------------------------------------------------
# 3. SELECT FEATURES AND TARGET
# -------------------------------------------------

X = data.drop("Final_Transit_Time", axis=1)
y = data["Final_Transit_Time"]

numerical_features = [
    "Distance_Miles",
    "Congestion_Index"
]

categorical_features = [
    "Carrier_Tier",
    "Weather_Condition",
    "Dispatch_Shift"
]


# -------------------------------------------------
# 4. DATA PREPROCESSING
# -------------------------------------------------

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


# -------------------------------------------------
# 5. SPLIT DATA INTO TRAINING AND TESTING
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# -------------------------------------------------
# 6. TRAIN DIFFERENT REGRESSION MODELS
# -------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("\nModel:", name)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))


# -------------------------------------------------
# 7. COMPARE MODEL PERFORMANCE
# -------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "generated/model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(results_df)


# -------------------------------------------------
# 8. HYPERPARAMETER TUNING
# -------------------------------------------------

random_forest_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ))
])

parameter_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [10, 20, None]
}

grid_search = GridSearchCV(
    random_forest_pipeline,
    parameter_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("\nPerforming hyperparameter tuning...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)


# -------------------------------------------------
# 9. EVALUATE BEST MODEL
# -------------------------------------------------

best_predictions = best_model.predict(X_test)

best_mae = mean_absolute_error(y_test, best_predictions)

best_rmse = np.sqrt(
    mean_squared_error(y_test, best_predictions)
)

best_r2 = r2_score(y_test, best_predictions)

print("\nBEST MODEL PERFORMANCE")
print("MAE:", round(best_mae, 3))
print("RMSE:", round(best_rmse, 3))
print("R2 Score:", round(best_r2, 3))


# -------------------------------------------------
# 10. CROSS VALIDATION
# -------------------------------------------------

cv_scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\nCROSS VALIDATION RESULTS")
print("Individual R2 Scores:", cv_scores)
print("Average R2 Score:", round(cv_scores.mean(), 3))


# -------------------------------------------------
# 11. SAVE TRAINED MODEL
# -------------------------------------------------

joblib.dump(
    best_model,
    "models/best_logistics_model.pkl"
)

print("\nTrained model saved successfully!")


# -------------------------------------------------
# 12. LOGISTICS OPTIMIZATION
# -------------------------------------------------

print("\nLOGISTICS OPTIMIZATION")

# Example shipment
shipment = pd.DataFrame({
    "Distance_Miles": [500, 500, 500],
    "Congestion_Index": [7, 7, 7],
    "Carrier_Tier": ["Economy", "Standard", "Premium"],
    "Weather_Condition": ["Rainy", "Rainy", "Rainy"],
    "Dispatch_Shift": ["Morning", "Morning", "Morning"]
})

shipment["Predicted_Transit_Time"] = best_model.predict(shipment)

print("\nCarrier comparison:")
print(shipment[["Carrier_Tier", "Predicted_Transit_Time"]])

# Select carrier with lowest predicted transit time
best_carrier = shipment.loc[
    shipment["Predicted_Transit_Time"].idxmin()
]

print("\nRecommended Carrier:")
print(best_carrier["Carrier_Tier"])

print(
    "Predicted Transit Time:",
    round(best_carrier["Predicted_Transit_Time"], 2),
    "hours"
)


# -------------------------------------------------
# 13. SAVE FINAL RESULTS
# -------------------------------------------------

with open("generated/model_results.txt", "w") as file:

    file.write("LOGISTICS PREDICTIVE MODELING RESULTS\n")
    file.write("-------------------------------------\n")

    file.write(f"Best Parameters: {grid_search.best_params_}\n")
    file.write(f"MAE: {best_mae:.3f}\n")
    file.write(f"RMSE: {best_rmse:.3f}\n")
    file.write(f"R2 Score: {best_r2:.3f}\n")
    file.write(f"Average CV R2: {cv_scores.mean():.3f}\n")
    file.write(f"Recommended Carrier: {best_carrier['Carrier_Tier']}\n")

print("\nAll results saved successfully!")
print("\nWEEK 4 PROJECT COMPLETED!")