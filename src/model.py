# import pandas as pd
# import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib

# # ---------- Load data ----------
# df = pd.read_csv("./Data/processed_data.csv")

# # Log-transform venue_capacity
# df["venue_capacity_log"] = np.log(df["venue_capacity"])

# # Features & target
# features = ['event_type', 'city', 'venue_capacity_log', 'is_indoor',
#             'is_weekend', 'holiday', 'start_time_hour', 'ticket_price',
#             'promotion_days', 'weather_temp_c', 'weather_is_rain', 'time_of_day']
# x = df[features]
# y = df["attendance"]

# # Categorical and numerical columns
# categorical_col = ['event_type', 'city', 'time_of_day']
# numerical_col = ['venue_capacity_log', 'start_time_hour', 'ticket_price',
#                  'promotion_days', 'weather_temp_c']

# # ---------- Preprocessing ----------
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numerical_col),
#         ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_col)
#     ],
#     remainder='passthrough'
# )

# # ---------- Pipeline ----------
# pipeline = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("model", LinearRegression())
# ])

# # ---------- Train/Test Split Evaluation ----------
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# print("----- Train/Test Split Evaluation -----")
# print(f"MAE: {mae:.2f}")
# print(f"RMSE: {rmse:.2f}")
# print(f"R² Score: {r2:.4f}")

# # ---------- Cross-Validation ----------
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring='neg_mean_squared_error')
# cv_rmse_scores = np.sqrt(-cv_scores)

# print("\n----- 5-Fold Cross-Validation -----")
# print("RMSE for each fold:", cv_rmse_scores)
# print("Average RMSE:", cv_rmse_scores.mean())

# # ---------- Feature Coefficients ----------
# lireg_model = pipeline.named_steps["model"]
# ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
# encoded_feature_names = ohe.get_feature_names_out(categorical_col)
# final_features = numerical_col + list(encoded_feature_names) + ['is_indoor', 'is_weekend', 'holiday', 'weather_is_rain']

# feature_coefficients = pd.DataFrame({
#     "feature": final_features,
#     "coefficients": lireg_model.coef_
# }).sort_values(by="coefficients", key=abs, ascending=False)

# print("\n----- Feature Importance (Coefficients) -----")
# print(feature_coefficients)

# # ---------- Save Model ----------
# joblib.dump(pipeline, "src/attendance_predictor.pkl")
# print("\nModel saved as 'attendance_predictor.pkl'")


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------- Load data ----------
df = pd.read_csv("./Data/processed_data.csv")

# Log-transform venue_capacity (optional, you can remove if causing issues)
# df["venue_capacity_log"] = np.log(df["venue_capacity"])
# df["venue_capacity_log"] = df["venue_capacity"]

# Features & target
features = ['event_type', 'city', 'venue_capacity', 'is_indoor',
            'is_weekend', 'holiday', 'start_time_hour', 'ticket_price',
            'promotion_days', 'weather_temp_c', 'weather_is_rain', 'time_of_day']
x = df[features]
y = df["attendance"]

# Categorical and numerical columns
categorical_col = ['event_type', 'city', 'time_of_day']
numerical_col = ['venue_capacity', 'start_time_hour', 'ticket_price',
                 'promotion_days', 'weather_temp_c']

# ---------- Preprocessing ----------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_col),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_col)
    ],
    remainder='passthrough'
)

# ---------- Pipeline with Random Forest ----------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42))
])

# ---------- Train/Test Split Evaluation ----------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("----- Train/Test Split Evaluation -----")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ---------- Cross-Validation ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, x, y, cv=cv, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

print("\n----- 5-Fold Cross-Validation -----")
print("RMSE for each fold:", cv_rmse_scores)
print("Average RMSE:", cv_rmse_scores.mean())

# ---------- Feature Importances ----------
rf_model = pipeline.named_steps["model"]
ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
encoded_feature_names = ohe.get_feature_names_out(categorical_col)
final_features = numerical_col + list(encoded_feature_names) + ['is_indoor', 'is_weekend', 'holiday', 'weather_is_rain']

feature_importances = pd.DataFrame({
    "feature": final_features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n----- Feature Importance -----")
print(feature_importances)

# ---------- Save Model ----------
joblib.dump(pipeline, "src/attendance_predictor.pkl")
print("\nModel saved as 'attendance_predictor.pkl'")

