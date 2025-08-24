import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold

df=pd.read_csv("./Data/processed_data.csv")

features = [
    'event_type', 'city', 'venue_capacity', 'is_indoor',
    'is_weekend', 'holiday', 'start_time_hour', 'ticket_price',
    'promotion_days', 'weather_temp_c', 'weather_is_rain',"time_of_day"
]
x = df[features] # features
y = df['attendance'] 

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

categorical_cols = ["event_type", "city", "time_of_day"]
numerical_cols=["venue_capacity","ticket_price","promotion_days","start_time_hour",'weather_temp_c']

preprocessor=ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numerical_cols),
        ("cat",OneHotEncoder(handle_unknown='ignore'),categorical_cols)
    ],
    remainder='passthrough'
)

#-----------------------------------
# linear regression vs random forest
#-----------------------------------

models={
    "linear_regression":LinearRegression(),
    "Random_Forest":RandomForestRegressor()
        }

for name,model in models.items():
    pipeline=Pipeline(steps=[("preprocessor",preprocessor),
                             ("model",model)])
    pipeline.fit(x_train,y_train)

    y_pred=pipeline.predict(x_test)

    mae=mean_absolute_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    r2=r2_score(y_test,y_pred)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate using negative mean squared error
    scores = cross_val_score(pipeline, x, y, cv=cv, scoring='neg_mean_squared_error')

    rmse_scores = np.sqrt(-scores)  # convert to RMSE

    print(f"{name} : mean absolute error - {mae} , mean squared error - {rmse} , r2_score - {r2}")
    print("Cross-validation RMSEs:", rmse_scores)
    print("Average RMSE:", rmse_scores.mean())
    
    # print(name)
    # plt.scatter(y_test, y_pred, alpha=0.5)
    # plt.xlabel("Actual Attendance")
    # plt.ylabel("Predicted Attendance")
    # plt.title("Actual vs Predicted")
    # plt.show()




