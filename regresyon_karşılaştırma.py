import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf

# MLflow server URL
mlflow_server_url = "http://127.0.0.1:5000"

# Read data into DataFrame
df = pd.read_csv("Feature_Engineering_Data.csv")

# Preparing the data
X = df.drop(columns=['Fiyat', 'log_Fiyat'])
y = df['log_Fiyat']
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

# Creating the 3 models
lm = LinearRegression()
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.values)
x_val_scaled = scaler.transform(x_val.values)
x_test_scaled = scaler.transform(x_test.values)
lm_reg = Ridge(alpha=22.21946860939524)
poly = PolynomialFeatures(degree=2) 
x_train_poly = poly.fit_transform(x_train.values)
x_val_poly = poly.transform(x_val.values)
x_test_poly = poly.transform(x_test.values)
lm_poly = LinearRegression()

# MLflow logging for each model
with mlflow.start_run():
    # Linear Regression
    lm.fit(x_train, y_train)
    val_r2_lm = r2_score(y_val, lm.predict(x_val))
    test_r2_lm = r2_score(y_test, lm.predict(x_test))
    
    mlflow.log_param("Model_Type", "Linear Regression")
    mlflow.log_metric("Validation_R2", val_r2_lm)
    mlflow.log_metric("Test_R2", test_r2_lm)

with mlflow.start_run():
    # Ridge Regression
    lm_reg.fit(x_train_scaled, y_train)
    val_r2_lm_reg = r2_score(y_val, lm_reg.predict(x_val_scaled))
    test_r2_lm_reg = r2_score(y_test, lm_reg.predict(x_test_scaled))
    
    mlflow.log_param("Model_Type", "Ridge Regression")
    mlflow.log_metric("Validation_R2", val_r2_lm_reg)
    mlflow.log_metric("Test_R2", test_r2_lm_reg)

with mlflow.start_run():
    # Polynomial Regression
    lm_poly.fit(x_train_poly, y_train)
    val_r2_lm_poly = r2_score(y_val, lm_poly.predict(x_val_poly))
    test_r2_lm_poly = r2_score(y_test, lm_poly.predict(x_test_poly))
    
    mlflow.log_param("Model_Type", "Polynomial Regression")
    mlflow.log_metric("Validation_R2", val_r2_lm_poly)
    mlflow.log_metric("Test_R2", test_r2_lm_poly)
