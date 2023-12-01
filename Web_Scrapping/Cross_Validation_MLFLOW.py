import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd

# MLflow server URL
mlflow_server_url = "http://127.0.0.1:5000"

# Read data into DataFrame
df = pd.read_csv("Feature_Engineering_Data.csv")

# Preparing the data
X = df.drop(columns=['Fiyat', 'log_Fiyat'])
y = df['log_Fiyat']

# Linear Regression
lr = LinearRegression()
lr_cv = cross_val_score(lr, X, y, cv=10, scoring='r2')

# Ridge Regression
ridge = Ridge()
X_scaled = StandardScaler().fit_transform(X)
ridge_cv = cross_val_score(ridge, X_scaled, y, cv=10, scoring='r2')

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
lm_poly = LinearRegression()
cv_scores_poly = cross_val_score(lm_poly, X_poly, y, cv=10, scoring='r2')

# MLflow logging for each model
with mlflow.start_run():
    mlflow.log_param("Model_Type", "Linear Regression")
    mlflow.log_metric("CV_Score_Mean", lr_cv.mean())
    mlflow.log_metric("CV_Score_Std", lr_cv.std())

with mlflow.start_run():
    mlflow.log_param("Model_Type", "Ridge Regression")
    mlflow.log_metric("CV_Score_Mean", ridge_cv.mean())
    mlflow.log_metric("CV_Score_Std", ridge_cv.std())

with mlflow.start_run():
    mlflow.log_param("Model_Type", "Polynomial Regression")
    mlflow.log_metric("CV_Score_Mean", cv_scores_poly.mean())
    mlflow.log_metric("CV_Score_Std", cv_scores_poly.std())
