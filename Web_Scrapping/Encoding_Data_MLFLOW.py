import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

import statsmodels.api as sm
import statsmodels.formula.api as smf

mlflow_server_url = "http://127.0.0.1:5000"

# Read data into DataFrame
df = pd.read_csv("Encoding_Data.csv")

# Assuming smaller_df1 is your DataFrame
lsm = smf.ols('log_Fiyat ~ Yaş + Kilometre + Q("Motor Hacmi") + Q("Motor Gücü") + Q("Ortalama Yakıt Tüketimi") + Q("Depo Hacmi")  + Ford + Opel + Renault + Skoda + Toyota + Volkswagen + Q("Vites Tipi") + Q("Hasar Durumu")  ', data=df)
fit1 = lsm.fit()


# Get the R-squared score
r2_score = fit1.rsquared

print("R-squared Score:", r2_score)

# Log R-squared score using MLflow
with mlflow.start_run():
    mlflow.log_param("Model", "Linear Regression")
    mlflow.log_metric("R2 Score", r2_score)