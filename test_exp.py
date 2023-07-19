# Databricks notebook source
import mlflow

# COMMAND ----------

# DBTITLE 1,expid = 1457383296188488
#mlflow.create_experiment("/Users/mohammad.subhani@databricks.com/test_repros_github_integration", artifact_location="dbfs:/subhani/test_repros_github_integration")

# COMMAND ----------

import mlflow.sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the Boston Housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with mlflow.start_run(experiment_id=1457383296188488):
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Log metrics to MLflow
    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))

    # Log the model to MLflow
    mlflow.sklearn.log_model(model, "test_model")

# COMMAND ----------


