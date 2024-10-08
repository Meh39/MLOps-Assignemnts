# train.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.DataFrame({
        'feature1': [i for i in range(100)],
        'feature2': [i ** 2 for i in range(100)],
        'target': [i * 2 + 3 for i in range(100)]
        })

X = data[['feature1', 'feature2']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

experiment_name = "MLFlow_Experiment"
mlflow.set_experiment(experiment_name)


def train_and_log_model(model, model_name):
    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        mlflow.log_param("model", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, model_name)
        print(f"{model_name} MSE: {mse}")

        # Log the run ID for future referen
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        # Register the model in the Model Registry
        model_uri = f"runs:/{run_id}/{model_name}"
        registered_model = mlflow.register_model(model_uri, model_name)
        print(f"Registered model: {registered_model.name}, Version: {registered_model.version}")


linear_model = LinearRegression()
train_and_log_model(linear_model, "Linear_Regression")

random_forest_model = RandomForestRegressor()
train_and_log_model(random_forest_model, "Random_Forest")
