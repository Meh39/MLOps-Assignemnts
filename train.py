import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the Titanic dataset directly from seaborn
data = sns.load_dataset('titanic')

# Replace missing values in the dataset
mean_age = data['age'].mean()
data['age'].fillna(mean_age, inplace=True)
mode_embarked = data['embarked'].mode()[0]
data['embarked'].fillna(mode_embarked, inplace=True)
data['fare'].fillna(0, inplace=True)
data['deck'] = data['deck'].cat.add_categories('Unknown')
data['deck'].fillna('Unknown', inplace=True)
mode_embark_town = data['embarked'].mode()[0]
data['embark_town'].fillna(mode_embark_town, inplace=True)

# Remove duplicates and handle data types
data.drop_duplicates(inplace=True)
data['age'] = data['age'].astype(float)
data['fare'] = data['fare'].astype(float)
data['sex'] = data['sex'].astype('category')
data['embarked'] = data['embarked'].astype('category')
data = data.copy()

# Standardize numerical features
scaler = StandardScaler()
data[['age', 'fare']] = scaler.fit_transform(data[['age', 'fare']])

# Create new features
data['family_size'] = data['sibsp'] + data['parch'] + 1

# Split into numerical and categorical features
numerical_features = ['age', 'sibsp', 'parch', 'family_size']
categorical_features = ['sex', 'embarked', 'class', 'adult_male', 'deck', 'embark_town', 'pclass']

# Prepare cleaned dataset for modeling
data_cleaned = data.drop(['survived', 'who', 'alive', 'alone'], axis=1)
X = data_cleaned.drop('fare', axis=1)
y = data_cleaned['fare']

# Handle categorical variables using one-hot encoding
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Standardize numerical features again after encoding
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import necessary libraries for model training and logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
mlflow.set_experiment("My_mlflow_exp")
# Train and log RandomForest model
def train_and_log_model():
    with mlflow.start_run() as run:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse1 = mean_squared_error(y_test, y_pred)
        r21 = r2_score(y_test, y_pred)
        mae1 = mean_absolute_error(y_test, y_pred)
        print(f"RandomForest - Mean Squared Error: {mse1}")
        print(f"RandomForest - R-squared: {r21}")
        print(f"RandomForest - Mean Absolute Error: {mae1}")
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_metric("mse", mse1)
        mlflow.log_metric("r2", r21)
        mlflow.sklearn.log_model(model, "model")                                                                        mlflow.end_run()

# Train and log Linear Regression model
def train_and_log_model2():
    with mlflow.start_run() as run:
        model = LinearRegression() 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"LinearRegression - Mean Squared Error: {mse}")
        print(f"LinearRegression - R-squared: {r2}")
        print(f"LinearRegression - Mean Absolute Error: {mae}")
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()

# Execute the functions
train_and_log_model()
train_and_log_model2()

