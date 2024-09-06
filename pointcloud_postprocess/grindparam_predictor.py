import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_data(file_path):
    #Load dataset from a CSV file.
    return pd.read_csv(file_path)

def preprocess_data(data, target_column):
    #Preprocess the data by splitting into features and target and then scaling.

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_svr(X_train, y_train):
    # Create and train the SVR model
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    wrapper = MultiOutputRegressor(svr)
    svr.fit(X_train, y_train)
    return svr

def evaluate_model(svr, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    y_pred = svr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Plot the results
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual RPM")
    plt.ylabel("Predicted RPM")
    plt.title("Actual vs Predicted Material Removal Volume")
    plt.show()
