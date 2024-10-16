import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


def load_data(file_path):
    #Load dataset from a CSV file.
    return pd.read_csv(file_path)

def preprocess_data(data, target_column):
    #Preprocess the data by splitting into features and target and then scaling.

    X = data.drop(columns=target_column)
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_multi_svr(X_train, y_train):
    """
    Train a Support Vector Regressor (SVR) for multi-output regression.
    Wrap the SVR with MultiOutputRegressor to handle multiple targets.
    """
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    
    # Wrap the SVR model with MultiOutputRegressor to handle multi-output regression
    multioutput_svr = MultiOutputRegressor(svr)
    multioutput_svr.fit(X_train, y_train)

    return multioutput_svr

def train_single_svr(X_train, y_train):
    """
    Train a Support Vector Regressor (SVR) for single-output regression.
    """
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr.fit(X_train, y_train)

    return svr

def evaluate_model_single(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Evaluate the model with Mean Squared Error and R^2 Score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Plot actual vs predicted for grind_eff (single output)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Grind Efficiency")
    plt.ylabel("Predicted Grind Efficiency")
    plt.title("Actual vs Predicted Grind Efficiency")
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Evaluate the model with Mean Squared Error and R^2 Score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    
    # Plot actual vs predicted for each output
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(y_test.columns):
        plt.subplot(1, len(y_test.columns), i + 1)
        plt.scatter(y_test[col], y_pred[:, i])
        plt.xlabel(f"Actual {col}")
        plt.ylabel(f"Predicted {col}")
        plt.title(f"Actual vs Predicted {col}")
    
    plt.tight_layout()
    plt.show()

def open_file_dialog():
    # Create a Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog and return selected file path
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    return file_path

def logarithmic_model(x):
    return -0.0629125446945545 * np.log(x) + 1.82752898018992

def main():
    #read wear data cumulative(RPM*Force*Time) vs Lost volume in mm^3
    file_path_wear = open_file_dialog()
    if not file_path_wear:
        print("No file selected. Exiting.")
        return

    data_wear = load_data(file_path_wear)

    #calculate grind efficiency from lost volume data and replace lost volume column in data_wear with grind_eff
    #max volume lost as maximum efficiency
    max_volume_loss = data_wear['Lost_Volume'].max()
    data_wear['grind_eff'] = (data_wear['Lost_Volume'] / max_volume_loss)
    # Replace Lost_Volume column with grind_eff
    data_wear.drop(columns=['Lost_Volume'], inplace=True)

    #fit curve/ML model with RPM*Force*Time as input and grind efficiency (knockdown factor) as output
    target_columns = ['grind_eff']

    # Preprocess the data (train the model using the CSV data, for example)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data_wear, target_columns)

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    #X_train = np.log(X_train)
    #X_test = np.log(X_test)

    y_pred = logarithmic_model(X_train)


    #Best parameters for SVR: {'C': 0.5, 'epsilon': 0.02, 'gamma': 1.25, 'kernel': 'rbf'}
    #Mean^2 error Score for the best model: 0.014481130596871603
    #R^2 Score for the best model: 0.4366693052537124

    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)

    print(f"Mean^2 error Score for the best model: {mse}")
    print(f"R^2 Score for the best model: {r2}")
    
    
    #read current belt's RPM*Force*time value

    #apply knockdown factor to predicted volume lost to get a more accurate RPM/force profile



if __name__ == "__main__":
    main()