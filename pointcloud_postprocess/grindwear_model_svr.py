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
import os
import joblib

def load_data(file_path):
    #Load dataset from a CSV file.
    return pd.read_csv(file_path)

def preprocess_data(data, target_column):
    #Preprocess the data by splitting into features and target and then scaling.

    X = data.drop(columns=target_column)
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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

def train_single_svr(X_train, y_train, ce=1, gam=0.1, eps=0.1):
    """
    Train a Support Vector Regressor (SVR) for single-output regression.
    """
    svr = SVR(kernel='rbf', C=ce, gamma=gam, epsilon=eps)
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
    
    # Set axis limits to be the same
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Plot reference diagonal line for perfect prediction
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

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
        
        # Set axis limits to be the same
        min_val = min(min(y_test[col]), min(y_pred[:, i]))
        max_val = max(max(y_test[col]), max(y_pred[:, i]))
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        
        # Plot reference diagonal line for perfect prediction
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

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

def visualize_svr_model(best_model, X_train, y_train, scaler):
    # Assuming 'X_train' has two columns: 'initial wear' and 'RPM'
    
    # Create a mesh grid for 'initial wear' and 'RPM'
    wear_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    rpm_range = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100)
    
    wear_grid, rpm_grid = np.meshgrid(wear_range, rpm_range)
    
    # Flatten the grid and stack the two features
    grid = np.c_[wear_grid.ravel(), rpm_grid.ravel()]
    

    
    # Predict 'grind_eff' using the SVR model
    grind_eff_pred = best_model.predict(grid)

    # Reshape the predicted values back to a grid for plotting
    grind_eff_grid = grind_eff_pred.reshape(wear_grid.shape)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface for the SVR predictions
    ax.plot_surface(wear_grid, rpm_grid, grind_eff_grid, color='r', alpha=0.7, label='SVR model')

    # Plot the original data points for 'initial wear', 'RPM', and 'grind_eff'
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, color='b', label='Actual data')

    # Set labels for axes
    ax.set_xlabel('Initial Wear')
    ax.set_ylabel('RPM')
    ax.set_zlabel('Grind Efficiency')

    plt.title("SVR Model Visualization (Initial Wear vs RPM vs Grind Efficiency)")
    plt.show()

def save_model(model, folder_name='saved_models', filename='svr_model.pkl'):
    # Get the current working directory
    current_dir = os.getcwd()

    # Create the full path by joining the current directory with the folder name
    folder_path = os.path.join(current_dir, folder_name)

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Create the full filepath to save the model
    filepath = os.path.join(folder_path, filename)

    # Save the model to the specified filepath
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(folder_name='saved_models', filename='svr_model.pkl'):
    # Get the current working directory
    current_dir = os.getcwd()

    # Create the full path by joining the current directory with the folder name
    folder_path = os.path.join(current_dir, folder_name)

    # Create the full filepath to load the model from
    filepath = os.path.join(folder_path, filename)

    # Load the model
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


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

    #Best parameters for SVR: {'C': 0.5, 'epsilon': 0.02, 'gamma': 1.25, 'kernel': 'rbf'}
    #Mean^2 error Score for the best model: 0.014481130596871603
    #R^2 Score for the best model: 0.4366693052537124

    # Define the parameter grid
    param_grid = {
        'C': [0.05, 0.1, 0.2, 0.5, 1, 5, 10],
        'gamma': [0.005, 0.01, 0.02, 0.05, 0.1],
        'epsilon': [0.005, 0.01, 0.02, 0.05, 0.1],
        'kernel': ['rbf']
    }

    # Initialize SVR model
    svr = SVR()

    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters:", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    #visualize_svr_model(best_model, X_train, y_train, scaler)

    # Optionally, evaluate the model on the test set
    evaluate_model_single(best_model, X_test, y_test)
    
    # save model
    #save_model(best_model, folder_name='wear_svr_model', filename='grind_eff_svr_V1.pkl')

    #read current belt's RPM*Force*time value

    #apply knockdown factor to predicted volume lost to get a more accurate RPM/force profile for desired volume



if __name__ == "__main__":
    main()