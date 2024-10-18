import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import GridSearchCV
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=np.random.randint(0,100))

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    return X_train, X_test, y_train, y_test, scaler

def train_multi_svr_with_grid_search(X_train, y_train):
    """
    Train a Support Vector Regressor (SVR) for multi-output regression.
    Wrap the SVR with MultiOutputRegressor to handle multiple targets.
    """
    # Define the parameter grid
    param_grid = {
        'estimator__C': [0.05, 0.1, 0.2, 0.5, 1, 5, 10, 20, 50, 100, 150],
        'estimator__gamma': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        'estimator__epsilon': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        'estimator__kernel': ['rbf']
    }


    # Initialize SVR model
    svr = SVR()
    multioutput_svr = MultiOutputRegressor(svr)

    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(multioutput_svr, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters:", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_


    return best_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Evaluate the model with Mean Squared Error and R^2 Score
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    mean_abs = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mean_abs}")
    print(f"RMS Error: {rmse}")
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

def save_model(model, scaler, folder_name='saved_models', modelname='svr_model.pkl', scalername='scaler.pkl'):
    # Get the current working directory
    current_dir = os.getcwd()

    # Create the full path by joining the current directory with the folder name
    folder_path = os.path.join(current_dir, folder_name)

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Create the full filepath to save the model
    model_filepath = os.path.join(folder_path, modelname)
    scaler_filepath = os.path.join(folder_path, scalername)

    # Save the model to the specified filepath
    joblib.dump(model, model_filepath)
    joblib.dump(scaler, scaler_filepath)
    print(f"Model saved to {model_filepath}")
    print(f"Scaler saved to {scaler_filepath}")

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
    #read grind data
    file_path = open_file_dialog()
    if not file_path:
        print("No file selected. Exiting.")
        return

    grind_data = load_data(file_path)

    '''
    #add data from another file
    another_file_path = open_file_dialog()
    if not another_file_path:
        print("No second file selected. Continuing with the first file data.")
    else:
        additional_data = load_data(another_file_path)
        # Assuming you're concatenating rows or merging based on a common column
        grind_data = pd.concat([grind_data, additional_data], ignore_index=True)
    '''
    # Delete rows where removed_material is less than 12
    grind_data = grind_data[grind_data['removed_material'] >= 5]

    # Filter out points which have mad of more than 1000
    grind_data = grind_data[grind_data['mad_rpm'] <= 1000]

    # Filter out avg rpm that is lower than half of rpm_setpoint
    grind_data = grind_data[grind_data['avg_rpm'] >= grind_data['rpm_setpoint'] / 2]

    grind_data = grind_data[pd.isna(grind_data['failure_msg'])]
    print(grind_data)

    #drop unrelated columns
    related_columns = [ 'grind_time', 'avg_rpm', 'avg_force', 'initial_wear', 'removed_material']
    grind_data = grind_data[related_columns]

    

    #desired output
    target_columns = ['removed_material']

    # Preprocess the data (train the model using the CSV data, for example)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(grind_data, target_columns)

    #y_train = y_train.values.ravel()
    #y_test = y_test.values.ravel()

    best_model = train_multi_svr_with_grid_search(X_train, y_train)

    # Optionally, evaluate the model on the test set
    evaluate_model(best_model, X_test, y_test)
 
    #save model
    save_model(best_model, scaler, folder_name='saved_models', modelname='volume_model_svr_V1.pkl', scalername='volume_scaler_svr_V1.pkl')

    '''
    latest model with data gathering 2
    Mean Absolute Error: 3.8609010601716998
    RMS Error: 5.733461811107904
    Mean Squared Error: 32.87258433943272
    R^2 Score: 0.9553432014709704
    '''


if __name__ == "__main__":
    main()