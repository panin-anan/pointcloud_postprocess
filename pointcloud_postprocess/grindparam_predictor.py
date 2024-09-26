import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_svr(X_train, y_train):
    """
    Train a Support Vector Regressor (SVR) for multi-output regression.
    Wrap the SVR with MultiOutputRegressor to handle multiple targets.
    """
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    
    # Wrap the SVR model with MultiOutputRegressor to handle multi-output regression
    multioutput_svr = MultiOutputRegressor(svr)
    multioutput_svr.fit(X_train, y_train)

    return multioutput_svr

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Evaluate the model with Mean Squared Error and R^2 Score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    '''
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
    '''

def create_grind_model(mstore):
    """
    Train the model if it does not exist yet in the MeshProcessor object.
    :param mstore: MeshProcessor object that holds the model and data.
    """
    if mstore.model is None:
        # Load the pre-trained model dataset
        file_path = '/workspaces/BrightSkyRepoLinux/mesh_sample/grinding_material_removal.csv'
        data = load_data(file_path)
        target_columns = ['RPM', 'Force']

        # Preprocess the data (train the model using the CSV data, for example)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data, target_columns)

        # Train the SVR model
        mstore.model = train_svr(X_train, y_train)
        mstore.scaler = scaler  # Save the scaler for scaling inputs later

        # Optionally, evaluate the model on the test set
        evaluate_model(mstore.model, X_test, y_test)
    else:
        print("Using previously trained model.")


def predict_grind_param(mstore, feed_rate):
    # Prepare inputs for the SVR model (lost volume + feed rate for each section)
    input_data = pd.DataFrame({
        'Feed_Rate': [feed_rate] * len(mstore.lost_volumes),
        'Lost_Volume': [param['lost_volume'] for param in mstore.lost_volumes]
    })

    # Scale the input_data using the stored scaler
    input_data_scaled = mstore.scaler.transform(input_data)

    # Predict the RPM and Force using the lost volume and feed rate
    predictions = mstore.model.predict(input_data_scaled)

    # Output predictions with segment and subsection indices
    predicted_rpm_force = pd.DataFrame(predictions, columns=['RPM', 'Force'])
    predicted_rpm_force['Segment'] = [param['section_idx'] for param in mstore.lost_volumes]
    predicted_rpm_force['Sub_Section'] = [param['sub_section_idx'] for param in mstore.lost_volumes]
    
    # Print results with correct segment and subsection designations
    print(predicted_rpm_force[['Segment', 'Sub_Section', 'RPM', 'Force']])
