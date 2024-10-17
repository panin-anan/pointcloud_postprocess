import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import joblib

def load_model(use_fixed_path=False, fixed_path='saved_models/svr_model.pkl'):
    if use_fixed_path:
        # If the argument is True, use the fixed path
        filepath = os.path.abspath(fixed_path)
        print(f"Using fixed path: {filepath}")
    else:
        # Open file dialog to manually select the model file
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open the file dialog and allow the user to select the model file
        filepath = filedialog.askopenfilename(title="Select Model File", filetypes=[("Pickle files", "*.pkl")])
        
        if not filepath:
            print("No file selected. Exiting.")
            return None

    # Load the model
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model



def load_scaler(use_fixed_path=False, fixed_path='saved_models/scaler.pkl'):
    if use_fixed_path:
        # If the argument is True, use the fixed path
        filepath = os.path.abspath(fixed_path)
        print(f"Using fixed path for scaler: {filepath}")
    else:
        # Open file dialog to manually select the scaler file
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open the file dialog and allow the user to select the scaler file
        filepath = filedialog.askopenfilename(title="Select Scaler File", filetypes=[("Pickle files", "*.pkl")])
        
        if not filepath:
            print("No file selected. Exiting.")
            return None

    # Load the scaler
    scaler = joblib.load(filepath)
    print(f"Scaler loaded from {filepath}")
    return scaler

def adjust_predictions_for_good_grind(predictions, mad_rpm_threshold):
    """
    Adjust the force and time predictions if the predicted mad_rpm exceeds the threshold.
    This is a simple rule-based adjustment, and it can be improved with more complex logic.
    """
    for i in range(len(predictions)):
        predicted_mad_rpm = predictions[i, 2]  # Assuming mad_rpm is the 3rd prediction

        # If mad_rpm is above the threshold, reduce the force and increase grind time slightly
        if predicted_mad_rpm > mad_rpm_threshold:
            predictions[i, 0] -= 0.5  # Reduce force
            predictions[i, 1] += 0.5  # Increase grind time

    return predictions


def main():
    #get grind model
    use_fixed_model_path = True# Set this to True or False based on your need
    
    if use_fixed_model_path:
        # Specify the fixed model and scaler paths
        fixed_grind_model_path = 'saved_models/grindparam_model_svr_V1.pkl'
        fixed_grind_scaler_path = 'saved_models/grindparam_scaler_svr_V1.pkl'
        fixed_volume_model_path = 'saved_models/volume_model_svr_V1.pkl'
        fixed_volume_scaler_path = 'saved_models/volume_scaler_svr_V1.pkl'
        
        grind_model = load_model(use_fixed_path=True, fixed_path=fixed_grind_model_path)
        grind_scaler = load_scaler(use_fixed_path=True, fixed_path=fixed_grind_scaler_path)
        volume_model = load_model(use_fixed_path=True, fixed_path=fixed_volume_model_path)
        volume_scaler = load_scaler(use_fixed_path=True, fixed_path=fixed_volume_scaler_path)
    else:
        # Load model and scaler using file dialogs
        grind_model = load_model(use_fixed_path=False)
        grind_scaler = load_scaler(use_fixed_path=False)
        volume_model = load_model(use_fixed_path=False)
        volume_scaler = load_scaler(use_fixed_path=False)

    #read current belt's 'initial wear', 'removed_volume', 'RPM' and predict 'Force' and 'grind_time'
    initial_wear = 1000000           
    removed_material = 50      # in mm^3
    avg_rpm = 10000

    # Create a DataFrame to store the input data
    input_grind_data_dict = {
        'avg_rpm': [avg_rpm],
        'initial_wear': [initial_wear],
        'removed_material': [removed_material]
    }
    input_df = pd.DataFrame(input_grind_data_dict)
    input_scaled = grind_scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
    # Predict grind
    predicted_force_time = grind_model.predict(input_scaled)

    # Assuming the model predicts two outputs: 'Force' and 'grind_time'
    predicted_grind_time = predicted_force_time[0, 0]
    predicted_force = predicted_force_time[0, 1]
    predicted_mad_rpm = predicted_force_time[0, 2]

    # Print the predictions
    print(f"Predicted Force: {predicted_force} N, Predicted Grind Time: {predicted_grind_time}s, Predicted mad_rpm: {predicted_mad_rpm}")


    # TODO use predicted force and time to input into volume_model_svr which predict volume_lost
    input_volume_data_dict = {
    'grind_time': [predicted_grind_time],
    'avg_rpm': [avg_rpm],
    'avg_force': [predicted_force],
    'initial_wear': [initial_wear]
    }
    input_df = pd.DataFrame(input_volume_data_dict)
    input_scaled = volume_scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
    # Predict volume
    predicted_volume = volume_model.predict(input_scaled)
    print(f"RPM: {avg_rpm}, Force: {predicted_force}N, Grind Time: {predicted_grind_time} sec --> Predicted Removed Volume: {predicted_volume[0]}")

    # TODO make decision rule for discrete force time selection

if __name__ == "__main__":
    main()
