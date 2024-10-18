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
        #print(f"Using fixed path: {filepath}")
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
        #print(f"Using fixed path for scaler: {filepath}")
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
    #print(f"Scaler loaded from {filepath}")
    return scaler

def adjust_force_with_volume_model(volume_model, volume_scaler, initial_wear, avg_rpm, predicted_grind_time, predicted_force, predicted_volume, target_volume, max_iterations=6):
    """
    Adjust the force if the grind time is lower than 10 or higher than 20,
    ensuring that the predicted volume remains constant.
    """
    # Set a small tolerance for volume prediction accuracy
    tolerance = 5e-0
    iteration_count = 0

    # Adjust the time to get closer to the predicted volume from volume model
    predicted_new_volume = predicted_volume
    adjusted_force = predicted_force

    print('adjusting force')
    # Adjust grind time to get closer to the predicted volume from the volume model
    while iteration_count < max_iterations:
        
        # Check if the predicted volume is close enough to the original volume
        if abs(predicted_new_volume - target_volume) < tolerance:
            break  # Stop adjusting if the volume is within tolerance

        # Adjust the grind time based on the difference
        if predicted_new_volume < target_volume:
            adjusted_force += 0.5  # Increase force
        else:
            adjusted_force -= 0.5  # Decrease force

        # Predict the volume with the current adjusted time
        predicted_new_volume = predict_volume(volume_model, volume_scaler, initial_wear, avg_rpm, adjusted_force, predicted_grind_time)
        iteration_count += 1
        
        print(f"RPM: {avg_rpm}, Force: {adjusted_force}N, Grind Time: {predicted_grind_time} sec --> Predicted Removed Volume: {predicted_new_volume}")
        
        if adjusted_force <= 3.0:
            print('minimum force reached')
            adjusted_force = 3.0
            predicted_new_volume = predict_volume(volume_model, volume_scaler, initial_wear, avg_rpm, adjusted_force, predicted_grind_time)
            print(f"RPM: {avg_rpm}, Force: {adjusted_force}N, Grind Time: {predicted_grind_time} sec --> Predicted Removed Volume: {predicted_new_volume}")
            break

    return adjusted_force, predicted_new_volume

def adjust_time_with_volume_model(volume_model, volume_scaler, initial_wear, avg_rpm, predicted_grind_time, predicted_force, predicted_volume, target_volume, max_iterations=10):
    """
    Adjust the force if the grind time is lower than 10 or higher than 20,
    ensuring that the predicted volume remains constant.
    """
    # Set a small tolerance for volume prediction accuracy
    tolerance = 1e-0
    iteration_count = 0

    # Adjust the time to get closer to the predicted volume from volume model
    predicted_new_volume = predicted_volume
    adjusted_time = predicted_grind_time
    print('adjusting time')

    # Adjust grind time to get closer to the predicted volume from the volume model
    while iteration_count < max_iterations:
        
        # Check if the predicted volume is close enough to the original volume
        if abs(predicted_new_volume - target_volume) < tolerance:
            break  # Stop adjusting if the volume is within tolerance

        # Adjust the grind time based on the difference
        if predicted_new_volume < target_volume:
            adjusted_time += 1.0  # Increase time if predicted volume is less than target volume
        else:
            adjusted_time -= 1.0  # Decrease time if predicted volume is greater than target volume
        iteration_count += 1

        # Predict the volume with the current adjusted time
        predicted_new_volume = predict_volume(volume_model, volume_scaler, initial_wear, avg_rpm, predicted_force, adjusted_time)

        print(f"RPM: {avg_rpm}, Force: {predicted_force}N, Grind Time: {adjusted_time} sec --> Predicted Removed Volume: {predicted_new_volume}")
        if adjusted_time <= 6.0:
            print('minimum time reached')
            break

    return adjusted_time, predicted_new_volume

def predict_volume(volume_model, volume_scaler, initial_wear, avg_rpm, avg_force, grind_time):
    """
    Utility function to predict volume based on current parameters.
    """
    input_volume_data_dict = {
        'grind_time': [grind_time],
        'avg_rpm': [avg_rpm],
        'avg_force': [avg_force],
        'initial_wear': [initial_wear]
    }
    input_df = pd.DataFrame(input_volume_data_dict)
    input_scaled = volume_scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
    predicted_volume = volume_model.predict(input_scaled)
    return predicted_volume[0]  # Return the predicted volume


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
    target_volume = 10      # in mm^3
    avg_rpm = 10000

    # Create a DataFrame to store the input data
    input_grind_data_dict = {
        'avg_rpm': [avg_rpm],
        'initial_wear': [initial_wear],
        'removed_material': [target_volume]
    }
    input_df = pd.DataFrame(input_grind_data_dict)
    input_scaled = grind_scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
    # Predict grind
    predicted_force_time = grind_model.predict(input_scaled)

    # Assuming the model predicts two outputs: 'Force' and 'grind_time'
    predicted_grind_time = predicted_force_time[0, 0]
    predicted_force = predicted_force_time[0, 1]
    #predicted_mad_rpm = predicted_force_time[0, 2]

    # Print the predictions
    print(f"Predicted Force: {predicted_force} N, Predicted Grind Time: {predicted_grind_time}s")


    # TODO add decision rule: for low volume use the current model, for high volume can scale volume lost with time
    # TODO use predicted force and time to input into volume_model_svr which predict volume_lost
    # Predict volume
    predicted_volume = predict_volume(volume_model, volume_scaler, initial_wear, avg_rpm, predicted_force, predicted_grind_time)
    print(f"RPM: {avg_rpm}, Force: {predicted_force}N, Grind Time: {predicted_grind_time} sec --> Predicted Removed Volume: {predicted_volume[0]}")

    # TODO adjust force and time for good grind and accurate volume with volume model
    adjusted_force, predicted_volume = adjust_force_with_volume_model(volume_model, volume_scaler, initial_wear, avg_rpm, predicted_grind_time, predicted_force, predicted_volume, target_volume)
    #print(f"RPM: {avg_rpm}, Force: {adjusted_force}N, Grind Time: {predicted_grind_time} sec --> Predicted Removed Volume: {predicted_volume[0]}")

    adjusted_time, predicted_volume = adjust_time_with_volume_model(volume_model, volume_scaler, initial_wear, avg_rpm, predicted_grind_time, adjusted_force, predicted_volume, target_volume)
    #print(f"RPM: {avg_rpm}, Force: {predicted_force}N, Grind Time: {adjusted_time} sec --> Predicted Removed Volume: {predicted_volume[0]}")



if __name__ == "__main__":
    main()
