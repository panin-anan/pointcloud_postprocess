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

def main():
    #get grind model
    use_fixed_model_path = True# Set this to True or False based on your need
    
    if use_fixed_model_path:
        # Specify the fixed model and scaler paths
        fixed_model_path = 'saved_models/volume_model_svr_V1.pkl'
        fixed_scaler_path = 'saved_models/volume_scaler_svr_V1.pkl'
        
        grind_model = load_model(use_fixed_path=True, fixed_path=fixed_model_path)
        scaler = load_scaler(use_fixed_path=True, fixed_path=fixed_scaler_path)
    else:
        # Load model and scaler using file dialogs
        grind_model = load_model(use_fixed_path=False)
        scaler = load_scaler(use_fixed_path=False)

    #read current belt's 'initial wear', 'removed_volume', 'RPM' and predict 'Force' and 'grind_time'
    rpm_range = np.arange(8500, 10001, 500)  # from 8500 to 10000 in steps of 500
    force_range = np.arange(3, 9, 1)  # from 3 to 9 in steps of 1
    time_range = np.arange(10, 20, 5)
    initial_wear_range = np.arange(1000000, 4000000, 500000)

    for avg_rpm in rpm_range:
       for avg_force in force_range:
           for grind_time in time_range:
                for initial_wear in initial_wear_range:
                    # Create a DataFrame to store the input data
                    input_data_dict = {
                        'grind_time': [grind_time],
                        'avg_rpm': [avg_rpm],
                        'avg_force': [avg_force],
                        'initial_wear': [initial_wear]
                    }
                    input_df = pd.DataFrame(input_data_dict)
                    input_scaled = scaler.transform(input_df)

                    input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)

                    # Predict volume
                    predicted_volume = grind_model.predict(input_scaled)

                    print(f"RPM: {avg_rpm}, Force: {avg_force}N, Grind Time: {grind_time} sec --> Predicted Removed Volume: {predicted_volume[0]}")
                


if __name__ == "__main__":
    main()