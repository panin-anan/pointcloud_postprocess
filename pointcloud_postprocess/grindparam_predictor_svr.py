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


def main():
    #get grind model
    use_fixed_model_path = False# Set this to True or False based on your need
    
    if use_fixed_model_path:
        # Specify the fixed model path
        fixed_path = 'saved_models/grind_model_svr_V1.pkl'
        grind_model = load_model(use_fixed_path=True, fixed_path=fixed_path)
    else:
        # Load model using file dialog
        grind_model = load_model(use_fixed_path=False)

    #read current belt's 'initial wear', 'removed_volume', 'RPM' and predict 'Force' and 'grind_time'
    initial_wear = 1000000           
    removed_volume = 200      # in mm^3
    avg_rpm = 11000

     # Combine input features into a 2D array (1 sample, 3 features)
    input_data = np.array([[avg_rpm, removed_volume, initial_wear]])

    # Predict 'Force' and 'grind_time'
    predictions = grind_model.predict(input_data)

    # Assuming the model predicts two outputs: 'Force' and 'grind_time'
    predicted_force = predictions[0, 0]
    predicted_grind_time = predictions[0, 1]

    # Print the predictions
    print(f"Predicted Force: {predicted_force}")
    print(f"Predicted Grind Time: {predicted_grind_time}")

if __name__ == "__main__":
    main()
