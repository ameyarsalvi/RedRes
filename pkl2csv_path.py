import pickle
import os
import pandas as pd

def extractData(EvalPath, specifier):
    # Path to the merged pickle file
    filename = os.path.join(EvalPath, f"{specifier}_all_logs.pkl")
    
    try:
        with open(filename, "rb") as fp:
            print(f"Loading file: {filename}")  # Debugging line
            data = pickle.load(fp)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        data = {}
    
    return data

# Define the base evaluation path
eval_path = '/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/Policies/AsynDump/'

# Define the X and Y values
#x_values = ['216', '288', '432', '864', '2160']
#x_values = ['idx1', 'idx2', 'idx3', 'idx4']
#x_values = ['Guided','ImgCent']
#y_values = ['0.15', '0.3', '0.45', '0.6', '0.75']
#y_values = ['0.75']
x_values = ['ICG_10','ICG_15','ICG_20','WPG_10','WPG_15','WPG_20']

# Prepare directory for saving CSV files
output_dir = os.path.join(eval_path, "output_csv")
os.makedirs(output_dir, exist_ok=True)

# Extract data and save to CSV files
for X in x_values:
    #for Y in y_values:
        #specifier = f"{X}_{Y}"
        specifier = f"{X}"
        print(f"Processing specifier: {specifier}")
        data = extractData(eval_path, specifier)
        
        if data:
            # Convert the data to a pandas DataFrame
            df = pd.DataFrame(data)
            
            # Replace '.' with 'p' in the specifier for the CSV filename
            csv_specifier = specifier.replace('.', 'p')
            
            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(output_dir, f"x{csv_specifier}.csv"), index=False)
        else:
            print(f"No data extracted for specifier {specifier}. Skipping CSV generation.")
