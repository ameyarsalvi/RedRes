import os
import pandas as pd

def extract_and_save_csv(input_dir, file_names, output_file):
    """
    Reads the first row and first three columns of each CSV file in the input directory,
    combines the extracted rows, and saves them into a new CSV file.

    Args:
        input_dir (str): Path to the directory containing the input CSV files.
        file_names (list): List of file names to be processed.
        output_file (str): Path to the output CSV file.
    """
    # Initialize an empty list to store the extracted rows
    extracted_data = []

    # Iterate over each file
    for file_name in file_names:
        file_path = os.path.join(input_dir, file_name)
        
        # Check if file exists
        if os.path.exists(file_path):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path, header=None)
            
            # Extract the first row and first three columns
            first_row = df.iloc[0, :3].tolist()
            extracted_data.append(first_row)
        else:
            print(f"Warning: File {file_path} does not exist. Skipping...")

    # Convert the extracted data into a DataFrame
    output_df = pd.DataFrame(extracted_data, columns=["Col1", "Col2", "Col3"])

    # Save the DataFrame to the output CSV file
    output_df.to_csv(output_file, index=False, header=False)
    print(f"Data has been successfully saved to {output_file}")

# Input directory containing CSV files
input_directory = "/home/asalvi/code_workspace/Husky_CS_SB3/SkidSteerRR/train/MixPathFlip/"

# List of 10 CSV file names
csv_file_names = [
    "ArcPath1.csv", "ArcPath1_.csv", "ArcPath2.csv", "ArcPath2_.csv", "ArcPath3.csv",
    "ArcPath3_.csv", "ArcPath4.csv", "ArcPath4_.csv", "ArcPath5.csv", "ArcPath5_.csv"
]

# Output file path
output_csv = "/home/asalvi/code_workspace/Husky_CS_SB3/SkidSteerRR/train/MixPathFlip/origin_list.csv"

# Call the function
extract_and_save_csv(input_directory, csv_file_names, output_csv)
