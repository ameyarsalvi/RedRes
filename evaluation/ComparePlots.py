

import pickle
import os
import matplotlib.pyplot as plt

def extract_data(filepath):
    """Load data from the given pickle file."""
    try:
        with open(filepath, "rb") as fp:
            print(f"Loading file: {filepath}")
            data = pickle.load(fp)
            return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

# Define the base evaluation path
eval_path = '/home/asalvi/code_workspace/tmp/RedRes/Data4Plot/'

# Get all files in the directory
all_files = [f for f in os.listdir(eval_path) if f.endswith(".pkl")]

# Initialize a dictionary to organize files by specifier
file_dict = {}
for file in all_files:
    if "rel_poseX" in file or "rel_poseY" in file:
        # Extract the common specifier (e.g., 2W1C, 2W)
        specifier = file.split("_rel_")[0]
        if specifier not in file_dict:
            file_dict[specifier] = {}
        if "rel_poseX" in file:
            file_dict[specifier]["poseX"] = os.path.join(eval_path, file)
        if "rel_poseY" in file:
            file_dict[specifier]["poseY"] = os.path.join(eval_path, file)

# Initialize a plot
plt.figure(figsize=(10, 8))
plt.title("Trajectories Comparison (rel_PoseX vs rel_PoseY)")
plt.xlabel("rel_PoseX")
plt.ylabel("rel_PoseY")

# Iterate over specifiers and plot their trajectories
for specifier, files in file_dict.items():
    if "poseX" in files and "poseY" in files:
        # Load the data for poseX and poseY
        rel_pose_x = extract_data(files["poseX"])
        rel_pose_y = extract_data(files["poseY"])
        
        # Check if data is valid
        if len(rel_pose_x) > 0 and len(rel_pose_y) > 0:
            # Plot the trajectory
            plt.plot(rel_pose_x, rel_pose_y, label=specifier)
        else:
            print(f"Missing or invalid data for specifier: {specifier}")
    else:
        print(f"Missing poseX or poseY file for specifier: {specifier}")

# Add legend and grid
plt.legend()
plt.grid(True)
plt.show()

