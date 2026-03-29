import subprocess
import os

# Specify the folder containing the scripts
script_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/scripts"

# List of scripts to execute in order
scripts = [
    "04_tune_baselearners.py",
    "05_run_repeated_cv.py",
    "06_format_cv_summary_table.py"
]

for script in scripts:
    script_path = os.path.join(script_folder, script)
    print(f"Executing {script_path}...")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    
    # Print output and errors (if any)
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
    
    print(f"Finished executing {script}\n")

print("All scripts executed successfully.")
