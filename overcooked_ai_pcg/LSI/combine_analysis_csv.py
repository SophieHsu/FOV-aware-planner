import os
import pandas as pd
from overcooked_ai_pcg import LSI_STEAK_STUDY_RESULT_DIR

# Set the directory where your CSV files are located
base_dir = LSI_STEAK_STUDY_RESULT_DIR
LSI_FOLDER = '/home/icaros/Documents/sophie/overcooked_ai/overcooked_ai_pcg/LSI'
# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Iterate through subdirectories and files
# for dir in os.listdir(base_dir):
for i in range(37):
    # os.system("python {}/steak_study.py --replay -type all --gen_vid -l {}".format(LSI_FOLDER, i))
    # if str(i) not in ['4', '9','11','17','21','24','28','29','31','34']:
    if str(i) not in ['0', '4', '10', '6', '12', '16', '19', '23', '26', '28', '27']:
    # if str(i) in ['36', '33', '34', '35', '18', '20', '21', '22', '25', '29', '30', '31', '32', '3', '8']:
    # if os.path.isdir(os.path.join(base_dir, str(i))): 
        print(os.path.join(base_dir, str(i)))
        for doc in os.listdir(os.path.join(base_dir, str(i))):
            if doc.endswith('analysis_log.csv'):
                # Create the full path to the CSV file
                file_path = os.path.join(base_dir, str(i), doc)
                
                # Read the CSV file into a DataFrame, skipping the first row
                df = pd.read_csv(file_path)#, skiprows=1)
                
                # Append the data to the combined DataFrame
                combined_data = combined_data.append(df, ignore_index=True)

# Specify the path where you want to save the combined CSV file
output_csv_path = os.path.join(LSI_STEAK_STUDY_RESULT_DIR, 'analysis_log.csv')

# Save the combined data to a single CSV file
combined_data.to_csv(output_csv_path, index=False)

print(f"Combined data saved to {output_csv_path}")