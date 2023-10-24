import os
import pandas as pd
from overcooked_ai_pcg import LSI_STEAK_STUDY_RESULT_DIR

# Set the directory where your CSV files are located
base_dir = os.path.join(os.getcwd(), 'overcooked_ai_py/data/logs/vr_analysis/')
# LSI_FOLDER = '/home/icaros/Documents/sophie/overcooked_ai/overcooked_ai_pcg/LSI'
# Initialize an empty DataFrame to store the combined data
combined_data = pd.DataFrame()
mid1_data = pd.DataFrame()
mid2_data = pd.DataFrame()
none3_data = pd.DataFrame()
tmp_data = pd.DataFrame()
tmp_mid2_data = pd.DataFrame()
tmp_none3_data = pd.DataFrame()
tmp_mid1_data = pd.DataFrame()

# Iterate through subdirectories and files
# for dir in os.listdir(base_dir):
for i in range(5,16):
    # os.system("python {}/steak_study.py --replay -type all --gen_vid -l {}".format(LSI_FOLDER, i))
    if True:
        print(os.path.join(base_dir, str(i)))
        for doc in os.listdir(os.path.join(base_dir, str(i))):
            if doc.endswith('analysis_log.csv'):
                # Create the full path to the CSV file
                file_path = os.path.join(base_dir, str(i), doc)
                
                # Read the CSV file into a DataFrame, skipping the first row
                df = pd.read_csv(file_path)#, skiprows=1)
                
                # Append the data to the combined DataFrame
                for index, lvl_config in df.iterrows():
                    if 'mid' in lvl_config['lvl_type']:
                        tmp_data = pd.concat([tmp_data, pd.DataFrame([lvl_config])], ignore_index=True)
                        tmp_mid2_data = pd.concat([tmp_mid2_data, pd.DataFrame([lvl_config])], ignore_index=True)
                    if 'none' in lvl_config['lvl_type']:
                        tmp_data = pd.concat([tmp_data, pd.DataFrame([lvl_config])], ignore_index=True)
                        tmp_none3_data = pd.concat([tmp_none3_data, pd.DataFrame([lvl_config])], ignore_index=True)

                combined_data = pd.concat([combined_data, tmp_data], ignore_index=True)
                mid2_data = pd.concat([mid2_data, tmp_mid2_data], ignore_index=True)
                none3_data = pd.concat([none3_data, tmp_none3_data], ignore_index=True)

                tmp_data = pd.DataFrame()
                tmp_mid2_data = pd.DataFrame()
                tmp_none3_data = pd.DataFrame()

# Specify the path where you want to save the combined CSV file
output_csv_path = os.path.join(base_dir, 'analysis_log.csv')
mid2_csv_path = os.path.join(base_dir, 'analysis_log_mid2.csv')
none3_csv_path = os.path.join(base_dir, 'analysis_log_none3.csv')

# Save the combined data to a single CSV file
combined_data.to_csv(output_csv_path, index=False)
mid2_data.to_csv(mid2_csv_path, index=False)
none3_data.to_csv(none3_csv_path, index=False)

print(f"Combined data saved to {output_csv_path}")