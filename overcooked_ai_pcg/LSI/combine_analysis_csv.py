import os
import pandas as pd
from overcooked_ai_pcg import LSI_STEAK_STUDY_RESULT_DIR

# Set the directory where your CSV files are located
base_dir = LSI_STEAK_STUDY_RESULT_DIR
LSI_FOLDER = '/home/icaros/Documents/sophie/overcooked_ai/overcooked_ai_pcg/LSI'
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
for i in range(37):
    # os.system("python {}/steak_study.py --replay -type all --gen_vid -l {}".format(LSI_FOLDER, i))
    # if str(i) not in ['4', '9','11','17','21','24','28','29','31','34']:
    if str(i) not in ['0', '4', '10', '6', '12', '16', '19', '23', '26', '28', '27', '9', '11', '17','29']:
    # if str(i) in ['36', '33', '34', '35', '18', '20', '21', '22', '25', '29', '30', '31', '32', '3', '8']:
    # if True:
        print(os.path.join(base_dir, str(i)))
        for doc in os.listdir(os.path.join(base_dir, str(i))):
            if doc.endswith('analysis_log.csv'):
                # Create the full path to the CSV file
                file_path = os.path.join(base_dir, str(i), doc)
                
                # Read the CSV file into a DataFrame, skipping the first row
                df = pd.read_csv(file_path)#, skiprows=1)
                
                # Append the data to the combined DataFrame
                for index, lvl_config in df.iterrows():
                    if lvl_config['lvl_type'] in ['Mid-2_120_not_aware_3', 'Mid-2_120_not_aware','Mid-2-120', 'Mid-2-120_3']:
                        tmp_data = tmp_data.append(lvl_config)
                        tmp_mid2_data = tmp_mid2_data.append(lvl_config)
                    if lvl_config['lvl_type'] in ['None-3-120', 'None-3-120_3','None-3_120_not_aware', 'None-3_120_not_aware_3']:
                        tmp_data = tmp_data.append(lvl_config)
                        tmp_none3_data = tmp_none3_data.append(lvl_config)
                    if lvl_config['lvl_type'] in ['Mid-1_120_not_aware_3','Mid-1-120_3']:
                        tmp_data = tmp_data.append(lvl_config)
                        tmp_mid1_data = tmp_mid1_data.append(lvl_config)

                combined_data = combined_data.append(tmp_data)#, ignore_index=True)
                mid1_data = mid1_data.append(tmp_mid1_data, ignore_index=True)
                mid2_data = mid2_data.append(tmp_mid2_data)#, ignore_index=True)
                none3_data = none3_data.append(tmp_none3_data)#, ignore_index=True)

                tmp_data = pd.DataFrame()
                tmp_mid2_data = pd.DataFrame()
                tmp_none3_data = pd.DataFrame()
                tmp_mid1_data = pd.DataFrame()

# Specify the path where you want to save the combined CSV file
output_csv_path = os.path.join(LSI_STEAK_STUDY_RESULT_DIR, 'analysis_log.csv')
mid1_csv_path = os.path.join(LSI_STEAK_STUDY_RESULT_DIR, 'analysis_log_mid1.csv')
mid2_csv_path = os.path.join(LSI_STEAK_STUDY_RESULT_DIR, 'analysis_log_mid2.csv')
none3_csv_path = os.path.join(LSI_STEAK_STUDY_RESULT_DIR, 'analysis_log_none3.csv')

# Save the combined data to a single CSV file
combined_data.to_csv(output_csv_path, index=False)
mid2_data.to_csv(mid2_csv_path, index=False)
none3_data.to_csv(none3_csv_path, index=False)
mid1_data.to_csv(mid1_csv_path, index=False)

print(f"Combined data saved to {output_csv_path}")