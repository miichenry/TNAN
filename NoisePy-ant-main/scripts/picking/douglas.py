import os
import random
import shutil
import logging

print('Yeah')
# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

def copy_files(input_dir, input_dir_2, output_dir_1,  output_dir_2, num_files=100):
    logging.info(f"Starting to copy files from {input_dir} to {output_dir_2}.")

    # Get a list of all .dat files in the input directory
    dat_files = [f for f in os.listdir(input_dir) if f.endswith('.dat')]

    if not dat_files:
        logging.warning("No .dat files found in the input directory.")
        return

    # Select random files
    try:
        selected_files = random.sample(dat_files, min(num_files, len(dat_files)))
    except ValueError as e:
        logging.error(f"Error selecting files: {e}")
        return

    # Copy selected .dat files to the output directory
    for file in selected_files:
        src_file_path = os.path.join(input_dir, file)
        try:
            shutil.copy(src_file_path, output_dir_1)
            logging.info(f"Copied {file} to {output_dir_1}.")
            
            # Extract the numeric parts from the filename
            parts = file.split('.')
            if len(parts) == 3:
                prefix = f"SS.{parts[0]}_SS.{parts[1]}_group_all.csv"
                csv_file_path = os.path.join(input_dir_2, prefix)

                # Check if the corresponding CSV file exists and copy it
                if os.path.exists(csv_file_path):
                    shutil.copy(csv_file_path, output_dir_2)
                    logging.info(f"Copied {prefix} to {output_dir_2}.")
                else:
                    logging.warning(f"CSV file {prefix} not found for {file}.")
        except Exception as e:
            logging.error(f"Error copying file {file}: {e}")

    logging.info("File copying process completed.")

# Specify your input and output directories here
input_directory = '/srv/beegfs/scratch/shares/cdff/DPM/DisperPicker/data/TestData/group_image'
input_directory_2 = '/srv/beegfs/scratch/shares/cdff/DPM/Postprocessing/dispersion/disp_pws_linear_ZZ_v0.2-5.0_Tmin0.2_dT0.1_dvel0.01'
output_directory_1 = './douglas_csv/DisperPicker'
output_directory_2 = './douglas_csv/pasDisperPicker'

# Call the function to copy files
copy_files(input_directory, input_directory_2, output_directory_1, output_directory_2)
