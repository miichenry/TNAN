#!/bin/bash
#SBATCH --job-name=disp  # create a name for your job
#SBATCH --partition=shared-cpu,public-cpu,public-bigmem,shared-bigmem
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=3        # cpu-cores per task
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)
##SBATCH --output="./outslurm/"
#SBATCH --output="outslurm/%x_%A_%a.out"
##SBATCH --mail=END
#SBATCH --array=0-230 #max 22146

source /opt/ebsofts/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate picking

fname="text_files/stack_filelist_SS_rma_full.txt"
log_file="log/array_log_rma.txt"
processed_log_file="log/processed_array_log_rma.txt" # Log file for processed files
failed_log_file="log/failed_array_log_rma.txt"       # Log file for failed files

# Initialize log files only once when SLURM_ARRAY_TASK_ID is 0
if [ "${SLURM_ARRAY_TASK_ID}" -eq "0" ]; then
    echo "Initializing log files: $log_file, $processed_log_file, $failed_log_file"
    echo "ROW, FILEPATH, STATUS, SLURM_ARRAY_TASK_ID" > "$log_file"
    echo "ROW, FILEPATH, SLURM_ARRAY_TASK_ID" > "$processed_log_file"
    echo "ROW, FILEPATH, SLURM_ARRAY_TASK_ID" > "$failed_log_file"
fi

# Loop over the sequence 0-99
for k in $(seq 0 99); do
    # Calculate ROW value based on SLURM_ARRAY_TASK_ID
    ROW=$(( ${SLURM_ARRAY_TASK_ID} * 100 + $k ))

#    # Skip ROW if it's 0
#    if [ "$ROW" -eq "0" ]; then
#        continue
#    fi
#
    # Skip if ROW is greater than the max value of 21529
    if [ "$ROW" -gt "21321" ]; then
        continue
    fi

    # Extract the file corresponding to the ROW from the file list
    sfile=$(sed -n "${ROW}p" $fname)

    # Check if the file exists
    if [ -z "$sfile" ]; then
        echo "$ROW, No file found, SKIPPED, ${SLURM_ARRAY_TASK_ID}" >> "$log_file"
        continue
    fi

    # Print the current task details
    echo $k $ROW $sfile

    # Run the Python script and log the status
    if python dispersion_curves_V2.py "$sfile"; then
        echo "$ROW, $sfile, PROCESSED, ${SLURM_ARRAY_TASK_ID}" >> "$log_file"
        echo "$ROW, $sfile, ${SLURM_ARRAY_TASK_ID}" >> "$processed_log_file"
    else
        echo "$ROW, $sfile, FAILED, ${SLURM_ARRAY_TASK_ID}" >> "$log_file"
        echo "$ROW, $sfile, ${SLURM_ARRAY_TASK_ID}" >> "$failed_log_file"
    fi
done
