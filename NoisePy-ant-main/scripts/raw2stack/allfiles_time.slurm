#!/bin/bash
#SBATCH --job-name=allfilles_time  # create a name for your job
#SBATCH --partition=public-cpu,shared-cpu,shared-bigmem
#SBATCH --ntasks=1              # total number of parallel tasks in MPI (increase for more speed, but see below)
#SBATCH --cpus-per-task=1        # cpu-cores per task (always choose 1 here)
##SBATCH --mem-per-cpu=20G               # Total memory requested, shared among all CPUs (divide mem 
#SBATCH --mem=800G                # by ntasks to get memory allocated per task) also option to use mem = 800G (total mem needed for all ntasks in 1 node)
#SBATCH --time=7:00:00          # total run time limit (HH:MM:SS) (keep in mind limits of partition requested)
#SBATCH --output="outslurm/%x-%j.out"   # Path where to write output files. Change as you want. %j is job ID, %x is job-name defined above.
#SBATCH --mail-type=END         # if you want to get an email when your job is done
##SBATCH --mail-user=michail.henry@etu.unige.ch  # Your email

echo "Job started at: $(date)":
source $(conda info --base)/etc/profile.d/conda.sh
conda activate noisepy

python create_allfiles_time_csv.py


