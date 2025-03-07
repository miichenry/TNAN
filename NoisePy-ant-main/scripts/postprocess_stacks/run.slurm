#!/bin/bash
#SBATCH --partition=public-cpu,shared-cpu,shared-bigmem,public-bigmem #,public-cpu,public-bigmem
#SBATCH --job-name=extract_ncfs   # job name
##SBATCH --ntasks=1              # total number of parallel tasks in MPI (increase for more speed, but see below)
#SBATCH --cpus-per-task=20       # cpu-cores per task (always choose 1 here)
##SBATCH --mem=300G               # (ntasks get memory allocated per task) also option to use mem = 800G (total mem needed for all ntasks in 1 node)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS) (keep in mind limits of partition requested4
#SBATCH --output="outslurm/%x-%j.out"   # Path where to write output files. Change as you want. %j is job ID, %x is job-name defined above.
#SBATCH --mail-type=BEGIN,END
###SBATCH --mem-per-cpu=20G               # Total memory requested, shared among all CPUs (divide mem 

source /opt/ebsofts/Anaconda3/2022.05/etc/profile.d/conda.sh #Loads conda command
conda activate noisepy

python -u extract_ncfs.py ZZ #beamform.py #extract_ncfs.py
python -u extract_ncfs.py RR
python -u extract_ncfs.py TT
python -u extract_ncfs.py ZR
python -u extract_ncfs.py RZ
