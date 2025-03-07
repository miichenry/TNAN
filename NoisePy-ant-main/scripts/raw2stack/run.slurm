#!/bin/bash
#SBATCH --partition=public-cpu,shared-cpu,shared-bigmem,public-bigmem #,public-cpu,public-bigmem
#SBATCH --job-name=S2   # job name
#SBATCH --ntasks=40              # total number of parallel tasks in MPI (increase for more speed, but see below)
#SBATCH --cpus-per-task=1        # cpu-cores per task (always choose 1 here)
#SBATCH --mem=200G               # (ntasks get memory allocated per task) also option to use mem = 800G (total mem needed for all ntasks in 1 node)
#SBATCH --time=1-00:00:00          # total run time limit (HH:MM:SS) (keep in mind limits of partition requested4
#SBATCH --output="outslurm/%x-%j.out"   # Path where to write output files. Change as you want. %j is job ID, %x is job-name defined above.
#SBATCH --mail-type=BEGIN,END
##SBATCH --mail-user=michail.henry@etu.unige.ch
###SBATCH --mem-per-cpu=20G               # Total memory requested, shared among all CPUs (divide mem 

# Copy this slurm files and the param files in your project directory and modify them there. Adjust the path to the script below

# *** Choice of NTASKS (NPROCS): ***
# Step 2: Do not specify more than the number of station pairs.
# Step 1: Do not specify more than the number of .h5 files in RAW_DATA
# Step 0: Do not specify more than the is number of time chunks (i.e. total dataset duration in hours divided by inc_hours)

source /opt/ebsofts/Anaconda3/2022.05/etc/profile.d/conda.sh #Loads conda command
conda activate noisepy

# *** Define the name of the input parameter file. ***
#PARAMS="S0B_params.yaml" 	#ntasks = 40, mem = 500G
#PARAMS="S1_params.yaml" 	#ntasks = 40, mem = 400G
PARAMS="S2_params.yaml"		#ntasks = 40, mem = 200G, time=3-00:00:00 

# *** Define the name of the script to use ***
#SCRIPT=S0B_to_ASDF.py   	# Convert mseed dataset to H5 files in time chunks
#SCRIPT=S1_fft_cc_MPI.py 	# Get cross-correlations  
#SCRIPT=S1_fft_cc_MPI_normZ.py 	# Get cross-correlations, with 3C data normalized by Z component (preserves relative amplitudes)
SCRIPT=S2_stacking.py  		# Stacking of the cross-correlations
#SCRIPT=S2_stacking_acorr.py 	# Stacking for autocorrelations only

#** Now call run the script with MPI! ***
mpiexec -n $SLURM_NPROCS python $SCRIPT $PARAMS
