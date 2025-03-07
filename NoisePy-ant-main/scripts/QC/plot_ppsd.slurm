#!/bin/bash
#SBATCH --job-name=ppsd_riehen # create a name for your job
#SBATCH --partition=shared-cpu,public-cpu,public-bigmem,shared-bigmem
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=16G         # memory per cpu-core
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output="outslurm/%A_%a-%x.out"
#SBATCH --array=2-198
#SBATCH --mail-type=START,END
#SBATCH --mail-user=genevieve.savard@unige.ch

fname="/home/users/h/henrymi2/scratch/DPM/nodes_stats/station_locations_noisepy.csv"
station=`sed -n "${SLURM_ARRAY_TASK_ID}p" $fname | awk -F, '{print $1}'`
echo $station 

python ppsd_1sta.py $station DPZ
python ppsd_1sta.py $station DPN
python psd_1sta.py $station DPE
