# submit multiple Phono3py BTE calculations of thermal conductivity, requires a conda environment for phono3py to be setup
# afterwards read to a single file using phono3py-load --mesh $MESH $MESH $MESH --br --read-gamma

#!/bin/bash

args=("$@")
# check for number of args
if ! ((${#args[@]}>=3)); then
  echo -e "Error: Need minimum 3 arguments --- suggested usage\ncalc_phono3py_tc.sh <job-name> <mesh-n-for-nxnxn> <num-jobs> <num-cores-per-job> \n"
  exit 64
fi
# rename args for legibility
# What you'd like your job to be called (e.g. final_final_job_v4):
JOB_NAME=${args[0]}
# mesh size
MESH=${args[1]}
# number of jobs in the array
NJOB=${args[2]}
# Number of OMP threads per MPI process:
CPUS=${args[3]:-1}

# check for ir_grid_points.yaml
if [ ! -f "ir_grid_points.yaml" ]; then
    echo -e "Cannot find ir_grid_points.yaml\nCreating ir_grid_points.yaml with $MESH $MESH $MESH mesh."
    # activate phono3py conda environment
    conda init
    conda activate /cluster/medbow/project/design-lab/software/Phono3py
    phono3py-load --mesh $MESH $MESH $MESH --wgp
fi

# Variables
# User email
USER_MAIL=""
# Number of Nodes
NODE=1
# NCORE = NTASK*NCPUS
# Number of MPI tasks per node:
TASK=1 

# Memory per node:
MEM=24GB
# Time for the job:
TIME=6:00:00

PROC=$((NODE * TASK))

j=0
for gps in `python -c "import sys; import yaml; from yaml import Loader; num = int(sys.argv[1]) if len(sys.argv) > 1 else 1; data = yaml.load(open('ir_grid_points.yaml'), Loader=Loader); gps = [gp['grid_point'] for gp in data['ir_grid_points']]; gp_lists = [[] for i in range(num)]; [gp_lists[i % num].append(gp) for i, gp in enumerate(gps)]; [print(','.join([str(gp) for gp in gp_set])) for gp_set in gp_lists]" "$NJOB"`;do 
# Generate particular job submit script
SCRIPT_NAME=run_${JOB_NAME}_${j}

cat > $SCRIPT_NAME<<!
#!/bin/bash
#SBATCH --job-name=${j}_${JOB_NAME}
#SBATCH --time=$TIME
#SBATCH --nodes=$NODE
#SBATCH --ntasks-per-node=$TASK
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --account=design-lab
#SBATCH --partition=inv-desousa
##SBATCH --partition=teton
#SBATCH --output=output/${JOB_NAME}_${j}.out
#SBATCH --error=output/${JOB_NAME}_${j}.err
#SBATCH --mail-type=ALL
# #SBATCH --mail-user=$USER_MAIL # change to your email here, if you want to receive emails

cd \$SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=$CPUS

module load gcc/14.2.0 openmpi/5.0.5 fftw/3.3.10-ompi openblas/0.3.24 netlib-scalapack/2.2.0-ompi wannier90/3.1.0 hdf5/1.14.3__hl_True__fortran_True-ompi

# activate phono3py conda environment
conda activate /cluster/medbow/project/design-lab/software/Phono3py

phono3py-load --mesh $MESH $MESH $MESH --br --gp "$gps" --write-gamma

!
# echo $gps

JOBID=($(sbatch $SCRIPT_NAME))
JOBID=${JOBID[3]}
printf "\n"
echo "  Job is queued. Job ID is: $JOBID"

start_secs=$(date --date="$start" '+%s')
end_secs=$(date --date="$end"   '+%s')
duration=$((end_secs - start_secs))

cat >> $SCRIPT_NAME<<!
#$JOBID

!

j=$(( j + 1 ))

done

echo -e "Submitted $NJOB jobs for $MESH mesh.\nRun [phono3py-load --mesh $MESH $MESH $MESH --br --read-gamma] to generate kappa-m${MESH}${MESH}${MESH}hdf5 Once the jobs are done."