#!/bin/bash
#SBATCH --job-name=cmame-workloads-diff-rl-rand-vs-mdp-small
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --account=nikolaid_548
#SBATCH --output ./hpc/logs/slurm-%j.out
#SBATCH --error ./hpc/logs/slurm-%j.out

# The configuration of this main job is very small because Dask spawns other
# jobs.

echo "========== SLURM JOB INFO =========="
echo
echo "The job will be started on the following node(s):"
echo $SLURM_JOB_NODELIST
echo
echo "Slurm User:         $SLURM_JOB_USER"
echo "Run Directory:      $(pwd)"
echo "Job ID:             $SLURM_JOB_ID"
echo "Job Name:           $SLURM_JOB_NAME"
echo "Partition:          $SLURM_JOB_PARTITION"
echo "Number of nodes:    $SLURM_JOB_NUM_NODES"
echo "Number of tasks:    $SLURM_NTASKS"
echo "Submitted From:     $SLURM_SUBMIT_HOST"
echo "Submit directory:   $SLURM_SUBMIT_DIR"
echo "Hostname:           $(hostname)"
echo
echo "Dashboard Host:     |$(hostname):8787|"
echo

echo
echo "========== Start =========="
date

echo
echo "========== Setup =========="

module load gcc/8.3.0
module load anaconda3

eval "$(conda shell.bash hook)"
conda activate overcooked_ai
conda env list

echo
echo "========== Starting Dask Python script =========="

python -u run_search.py \
  -s "small" \
  -c "data/config/experiment/CMAME_workloads_diff_rl_rand_small_vs_mdp_slurm.tml"

echo
echo "========== Done =========="
date
