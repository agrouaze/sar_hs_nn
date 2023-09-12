#!/bin/bash
#SBATCH --account=deu@gpu
#SBATCH -A deu@gpu
#SBATCH --job-name=HsWVsadowski_exp0     # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1         # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=1           # number of cores per tasks
##SBATCH --hint=nomultithread         # we get physical cores not logical
##SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=1:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=HsWVsadowski_exp0%j.out # output file name
#SBATCH --error=HsWVsadowski_exp0%j.out  # error file name
#SBATCH --qos=qos_gpu-dev         # we are submitting a test job
#SBATCH --mem=60G           # total memory per node
# asks SLURM to send the USR1 signal 20 seconds before the end of the time limit
#SBATCH --signal=USR1@20
### SBATCH --qos=qos_gpu-dev         # we are submitting a test job
set -x
module purge
export PYTHONPATH=/linkhome/rech/genlop01/utt19ve/sources/sar_hs_nn
#pybin=/gpfswork/rech/deu/utt19ve/conda/envs/py3keras_v2/bin/python
#bcp plus rapide et fiable!!
module load tensorflow-gpu/py3/2.2.0
which python
exe=/gpfs7kw/linkhome/rech/genlop01/utt19ve/sources/sar_hs_nn/sarhspredictor/bin/replicate_training_sadowsky_feb2021.py
srun python $exe --zay

wait


# tested jean zay
# srun --account=deu@gpu --ntasks=1 --cpus-per-task=1 --mem=120G --gres=gpu:1 --time=01:00:00 --qos=qos_gpu-dev --pty bash -i
# note je peux augmenter les cpus-per-task (up to 63 tested)

# une autre queue on il y a plus de temps:
# srun --account=deu@gpu --ntasks=1 --cpus-per-task=63 --mem=120G --gres=gpu:1 --time=10:00:00 --qos=qos_gpu-t3 --pty bash -i
#dernier test en interactif (jutilise le slurm uniquement pour aide memo)
# srun --account=deu@gpu --ntasks=1 --cpus-per-task=10 --mem=30G --gres=gpu:1 --time=19:00:00 --qos=qos_gpu-t3 --pty bash -i
