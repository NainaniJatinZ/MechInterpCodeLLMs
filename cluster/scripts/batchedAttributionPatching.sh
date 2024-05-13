#!/bin/bash
#SBATCH  -t 3-00:00:00
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH --mem 512GB
#SBATCH --constraint a100-80g
#SBATCH --output=cluster/logs/infoRet.out
#SBATCH --error=cluster/logs/infoRet.err

module load miniconda/22.11.1-1
conda activate finetuning
echo "Activated virtual environment"
cd /home/jnainani_umass_edu/codellm/MechInterpCodeLLMs
echo "Changed the directory to: ${PWD}"

python -u seqAttPatching.py --logit_difference 0.5171 --NumberOfPrompts 8