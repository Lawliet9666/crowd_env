#!/bin/bash

#SBATCH --job-name=newrl_cvar_bf_adaptive
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/daiyp/CODE/crowd_env/trained_models/logs/%x-%j.log
#SBATCH --mail-user=xinywa@umich.edu
#SBATCH --mail-type=BEGIN,END

source /home/daiyp/.bashrc
cd /home/daiyp/CODE/crowd_env
micromamba activate ppo
unset LEROBOT_HOME
unset TRANSFORMERS_CACHE

export WANDB_API_KEY=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

srun --jobid $SLURM_JOBID bash -c 'python scripts/run_crowdsim_ppo_base.py run_name=newrl_ppo_cvar_2nets wandb_entity=xinywa_umich wandb_project=newrl_ppo method=rl trainer.total_steps=20000000 trainer.batch_size=8192 trainer.minibatch_size=256 trainer.num_envs=8 env.topk=1 save_dir=xy_runs overwrite=true'
# sbatch /home/daiyp/CODE/crowd_env/slurm/run_crowdsim_ppo_newrl.sh
# squeue -a | grep daiyp 
# scancel 44347755


# python scripts/run_crowdsim_ppo_base.py \
#   run_name=newrl_ppo_cvar_2nets \
#   wandb_entity=xinywa_umich \
#   wandb_project=newrl_ppo \
#   method=rlcvarbetaradius_2nets \
#   trainer.total_steps=20000000 \
#   trainer.batch_size=8192 \
#   trainer.minibatch_size=256 \
#   trainer.num_envs=32 \
#   env.topk=1 \
#   save_dir=xy_runs \
#   overwrite=true

# python scripts/run_crowdsim_ppo_base.py run_name=newrl_ppo_cvar_2nets wandb_entity=xinywa_umich wandb_project=newrl_ppo method=rlcvarbetaradius_2nets trainer.total_steps=20000000 trainer.batch_size=8192 trainer.minibatch_size=256 trainer.num_envs=32 env.topk=1 save_dir=xy_runs overwrite=true
# python scripts/run_crowdsim_ppo_base.py run_name=newrl_ppo_cvar_2nets wandb_entity=xinywa_umich wandb_project=newrl_ppo method=rlcvarbetaradius trainer.total_steps=20000000 trainer.batch_size=8192 trainer.minibatch_size=256 trainer.num_envs=32 env.topk=1 save_dir=xy_runs overwrite=true
# python scripts/run_crowdsim_ppo_base.py run_name=newrl_ppo_cvar_2nets wandb_entity=xinywa_umich wandb_project=newrl_ppo method=rlcbfgamma trainer.total_steps=20000000 trainer.batch_size=8192 trainer.minibatch_size=256 trainer.num_envs=32 env.topk=1 save_dir=xy_runs overwrite=true
# python scripts/run_crowdsim_ppo_base.py run_name=newrl_ppo_cvar_2nets wandb_entity=xinywa_umich wandb_project=newrl_ppo method=rlcbfgamma_2nets trainer.total_steps=20000000 trainer.batch_size=8192 trainer.minibatch_size=256 trainer.num_envs=32 env.topk=1 save_dir=xy_runs overwrite=true


#ls -l /home/daiyp/CODE/crowd_env/trained_models