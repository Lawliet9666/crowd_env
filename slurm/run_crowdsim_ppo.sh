#!/bin/bash

#SBATCH --job-name=rl_cvar_bf_adaptive
#SBATCH --account=chaijy2
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/daiyp/CODE/crowd_env/xy_runs/logs/%x-%j.log
#SBATCH --mail-user=xinywa@umich.edu
#SBATCH --mail-type=BEGIN,END

source /home/daiyp/.bashrc
cd /home/daiyp/CODE/crowd_env
micromamba activate irsim39
unset LEROBOT_HOME
unset TRANSFORMERS_CACHE

export WANDB_API_KEY=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

srun --jobid $SLURM_JOBID bash -c 'python main_vec.py trainer=ppo method=rlcvarbetaradius_2nets total_timesteps=2000000 num_envs=16 obs_topk=5 run_name=rl_cvar_bf_adaptive'
# sbatch /home/daiyp/CODE/crowd_env/slurm/run_crowdsim_ppo.sh
# squeue -a | grep daiyp 
# scancel 44347755

# python main_vec.py trainer=ppo method=rlcbfgamma total_timesteps=2000000 num_envs=16 obs_topk=1 run_name=rl_cvar_bf_adaptive
# python main_vec.py trainer=ppo method=rlcbfgamma total_timesteps=2000000 num_envs=16 obs_topk=5 run_name=rl_cvar_bf_adaptive

# python main_vec.py trainer=ppo method=rlcbfgamma_2nets total_timesteps=2000000 num_envs=16 obs_topk=1 run_name=rl_cvar_bf_adaptive
# python main_vec.py trainer=ppo method=rlcbfgamma_2nets total_timesteps=2000000 num_envs=16 obs_topk=5 run_name=rl_cvar_bf_adaptive

# python main_vec.py trainer=sac method=rlcvarbetaradius total_timesteps=2000000 num_envs=16 obs_topk=1 run_name=rl_cvar_bf_adaptive
# python main_vec.py trainer=ppo method=rlcvarbetaradius total_timesteps=2000000 num_envs=16 obs_topk=5 run_name=rl_cvar_bf_adaptive

# python main_vec.py trainer=ppo method=rlcvarbetaradius_2nets total_timesteps=2000000 num_envs=16 obs_topk=1 run_name=rl_cvar_bf_adaptive
# python main_vec.py trainer=ppo method=rlcvarbetaradius_2nets total_timesteps=2000000 num_envs=16 obs_topk=5 run_name=rl_cvar_bf_adaptive





