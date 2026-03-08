#!/bin/bash

#SBATCH --job-name=v2_rl_cvar_bf_adaptive
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

srun --jobid $SLURM_JOBID bash -c 'python main_vec.py run_name=v2_rl_cvar_bf_adaptive trainer=sac method=rlcbfgamma_2nets_risk total_timesteps=5_000_000 num_envs=8 obs_topk=1'
# sbatch /home/daiyp/CODE/crowd_env/slurm/run_crowdsim_sac.sh
# squeue -a | grep daiyp 
# scancel 44347755

# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcbfgamma total_timesteps=5_000_000 num_envs=8 obs_topk=1
# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcbfgamma_2nets total_timesteps=5_000_000 num_envs=8 obs_topk=1
# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcvarbetaradius total_timesteps=5_000_000 num_envs=8 obs_topk=1
# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcvarbetaradius_2nets total_timesteps=5_000_000 num_envs=8 obs_topk=1
# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcvarbetaradius_2nets_risk total_timesteps=5_000_000 num_envs=8 obs_topk=1

# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcvarbetaradius_2nets total_timesteps=5_000_000 num_envs=8 obs_topk=5
# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcvarbetaradius total_timesteps=5_000_000 num_envs=8 obs_topk=5
# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcbfgamma_2nets total_timesteps=5_000_000 num_envs=8 obs_topk=5
# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcbfgamma total_timesteps=5_000_000 num_envs=8 obs_topk=5
# python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rlcbfgamma_2nets_risk total_timesteps=5_000_000 num_envs=8 obs_topk=5


#ls -l /home/daiyp/CODE/crowd_env/trained_models