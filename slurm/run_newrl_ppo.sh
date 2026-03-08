#!/bin/bash
export WANDB_API_KEY=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

# pip install -e .

python scripts/run_crowdsim_ppo_base.py \
  run_name=ppo_cvar_2nets \
  wandb_entity=xinywa_umich \
  wandb_project=crowdsim_ppo \
  method=rlcvarbetaradius_2nets \
  trainer.total_steps=20000000 \
  trainer.batch_size=8192 \
  trainer.minibatch_size=256 \
  trainer.num_envs=32 \
  
# bash slurm/run.sh