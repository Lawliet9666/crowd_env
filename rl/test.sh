python main_vec.py trainer=ppo method=rlcbfgamma total_timesteps=2000000 num_envs=16 obs_topk=1 obs_farest_dist=5.0 qp_start_timesteps=0 run_name=rlcbfgamma_ppo_k1
python main_vec.py trainer=ppo method=rlcbfgamma total_timesteps=2000000 num_envs=16 obs_topk=5 obs_farest_dist=5.0 qp_start_timesteps=0 run_name=rlcbfgamma_ppo_k5

python main_vec.py trainer=ppo method=rlcbfgamma_2nets total_timesteps=2000000 num_envs=16 obs_topk=1 obs_farest_dist=5.0 qp_start_timesteps=0 run_name=rlcbfgamma_2nets_ppo_k1
python main_vec.py trainer=ppo method=rlcbfgamma_2nets total_timesteps=2000000 num_envs=16 obs_topk=5 obs_farest_dist=5.0 qp_start_timesteps=0 run_name=rlcbfgamma_2nets_ppo_k5

python main_vec.py trainer=sac method=rlcvarbetaradius total_timesteps=2000000 num_envs=16 obs_topk=1 obs_farest_dist=5.0 qp_start_timesteps=0 run_name=rlcvarbetaradius_ppo_k1
python main_vec.py trainer=ppo method=rlcvarbetaradius total_timesteps=2000000 num_envs=16 obs_topk=5 obs_farest_dist=5.0 qp_start_timesteps=0 run_name=rlcvarbetaradius_ppo_k5

python main_vec.py trainer=ppo method=rlcvarbetaradius_2nets total_timesteps=2000000 num_envs=16 obs_topk=1 obs_farest_dist=5.0 qp_start_timesteps=0 run_name=rlcvarbetaradius_2nets_ppo_k1
python main_vec.py trainer=ppo method=rlcvarbetaradius_2nets total_timesteps=2000000 num_envs=16 obs_topk=5 obs_farest_dist=5.0 qp_start_timesteps=0 run_name=rlcvarbetaradius_2nets_ppo_k5
