# New RL

## Install
```
pip install -r requirement.txt
pip install -e .
```

## train
```
WANDB_API_KEY=xxx  python scripts/run_crowdsim_ppo_base.py run_name=xxx wandb_entity=xxx wandb_project=xxx trainer.total_steps=10000000 trainer.batch_size=8192 trainer.minibatch_size=256


WANDB_API_KEY=xxx  python scripts/run_crowdsim_ppo_base.py run_name=xxx wandb_entity=xxx wandb_project=xxx trainer.total_steps=2000000 trainer.batch_size=256  trainer.num_envs=4 trainer.update_per_step=2 trainer.alpha=0.001
```

### Best practice for PPO
Use large batch_size (i.e rollout 8196) and large minibatch_size (e.g., 256 or larger)
A good config example is `runs/multi-crowdsim-20m-ppo_base-bs8192-256-ep4-lr2.5e-04-cons-vf0.5-env_clip-ent0.02decay/config.yaml`

| Total Steps | Batch size| minibatch size| best success| 
|:-:|:-:|:-:|:-:|
| 10M | 2048 | 64 | 58%  |  
| 10M | 4096 | 64 | 61%  |
| 10M | 8192 | 256 | 64% |
| 20M | 4096 | 64 | 66% |
| 20M | 8192 | 256 | 72.6% |
| 50M | 4096 | 64 | 69.4%  |
| 50M | 8192 | 256 | 74.2% |

10M-20M should be enough for your experiment. PPO 20M takes 4.3 hours (when num_env=8). You can increase num_envs=16/32 if needed.



### Best practice for SAC

Set small alpha (e.g., 0.001 or auto), batch size 128-256, do not use larger batch size.
A good config example is `runs/SACabl-crowdsim-silu-env8-sac_base-bs128-a0.001-alr1.0e-03-clr1.0e-03-model_tanh/config.yaml`

| Total Steps | Batch size | alpha | activation |best success| 
|:-:|:-:|:-:|:-:|:-:|
| 1M | 256 | auto |relu| 61 % | 
| 1M | 256 | 0.001 |relu| 57 % |
| 2M | 256 | auto |relu| 77% | 
| 2M | 256 | 0.001 |relu| 78.6% | 
| 2M | 256 | 0.001 |silu|  84% | 
| 2M | 128 | 0.001 |silu|  85% |
| 3M | 256 | auto |relu| 78% |
| 3M | 256 | 0.001 |relu| 81% | 



1M-2M should be enough for your experiment. SAC 2M takes 3 hours (when num_env=8), larger num_env>8 is not verified.

## test
```
python eval/eval_batch.py actor_model=trained_models/<run_name> episodes_per_seed=50
```




# Crowd Navigation Test Environment

## Install (Modular)

If running on vail:

```bash
micromamba activate irsim39
```

Verified on Python `3.10` to `3.12`. Or create a fresh one:

```bash
micromamba create -n env-name python=3.11 -y
micromamba activate env-name
```

### 1) Install PyTorch (required)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

### 2) Core dependencies (required)

```bash
pip install numpy gymnasium wandb imageio matplotlib cvxopt
```

### 3) ORCA / RVO2 (conditional)

If human policy is `orca`, install `rvo2` from:
- [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)

### 4) Optional modules

QP / CVaR network experiments (only needed for qpth-based models):

```bash
pip install scipy qpth
```

## Quick Environment Test

Run a simple rollout and save a GIF:

```bash
# Variable-number crowd (default)
python test_env.py --env_name social_nav_var_num --steps 200 --obs_num 20
```


### `test_env.py` Arguments

| Argument | Default | Description |
|---|---|---|
| `--env_name` | `social_nav_var_num` | Environment to test |
| `--seed` | `0` | Random seed |
| `--steps` | `200` | Max steps per episode |
| `--save_path` | `trained_models/<env>.gif` | Output GIF path |
| `--human_num` | `20` | Number of pedestrians |

## Observation Format

### A) Environment Output (`env.reset()` / `env.step()`)

Env returns **absolute-format** observation:

```text
obs_env_dim = 8 + 6 * K
K = config.env["max_obstacles_obs"]
```

Robot+goal block (`obs[0:8]`):

```text
[rx, ry, gx, gy, rvx, rvy, rtheta, r_radius]
```

Obstacle blocks (`K` slots, each 6 dims, `obs[8:]`):

```text
[hx, hy, hvx, hvy, h_radius, mask]
```

Notes:
- Only visible humans (`distance <= sensing_radius`) are packed.
- Nearest visible humans are selected up to `K`.
- Empty slots are zero-padded with `mask=0`; valid slots use `mask=1`.
<!-- - If `config.env["normalize_obs"]=True`, values are normalized but layout stays the same. -->

### B) Network Input (what policy actually sees)

#### PPO / VecPPO / SAC / VecSAC input
Actor/critic input is **polar-format**:

```text
obs_actor_dim = 3 + 4 * K
```

for each obstacle slot:

```text
[distance, angle, speed_norm, heading_diff]
```

For `rlcbfgamma*` and `rlcvarbetaradius*`, the QP layer additionally consumes an internal **relative-format** tensor (`6 + 6*K`) computed from the same raw env observation at the same timestep.
 

## Training and Evaluation

Training entrypoint is Hydra-native:
- `main_vec.py` (vectorized, PPO or SAC)
- `main_opt.py` (non-RL controller evaluation)

Print resolved config:

```bash
python main_vec.py --cfg job --resolve
python main_opt.py --cfg job --resolve
python eval/eval_test.py --cfg job --resolve
```

### `main_vec.py` (Vectorized)

Train PPO:

```bash
# python main_vec.py trainer=ppo method=rlcbfgamma \
#   total_timesteps=2000000 num_envs=16
python main_vec.py run_name=rl_cvar_bf_adaptive trainer=ppo method=rl total_timesteps=20_000_000 num_envs=8 obs_topk=1 
```

Train SAC:

```bash
# python main_vec.py trainer=sac method=rl \
#   total_timesteps=500000 num_envs=8
python main_vec.py run_name=rl_cvar_bf_adaptive trainer=sac method=rl total_timesteps=5_000_000 num_envs=8 obs_topk=1 
```

### `main_opt.py` (Controller Baselines)

Run non-RL controllers:

```bash
python main_opt.py method=cbfqp test_mode=both test_ep=100 test_viz_ep=20
```

- Output directory:
  - if `actor_model` is set: `dirname(actor_model)/{robot_type}_obs_{obstacle_count}_vmax_{vmax}_omegamax_{omega_max}`
  - otherwise: `trained_models/<model_folder>/{robot_type}_obs_{obstacle_count}_vmax_{vmax}_omegamax_{omega_max}`



## Evaluation

All evaluation scripts default to CPU. 

### A) Single-checkpoint visual/metric test (`eval/eval_test.py`)

Use this for one actor checkpoint.

```bash
python eval/eval_test.py \
  method=rl \
  actor_model=trained_models/default/rl_cvar_bf_adaptive-unicycle-rl-ppo-bs8192-ep8-mbsz512-k1-env8_1/ppo_actor_best.pth \
  test_mode=both \
  test_ep=100 \
  test_viz_ep=10 \
  eval_seed=100 \
  obs_topk=1

python eval/eval_test.py \
  method=rlcvarbetaradius_2nets \
  actor_model=trained_models/default2/v2_rl_cvar_bf_adaptive_beta09-unicycle-rlcvarbetaradius_2nets-ppo-bs8192-ep8-mbsz512-k1-env32-annA0B0R0/ppo_actor_step_10000000.pth \
  test_mode=both \
  test_ep=100 \
  test_viz_ep=10 \
  eval_seed=100 \
  obs_topk=1
```

- `test_mode=eval`: batch episode evaluation.
- `test_mode=crossing`: fixed crossing scenario visualization.
- `test_mode=both`: run both.
- If `save_path` is empty, outputs go to `<dirname(actor_model)>/<robot_type>_obs_<obstacle_count>_vmax_<vmax>_omegamax_<omega_max>/`.
- Output files include `eval_ep_<ep>_succ_<0|1>_coll_<0|1>.gif`, `crossing.gif`, and `eval_log.json`.


### B) Batch eval for all actor checkpoints (`eval/eval_batch.py`)

Use this to evaluate all actor checkpoints under one run folder with fixed seeds.

```bash
  python eval/eval_batch.py \
  method=rl \
  actor_model=trained_models/default/rl_cvar_bf_adaptive-unicycle-rl-ppo-bs8192-ep8-mbsz512-k1-env8_1/ \
  episodes_per_seed=50

  python eval/eval_batch.py \
  method=rlcvarbetaradius_2nets \
  actor_model=trained_models/default2/v2_rl_cvar_bf_adaptive_beta09-unicycle-rlcvarbetaradius_2nets-ppo-bs8192-ep8-mbsz512-k1-env32-annA0B0R0 \
  episodes_per_seed=50
```

- Hydra entrypoint (`key=value` overrides, not `--flag` style).
- Evaluates every `*actor*.pth` checkpoint in the folder.
- Uses fixed multi-seed rollout (`FIXED_EVAL_SEEDS` default).
- `actor_model` is the run directory path.
- Optional overrides: `method`, `robot_type`, `num_humans`, `obs_topk`, `obs_farest_dist`,
  `nHidden1`, `nHidden21`, `nHidden22`, `alpha_hidden1`, `alpha_hidden2`.
- Output files include `test_config.json` and `checkpoint_eval_all_multiseed.json`.

### C) Scenario matrix compare (`eval/run_eval_compare.py`)

Use this for batch comparisons across robot types, obstacle counts, and methods.

```bash
python eval/run_eval_compare.py
```

- Iterates scenario grid configured in the script.
- Calls `eval/eval_batch.py` per method/scenario.
- Saves outputs under `trained_models/compare/<robot_type>_obs_<obstacle_count>/<method>/`.
- Per-method files are produced by `eval/eval_batch.py` (`test_config.json`, `checkpoint_eval_all_multiseed.json`).

### D) Seed-advantage GIF analysis (`eval/run_compare_seed_advantage_gifs.py`)

Use this to find seeds where one target method clearly outperforms others and export gifs.

```bash
python eval/run_compare_seed_advantage_gifs.py \
  seed=100 \
  target_method=rlcvarbetaradius \
  robot_type=unicycle \
  obstacle_number=25
```

- Hydra entrypoint (`key=value` overrides, not `--flag` style).
- Evaluates seeds in `[seed, seed + 99]`.
- Finds seeds where target succeeds and all other compared methods fail.
- Exports per-method gifs for matched seeds.
- If `out_dir` is empty, output root is `<compare_root>/<robot_type>_obs_<obstacle_number>/seed_<start>_<end>/`.
- Matched-seed gifs are saved as `seed_<seed>/<method>_seed_<seed>_succ_<0|1>_coll_<0|1>.gif`.
- Summary file is `<scenario>_seed_<start>_<end>_advantage_summary.json`.

### E) Compare-result summary report (`eval/analyze_compare_results.py`)

Use this to aggregate `trained_models/compare` into CSV/Markdown plus figures.

```bash
python eval/analyze_compare_results.py \
  --compare_root trained_models/compare \
  --out_dir trained_models/compare \
  --dpi 160
```

- Writes `summary_compare_metrics.csv` and `summary_compare_report.md`.
- Writes success-rate plots like `success_rate_vs_obstacles_<robot_type>.png`.
- Writes radar outputs `radar_metrics_summary.csv` and `radar_metrics_<robot_type>.png`.
- If `--out_dir` is empty, outputs are written into `--compare_root`.

## Hydra Overrides

### Common Keys

- `method` (`rl`, `rlcbfgamma`, `rlcbfgamma_2nets`, `rlcbfgamma_2nets_risk`, `rlcvarbetaradius`, `rlcvarbetaradius_2nets`, `rlcvarbetaradius_2nets_risk`, `rlcvarbetaradiusalpha_2nets`, `rlcvarbetaradiusalpha_2nets_risk`)
- `actor_model`, `critic_model`
- `device` (`cuda` or `cpu`)
- `seed`, `eval_seed`
- `model_folder`
- `obs_topk`, `obs_farest_dist`
- `qp_start_timesteps`

### Trainer Keys

- Shared: `total_timesteps`, `gamma`, `render_every_i`, `save_after_timesteps`, `save_freq`
- PPO (`trainer=ppo`): `n_updates_per_iteration`, `timesteps_per_batch`, `lr`, `clip`, `lam`, `num_minibatches`, `ent_coef`, `target_kl`, `max_grad_norm`, `action_std_init`, `eval_freq_timesteps`, `eval_episodes`
- SAC (`trainer=sac`): `sac_timesteps_per_batch`, `sac_buffer_size`, `sac_batch_size`, `sac_start_timesteps`, `sac_updates_per_step`, `sac_hidden_sizes`, `sac_tau`, `sac_actor_lr`, `sac_critic_lr`, `sac_max_grad_norm`, `sac_auto_alpha`, `sac_alpha`, `sac_alpha_lr`, `sac_target_entropy`, `sac_action_std_init`, `sac_eval_freq_episodes`, `sac_eval_episodes`
- Vectorized env count (`main_vec.py`): `num_envs` (`0` means fallback to `cpu_count`)

Evaluation in training can be disabled by setting frequency to `0`:
- SAC: `sac_eval_freq_episodes=0`
- PPO: `eval_freq_timesteps=0`

### Config Files

- `config/main_vec.yaml` for `main_vec.py`
- `config/eval_test.yaml` for `eval/eval_test.py`
- `config/env/crowdsim.yaml` for observation-related env overrides
- `config/model/default.yaml` for method/checkpoint defaults
- `config/trainer/common.yaml`, `config/trainer/ppo.yaml`, `config/trainer/sac.yaml`
- `eval/eval_test.py` standalone single-checkpoint evaluation script

## Checkpoints and Output Folder

Training outputs are saved to:

```text
trained_models/<model_folder>/<timestamp>_<robot_type>_<method>_<algo>/
```

Warm-start options:
- `actor_model`: load actor before training
- `critic_model`: load critic before training

Device behavior:
- `device=cuda` uses GPU when available
- if CUDA is unavailable, code falls back to CPU automatically

## Configuration

All environment/controller defaults are in `config/config.py`:

- `config.env` for timestep, max steps, sensing radius, normalization
- `config.robot` for robot type and limits (`vmax`, `amax`, `omega_max`)
- `config.human` for crowd setup and human policy
- `config.reward` for success/collision/discomfort shaping
- `config.controller` for CBF/CVaR parameters

## Robot Types

Robot-related config is under `config.robot` in `config/config.py`:
- `type`
- `radius`
- `vmax`
- `omega_max`

| Type (`config.robot.type`) | Action Space | Uses From `config.robot` | Notes |
|---|---|---|---|
| `single_integrator` | `[vx, vy]` | `radius`, `vmax` | Action norm is clipped to `vmax`. `omega_max` / `amax` are not used. |
| `unicycle` | `[v, omega]` | `radius`, `vmax`, `omega_max` | `v` clipped to `[-vmax, vmax]`, `omega` clipped to `[-omega_max, omega_max]`. |

Extra note for RL with unicycle:
- If `config.env.rl_xy_to_unicycle=True`, policy outputs `[vx, vy]` and env converts it to `[v, omega]` internally. and in this case, the traing curve is a slow than  `[v, omega]`.


great lake: https://vscode.dev/tunnel/gl3025arc-tsumichedu/home/daiyp/CODE/crowd_env/
