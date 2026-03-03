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
Use large batch_size (i.e rollout 8196) and large minibatch_size (e.g., 256 or larger).  
A good config example is `runs/multi-crowdsim-20m-ppo_base-bs8192-256-ep4-lr2.5e-04-cons-vf0.5-env_clip-ent0.02decay/config.yaml`

| Total Steps | Batch size| minibatch size| best success| 
|:-:|:-:|:-:|:-:|
| 10M | 2048 | 64 | 58%  |  
| 10M | 4096 | 64 | 61%  |
| 10M | 8192 | 256 | 64% |
| 20M | 4096 | 64 | 66% |
| 20M | 8192 | 256 | 73% |
| 50M | 4096 | 64 | 69%  |
| 50M | 8192 | 256 | 74% |
| 100M | 8192 | 256 | 75% |

10M-20M should be enough for your experiment. PPO 20M takes 4.3 hours (when num_env=8). You can increase num_envs=16/32 if needed.



### Best practice for SAC

Set small alpha (e.g., 0.001 or auto), batch size 128-256, do not use larger batch size.  
A good config example is `runs/SACabl-crowdsim-silu-env8-sac_base-bs128-a0.001-alr1.0e-03-clr1.0e-03-model_tanh/config.yaml`

| Total Steps | Batch size | alpha | activation |best success| 
|:-:|:-:|:-:|:-:|:-:|
| 1M | 256 | auto |relu| 61 % | 
| 1M | 256 | 0.001 |relu| 57 % |
| 2M | 256 | auto |relu| 77% | 
| 2M | 256 | 0.001 |relu| 79% | 
| 2M | 256 | 0.001 |silu|  84% | 
| 2M | 128 | 0.001 |silu|  85% |
| 3M | 256 | auto |relu| 78% |
| 3M | 256 | 0.001 |relu| 81% | 
| 10M | 128 | 0.001 |silu|  86% |



1M-2M should be enough for your experiment. SAC 2M takes 3 hours (when num_env=8), larger num_env>8 is not verified.

## test
```
python scripts/eval.py --save-dir xxx --use-all-obs --visualize
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

#### PPO / VecPPO / SAC / VecSAC  input
PPO converts env absolute obs to **relative-format** before actor/critic:

```text
obs_net_dim = 6 + 6 * K
```

Robot block:

```text
[goal_rel_x, goal_rel_y, rvx, rvy, rtheta, r_radius]
goal_rel = [rx - gx, ry - gy]
```

Obstacle block:

```text
[rel_x, rel_y, hvx, hvy, h_radius, mask]
rel = [rx - hx, ry - hy]
```
 

## Training and Evaluation

This repo now supports both `SAC` and `PPO` in:
- `main.py` (single environment)
- `main_vec.py` (vectorized environments)

Algorithm is selected with:

```bash
--algo sac   # default
--algo ppo
```

### `main.py` (Single-Environment)

Train SAC:

```bash
python main.py --mode train --algo sac --method rl
```

Train PPO:

```bash
python main.py --mode train --algo ppo --method rl
```

Test a trained model:

```bash
python main.py --mode test --algo sac --actor_model trained_models/.../sac_actor.pth --test_ep 100
python main.py --mode test --algo ppo --actor_model trained_models/.../ppo_actor.pth --test_ep 100
```

### `main_vec.py` (Vectorized)

`main_vec.py` uses `AsyncVectorEnv` and auto-sets the number of envs to `multiprocessing.cpu_count()`.

Train SAC (vectorized):

```bash
python main_vec.py --mode train --algo sac --method rl
```

Train PPO (vectorized):

```bash
python main_vec.py --mode train --algo ppo --method rl
```

Test (vectorized entrypoint):

```bash
python main_vec.py --mode test --algo sac --method rl --actor_model trained_models/.../sac_actor.pth
python main_vec.py --mode test --algo ppo --method rl --actor_model trained_models/.../ppo_actor.pth
```

Note: `main_vec.py` currently enforces `--method rl`.



## Evaluation

### A) Visual eval for one checkpoint (`main_vec.py`，`main.py`)

Use this when you want to directly inspect trajectory behavior and GIFs for a specific actor checkpoint.
default algo is sac
```bash
python main_vec.py \
  --mode test \
  --method rl \
  --algo ppo \
  --actor_model trained_models/default/20260227_111328_unicycle_rl_ppo/ppo_actor_step_500000.pth \
  --test_ep 100 \
  --test_viz_ep 50 \
  --eval_seed 100
```

- Output folder: `<checkpoint_dir>/<timestamp>/` (contains GIFs and `eval_log.json`).
- Render behavior is controlled by `render_mode` when creating test env.
- Default is `rgb_array` (save GIFs).
- Add `--render` to use `human` mode (show window directly):


### B) Batch eval for all actor checkpoints (`eval.py`)

Use this when you want robust model selection across many checkpoints and seeds.

```bash
python eval.py \
  --actor_model 20260227_111328_unicycle_rl_ppo \
  --eval_seeds 100,200,300,400,500,600,700,800,900,1000 \
  --episodes_per_seed 50
```

- Automatically evaluates all `*actor*.pth` in the run directory.
- Outputs one summary JSON in run dir (default: `checkpoint_eval_all_multiseed.json`).


## Arguments (Current)

### Common Runtime Arguments

- `--mode` (`train` or `test`)
- `--algo` (`sac` or `ppo`)
- `--method` (`rl`)
- `--actor_model`, `--critic_model`

### Common Training / System Arguments

- `--total_timesteps`
- `--timesteps_per_batch`
- `--max_timesteps_per_episode`
- `--gamma`
- `--test_ep`, `--test_viz_ep`
- `--render`, `--render_every_i`
- `--save_after_timesteps`, `--save_freq`
- `--model_folder`
- `--device`
- `--seed`, `--eval_seed`

### SAC-Specific Arguments

- `--buffer_size`
- `--batch_size`
- `--start_timesteps`
- `--updates_per_step`
- `--hidden_sizes`
- `--tau`
- `--actor_lr`, `--critic_lr`
- `--sac_max_grad_norm`
- `--sac_eval_freq_episodes`, `--sac_eval_episodes`
- `--no_auto_alpha` (default is auto-alpha enabled; pass this flag to disable)
- `--alpha`, `--alpha_lr`, `--target_entropy`
- `--action_std_init`

### PPO-Specific Arguments

- `--ppo_n_updates_per_iteration`
- `--ppo_lr`
- `--ppo_clip`
- `--ppo_lam`
- `--ppo_num_minibatches`
- `--ppo_ent_coef`
- `--ppo_target_kl`
- `--ppo_max_grad_norm`
- `--ppo_action_std_init`
- `--ppo_eval_freq_timesteps`, `--ppo_eval_episodes`

Evaluation in training can be disabled by setting frequency to `0`:
- SAC: `--sac_eval_freq_episodes 0`
- PPO: `--ppo_eval_freq_timesteps 0`

## Checkpoints and Output Folder

Training outputs are saved to:

```text
trained_models/<model_folder>/<timestamp>_<robot_type>_<method>_<algo>/
```

Warm-start options:
- `--actor_model`: load actor before training
- `--critic_model`: load critic before training

Device behavior:
- `--device cuda` uses GPU when available
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
- If `config.env.rl_xy_to_unicycle=True`, policy outputs `[vx, vy]` and env converts it to `[v, omega]` internally.
