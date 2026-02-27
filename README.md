# Crowd Navigation Test Environment

## Install

```bash
micromamba activate irsim39
```

## Quick Environment Test

Run a simple rollout and save a GIF:

```bash
# Variable-number crowd (default)
python test_env.py --env_name social_nav_var_num --steps 200 --obs_num 20

# Fixed crowd
python test_env.py --env_name social_nav --steps 200 --obs_num 10

# Custom seed and output path
python test_env.py --seed 42 --save_path trained_models/test.gif
```

### `test_env.py` Arguments

| Argument | Default | Description |
|---|---|---|
| `--env_name` | `social_nav_var_num` | Environment to test |
| `--seed` | `0` | Random seed |
| `--steps` | `200` | Max steps per episode |
| `--save_path` | `trained_models/<env>.gif` | Output GIF path |
| `--human_num` | `20` | Number of pedestrians |

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

## Hyperparameters: SAC vs PPO

### Shared/Common Arguments

- `--total_timesteps`
- `--timesteps_per_batch`
- `--max_timesteps_per_episode`
- `--gamma`
- `--max_grad_norm`
- `--seed`, `--eval_seed`
- `--device` (`cuda` or `cpu`)
- `--save_after_timesteps`, `--save_freq`, `--model_folder`

### SAC-Oriented Arguments

- `--buffer_size`
- `--batch_size`
- `--start_timesteps`
- `--updates_per_step`
- `--hidden_sizes`
- `--tau`
- `--actor_lr`, `--critic_lr`
- `--auto_alpha`, `--no_auto_alpha`
- `--alpha`, `--alpha_lr`, `--target_entropy`
- `--action_std_init`

### PPO-Oriented Arguments

- `--ppo_n_updates_per_iteration`
- `--ppo_lr`
- `--ppo_clip`
- `--ppo_lam`
- `--ppo_num_minibatches`
- `--ppo_ent_coef`
- `--ppo_target_kl`
- `--ppo_action_std_init`
- `--ppo_use_ema`
- `--ppo_ema_decay`
- `--ppo_eval_freq_episodes`

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

| Type | Action Space | Description |
|---|---|---|
| `single_integrator` | `[vx, vy]` | Velocity-controlled |
| `unicycle` | `[v, ω]` | Heading + speed controlled |
