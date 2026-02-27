# Crowd Navigation Test Environment

## Install

```bash
 micromamba activate irsim39  
```
## Usage

Run the environment test and save a visualization GIF:

```bash
# Variable-number crowd (default)
python test_env.py --env_name social_nav_var_num --steps 200 --obs_num 20
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--env_name` | `social_nav_var_num` | Environment to test |
| `--seed` | `0` | Random seed |
| `--steps` | `200` | Max steps per episode |
| `--save_path` | `trained_models/<env>.gif` | Output GIF path |
| `--human_num` | `20` | Number of pedestrians |

## Configuration

All parameters are defined in [config/config.py](config/config.py):

- **`config.env`** — timestep `dt`, `max_steps`, `sensing_radius`, arena settings
- **`config.robot`** — radius, `vmax`, `amax`, `omega_max`, robot `type`
- **`config.human`** — crowd size, velocity range, navigation `policy` (`orca` / `social_force`)
- **`config.reward`** — success/collision rewards, discomfort penalty
- **`config.controller`** — CBF parameters (`cbf_alpha`, `cvar_beta`, `safety_margin`)

## Robot Types

| Type | Action Space | Description |
|---|---|---|
| `single_integrator` | `[vx, vy]` | Velocity-controlled |
| `unicycle` | `[v, ω]` | Heading + speed controlled |
