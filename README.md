# Reinforcement Learning for Highway Tactical Decision-Making in CARLA

This repository asks whether a state-based PPO agent can make safer and more
efficient highway overtaking and lane-change decisions than random, fixed
keep-lane, and simple rule-based policies under varying CARLA traffic
conditions. The implementation deliberately stays narrow: CARLA 0.9.16,
Town04, one Gymnasium environment, one Stable-Baselines3 PPO policy, and three
non-learning baselines.

The learned component is high-level tactical decision-making. PPO never emits
steering, throttle, or brake values. A shared waypoint/PID controller executes
target-lane and target-speed decisions for every policy, with the same minimal
car-following override to prevent the task from collapsing into emergency
braking. This is not end-to-end autonomous driving and provides no safety
guarantee.

> **Project question:** Can a PPO agent learn safer and more efficient highway
> overtaking and lane-change decisions than random, fixed keep-lane, and simple
> rule-based policies under varying CARLA traffic conditions?

## Design

The PPO policy observes a normalized 20-dimensional state vector: ego speed,
target speed, lane offset, heading error, adjacent-lane availability,
lane-change state, and front/rear gap and relative-speed features for the left,
current, and right lanes.

The five tactical actions are `MAINTAIN`, `ACCELERATE`, `DECELERATE`,
`CHANGE_LEFT`, and `CHANGE_RIGHT`. Lane-change requests are checked for legal
same-direction lanes, front/rear gaps, and rear TTC. Accepted decisions are
tracked by a waypoint controller; rejected decisions execute as maintain and
receive the configured penalty.

```mermaid
flowchart LR
    A[CARLA state] --> B[20-D observation]
    B --> C[PPO or baseline tactical policy]
    C --> D[target speed / target lane]
    D --> E[shared PID and car-following controller]
    E --> F[CARLA vehicle]
    F --> G[metrics and RGB rendering]
```

The three baselines are:

- uniformly seeded random tactical actions;
- fixed keep-lane, with shared car following behind slower traffic;
- a transparent rule-based overtaking policy that prefers a safe left pass.

Evaluation uses a frozen paired manifest spanning three traffic densities,
three weather presets, and ten held-out seeds: 90 identical conditions per
policy and 360 episodes overall. Analysis reports Wilson intervals for binary
rates, seeded bootstrap intervals for continuous metrics, and condition-paired
PPO-minus-baseline differences. RGB chase-camera video is used only during
rendering.

## Repository

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── carla_env.py
├── policies.py
├── run.py
├── notebooks/
│   ├── 01_train_and_evaluate.ipynb
│   └── 02_render_and_report.ipynb
├── reports/
│   ├── main_report.tex
│   ├── technical_log.tex
│   └── generated/
│       └── .gitkeep
└── artifacts/
    └── .gitkeep
```

Generated models, logs, manifests, evaluation CSVs, plots, recordings, frames,
and videos live below `artifacts/` and are gitignored. Generated LaTeX inputs
live below `reports/generated/`.

## Setup

Use Python 3.12 and the packaged CARLA 0.9.16 UE4.26 release. Install
dependencies before importing NumPy or CARLA:

```bash
python -m pip install -r requirements.txt
python run.py validate-config --config config.yaml
python run.py offline-self-test --config config.yaml
```

The default **external** mode connects to an already running CARLA server at
`127.0.0.1:2000`. Override connection settings with `CARLA_HOST`,
`CARLA_PORT`, and `CARLA_TM_PORT`. The optional **managed** mode launches only
the packaged local process named by `CARLA_ROOT`; set
`carla.server.mode: managed`, then use:

```bash
python run.py server start --config config.yaml
python run.py server status --config config.yaml
python run.py server stop --config config.yaml
```

Use `--rendering` on `server start` for RGB evaluation. The repository never
downloads CARLA, builds Unreal Engine, or kills a server it did not start.

## Core workflow

```bash
python run.py doctor --config config.yaml
python run.py smoke --config config.yaml

python run.py train --config config.yaml --run-name ppo_seed_0 \
  --seed 0 --total-timesteps 25000

python run.py train --config config.yaml --run-name ppo_seed_0 \
  --seed 0 \
  --resume artifacts/models/ppo_seed_0/checkpoints/CHECKPOINT.zip \
  --additional-timesteps 25000

python run.py make-eval-manifest --config config.yaml
python run.py evaluate --config config.yaml \
  --manifest artifacts/manifests/evaluation_manifest.json \
  --model artifacts/models/ppo_seed_0/final_model.zip \
  --policies ppo random keep_lane rule_based
python run.py analyze --config config.yaml \
  --episodes artifacts/evaluations/episode_results.csv
python run.py report-data --config config.yaml
```

Use `--quick` when creating or running a three-seed debugging manifest.
Evaluation can be sliced with `--start-index` and `--end-index`, then safely
continued with `--resume-existing`. Video selection and rendering are:

```bash
python run.py select-videos --config config.yaml \
  --episodes artifacts/evaluations/episode_results.csv \
  --model artifacts/models/ppo_seed_0/final_model.zip
python run.py render-videos --config config.yaml \
  --manifest artifacts/manifests/evaluation_manifest.json \
  --model artifacts/models/ppo_seed_0/final_model.zip --resume-existing
```

Google Drive synchronization copies without deleting destination files:

```bash
python run.py sync --config config.yaml --to-drive \
  --local-root artifacts \
  --drive-root /content/drive/MyDrive/CARLA_Highway_RL
```

## Notebooks

Run [01_train_and_evaluate.ipynb](notebooks/01_train_and_evaluate.ipynb) first.
It installs dependencies, restores Drive artifacts, diagnoses CARLA, performs
smoke tests, trains in 25K stages, freezes the paired manifest, evaluates, and
analyzes. Then run
[02_render_and_report.ipynb](notebooks/02_render_and_report.ipynb) with
rendering enabled to select scenarios, create six honest qualitative videos,
generate report inputs, and optionally compile the reports. Each major section
is restartable after rerunning its notebook's common initialization.

## Results

**Experiments pending.** The repository contains no fabricated training or
evaluation values. After experiments, `report-data` conditionally populates the
reports from the generated CSVs.

## Reproducibility and limitations

Python, NumPy, Traffic Manager, environment, and SB3 seeds are fixed; CARLA uses
synchronous 0.05-second steps; every run saves a resolved configuration and
hash; and all policies replay the same manifest rows. Exact GPU/rendering
behavior can still vary across CARLA installations. The environment uses
privileged simulator state, approximate lane-relative vehicle classification,
a compact conventional controller, one primary training seed, and a limited
held-out scenario grid. The car-following override is intentionally minimal,
not a complete safety system.

An optional future comparison may add CARLA's BehaviorAgent as a clearly
separate reference, but it is not part of the implemented four-policy
experiment.
