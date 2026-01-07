# Reinforcement Learning on MiniGrid MultiRoom

## Quick Start (tabular)
- Q-learning (FullyObs recommended):
  `python -m training.train --agent qlearning --env MiniGrid-MultiRoom-N2-S4-v0 --fully-obs --log-dir logs/qlearning_run`
- SARSA (FullyObs recommended):
  `python -m training.train --agent sarsa --env MiniGrid-MultiRoom-N2-S4-v0 --fully-obs --log-dir logs/sarsa_run`

Use `--episodes` and `--max-steps` to override defaults; all other hyperparameters live in [config/hyperparams.yaml](config/hyperparams.yaml).

## Visualization
- Q-learning: `python evaluation/visualize.py --agent qlearning --q-table logs/qlearning_run/q_table_final.pkl --fully-obs`
- SARSA: `python evaluation/visualize.py --agent sarsa --q-table logs/sarsa_run/sarsa_table_final.pkl --fully-obs`

## TensorBoard
1) Train with `--log-dir` enabled (see above).
2) From the repo root: `tensorboard --logdir logs --port 6006`
3) Open http://localhost:6006 (Scalars tab).

## DQN
- Training: `python -m training.train --agent dqn --env MiniGrid-MultiRoom-N2-S4-v0 --wrapper dqn --log-dir logs/dqn_run`
- Visualization (random or with a saved model): `python evaluation/visualize.py --agent dqn --wrapper dqn --env MiniGrid-MultiRoom-N2-S4-v0 --model logs/dqn_run/model.pt`
- Notes: always use the `dqn` wrapper (flat float state). Adjust `--episodes`, `--seed`, or other hyperparameters in [config/hyperparams.yaml](config/hyperparams.yaml).

Notes:
- `--shaped` and `--curriculum` apply only to the DQN pipeline; tabular agents already include built-in shaping.
- For deterministic runs: add `--seed 0`.
