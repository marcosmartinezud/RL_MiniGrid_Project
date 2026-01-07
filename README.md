# Reinforcement Learning on MiniGrid MultiRoom

## Entrenamiento rápido (tabular)
- Q-learning (FullyObs recomendado):
	`python -m training.train --agent qlearning --env MiniGrid-MultiRoom-N2-S4-v0 --fully-obs --log-dir logs/qlearning_run`
- SARSA (FullyObs recomendado):
	`python -m training.train --agent sarsa --env MiniGrid-MultiRoom-N2-S4-v0 --fully-obs --log-dir logs/sarsa_run`

Usa `--episodes` y `--max-steps` para overrides; el resto de hiperparámetros está en [config/hyperparams.yaml](config/hyperparams.yaml).

## Visualización
- Q-learning: `python evaluation/visualize.py --agent qlearning --q-table logs/qlearning_run/q_table_final.pkl --fully-obs`
- SARSA: `python evaluation/visualize.py --agent sarsa --q-table logs/sarsa_run/sarsa_table_final.pkl --fully-obs`

## TensorBoard
1) Lanza entrenamiento con `--log-dir` (como arriba).
2) Desde la raíz: `tensorboard --logdir logs --port 6006`
3) Abre http://localhost:6006 (pestaña Scalars).

## DQN
- `python -m training.train --agent dqn --env MiniGrid-MultiRoom-N2-S4-v0 --wrapper dqn --log-dir logs/dqn_run`

Notas:
- `--shaped` y `--curriculum` aplican solo al pipeline DQN; los tabulares ya incluyen shaping interno.
- Para semilla fija: añade `--seed 0`.
