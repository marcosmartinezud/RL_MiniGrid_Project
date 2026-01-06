# Reinforcement Learning on MiniGrid MultiRoom

tensorboard --logdir logs --port 6006

Baseline Q-learning: python -m training.train --agent qlearning --episodes 10000
Q-learning shaped: python -m training.train --agent qlearning --episodes 10000 --shaped
SARSA baseline: python -m training.train --agent sarsa --episodes 10000
SARSA shaped: python -m training.train --agent sarsa --episodes 10000 --shaped
Curriculum + shaping (avanzado): python -m training.train --agent qlearning --curriculum --shaped