# Reinforcement Learning on MiniGrid MultiRoom

tensorboard --logdir logs --port 6006

1. Q-learning Vanilla (Baseline)
python -m training.train --agent qlearning --wrapper simple --episodes 10000

2. SARSA Vanilla
python -m training.train --agent sarsa --wrapper simple --episodes 10000

3. Q-learning con Reward Shaping
python -m training.train --agent qlearning --wrapper simple --episodes 10000 --shaped

4. SARSA con Reward Shaping
python -m training.train --agent sarsa --wrapper simple --episodes 10000 --shaped

5. Q-learning con Curriculum Learning
python -m training.train --agent qlearning --wrapper simple --curriculum --shaped

6. Q-learning con PositionAware (falla esperada)
python -m training.train --agent qlearning --wrapper position --episodes 10000

7. DQN (Solución Final)
python -m training.train --agent dqn --wrapper dqn --episodes 5000