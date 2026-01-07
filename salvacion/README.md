# Salvacion: baselines Q-learning y Dueling DQN para MultiRoom

Objetivo: mantener un baseline tabular simple (Q-learning) y una version
Dueling DQN inspirada en el repositorio de referencia para generalizar a
layouts aleatorios en `MiniGrid-MultiRoom-N4-S5-v0`.

## Q-learning (generaliza a layouts aleatorios)
```bash
.venv/bin/python -m salvacion.train
```

Configuracion: `salvacion/config.yaml` (usa `random_layout: true`, estado
`bfs` y guia experta para acelerar el aprendizaje).

El estado `bfs` combina observacion local (frente/izq/der, puertas visibles)
con distancia al objetivo y la direccion recomendada por el planner BFS.

## Dueling DQN (generaliza a layouts aleatorios)
```bash
.venv/bin/python -m salvacion.train_dqn
```

Configuracion: `salvacion/dqn_config.yaml` (usa `random_layout: true`,
observacion RGB parcial y downsample a 14x14). Incluye un **prefill con
expert planner** para acelerar el aprendizaje.

Para entrenar mas rapido o probar, puedes reducir episodios:
```bash
.venv/bin/python -m salvacion.train_dqn --episodes 5000
```

## Evaluacion
```bash
.venv/bin/python -m salvacion.evaluate
```

## Visualizacion
```bash
.venv/bin/python -m salvacion.visualize --random-layout --episodes 5
```

```bash
.venv/bin/python -m salvacion.visualize_dqn --episodes 5 --random-layout
```

```bash
.venv/bin/python -m salvacion.visualize_expert --episodes 5 --random-layout
```

## Archivos de salida
- `salvacion/output/q_table.pkl`: Q-table entrenada
- `salvacion/output/train_metrics.csv`: log de entrenamiento
- `salvacion/output/eval_metrics.json`: metricas de evaluacion
- `salvacion/dqn_output/dueling_dqn.pt`: pesos del DQN
- `salvacion/dqn_output/train_metrics.csv`: log DQN
- `salvacion/dqn_output/eval_metrics.json`: metricas DQN

## Notas
- Acciones usadas: izquierda, derecha, avanzar, toggle (abre puertas).
- Q-learning usa estado `bfs` y prefill experto con mezclado de acciones.
- DQN usa reward shaping suave (exploracion, puertas, repeticion, step penalty)
  para lidiar con recompensa dispersa.
