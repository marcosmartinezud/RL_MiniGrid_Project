# Installation Guide

## Requirements
- Python 3.10+
- pip
- Virtual environment recommended (venv or conda)

## Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Verify installation
```bash
python -c "import gymnasium, minigrid, numpy, yaml; print('ok')"
```

## Optional: PyTorch for TensorBoard logging
- The simple logger uses `torch.utils.tensorboard`. Install with:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- GPU wheels vary by machine; check pytorch.org for the right command.

## Common issues (Windows)
- If MiniGrid fails to build, make sure Microsoft C++ Build Tools are installed.
- Sometimes `gymnasium` extras complain about box2d; not needed here, so you can ignore that warning.
- If TensorBoard command is not found, ensure the virtualenv `Scripts` directory is on PATH.
