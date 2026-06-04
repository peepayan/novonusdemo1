# Novonus

Novonus: EMG-augmented imitation learning pipeline for contact-rich robotic manipulation.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows bash
pip install -r requirements.txt
python verify_setup.py
```

## Layout

- `src/` — source code
- `data/` — datasets (gitignored, structure tracked via `.gitkeep`)
- `outputs/` — checkpoints, plots, artifacts (gitignored)
- `notebooks/` — exploration

## Hardware target

NVIDIA RTX 5060 (Blackwell, 8 GB VRAM). Keep batch sizes and memory conservative.
