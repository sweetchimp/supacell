# supacell

supacell is a research prototype for AI-assisted household energy management, exploring how household-level data, short-term forecasting, and battery control models can support more informed energy decisions while keeping raw data, machine-learning experiments, API services, and frontend work in separate parts of the repository.

## Setup

Requires Python 3.9 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` if you prefer running `python` and `pytest` directly.

## Repository layout

- `data/raw/` and `data/processed/`: ignored dataset storage; label every dataset as `real`, `synthetic`, or `simulated`.
- `ml/`: Python package, simulation/forecasting/battery modules, and experiment scripts.
- `api/`: FastAPI backend only.
- `web/`: reserved for a later Next.js + TypeScript + Tailwind frontend.
- `docs/`: research write-up and methodology notes.
- `results/`: ignored storage for saved experiment outputs.

## Project rules

- Never invent results; report only computed outputs and preserve the code and inputs needed to reproduce them.
- Label all data as `real`, `synthetic`, or `simulated` in filenames, manifests, or write-up tables.
- Keep ML, API, and frontend responsibilities separated.
