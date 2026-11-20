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

## API

The FastAPI service is read-only: it serves saved experiment outputs and does not retrain models or recompute battery behavior at request time.

```powershell
.\.venv\Scripts\python api\scripts\build_battery_lookup.py
.\.venv\Scripts\python -m uvicorn api.app.main:app --reload --port 8000
```

OpenAPI docs are available at `http://127.0.0.1:8000/docs`. The API allows browser requests from `http://localhost:3000` and `http://127.0.0.1:3000` by default; override them with `SUPACELL_CORS_ORIGINS` as a comma-separated list.

| Endpoint | Example |
| --- | --- |
| Health and artifact status | `GET /health` |
| Available profiles | `GET /profiles` |
| REAL forecast with 90% band | `GET /forecast?profile=real&date=2010-04-20&model=xgboost` |
| SYNTHETIC solar forecast band | `GET /forecast?profile=grid_plus_solar_plus_battery&date=2009-11-07` |
| Battery runtime lookup | `GET /battery/runtime?profile=grid_only&state_of_charge=0.8&scenario=correct` |
| Degradation summary | `GET /degradation/summary?level=full` |
| Saved-output recommendations | `GET /recommendations?profile=grid_plus_solar_plus_battery&date=2009-11-07&state_of_charge=0.8` |

Dates use `YYYY-MM-DD`. Saved battery state-of-charge values are `0.2`, `0.4`, `0.6`, `0.8`, and `1.0`. Unsupported profiles, unavailable dates, invalid input, and missing artifacts return JSON errors with `code`, `message`, and `detail` fields. API tests run with `.\.venv\Scripts\python -m pytest api\tests -q`.

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
