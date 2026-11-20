from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from api.app.config import ArtifactPaths


@dataclass
class ArtifactStore:
    forecast_predictions: Optional[pd.DataFrame] = None
    solar_predictions: Optional[pd.DataFrame] = None
    battery_lookup: Optional[dict[str, Any]] = None
    degradation_results: Optional[pd.DataFrame] = None
    degradation_metadata: Optional[dict[str, Any]] = None
    status: dict[str, bool] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def load(self) -> None:
        self.forecast_predictions = self._read_csv(
            "forecast_predictions", ArtifactPaths.forecast_predictions, ["timestamp"]
        )
        if self.forecast_predictions is not None:
            self.forecast_predictions = self.forecast_predictions.set_index("timestamp").sort_index()

        self.solar_predictions = self._read_csv(
            "solar_predictions", ArtifactPaths.solar_predictions, ["timestamp"]
        )
        self.battery_lookup = self._read_json(
            "battery_lookup", ArtifactPaths.battery_lookup
        )
        self.degradation_results = self._read_csv(
            "degradation_results", ArtifactPaths.degradation_results
        )
        self.degradation_metadata = self._read_json(
            "degradation_metadata", ArtifactPaths.degradation_metadata
        )

    def _read_csv(self, name: str, path, parse_dates: Optional[list[str]] = None):
        self.paths[name] = str(path)
        try:
            frame = pd.read_csv(path, parse_dates=parse_dates)
        except FileNotFoundError:
            self.status[name] = False
            self.errors[name] = f"missing artifact: {path}"
            return None
        except Exception as exc:
            self.status[name] = False
            self.errors[name] = f"could not read {path}: {exc}"
            return None
        self.status[name] = True
        self.errors.pop(name, None)
        return frame

    def _read_json(self, name: str, path):
        self.paths[name] = str(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.status[name] = False
            self.errors[name] = f"missing artifact: {path}"
            return None
        except Exception as exc:
            self.status[name] = False
            self.errors[name] = f"could not read {path}: {exc}"
            return None
        self.status[name] = True
        self.errors.pop(name, None)
        return payload

    @property
    def missing_artifacts(self) -> list[str]:
        return sorted(name for name, available in self.status.items() if not available)

    @property
    def health_status(self) -> str:
        return "degraded" if self.missing_artifacts else "ok"


STORE = ArtifactStore()
