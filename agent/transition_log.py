"""JSONL transition logging for offline ARC-AGI-3 runs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .world_model import Transition, available_actions, score_of


class TransitionLogger:
    """Append one compact JSON record for each submitted env.step action."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8", buffering=1)

    def log(
        self,
        *,
        game_id: str,
        phase: str,
        action: str,
        data: dict[str, Any] | None,
        transition: Transition,
        next_frame: Any,
        elapsed_ms: float,
        novel_state: bool = False,
        distinct_grids_seen: int = 0,
    ) -> None:
        record = {
            "ts": time.time(),
            "game_id": game_id,
            "turn": transition.turn,
            "phase": phase,
            "action": action,
            "data": data or {},
            "state": _state_name(getattr(next_frame, "state", None)),
            "score": score_of(next_frame),
            "score_delta": transition.score_delta,
            "changed_cells": transition.changed_cells,
            "novel_state": bool(novel_state),
            "distinct_grids_seen": int(distinct_grids_seen),
            "levels_completed": _safe_attr(next_frame, "levels_completed", 0),
            "level_actions": _safe_attr(next_frame, "level_actions", []),
            "available_actions": available_actions(next_frame),
            "elapsed_ms": round(float(elapsed_ms), 3),
        }
        self._fp.write(json.dumps(record, default=_json_default) + "\n")

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> "TransitionLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _state_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value).split(".")[-1] if value is not None else ""


def _safe_attr(value: Any, name: str, default: Any) -> Any:
    found = getattr(value, name, default)
    if found is None:
        return default
    return found


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "tolist"):
        return value.tolist()
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value)
