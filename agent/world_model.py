"""Evolving theory of an ARC-AGI-3 game.

Stores what the agent has observed: which actions cause which grid changes,
score events, hypotheses about the goal, recent frame history. Knows nothing
about LLMs — the llm_agent module reads .summary() and writes hypotheses.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

# Optional SDK imports — keep module importable on Python <3.12 / without ARC SDKs.
try:  # pragma: no cover - exercised only when the SDK is installed
    from arc_agi_3 import FrameData as _FrameData  # type: ignore
    from arc_agi_3 import GameState as _GameState  # type: ignore
except Exception:  # noqa: BLE001
    _FrameData = None  # type: ignore[assignment]
    _GameState = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised only when the toolkit is installed
    from arcengine import FrameDataRaw as _FrameDataRaw  # type: ignore
    from arcengine import GameState as _EngineGameState  # type: ignore
except Exception:  # noqa: BLE001
    _FrameDataRaw = None  # type: ignore[assignment]
    _EngineGameState = None  # type: ignore[assignment]


def _is_frame_data(frame: Any) -> bool:
    """True if ``frame`` looks like an ``arc_agi_3.FrameData`` instance.

    We don't rely on isinstance against ``_FrameData`` exclusively because
    smoke tests may pass dicts. But when the SDK is installed and the object
    is genuinely a FrameData, we route through the typed accessors.
    """
    if _FrameData is not None and isinstance(frame, _FrameData):
        return True
    if _FrameDataRaw is not None and isinstance(frame, _FrameDataRaw):
        return True
    return False


def _get(frame: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(frame, dict) and key in frame:
            return frame[key]
        if hasattr(frame, key):
            return getattr(frame, key)
    return default


def grid_of(frame: Any) -> list[list[int]] | None:
    """Return a 2D grid (list of rows) extracted from a frame.

    ``FrameData.frame`` is 3D (``list[list[list[int]]]``); the visible 2D grid
    is the first slice. Falls back to dict-style access for legacy / mock
    frames used by smoke tests.
    """
    if _is_frame_data(frame):
        grid = frame.frame
    else:
        grid = _get(frame, "grid", "frame", "observation", "obs", "state")
    if grid is None:
        return None
    if isinstance(grid, tuple) and grid:
        grid = grid[0]

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(grid)
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim == 2:
            return arr.tolist()
        return None
    except Exception:  # noqa: BLE001
        pass

    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if (
        isinstance(grid, list)
        and grid
        and isinstance(grid[0], list)
        and grid[0]
        and isinstance(grid[0][0], list)
    ):
        grid = grid[0]
    return grid


def score_of(frame: Any) -> int:
    if _is_frame_data(frame):
        try:
            return int(getattr(frame, "score", 0) or 0)
        except (TypeError, ValueError):
            return 0
    value = _get(frame, "score", default=0)
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def is_done(frame: Any) -> bool:
    if _is_frame_data(frame):
        return _state_name(_get(frame, "state")) in {"WIN", "GAME_OVER", "NOT_STARTED"}
    explicit = _get(frame, "done", "game_over", "terminal", "finished", default=None)
    if explicit is not None:
        return bool(explicit)
    return _state_name(_get(frame, "state")) in {"WIN", "GAME_OVER", "NOT_STARTED"}


def available_actions(frame: Any) -> list[str]:
    if _is_frame_data(frame):
        actions = frame.available_actions or []
        return [_action_name(a) for a in actions]
    actions = _get(frame, "available_actions", "legal_actions", "actions")
    if not actions:
        return []
    return [_action_name(a) for a in actions]


def _state_name(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if name:
        return str(name).upper()
    return str(value).split(".")[-1].upper()


def _action_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name).upper()
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        text = str(value).split(".")[-1].upper()
        return text
    if as_int == 0:
        return "RESET"
    return f"ACTION{as_int}"


def _count_changed_cells(prev: list[list[int]] | None, curr: list[list[int]] | None) -> int:
    if prev is None or curr is None:
        return 0
    changed = 0
    for r, row in enumerate(curr):
        if r >= len(prev):
            changed += len(row)
            continue
        prev_row = prev[r]
        for c, value in enumerate(row):
            if c >= len(prev_row) or prev_row[c] != value:
                changed += 1
    return changed


def _grid_hash(grid: list[list[int]] | None) -> int | None:
    """Stable in-process hash of a 2D grid, or None for empty/missing.

    Used by ``WorldModel`` to count distinct visited states and to attribute
    novelty to whichever action produced each unseen state. Hash randomization
    is fine because the value is only compared within a single process.
    """
    if not grid:
        return None
    try:
        return hash(tuple(tuple(row) for row in grid))
    except TypeError:
        return None


@dataclass
class Transition:
    action: str
    score_delta: int
    changed_cells: int
    turn: int
    novel_next: bool = False

    @property
    def grid_changed(self) -> bool:
        return self.changed_cells > 0


@dataclass
class ScoreEvent:
    turn: int
    score_before: int
    score_after: int
    action: str


@dataclass
class ObjectTrace:
    color: int
    cell_count: int
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)


@dataclass
class Hypothesis:
    text: str
    confidence: float = 0.5


@dataclass
class WorldModel:
    action_effects: dict[str, list[Transition]] = field(default_factory=dict)
    objects_seen: list[ObjectTrace] = field(default_factory=list)
    score_events: list[ScoreEvent] = field(default_factory=list)
    goal_hypotheses: list[Hypothesis] = field(default_factory=list)
    frame_history: deque = field(default_factory=lambda: deque(maxlen=16))
    last_prev_grid: list[list[int]] | None = None
    last_action: str = ""
    turn: int = 0
    current_score: int = 0
    game_id: str = ""
    distinct_grid_hashes: set[int] = field(default_factory=set)
    action_novelty: dict[str, int] = field(default_factory=dict)

    def update(self, prev_frame: Any, action: str, next_frame: Any) -> Transition:
        prev_grid = grid_of(prev_frame)
        next_grid = grid_of(next_frame)
        score_before = score_of(prev_frame)
        score_after = score_of(next_frame)
        delta = score_after - score_before
        changed = _count_changed_cells(prev_grid, next_grid)

        transition = Transition(
            action=action,
            score_delta=delta,
            changed_cells=changed,
            turn=self.turn,
        )
        key = _action_key(action)
        self.action_effects.setdefault(key, []).append(transition)
        if delta != 0:
            self.score_events.append(
                ScoreEvent(self.turn, score_before, score_after, action)
            )
        prev_hash = _grid_hash(prev_grid)
        if prev_hash is not None:
            self.distinct_grid_hashes.add(prev_hash)
        next_hash = _grid_hash(next_grid)
        novel = next_hash is not None and next_hash not in self.distinct_grid_hashes
        if novel:
            self.distinct_grid_hashes.add(next_hash)
            self.action_novelty[key] = self.action_novelty.get(key, 0) + 1
        transition.novel_next = novel
        self.frame_history.append(next_frame)
        self.last_prev_grid = [row[:] for row in prev_grid] if prev_grid else None
        self.last_action = action
        self.current_score = score_after
        self.turn += 1
        return transition

    def add_hypothesis(self, text: str, confidence: float = 0.5) -> int:
        self.goal_hypotheses.append(Hypothesis(text.strip(), float(confidence)))
        return len(self.goal_hypotheses) - 1

    def invalidate_hypothesis(self, index: int, reason: str = "") -> bool:
        if 0 <= index < len(self.goal_hypotheses):
            self.goal_hypotheses.pop(index)
            return True
        return False

    def record_objects(self, objects: Iterable[ObjectTrace]) -> None:
        seen_keys = {(o.color, o.bbox) for o in self.objects_seen}
        for obj in objects:
            key = (obj.color, obj.bbox)
            if key not in seen_keys:
                self.objects_seen.append(obj)
                seen_keys.add(key)

    def summary(self, max_chars: int = 4000) -> str:
        lines: list[str] = []
        lines.append(f"GAME: {self.game_id or '<unknown>'}  TURN: {self.turn}  SCORE: {self.current_score}")

        if self.goal_hypotheses:
            lines.append("\nHYPOTHESES (most recent first):")
            for i, h in enumerate(reversed(self.goal_hypotheses)):
                real_idx = len(self.goal_hypotheses) - 1 - i
                lines.append(f"  [{real_idx}] ({h.confidence:.2f}) {h.text}")
        else:
            lines.append("\nHYPOTHESES: none yet.")

        if self.action_effects:
            lines.append("\nACTION EFFECTS (sample count, avg Δscore, avg cells changed):")
            for key in sorted(self.action_effects):
                trs = self.action_effects[key]
                n = len(trs)
                avg_delta = sum(t.score_delta for t in trs) / n
                avg_cells = sum(t.changed_cells for t in trs) / n
                lines.append(f"  {key:>18}  n={n:3d}  Δscore={avg_delta:+.2f}  cells={avg_cells:5.1f}")
        else:
            lines.append("\nACTION EFFECTS: none recorded.")

        if self.score_events:
            lines.append("\nSCORE EVENTS (last 5):")
            for ev in self.score_events[-5:]:
                lines.append(f"  turn {ev.turn:4d}  {ev.action:>14}  {ev.score_before} -> {ev.score_after}")

        last_frame = self.frame_history[-1] if self.frame_history else None
        curr_grid = grid_of(last_frame) if last_frame is not None else None
        prev_grid = self.last_prev_grid

        if self.last_action and prev_grid is not None and curr_grid is not None:
            lines.append(f"\nLAST TRANSITION: {self.last_action}")
            lines.append("BEFORE:")
            lines.append(render_grid(prev_grid))
            lines.append("AFTER:")
            lines.append(render_grid(curr_grid))
            diff = render_diff(prev_grid, curr_grid)
            if diff:
                lines.append("DIFF (* = changed cell):")
                lines.append(diff)
        elif curr_grid is not None:
            lines.append("\nCURRENT GRID:")
            lines.append(render_grid(curr_grid))

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[: max_chars - 20] + "\n...(truncated)"
        return text


def _action_key(action: str) -> str:
    base = action.split("(")[0].strip().upper()
    return base


def render_grid(grid: list[list[int]]) -> str:
    if not grid:
        return "<empty>"
    width = len(grid[0])
    header = "   " + "".join(f"{c % 10}" for c in range(width))
    rows = [header]
    for r, row in enumerate(grid):
        rows.append(f"{r:2d} " + "".join(_cell_char(v) for v in row))
    return "\n".join(rows)


def _cell_char(value: int) -> str:
    if value == 0:
        return "."
    if 1 <= value <= 9:
        return str(value)
    table = "abcdef"  # 10..15
    if 10 <= value <= 15:
        return table[value - 10]
    return "?"


def render_diff(prev: list[list[int]], curr: list[list[int]]) -> str:
    if not prev or not curr:
        return ""
    h = max(len(prev), len(curr))
    width = max(
        max((len(r) for r in prev), default=0),
        max((len(r) for r in curr), default=0),
    )
    if width == 0:
        return ""
    header = "   " + "".join(f"{c % 10}" for c in range(width))
    rows = [header]
    for r in range(h):
        prev_row = prev[r] if r < len(prev) else []
        curr_row = curr[r] if r < len(curr) else []
        chars = []
        for c in range(width):
            pv = prev_row[c] if c < len(prev_row) else None
            cv = curr_row[c] if c < len(curr_row) else None
            chars.append("*" if pv != cv else ".")
        rows.append(f"{r:2d} " + "".join(chars))
    return "\n".join(rows)
