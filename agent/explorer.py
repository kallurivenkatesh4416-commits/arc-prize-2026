"""Helpers for probing ARC-AGI-3 toolkit environments."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .world_model import (
    ObjectTrace,
    Transition,
    WorldModel,
    available_actions,
    grid_of,
    is_done,
    score_of,
)

DIRECTIONAL = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]
CLICK_ACTION = "ACTION6"
UNDO_ACTION = "ACTION7"
RESET_ACTION = "RESET"


@dataclass
class ProbeReport:
    game_id: str
    legal_actions: list[str] = field(default_factory=list)
    directional_transitions: list[Transition] = field(default_factory=list)
    action6_live_coords: list[tuple[int, int]] = field(default_factory=list)
    objects: list[ObjectTrace] = field(default_factory=list)
    initial_score: int = 0

    def as_text(self) -> str:
        lines = [f"PROBE REPORT for {self.game_id}"]
        lines.append(f"  legal_actions: {self.legal_actions}")
        lines.append(f"  initial_score: {self.initial_score}")
        lines.append("  directional probe:")
        for t in self.directional_transitions:
            lines.append(
                f"    {t.action:>8}  Δscore={t.score_delta:+d}  cells_changed={t.changed_cells}"
            )
        if self.action6_live_coords:
            lines.append(f"  ACTION6 live coords (sampled): {self.action6_live_coords[:16]}"
                         + (" ..." if len(self.action6_live_coords) > 16 else ""))
        if self.objects:
            lines.append(f"  objects (first 8 of {len(self.objects)}):")
            for obj in self.objects[:8]:
                lines.append(f"    color={obj.color} cells={obj.cell_count} bbox={obj.bbox}")
        return "\n".join(lines)


@dataclass
class StepOutcome:
    frame: Any
    reward: float | None = None
    done: bool = False
    info: Any = None


def normalize_action_names(actions: Any) -> list[str]:
    if not actions:
        return []
    names: list[str] = []
    for action in actions:
        name = getattr(action, "name", None)
        if name:
            names.append(str(name).upper())
            continue
        try:
            action_id = int(action)
        except (TypeError, ValueError):
            names.append(str(action).split(".")[-1].upper())
        else:
            names.append("RESET" if action_id == 0 else f"ACTION{action_id}")
    return names


def legal_actions(env: Any, frame: Any | None = None) -> list[str]:
    frame_actions = available_actions(frame) if frame is not None else []
    if frame_actions:
        return frame_actions
    return normalize_action_names(getattr(env, "action_space", None))


def unwrap_step_result(result: Any) -> StepOutcome:
    if isinstance(result, tuple):
        frame = result[0] if result else None
        reward = result[1] if len(result) > 1 else None
        if len(result) > 4:
            done = bool(result[2]) or bool(result[3])
            info = result[4]
        else:
            done = bool(result[2]) if len(result) > 2 else is_done(frame)
            info = result[3] if len(result) > 3 else None
        return StepOutcome(frame=frame, reward=reward, done=done, info=info)
    return StepOutcome(frame=result, done=is_done(result))


def step_action(
    env: Any,
    action_name: str,
    data: dict[str, Any] | None = None,
    reasoning: Any = None,
) -> StepOutcome:
    """Submit an action to an ``arc_agi`` EnvironmentWrapper.

    The 0.9.x toolkit accepts ``step(GameAction, data=..., reasoning=...)``.
    The fallbacks keep tests and slightly older builds easy to exercise.
    """
    from arcengine import GameAction  # type: ignore

    action = getattr(GameAction, action_name)
    kwargs: dict[str, Any] = {}
    if data:
        kwargs["data"] = data
    if reasoning is not None:
        kwargs["reasoning"] = reasoning

    try:
        return unwrap_step_result(env.step(action, **kwargs))
    except TypeError:
        pass

    if data:
        try:
            return unwrap_step_result(env.step(action, data))
        except TypeError:
            if {"x", "y"} <= set(data):
                return unwrap_step_result(env.step(action, int(data["x"]), int(data["y"])))
            raise
    return unwrap_step_result(env.step(action))


def run_probe(
    env: Any,
    game_id: str,
    *,
    world: WorldModel | None = None,
    initial_frame: Any | None = None,
    probe_budget: int = 24,
    max_probe_clicks: int = 8,
    on_transition: Callable[[str, str, dict[str, Any] | None, Transition, Any, float], None] | None = None,
) -> tuple[WorldModel, ProbeReport, Any]:
    """Run a short scripted probe and return the updated world model."""
    world = world or WorldModel(game_id=game_id)
    if not world.game_id:
        world.game_id = game_id

    current = initial_frame
    if current is None:
        current = unwrap_step_result(env.reset()).frame

    report = ProbeReport(
        game_id=game_id,
        legal_actions=legal_actions(env, current),
        initial_score=score_of(current),
    )

    steps = 0
    for action_name in DIRECTIONAL:
        legal = legal_actions(env, current)
        if legal and action_name not in legal:
            continue
        started_at = time.monotonic()
        outcome = step_action(
            env,
            action_name,
            reasoning={"phase": "probe", "kind": "directional", "action": action_name},
        )
        elapsed_ms = (time.monotonic() - started_at) * 1000
        transition = world.update(current, action_name, outcome.frame)
        if on_transition is not None:
            on_transition("probe", action_name, None, transition, outcome.frame, elapsed_ms)
        report.directional_transitions.append(transition)
        current = outcome.frame
        steps += 1
        if steps >= probe_budget or outcome.done or is_done(current):
            return world, report, current

    objects = detect_objects(grid_of(current))
    report.objects = objects
    world.record_objects(objects)

    seen: set[tuple[int, int]] = set()
    coords: list[tuple[int, int]] = []
    for obj in objects:
        coord = _centroid(obj)
        if coord in seen:
            continue
        seen.add(coord)
        coords.append(coord)
        if len(coords) >= max_probe_clicks:
            break

    for x, y in coords:
        legal = legal_actions(env, current)
        if legal and CLICK_ACTION not in legal:
            break
        data = {"x": int(x), "y": int(y)}
        started_at = time.monotonic()
        outcome = step_action(
            env,
            CLICK_ACTION,
            data=data,
            reasoning={"phase": "probe", "kind": "centroid_click", "x": int(x), "y": int(y)},
        )
        elapsed_ms = (time.monotonic() - started_at) * 1000
        transition = world.update(current, f"ACTION6({int(x)},{int(y)})", outcome.frame)
        if on_transition is not None:
            on_transition("probe", CLICK_ACTION, data, transition, outcome.frame, elapsed_ms)
        if transition.score_delta != 0 or transition.changed_cells != 0:
            report.action6_live_coords.append((int(x), int(y)))
        current = outcome.frame
        steps += 1
        if steps >= probe_budget or outcome.done or is_done(current):
            break

    return world, report, current


def detect_objects(grid: list[list[int]] | None, background: int = 0) -> list[ObjectTrace]:
    if not grid:
        return []
    h = len(grid)
    w = len(grid[0]) if h else 0
    visited = [[False] * w for _ in range(h)]
    objects: list[ObjectTrace] = []

    for r in range(h):
        for c in range(w):
            if visited[r][c] or grid[r][c] == background:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    continue
                if visited[rr][cc] or grid[rr][cc] != color:
                    continue
                visited[rr][cc] = True
                cells.append((rr, cc))
                stack.extend([(rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)])
            xs = [c for _, c in cells]
            ys = [r for r, _ in cells]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            objects.append(ObjectTrace(color=color, cell_count=len(cells), bbox=bbox))

    objects.sort(key=lambda o: -o.cell_count)
    return objects


def _centroid(obj: ObjectTrace) -> tuple[int, int]:
    """Return (x, y) centroid of an object's bounding box."""
    x0, y0, x1, y1 = obj.bbox
    return ((x0 + x1) // 2, (y0 + y1) // 2)
