"""Cheap scripted probe that seeds the world model before Claude takes over.

Runs each legal action, records frame diffs + score deltas, sparsely scans
the 64x64 click grid for ACTION6, and extracts connected-component objects.
All functions take pure data in, return pure data out — the caller drives
the env.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Any

from .world_model import (
    ObjectTrace,
    Transition,
    WorldModel,
    available_actions,
    grid_of,
    is_done,
    score_of,
)

log = logging.getLogger(__name__)

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


def _step(env: Any, action_name: str, *coords: int) -> Any:
    """Call env.step with the correct GameAction enum variant.

    Imported lazily so unit tests can mock env without arcengine installed.
    """
    from arcengine import GameAction  # type: ignore

    action = getattr(GameAction, action_name)
    if coords:
        return env.step(action, *coords)
    return env.step(action)


def _reset(env: Any) -> Any:
    return env.reset()


def probe_action_effects(env: Any, world: WorldModel, budget_steps: int = 40) -> list[Transition]:
    """Try each legal action once from a fresh reset, record effects."""
    transitions: list[Transition] = []
    frame = _reset(env)
    legal = available_actions(frame) or DIRECTIONAL + [CLICK_ACTION]

    steps_used = 0
    for action in legal:
        if steps_used >= budget_steps:
            break
        if action == CLICK_ACTION:
            continue  # handled separately by sample_action6_grid
        if action in (RESET_ACTION, UNDO_ACTION):
            continue

        try:
            next_frame = _step(env, action)
        except Exception as exc:  # noqa: BLE001
            log.warning("action %s raised %s; skipping", action, exc)
            continue

        t = world.update(frame, action, next_frame)
        transitions.append(t)
        steps_used += 1

        if is_done(next_frame):
            frame = _reset(env)
        else:
            # try to undo; fall back to reset if undo isn't available
            if UNDO_ACTION in legal:
                try:
                    frame = _step(env, UNDO_ACTION)
                    steps_used += 1
                    continue
                except Exception:  # noqa: BLE001
                    pass
            frame = _reset(env)

    return transitions


def _centroid(obj: ObjectTrace) -> tuple[int, int]:
    x0, y0, x1, y1 = obj.bbox
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def sample_action6_on_objects(
    env: Any,
    world: WorldModel,
    objects: list[ObjectTrace],
    budget_clicks: int = 16,
) -> list[tuple[int, int]]:
    """Click the centroid of each detected object. Far cheaper and higher-signal
    than a blind grid scan: targets the ~5-15 things that actually exist."""
    live: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    frame = _reset(env)
    clicks = 0

    for obj in objects:
        if clicks >= budget_clicks:
            break
        coord = _centroid(obj)
        if coord in seen:
            continue
        seen.add(coord)
        x, y = coord
        try:
            next_frame = _step(env, CLICK_ACTION, x, y)
        except Exception as exc:  # noqa: BLE001
            log.debug("ACTION6(%d,%d) raised %s", x, y, exc)
            continue
        t = world.update(frame, f"ACTION6({x},{y})", next_frame)
        clicks += 1
        if t.changed_cells > 0 or t.score_delta != 0:
            live.append(coord)
        frame = _reset(env)
    return live


def sample_action6_grid(
    env: Any,
    world: WorldModel,
    stride: int = 16,
    budget_clicks: int = 16,
) -> list[tuple[int, int]]:
    """Fallback sparse scan when no objects are detected. Conservative budget."""
    live: list[tuple[int, int]] = []
    frame = _reset(env)
    clicks = 0

    for y in range(0, 64, stride):
        for x in range(0, 64, stride):
            if clicks >= budget_clicks:
                return live
            try:
                next_frame = _step(env, CLICK_ACTION, x, y)
            except Exception as exc:  # noqa: BLE001
                log.debug("ACTION6(%d,%d) raised %s", x, y, exc)
                continue
            t = world.update(frame, f"ACTION6({x},{y})", next_frame)
            clicks += 1
            if t.changed_cells > 0 or t.score_delta != 0:
                live.append((x, y))
            frame = _reset(env)
    return live


def run_probe(env: Any, game_id: str, stride: int = 16) -> tuple[WorldModel, ProbeReport]:
    """Convenience: reset, probe directionals, sample clicks, extract objects.

    Tolerant of partial failures — returns whatever was gathered before a crash.
    """
    world = WorldModel(game_id=game_id)
    report = ProbeReport(game_id=game_id)
    try:
        initial = _reset(env)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: initial reset failed: %s", game_id, exc)
        return world, report

    world.current_score = score_of(initial)
    legal = available_actions(initial) or DIRECTIONAL + [CLICK_ACTION]
    report.legal_actions = legal
    report.initial_score = world.current_score

    try:
        report.directional_transitions = probe_action_effects(env, world)
    except Exception as exc:  # noqa: BLE001
        log.warning("%s: directional probe failed mid-flight: %s", game_id, exc)

    initial_objects = detect_objects(grid_of(initial))
    if CLICK_ACTION in legal:
        try:
            if initial_objects:
                report.action6_live_coords = sample_action6_on_objects(env, world, initial_objects)
            else:
                report.action6_live_coords = sample_action6_grid(env, world, stride=stride)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: ACTION6 probe failed mid-flight: %s", game_id, exc)

    latest = world.frame_history[-1] if world.frame_history else initial
    objects = detect_objects(grid_of(latest))
    world.record_objects(initial_objects + objects)
    report.objects = world.objects_seen
    return world, report


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the scripted probe on one ARC-AGI-3 game.")
    parser.add_argument("--game", required=True, help="game_id, e.g. ls20")
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import arc_agi  # type: ignore

    arc = arc_agi.Arcade()
    env = arc.make(args.game)
    _, report = run_probe(env, args.game, stride=args.stride)
    print(report.as_text())


if __name__ == "__main__":
    _cli()
