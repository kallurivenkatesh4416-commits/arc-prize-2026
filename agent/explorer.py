"""Pure helpers for probing ARC-AGI-3 grids.

The previous version of this module drove ``env.step()`` / ``env.reset()``
against a fictional SDK shape. The real ``arc_agi_3`` SDK owns the loop
(see ``OfflineControllerAgent`` in ``offline_controller.py``), so the
env-driving helpers have been removed. What remains here are the pure data
helpers that the agent uses internally during its probe phase:

- ``ProbeReport`` — dataclass summarising what was observed.
- ``detect_objects`` — connected-component extraction from a 2D grid.
- ``_centroid`` — bounding-box centroid for an :class:`ObjectTrace`.
- ``DIRECTIONAL`` / ``CLICK_ACTION`` / ``RESET_ACTION`` / ``UNDO_ACTION``
  constants used by the agent's policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .world_model import ObjectTrace, Transition

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


def _centroid(obj: ObjectTrace) -> tuple[int, int]:
    """Return (x, y) centroid of an object's bounding box."""
    x0, y0, x1, y1 = obj.bbox
    return ((x0 + x1) // 2, (y0 + y1) // 2)
