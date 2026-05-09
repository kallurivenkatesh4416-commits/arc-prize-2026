"""ls20 trajectory + per-action distance analyzer.

Reads an existing ARC Toolkit recording and optional transition log, then emits
trajectory CSV, per-action aggregate CSV, SVG overlay, and a short summary.

No API calls are made.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from inspect_recording_grids import (  # type: ignore
    PALETTE,
    cell_char,
    extract_frame_object,
    extract_grid,
    get_value,
    is_wrapped_grid,
)


DEFAULT_OUT_DIR = Path("runs/ls20-traj")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ls20 block trajectory from a recording.")
    parser.add_argument("recording", nargs="?", help="recording JSONL path; defaults to newest under recordings/")
    parser.add_argument(
        "--transitions",
        default=None,
        help="transition JSONL path; defaults to newest runs/transitions-*.jsonl if present",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output directory")
    args = parser.parse_args()

    root = Path.cwd()
    recording_path = resolve_recording(root, args.recording)
    transitions_path = resolve_transitions(root, args.transitions)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_recording_frames(recording_path)
    if not frames:
        raise RuntimeError(f"No parseable grid frames found in {recording_path}")

    action_by_turn = load_action_map(transitions_path) if transitions_path else {}
    target = target_centroid(frames[0]["grid"])
    rows = build_trajectory_rows(frames, action_by_turn, target)
    per_action_rows = build_per_action_rows(rows)

    trajectory_path = out_dir / "trajectory.csv"
    per_action_path = out_dir / "per_action.csv"
    svg_path = out_dir / "trajectory.svg"
    summary_path = out_dir / "summary.txt"

    write_trajectory_csv(trajectory_path, rows)
    write_per_action_csv(per_action_path, per_action_rows)
    write_trajectory_svg(
        svg_path,
        frames[0]["grid"],
        rows,
        target=target,
        title=f"ls20 trajectory: {recording_path.name}",
    )

    summary = render_summary(
        recording_path=recording_path,
        transitions_path=transitions_path,
        rows=rows,
        per_action_rows=per_action_rows,
        target=target,
    )
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    return 0


def resolve_recording(root: Path, supplied: str | None) -> Path:
    if supplied:
        path = Path(supplied)
        return path if path.is_absolute() else root / path

    candidates = list((root / "recordings").glob("**/*.jsonl"))
    if not candidates:
        raise FileNotFoundError("No recordings/**/*.jsonl files found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_transitions(root: Path, supplied: str | None) -> Path | None:
    if supplied:
        path = Path(supplied)
        return path if path.is_absolute() else root / path

    candidates = list((root / "runs").glob("transitions-*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_recording_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            frame_obj = extract_frame_object(obj)
            grid = extract_grid(frame_obj)
            if grid is None:
                continue

            # Keep these imports/use-sites live: extraction helpers are shared
            # with inspect_recording_grids.py by design.
            _ = is_wrapped_grid([grid])
            _ = cell_char(grid[0][0] if grid and grid[0] else 0)

            frames.append(
                {
                    "line_number": line_number,
                    "raw": frame_obj,
                    "grid": grid,
                    "score": safe_float(get_value(frame_obj, "score", default=0)),
                    "state": stringify_state(get_value(frame_obj, "state", default="")),
                    "levels_completed": safe_int(get_value(frame_obj, "levels_completed", default=0)),
                }
            )
    return frames


def load_action_map(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(path)

    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = str(record.get("action", "") or "")
            if action == "RESET":
                continue
            try:
                turn = int(record["turn"])
            except (KeyError, TypeError, ValueError):
                continue
            out[turn] = action
    return out


def build_trajectory_rows(
    frames: list[dict[str, Any]],
    action_by_turn: dict[int, str],
    target: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_grid: list[list[int]] | None = None
    prev_score = 0.0

    for turn, frame in enumerate(frames):
        grid = frame["grid"]
        block = block_centroid(grid)
        score = safe_float(frame.get("score", 0))
        score_delta = score - prev_score if turn > 0 else 0.0
        changed_cells = count_changed_cells(prev_grid, grid) if prev_grid is not None else 0
        counts = color_counts(grid)

        distance = None
        if block is not None and target is not None:
            distance = abs(block[0] - target[0]) + abs(block[1] - target[1])

        rows.append(
            {
                "turn": turn,
                "action": action_by_turn.get(turn, ""),
                "block_y": block[0] if block is not None else None,
                "block_x": block[1] if block is not None else None,
                "target_y": target[0] if target is not None else None,
                "target_x": target[1] if target is not None else None,
                "distance": distance,
                "score": score,
                "score_delta": score_delta,
                "changed_cells": changed_cells,
                "color_11_count": counts.get(11, 0),
                "color_3_count": counts.get(3, 0),
            }
        )
        prev_grid = grid
        prev_score = score

    return rows


def build_per_action_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: dict[str, list[dict[str, float | bool]]] = defaultdict(list)

    prev_by_turn = {int(row["turn"]): row for row in rows}
    for row in rows:
        turn = int(row["turn"])
        if turn == 0:
            continue
        action = str(row.get("action") or "")
        if not action:
            continue
        prev = prev_by_turn.get(turn - 1)
        if prev is None:
            continue
        if row.get("block_y") is None or prev.get("block_y") is None:
            continue

        dy = float(row["block_y"]) - float(prev["block_y"])
        dx = float(row["block_x"]) - float(prev["block_x"])
        distance_change = None
        if row.get("distance") is not None and prev.get("distance") is not None:
            distance_change = float(row["distance"]) - float(prev["distance"])
        no_movement = dy == 0 and dx == 0
        decreased = distance_change is not None and distance_change < 0
        samples[action].append(
            {
                "dy": dy,
                "dx": dx,
                "distance_change": distance_change if distance_change is not None else 0.0,
                "has_distance": distance_change is not None,
                "no_movement": no_movement,
                "decreased": decreased,
            }
        )

    out: list[dict[str, Any]] = []
    for action in sorted(samples):
        items = samples[action]
        distance_items = [item for item in items if item["has_distance"]]
        n = len(items)
        out.append(
            {
                "action": action,
                "n": n,
                "avg_block_dy": mean(item["dy"] for item in items),
                "avg_block_dx": mean(item["dx"] for item in items),
                "avg_distance_change": mean(item["distance_change"] for item in distance_items)
                if distance_items
                else None,
                "frac_decreased_distance": mean(1.0 if item["decreased"] else 0.0 for item in distance_items)
                if distance_items
                else None,
                "frac_no_movement": mean(1.0 if item["no_movement"] else 0.0 for item in items),
            }
        )
    return out


def connected_components(grid: list[list[int]], color: int) -> list[list[tuple[int, int]]]:
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    visited = [[False] * width for _ in range(height)]
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(len(grid[r])):
            if visited[r][c] or grid[r][c] != color:
                continue
            stack = [(r, c)]
            component: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= height or cc < 0 or cc >= len(grid[rr]):
                    continue
                if visited[rr][cc] or grid[rr][cc] != color:
                    continue
                visited[rr][cc] = True
                component.append((rr, cc))
                stack.extend([(rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)])
            components.append(component)
    return components


def block_centroid(grid: list[list[int]]) -> tuple[float, float] | None:
    components = connected_components(grid, color=12)
    if not components:
        return None
    comp = max(components, key=len)
    ys = [r for r, _ in comp]
    xs = [c for _, c in comp]
    return (sum(ys) / len(ys), sum(xs) / len(xs))


def target_centroid(grid: list[list[int]]) -> tuple[float, float] | None:
    components = connected_components(grid, color=9)
    upper = [
        comp
        for comp in components
        if max(r for r, _ in comp) <= 30 and len(comp) <= 12
    ]
    if not upper:
        return None
    comp = min(upper, key=len)
    ys = [r for r, _ in comp]
    xs = [c for _, c in comp]
    return (sum(ys) / len(ys), sum(xs) / len(xs))


def count_changed_cells(prev: list[list[int]], curr: list[list[int]]) -> int:
    height = max(len(prev), len(curr))
    changed = 0
    for r in range(height):
        prev_row = prev[r] if r < len(prev) else []
        curr_row = curr[r] if r < len(curr) else []
        width = max(len(prev_row), len(curr_row))
        for c in range(width):
            pv = prev_row[c] if c < len(prev_row) else None
            cv = curr_row[c] if c < len(curr_row) else None
            if pv != cv:
                changed += 1
    return changed


def color_counts(grid: list[list[int]]) -> Counter[int]:
    return Counter(value for row in grid for value in row)


def write_trajectory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "turn",
        "action",
        "block_y",
        "block_x",
        "target_y",
        "target_x",
        "distance",
        "score",
        "score_delta",
        "changed_cells",
        "color_11_count",
        "color_3_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def write_per_action_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "action",
        "n",
        "avg_block_dy",
        "avg_block_dx",
        "avg_distance_change",
        "frac_decreased_distance",
        "frac_no_movement",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def write_trajectory_svg(
    path: Path,
    grid: list[list[int]],
    rows: list[dict[str, Any]],
    *,
    target: tuple[float, float] | None,
    title: str,
) -> None:
    cell = 10
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    svg_width = width * cell
    svg_height = height * cell + 26
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        f'<text x="4" y="17" fill="#e5e7eb" font-family="monospace" font-size="12">{escape_xml(title)}</text>',
    ]
    y_offset = 26
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            color = PALETTE.get(value, "#64748b")
            lines.append(
                f'<rect x="{c * cell}" y="{y_offset + r * cell}" width="{cell}" height="{cell}" fill="{color}"/>'
            )

    points = [
        (int(row["turn"]), float(row["block_y"]), float(row["block_x"]))
        for row in rows
        if row.get("block_y") is not None and row.get("block_x") is not None
    ]
    denom = max(len(points) - 1, 1)
    for i, (turn, y, x) in enumerate(points):
        color = gradient_blue_to_red(i / denom)
        lines.append(
            f'<circle cx="{(x + 0.5) * cell:.2f}" cy="{y_offset + (y + 0.5) * cell:.2f}" r="3.2" '
            f'fill="{color}" stroke="#111827" stroke-width="0.6">'
            f"<title>turn {turn}: ({y:.1f}, {x:.1f})</title></circle>"
        )

    if target is not None:
        ty, tx = target
        cx = (tx + 0.5) * cell
        cy = y_offset + (ty + 0.5) * cell
        lines.append(
            f'<text x="{cx:.2f}" y="{cy + 4:.2f}" text-anchor="middle" '
            'fill="#facc15" stroke="#111827" stroke-width="0.8" '
            'font-family="monospace" font-size="16">*</text>'
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_summary(
    *,
    recording_path: Path,
    transitions_path: Path | None,
    rows: list[dict[str, Any]],
    per_action_rows: list[dict[str, Any]],
    target: tuple[float, float] | None,
) -> str:
    first_block = first_non_empty_block(rows)
    last_block = last_non_empty_block(rows)
    start_distance = first_non_empty_distance(rows)
    end_distance = last_non_empty_distance(rows)

    dy = dx = None
    if first_block is not None and last_block is not None:
        dy = last_block[0] - first_block[0]
        dx = last_block[1] - first_block[1]

    distance_delta = None
    if start_distance is not None and end_distance is not None:
        distance_delta = end_distance - start_distance

    ranked = sorted(
        per_action_rows,
        key=lambda row: (
            none_to_low(row.get("frac_decreased_distance")),
            -abs(float(row.get("avg_distance_change") or 0)),
        ),
        reverse=True,
    )

    lines = [
        f"recording: {recording_path}",
        f"transitions: {transitions_path if transitions_path else 'none'}",
        f"frames: {len(rows)}",
        f"target: (y, x) = {format_point(target)}",
        f"block start: {format_point(first_block)} at turn {first_block_turn(rows)}",
        f"block end:   {format_point(last_block)} at turn {last_block_turn(rows)}",
        f"net displacement: dy={format_signed(dy)} dx={format_signed(dx)} {direction_phrase(dy, dx)}",
        f"distance start -> end: {format_number(start_distance)} -> {format_number(end_distance)} (delta={format_signed(distance_delta)})",
        "per-action effects (top 5 by frac_decreased_distance):",
    ]
    for row in ranked[:5]:
        lines.append(
            "  "
            f"{row['action']}: "
            f"n={row['n']} "
            f"avg_dy={format_signed(row.get('avg_block_dy'))} "
            f"avg_dx={format_signed(row.get('avg_block_dx'))} "
            f"avg_d={format_signed(row.get('avg_distance_change'))} "
            f"frac_decreased={format_number(row.get('frac_decreased_distance'))}"
        )
    return "\n".join(lines) + "\n"


def first_non_empty_block(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    for row in rows:
        if row.get("block_y") is not None:
            return (float(row["block_y"]), float(row["block_x"]))
    return None


def last_non_empty_block(rows: list[dict[str, Any]]) -> tuple[float, float] | None:
    for row in reversed(rows):
        if row.get("block_y") is not None:
            return (float(row["block_y"]), float(row["block_x"]))
    return None


def first_non_empty_distance(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        if row.get("distance") is not None:
            return float(row["distance"])
    return None


def last_non_empty_distance(rows: list[dict[str, Any]]) -> float | None:
    for row in reversed(rows):
        if row.get("distance") is not None:
            return float(row["distance"])
    return None


def first_block_turn(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        if row.get("block_y") is not None:
            return str(row["turn"])
    return "n/a"


def last_block_turn(rows: list[dict[str, Any]]) -> str:
    for row in reversed(rows):
        if row.get("block_y") is not None:
            return str(row["turn"])
    return "n/a"


def direction_phrase(dy: Any, dx: Any) -> str:
    if dy is None and dx is None:
        return ""
    parts: list[str] = []
    if dy is not None:
        if dy < 0:
            parts.append(f"moved UP {abs(dy):.1f} rows")
        elif dy > 0:
            parts.append(f"moved DOWN {abs(dy):.1f} rows")
    if dx is not None:
        if dx < 0:
            parts.append(f"moved LEFT {abs(dx):.1f} cols")
        elif dx > 0:
            parts.append(f"moved RIGHT {abs(dx):.1f} cols")
    return "-> " + ", ".join(parts) if parts else ""


def mean(values: Iterable[float | int | bool]) -> float:
    nums = [float(value) for value in values]
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def stringify_state(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("name") or value.get("value") or value)
    return str(value).split(".")[-1]


def csv_value(value: Any) -> str | int:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.1f}"
    return value


def format_point(point: tuple[float, float] | None) -> str:
    if point is None:
        return "n/a"
    return f"({point[0]:.1f}, {point[1]:.1f})"


def format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def format_signed(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) < 0.005:
        number = 0.0
    return f"{number:+.2f}".rstrip("0").rstrip(".")


def none_to_low(value: Any) -> float:
    if value is None:
        return -1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def gradient_blue_to_red(t: float) -> str:
    t = max(0.0, min(1.0, t))
    # Blue (#2563eb) to red (#ef4444).
    start = (37, 99, 235)
    end = (239, 68, 68)
    rgb = tuple(round(start[i] + (end[i] - start[i]) * t) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
