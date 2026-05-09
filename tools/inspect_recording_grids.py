"""Inspect ARC Toolkit recording grids without making API calls.

Examples:
    python tools/inspect_recording_grids.py
    python tools/inspect_recording_grids.py recordings/<card>/<game>.jsonl
    python tools/inspect_recording_grids.py --turns 0,32,64,96,128,last --out runs/grid-inspect
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PALETTE = {
    0: "#111827",
    1: "#f9fafb",
    2: "#ef4444",
    3: "#22c55e",
    4: "#3b82f6",
    5: "#eab308",
    6: "#a855f7",
    7: "#ec4899",
    8: "#14b8a6",
    9: "#f97316",
    10: "#94a3b8",
    11: "#84cc16",
    12: "#06b6d4",
    13: "#f43f5e",
    14: "#8b5cf6",
    15: "#facc15",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect grids in an ARC recording JSONL file.")
    parser.add_argument("recording", nargs="?", help="recording JSONL path; defaults to newest recording")
    parser.add_argument(
        "--turns",
        default="0,32,64,96,128,last",
        help="comma-separated turns to sample; supports last and pct:N, e.g. pct:50",
    )
    parser.add_argument("--out", default="runs/grid-inspect", help="directory for text/SVG output")
    parser.add_argument("--ascii-size", type=int, default=32, help="max rows/cols in terminal preview")
    args = parser.parse_args()

    root = Path.cwd()
    recording_path = Path(args.recording) if args.recording else newest_recording(root)
    if not recording_path.is_absolute():
        recording_path = root / recording_path
    if not recording_path.exists():
        raise FileNotFoundError(recording_path)

    frames = load_frames(recording_path)
    if not frames:
        raise RuntimeError(f"No grid frames found in {recording_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = resolve_turns(args.turns, len(frames))
    summary_lines = [
        f"recording: {recording_path}",
        f"frames_with_grids: {len(frames)}",
        f"sample_turns: {sample_indices}",
        "",
    ]

    for idx in sample_indices:
        frame = frames[idx]
        grid = frame["grid"]
        stats = grid_stats(grid)
        prefix = f"turn-{idx:04d}"
        svg_path = out_dir / f"{prefix}.svg"
        txt_path = out_dir / f"{prefix}.txt"

        ascii_grid = render_ascii(grid, max_size=args.ascii_size)
        write_svg(svg_path, grid, title=f"{recording_path.name} turn {idx}")
        txt_path.write_text(
            "\n".join(
                [
                    f"recording: {recording_path}",
                    f"turn: {idx}",
                    f"game_id: {frame.get('game_id', '')}",
                    f"state: {frame.get('state', '')}",
                    f"score: {frame.get('score', '')}",
                    f"levels_completed: {frame.get('levels_completed', '')}",
                    f"shape: {stats['height']}x{stats['width']}",
                    f"colors: {stats['colors']}",
                    "",
                    ascii_grid,
                    "",
                ]
            ),
            encoding="utf-8",
        )

        summary_lines.extend(
            [
                f"TURN {idx}",
                f"  game_id: {frame.get('game_id', '')}",
                f"  state: {frame.get('state', '')}",
                f"  score: {frame.get('score', '')}",
                f"  levels_completed: {frame.get('levels_completed', '')}",
                f"  shape: {stats['height']}x{stats['width']}",
                f"  colors: {stats['colors']}",
                f"  svg: {svg_path}",
                f"  txt: {txt_path}",
                "",
                ascii_grid,
                "",
            ]
        )

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"wrote {summary_path}")
    return 0


def newest_recording(root: Path) -> Path:
    candidates = list((root / "recordings").glob("**/*.jsonl"))
    candidates.extend(root.glob("*.recording.jsonl"))
    if not candidates:
        raise FileNotFoundError("No recordings/**/*.jsonl or *.recording.jsonl files found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_frames(path: Path) -> list[dict[str, Any]]:
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
            frames.append(
                {
                    "line_number": line_number,
                    "grid": grid,
                    "game_id": get_value(frame_obj, "game_id"),
                    "state": stringify_state(get_value(frame_obj, "state")),
                    "score": get_value(frame_obj, "score", default=0),
                    "levels_completed": get_value(frame_obj, "levels_completed", default=0),
                }
            )
    return frames


def extract_frame_object(obj: Any) -> Any:
    if isinstance(obj, dict):
        for key in ("data", "frame_data", "frameData", "observation", "obs"):
            value = obj.get(key)
            if isinstance(value, dict) and any(k in value for k in ("frame", "grid", "observation")):
                return value
        return obj
    return obj


def extract_grid(obj: Any) -> list[list[int]] | None:
    raw = get_value(obj, "grid", default=None)
    if raw is None:
        raw = get_value(obj, "frame", default=None)
    if raw is None:
        raw = get_value(obj, "observation", default=None)
    if raw is None:
        return None

    while is_wrapped_grid(raw):
        raw = raw[0]
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], list):
        return None

    grid: list[list[int]] = []
    for row in raw:
        if not isinstance(row, list):
            return None
        try:
            grid.append([int(value) for value in row])
        except (TypeError, ValueError):
            return None
    return grid


def is_wrapped_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
        and value[0]
        and isinstance(value[0][0], list)
    )


def get_value(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def stringify_state(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("name") or value.get("value") or value)
    return str(value).split(".")[-1]


def resolve_turns(spec: str, total: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token == "last":
            idx = total - 1
        elif token.startswith("pct:"):
            pct = float(token.split(":", 1)[1])
            idx = round((pct / 100.0) * (total - 1))
        else:
            idx = int(token)
        idx = max(0, min(total - 1, idx))
        if idx not in out:
            out.append(idx)
    return out


def grid_stats(grid: list[list[int]]) -> dict[str, Any]:
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    counts = Counter(value for row in grid for value in row)
    colors = " ".join(f"{color}:{count}" for color, count in sorted(counts.items()))
    return {"height": height, "width": width, "colors": colors}


def render_ascii(grid: list[list[int]], max_size: int = 32) -> str:
    if not grid:
        return "<empty>"
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    row_step = max(1, math.ceil(height / max_size))
    col_step = max(1, math.ceil(width / max_size))
    lines: list[str] = []
    for r in range(0, height, row_step):
        chars: list[str] = []
        row = grid[r]
        for c in range(0, width, col_step):
            value = row[c] if c < len(row) else 0
            chars.append(cell_char(value))
        lines.append("".join(chars))
    return "\n".join(lines)


def cell_char(value: int) -> str:
    if value == 0:
        return "."
    if 1 <= value <= 9:
        return str(value)
    letters = "abcdef"
    if 10 <= value <= 15:
        return letters[value - 10]
    return "?"


def write_svg(path: Path, grid: list[list[int]], title: str) -> None:
    cell = 10
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    svg_width = width * cell
    svg_height = height * cell + 24
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        f'<text x="4" y="16" fill="#e5e7eb" font-family="monospace" font-size="12">{escape_xml(title)}</text>',
    ]
    y_offset = 24
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            color = PALETTE.get(value, "#64748b")
            lines.append(
                f'<rect x="{c * cell}" y="{y_offset + r * cell}" width="{cell}" height="{cell}" fill="{color}"/>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
