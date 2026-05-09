"""Offline ARC-AGI-3 controller for prize-eligible evaluation.

This module deliberately avoids external model/API calls. It uses the scripted
probe as a cheap evidence pass, then plays with a small score-delta policy:
prefer actions that changed score, prefer actions that changed state, and
explicitly replay useful ACTION6 coordinates discovered during probing.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .explorer import (
    CLICK_ACTION,
    DIRECTIONAL,
    RESET_ACTION,
    UNDO_ACTION,
    ProbeReport,
    run_probe,
)
from .world_model import (
    Transition,
    WorldModel,
    available_actions,
    is_done,
    score_of,
)

log = logging.getLogger(__name__)

DEFAULT_PER_GAME_WALL_SECONDS = 240
DEFAULT_PER_GAME_TURNS = 240
ACTION6_LABEL = re.compile(r"ACTION6\((\d+),(\d+)\)")


@dataclass
class OfflineGameResult:
    game_id: str
    final_score: int
    turns_used: int
    wall_seconds: float
    finished: bool = False
    probe_actions: list[str] = field(default_factory=list)
    action6_coords: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class OfflineScorecard:
    results: list[OfflineGameResult] = field(default_factory=list)
    raw: Any = None

    def to_dict(self) -> dict:
        return {
            "results": [r.__dict__ for r in self.results],
            "raw": self.raw,
        }


def _step(env: Any, action_name: str, *coords: int) -> Any:
    from arcengine import GameAction  # type: ignore

    action = getattr(GameAction, action_name)
    if coords:
        return env.step(action, *coords)
    return env.step(action)


def _transition_rank(t: Transition) -> tuple[int, int]:
    return (t.score_delta, t.changed_cells)


def _rank_non_click_actions(world: WorldModel, legal: list[str]) -> list[str]:
    candidates: list[tuple[float, float, str]] = []
    for action, transitions in world.action_effects.items():
        if action == CLICK_ACTION or action in (RESET_ACTION, UNDO_ACTION):
            continue
        if legal and action not in legal:
            continue
        if not transitions:
            continue
        avg_delta = sum(t.score_delta for t in transitions) / len(transitions)
        avg_changed = sum(t.changed_cells for t in transitions) / len(transitions)
        candidates.append((avg_delta, avg_changed, action))
    candidates.sort(reverse=True)
    return [action for _, _, action in candidates]


def _rank_action6_coords(world: WorldModel, report: ProbeReport) -> list[tuple[int, int]]:
    by_coord: dict[tuple[int, int], tuple[int, int]] = {}
    for t in world.action_effects.get(CLICK_ACTION, []):
        match = ACTION6_LABEL.fullmatch(t.action)
        if not match:
            continue
        coord = (int(match.group(1)), int(match.group(2)))
        rank = _transition_rank(t)
        if coord not in by_coord or rank > by_coord[coord]:
            by_coord[coord] = rank

    for coord in report.action6_live_coords:
        by_coord.setdefault(coord, (0, 1))

    ranked = sorted(by_coord.items(), key=lambda item: item[1], reverse=True)
    return [coord for coord, _ in ranked]


def _choose_action(
    *,
    world: WorldModel,
    report: ProbeReport,
    legal: list[str],
    action6_coords: list[tuple[int, int]],
    action_cursor: int,
    click_cursor: int,
    no_progress_turns: int,
) -> tuple[str, tuple[int, ...], int, int]:
    legal_set = set(legal or report.legal_actions or DIRECTIONAL + [CLICK_ACTION])

    if CLICK_ACTION in legal_set and action6_coords and (no_progress_turns > 0 or click_cursor == 0):
        coord = action6_coords[click_cursor % len(action6_coords)]
        return CLICK_ACTION, coord, action_cursor, click_cursor + 1

    ranked_actions = _rank_non_click_actions(world, list(legal_set))
    exploratory = [a for a in DIRECTIONAL if a in legal_set and a not in ranked_actions]
    candidates = ranked_actions + exploratory

    if candidates:
        action = candidates[action_cursor % len(candidates)]
        return action, (), action_cursor + 1, click_cursor

    if CLICK_ACTION in legal_set and action6_coords:
        coord = action6_coords[click_cursor % len(action6_coords)]
        return CLICK_ACTION, coord, action_cursor, click_cursor + 1

    fallback = random.choice(list(legal_set) or DIRECTIONAL)
    return fallback, (), action_cursor + 1, click_cursor


def play_game_offline(
    arc: Any,
    game_id: str,
    *,
    max_turns: int = DEFAULT_PER_GAME_TURNS,
    max_seconds: int = DEFAULT_PER_GAME_WALL_SECONDS,
) -> OfflineGameResult:
    env = arc.make(game_id)
    wall_start = time.monotonic()

    world, report = run_probe(env, game_id)
    action6_coords = _rank_action6_coords(world, report)

    frame = env.reset()
    world.current_score = score_of(frame)
    turns = 0
    done = False
    action_cursor = 0
    click_cursor = 0
    no_progress_turns = 0

    while turns < max_turns and not done:
        if time.monotonic() - wall_start > max_seconds:
            log.info("%s: wall-clock budget exhausted", game_id)
            break

        legal = available_actions(frame) or report.legal_actions
        action, coords, action_cursor, click_cursor = _choose_action(
            world=world,
            report=report,
            legal=legal,
            action6_coords=action6_coords,
            action_cursor=action_cursor,
            click_cursor=click_cursor,
            no_progress_turns=no_progress_turns,
        )

        try:
            next_frame = _step(env, action, *coords)
        except Exception as exc:  # noqa: BLE001
            log.debug("%s: action %s%r failed: %s", game_id, action, coords, exc)
            no_progress_turns += 1
            if no_progress_turns >= 5 and CLICK_ACTION in (legal or []) and action6_coords:
                click_cursor += 1
            continue

        label = f"{action}({coords[0]},{coords[1]})" if action == CLICK_ACTION and coords else action
        transition = world.update(frame, label, next_frame)
        frame = next_frame
        turns += 1
        done = is_done(frame)

        if transition.score_delta > 0 or transition.changed_cells > 0:
            no_progress_turns = 0
        else:
            no_progress_turns += 1

    return OfflineGameResult(
        game_id=game_id,
        final_score=world.current_score,
        turns_used=turns,
        wall_seconds=time.monotonic() - wall_start,
        finished=done,
        probe_actions=report.legal_actions,
        action6_coords=action6_coords,
    )


def _list_games(arc: Any) -> list[str]:
    for attr in ("list_games", "games", "available_games"):
        value = getattr(arc, attr, None)
        if callable(value):
            try:
                return list(value())
            except Exception:  # noqa: BLE001
                continue
        if isinstance(value, (list, tuple)):
            return list(value)
    return []


def run_competition_offline(
    game_ids: list[str] | None = None,
    *,
    max_turns_per_game: int = DEFAULT_PER_GAME_TURNS,
    max_seconds_per_game: int = DEFAULT_PER_GAME_WALL_SECONDS,
) -> OfflineScorecard:
    import arc_agi  # type: ignore

    arc = arc_agi.Arcade()
    if game_ids is None:
        game_ids = _list_games(arc)
    if not game_ids:
        raise RuntimeError("No game_ids provided and Arcade did not expose a game list.")

    scorecard = OfflineScorecard()
    for gid in game_ids:
        log.info("== playing %s offline ==", gid)
        try:
            result = play_game_offline(
                arc,
                gid,
                max_turns=max_turns_per_game,
                max_seconds=max_seconds_per_game,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("offline game %s crashed: %s", gid, exc)
            result = OfflineGameResult(game_id=gid, final_score=0, turns_used=0, wall_seconds=0.0)
        scorecard.results.append(result)
        log.info("%s: score=%s turns=%s", gid, result.final_score, result.turns_used)

    try:
        scorecard.raw = arc.get_scorecard()
    except Exception as exc:  # noqa: BLE001
        log.warning("get_scorecard() failed: %s", exc)
    return scorecard


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the offline ARC-AGI-3 controller.")
    parser.add_argument("--game", action="append", help="game_id; pass multiple times for multiple games")
    parser.add_argument("--budget", type=int, default=DEFAULT_PER_GAME_WALL_SECONDS, help="seconds per game")
    parser.add_argument("--turns", type=int, default=DEFAULT_PER_GAME_TURNS, help="turns per game")
    parser.add_argument("--out", default=None, help="write scorecard JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    scorecard = run_competition_offline(
        game_ids=args.game,
        max_turns_per_game=args.turns,
        max_seconds_per_game=args.budget,
    )
    payload = scorecard.to_dict()
    print(json.dumps(payload, indent=2, default=str))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)


if __name__ == "__main__":
    _cli()
