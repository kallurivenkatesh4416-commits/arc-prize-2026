"""Offline ARC-AGI-3 controller using the current ``arc_agi`` Toolkit.

The important lifecycle rule is that one ``arc_agi.Arcade`` instance should
own scorecard creation, environment creation, action submission, and scorecard
closing. The Toolkit keeps the HTTP session, cookies, and game guid threaded
through its ``EnvironmentWrapper``; splitting scorecard open/close into a
separate process can produce unscored runs.
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
    legal_actions,
    run_probe,
    step_action,
    unwrap_step_result,
)
from .transition_log import TransitionLogger
from .world_model import Transition, WorldModel, is_done, score_of

log = logging.getLogger(__name__)

ACTION6_LABEL = re.compile(r"ACTION6\((\d+),(\d+)\)")


@dataclass
class GameRunSummary:
    game_id: str
    score: float
    actions: int
    levels_completed: int = 0
    state: str = ""
    won: bool = False
    elapsed_seconds: float = 0.0
    probe_actions: int = 0
    scorecard_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CompetitionResult:
    scorecard_id: str
    runs: list[GameRunSummary] = field(default_factory=list)
    scorecard: Any = None
    transitions_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "scorecard_id": self.scorecard_id,
            "runs": [run.to_dict() for run in self.runs],
        }
        if self.transitions_path:
            payload["transitions_path"] = self.transitions_path
        if self.scorecard is not None:
            payload["scorecard"] = _dump_model(self.scorecard)
        return payload


class OfflineControllerAgent:
    """Probe-then-policy controller for an ``arc_agi`` EnvironmentWrapper."""

    MAX_ACTIONS = 240
    PROBE_BUDGET = 24
    MAX_PROBE_CLICKS = 8
    NO_PROGRESS_CLICK_THRESHOLD = 5

    def __init__(
        self,
        *,
        game_id: str,
        env: Any | None = None,
        scorecard_id: str | None = None,
        max_actions: int | None = None,
        transition_logger: TransitionLogger | None = None,
    ) -> None:
        self.game_id = game_id
        self.env = env
        self.scorecard_id = scorecard_id
        self.max_actions = max_actions or self.MAX_ACTIONS
        self.transition_logger = transition_logger
        self.world = WorldModel(game_id=game_id)
        self.probe_report = None
        self._action_cursor = 0
        self._click_cursor = 0
        self._no_progress_turns = 0
        self._action6_coords_cache: list[tuple[int, int]] | None = None

    def run(self, env: Any | None = None) -> GameRunSummary:
        env = env or self.env
        if env is None:
            raise ValueError("OfflineControllerAgent.run requires an EnvironmentWrapper.")
        self.env = env

        started_at = time.monotonic()
        initial_frame = unwrap_step_result(env.reset()).frame
        self.world.frame_history.append(initial_frame)
        self.world.current_score = score_of(initial_frame)

        current = initial_frame
        if not is_done(current) and self.max_actions > 0:
            self.world, self.probe_report, current = run_probe(
                env,
                self.game_id,
                world=self.world,
                initial_frame=current,
                probe_budget=min(self.PROBE_BUDGET, self.max_actions),
                max_probe_clicks=self.MAX_PROBE_CLICKS,
                on_transition=self._log_transition,
            )
            self._action6_coords_cache = self._rank_action6_coords()

        while self.world.turn < self.max_actions and not is_done(current):
            action_name, data, reasoning = self._play_action(current)
            label = _action_label(action_name, data)
            started_at = time.monotonic()
            outcome = step_action(env, action_name, data=data, reasoning=reasoning)
            elapsed_ms = (time.monotonic() - started_at) * 1000
            transition = self.world.update(current, label, outcome.frame)
            self._log_transition("play", action_name, data, transition, outcome.frame, elapsed_ms)
            self._update_progress(transition)
            current = outcome.frame
            log.info(
                "%s - %s: count %s, score %s",
                self.game_id,
                label,
                self.world.turn,
                score_of(current),
            )
            if outcome.done:
                break

        return GameRunSummary(
            game_id=self.game_id,
            score=score_of(current),
            actions=self.world.turn,
            levels_completed=_int_attr(current, "levels_completed"),
            state=_state_name(getattr(current, "state", "")),
            won=_state_name(getattr(current, "state", "")) == "WIN",
            elapsed_seconds=time.monotonic() - started_at,
            probe_actions=self.probe_report and len(self.probe_report.directional_transitions) or 0,
            scorecard_id=self.scorecard_id,
        )

    def _log_transition(
        self,
        phase: str,
        action: str,
        data: dict[str, Any] | None,
        transition: Transition,
        next_frame: Any,
        elapsed_ms: float,
    ) -> None:
        if self.transition_logger is None:
            return
        self.transition_logger.log(
            game_id=self.game_id,
            phase=phase,
            action=action,
            data=data,
            transition=transition,
            next_frame=next_frame,
            elapsed_ms=elapsed_ms,
        )

    def _update_progress(self, transition: Transition) -> None:
        if transition.score_delta > 0 or transition.changed_cells > 0:
            self._no_progress_turns = 0
        else:
            self._no_progress_turns += 1

    def _transition_rank(self, t: Transition) -> tuple[float, int]:
        return (t.score_delta, t.changed_cells)

    def _rank_non_click_actions(self, legal: list[str]) -> list[str]:
        legal_set = set(legal)
        candidates: list[tuple[float, float, str]] = []
        for action_name, transitions in self.world.action_effects.items():
            if action_name == CLICK_ACTION or action_name in (RESET_ACTION, UNDO_ACTION):
                continue
            if legal_set and action_name not in legal_set:
                continue
            if not transitions:
                continue
            avg_delta = sum(t.score_delta for t in transitions) / len(transitions)
            avg_changed = sum(t.changed_cells for t in transitions) / len(transitions)
            candidates.append((avg_delta, avg_changed, action_name))
        candidates.sort(reverse=True)
        return [name for _, _, name in candidates]

    def _rank_action6_coords(self) -> list[tuple[int, int]]:
        by_coord: dict[tuple[int, int], tuple[float, int]] = {}
        for t in self.world.action_effects.get(CLICK_ACTION, []):
            if t.score_delta == 0 and t.changed_cells == 0:
                continue
            match = ACTION6_LABEL.fullmatch(t.action)
            if not match:
                continue
            coord = (int(match.group(1)), int(match.group(2)))
            rank = self._transition_rank(t)
            if coord not in by_coord or rank > by_coord[coord]:
                by_coord[coord] = rank
        ranked = sorted(by_coord.items(), key=lambda item: item[1], reverse=True)
        return [coord for coord, _ in ranked]

    def _play_action(self, latest_frame: Any) -> tuple[str, dict[str, Any] | None, Any]:
        legal = legal_actions(self.env, latest_frame) or list(DIRECTIONAL)
        legal_set = set(legal)
        coords = self._action6_coords_cache or []

        if (
            CLICK_ACTION in legal_set
            and coords
            and self._no_progress_turns >= self.NO_PROGRESS_CLICK_THRESHOLD
        ):
            x, y = coords[self._click_cursor % len(coords)]
            self._click_cursor += 1
            return (
                CLICK_ACTION,
                {"x": int(x), "y": int(y)},
                {"phase": "play", "kind": "replay_productive_click", "x": int(x), "y": int(y)},
            )

        ranked_actions = self._rank_non_click_actions(legal)
        exploratory = [a for a in DIRECTIONAL if a in legal_set and a not in ranked_actions]
        candidates = ranked_actions + exploratory

        if candidates:
            name = candidates[self._action_cursor % len(candidates)]
            self._action_cursor += 1
            return name, None, {"phase": "play", "kind": "ranked_candidate", "action": name}

        if CLICK_ACTION in legal_set and coords:
            x, y = coords[self._click_cursor % len(coords)]
            self._click_cursor += 1
            return CLICK_ACTION, {"x": int(x), "y": int(y)}, {"phase": "play", "kind": "fallback_click"}

        fallback = random.choice(legal or DIRECTIONAL)
        if fallback == CLICK_ACTION:
            data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}
            return fallback, data, {"phase": "play", "kind": "random_fallback"}
        return fallback, None, {"phase": "play", "kind": "random_fallback"}


def create_arcade(*, api_key: str | None = None, arc_base_url: str | None = None) -> Any:
    import arc_agi  # type: ignore

    kwargs: dict[str, Any] = {}
    if api_key:
        kwargs["arc_api_key"] = api_key
    if arc_base_url:
        kwargs["arc_base_url"] = arc_base_url
    try:
        from arc_agi import OperationMode  # type: ignore

        kwargs["operation_mode"] = OperationMode.ONLINE
    except Exception:  # noqa: BLE001
        pass
    return arc_agi.Arcade(**kwargs)


def make_env(
    arc: Any,
    *,
    game_id: str,
    scorecard_id: str,
    seed: int = 0,
    record: bool = False,
    include_frame_data: bool = True,
) -> Any:
    kwargs: dict[str, Any] = {
        "seed": seed,
        "scorecard_id": scorecard_id,
        "save_recording": record,
        "include_frame_data": include_frame_data,
    }
    while True:
        try:
            env = arc.make(game_id, **kwargs)
            if env is None:
                raise RuntimeError(f"arc.make({game_id!r}) returned None")
            return env
        except TypeError as exc:
            message = str(exc)
            removed = False
            for key in ("include_frame_data", "save_recording", "seed"):
                if key in kwargs and key in message:
                    kwargs.pop(key)
                    removed = True
                    break
            if not removed:
                raise


def list_game_ids(arc: Any) -> list[str]:
    envs = None
    for attr in ("get_environments", "list_games", "games", "available_games"):
        value = getattr(arc, attr, None)
        try:
            envs = value() if callable(value) else value
        except Exception:  # noqa: BLE001
            continue
        if envs:
            break
    if not envs:
        raise RuntimeError("Could not list ARC environments from Arcade.")

    ids: list[str] = []
    for env_info in envs:
        if isinstance(env_info, str):
            ids.append(env_info)
            continue
        if isinstance(env_info, dict):
            game_id = env_info.get("game_id") or env_info.get("id") or env_info.get("name")
        else:
            game_id = (
                getattr(env_info, "game_id", None)
                or getattr(env_info, "id", None)
                or getattr(env_info, "name", None)
            )
        if game_id:
            ids.append(str(game_id))
    return ids


def run_competition(
    game_ids: list[str] | None,
    *,
    api_key: str | None = None,
    arc_base_url: str | None = None,
    source_url: str | None = None,
    tags: list[str] | None = None,
    opaque: dict[str, Any] | None = None,
    seed: int = 0,
    record: bool = False,
    max_actions: int | None = None,
    scorecard_id: str | None = None,
    transitions_out: str | None = None,
) -> CompetitionResult:
    arc = create_arcade(api_key=api_key or os.getenv("ARC_API_KEY"), arc_base_url=arc_base_url)
    if game_ids is None:
        game_ids = list_game_ids(arc)
    if not game_ids:
        raise RuntimeError("No ARC games available for this API key.")

    tags = tags or ["offline-controller", "probe-policy"]
    if scorecard_id is None:
        scorecard_id = arc.create_scorecard(source_url=source_url, tags=tags, opaque=opaque)

    transition_logger: TransitionLogger | None = None
    transitions_path = transitions_out
    if transitions_path is None:
        transitions_path = "runs/transitions-{scorecard_id}.jsonl"
    if transitions_path:
        transitions_path = transitions_path.format(
            scorecard_id=str(scorecard_id),
            card_id=str(scorecard_id),
        )
        transition_logger = TransitionLogger(transitions_path)

    result = CompetitionResult(scorecard_id=str(scorecard_id), transitions_path=transitions_path)
    try:
        for game_id in game_ids:
            env = make_env(
                arc,
                game_id=game_id,
                scorecard_id=str(scorecard_id),
                seed=seed,
                record=record,
            )
            controller = OfflineControllerAgent(
                game_id=game_id,
                env=env,
                scorecard_id=str(scorecard_id),
                max_actions=max_actions,
                transition_logger=transition_logger,
            )
            try:
                result.runs.append(controller.run())
            except Exception as exc:  # noqa: BLE001
                log.exception("game %s failed: %s", game_id, exc)
                result.runs.append(GameRunSummary(game_id=game_id, score=0, actions=0, error=str(exc)))
    finally:
        if transition_logger is not None:
            transition_logger.close()
        try:
            result.scorecard = arc.close_scorecard(str(scorecard_id))
        except Exception as exc:  # noqa: BLE001
            log.warning("close_scorecard(%s) failed: %s", scorecard_id, exc)
            try:
                result.scorecard = arc.get_scorecard(str(scorecard_id))
            except Exception as get_exc:  # noqa: BLE001
                log.warning("get_scorecard(%s) failed: %s", scorecard_id, get_exc)
    return result


def _action_label(action_name: str, data: dict[str, Any] | None) -> str:
    if action_name == CLICK_ACTION and data:
        return f"ACTION6({data.get('x')},{data.get('y')})"
    return action_name


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _int_attr(value: Any, name: str, default: int = 0) -> int:
    try:
        return int(getattr(value, name, default) or default)
    except (TypeError, ValueError):
        return default


def _state_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value).split(".")[-1] if value is not None else ""


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the offline ARC-AGI-3 controller via arc_agi.Arcade.")
    parser.add_argument("--game", action="append", required=True, help="game_id; repeat for multiple games")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=OfflineControllerAgent.MAX_ACTIONS)
    parser.add_argument("--record", action="store_true", help="ask the Toolkit to save a recording")
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--scorecard-id", default=None, help="existing Toolkit scorecard id; normally omit")
    parser.add_argument(
        "--transitions-out",
        default=None,
        help="write transition JSONL here; supports {scorecard_id}",
    )
    parser.add_argument(
        "--arc-base-url",
        "--root-url",
        dest="arc_base_url",
        default=os.environ.get("ARC_BASE_URL", "https://three.arcprize.org"),
    )
    parser.add_argument("--out", default=None, help="write result JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:  # noqa: BLE001
        pass

    opaque = {"runner": "agent.offline_controller", "games": args.game}
    result = run_competition(
        args.game,
        api_key=os.getenv("ARC_API_KEY"),
        arc_base_url=args.arc_base_url,
        source_url=args.source_url,
        tags=args.tags,
        opaque=opaque,
        seed=args.seed,
        record=args.record,
        max_actions=args.max_actions,
        scorecard_id=args.scorecard_id,
        transitions_out=args.transitions_out,
    )

    payload = result.to_dict()
    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    _cli()
