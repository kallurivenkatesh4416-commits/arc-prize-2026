"""Offline ARC-AGI-3 controller as an ``arc_agi_3.Agent`` subclass.

The SDK owns the game loop (it calls ``choose_action`` per turn and handles
HTTP transport via ``do_action_request``). This file implements a probe-then-
policy controller that:

1. Returns ``RESET`` when the game is in ``NOT_PLAYED`` / ``GAME_OVER``.
2. Spends a small budget cycling directional actions and clicking object
   centroids while recording what changes (``PROBE`` phase).
3. Switches to a score-delta policy that prefers actions / ACTION6 coords
   that produced score deltas or cell changes during the probe (``PLAY`` phase).

The ACTION6 no-op filter from commit ``b9b4bd2`` is preserved: clicks whose
probe transition produced ``score_delta == 0 and changed_cells == 0`` are
NOT replayed during play.

NOTE on Kaggle's "no internet" rule: this agent uses the SDK's
``do_action_request`` HTTP calls against the ARC-AGI-3 evaluation API. ARC's
own API endpoint should be permitted under Kaggle's evaluation policy, but
external LLM APIs (OpenAI/Anthropic/etc.) are not — this controller is fully
offline w.r.t. external models. Verify against the latest Kaggle evaluation
runtime details before relying on this assumption for the final submission.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from typing import Any

import arc_agi_3
from arc_agi_3 import GameAction, GameState

from .explorer import (
    CLICK_ACTION,
    DIRECTIONAL,
    RESET_ACTION,
    UNDO_ACTION,
    _centroid,
    detect_objects,
)
from .world_model import (
    Transition,
    WorldModel,
    grid_of,
    score_of,
)

log = logging.getLogger(__name__)

ACTION6_LABEL = re.compile(r"ACTION6\((\d+),(\d+)\)")
_PROBE_DIRECTIONALS = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
]


class OfflineControllerAgent(arc_agi_3.Agent):
    """Probe-then-policy agent for ARC-AGI-3.

    State machine inside :py:meth:`choose_action`:

    - ``RESET_PENDING`` → returns ``GameAction.RESET`` whenever the game is
      ``NOT_PLAYED`` / ``GAME_OVER``; flips into ``PROBE`` on the first
      such reset.
    - ``PROBE`` → for the first ``PROBE_BUDGET`` productive turns, cycle
      directionals then click object centroids, recording transitions in
      :class:`WorldModel`.
    - ``PLAY`` → score-delta-driven policy: rank candidate non-click
      actions, replay productive ACTION6 coordinates, fall back to
      exploration when stuck.
    """

    MAX_ACTIONS = 240        # play-phase budget; SDK enforces this
    PROBE_BUDGET = 24        # turns spent probing before play
    NO_PROGRESS_CLICK_THRESHOLD = 1
    MAX_PROBE_CLICKS = 8

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.world = WorldModel(game_id=self.game_id)
        self._phase = "RESET_PENDING"
        self._probe_steps = 0
        self._probe_dir_idx = 0
        self._probe_click_idx = 0
        self._probe_click_coords: list[tuple[int, int]] = []
        self._action_cursor = 0
        self._click_cursor = 0
        self._no_progress_turns = 0
        self._last_grid: list[list[int]] | None = None
        self._last_score = 0
        # Cache of productive ACTION6 coords discovered during probing,
        # populated lazily on the first PLAY turn.
        self._action6_coords_cache: list[tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Agent contract
    # ------------------------------------------------------------------

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[Any], latest_frame: Any) -> GameAction:
        # 1. RESET handling — must come before any world-model update because
        #    the previous frame (if any) may belong to a finished game.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if self._phase == "RESET_PENDING":
                self._phase = "PROBE"
            return self._reasoned(GameAction.RESET, "reset before play")

        # 2. Update world model from prev → latest using the previous frame
        #    in ``frames`` (the SDK appends the latest frame to ``frames``).
        #
        #    NOTE: ``prev.action_input.id`` is the SDK ``GameAction`` enum.
        #    Keep this controller on that enum end-to-end so Agent.take_action
        #    sees the exact type it expects, especially for RESET card_id setup.
        if frames and len(frames) >= 2:
            prev = frames[-2]
            try:
                last_action_id = prev.action_input.id
                last_action_data = prev.action_input.data or {}
            except AttributeError:
                last_action_id = None
                last_action_data = {}
            if last_action_id is not None and getattr(last_action_id, "name", "") != "RESET":
                label = self._action_label(last_action_id, last_action_data)
                self.world.update(prev, label, latest_frame)

        self._last_grid = grid_of(latest_frame)
        self._last_score = latest_frame.score

        # 3. Phase: PROBE → cycle directionals, then click object centroids.
        if self._phase == "PROBE" and self._probe_steps < self.PROBE_BUDGET:
            self._probe_steps += 1
            return self._next_probe_action(latest_frame)

        # 4. Phase: PLAY → score-delta policy + ACTION6 replay.
        if self._phase != "PLAY":
            self._phase = "PLAY"
            self._action6_coords_cache = self._rank_action6_coords()
        return self._play_action(latest_frame)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reasoned(self, action: GameAction, reason: str) -> GameAction:
        if action.is_simple():
            action.reasoning = reason
        return action

    def _action_label(self, action_id: Any, data: dict[str, Any]) -> str:
        name = getattr(action_id, "name", str(action_id))
        if name == "ACTION6" and isinstance(data, dict):
            return f"ACTION6({data.get('x')},{data.get('y')})"
        return name

    def _legal_action_names(self, latest_frame: Any) -> list[str]:
        avail = getattr(latest_frame, "available_actions", None) or []
        return [a.name for a in avail]

    # ----- PROBE phase -------------------------------------------------

    def _next_probe_action(self, latest_frame: Any) -> GameAction:
        """Cycle directionals, then click up to ``MAX_PROBE_CLICKS`` centroids.

        We deliberately probe on the *current* frame's state so we don't
        burn turns on actions that aren't legal right now.
        """
        legal = set(self._legal_action_names(latest_frame))

        # Phase 3a: directionals — try each legal directional once.
        while self._probe_dir_idx < len(_PROBE_DIRECTIONALS):
            candidate = _PROBE_DIRECTIONALS[self._probe_dir_idx]
            self._probe_dir_idx += 1
            if not legal or candidate.name in legal:
                return self._reasoned(candidate, f"probe directional {candidate.name}")

        # Phase 3b: ACTION6 centroids — populate coord queue lazily.
        if not self._probe_click_coords:
            grid = grid_of(latest_frame)
            objects = detect_objects(grid)
            seen: set[tuple[int, int]] = set()
            for obj in objects:
                coord = _centroid(obj)
                if coord in seen:
                    continue
                seen.add(coord)
                self._probe_click_coords.append(coord)
                if len(self._probe_click_coords) >= self.MAX_PROBE_CLICKS:
                    break

        if (
            (not legal or CLICK_ACTION in legal)
            and self._probe_click_idx < len(self._probe_click_coords)
        ):
            x, y = self._probe_click_coords[self._probe_click_idx]
            self._probe_click_idx += 1
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {
                "phase": "probe",
                "kind": "centroid_click",
                "x": int(x),
                "y": int(y),
            }
            return action

        # Probe queue exhausted earlier than budget — flip to play.
        self._phase = "PLAY"
        self._action6_coords_cache = self._rank_action6_coords()
        return self._play_action(latest_frame)

    # ----- PLAY phase --------------------------------------------------

    def _transition_rank(self, t: Transition) -> tuple[int, int]:
        return (t.score_delta, t.changed_cells)

    def _rank_non_click_actions(self, legal: list[str]) -> list[str]:
        candidates: list[tuple[float, float, str]] = []
        for action_name, transitions in self.world.action_effects.items():
            if action_name == CLICK_ACTION or action_name in (RESET_ACTION, UNDO_ACTION):
                continue
            if legal and action_name not in legal:
                continue
            if not transitions:
                continue
            avg_delta = sum(t.score_delta for t in transitions) / len(transitions)
            avg_changed = sum(t.changed_cells for t in transitions) / len(transitions)
            candidates.append((avg_delta, avg_changed, action_name))
        candidates.sort(reverse=True)
        return [name for _, _, name in candidates]

    def _rank_action6_coords(self) -> list[tuple[int, int]]:
        """Productive ACTION6 coords ranked by (score_delta, changed_cells).

        Preserves the no-op filter from commit b9b4bd2: coords whose probe
        transition produced ``score_delta == 0 and changed_cells == 0`` are
        skipped entirely.
        """
        by_coord: dict[tuple[int, int], tuple[int, int]] = {}
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

    def _play_action(self, latest_frame: Any) -> GameAction:
        legal_names = self._legal_action_names(latest_frame)
        legal_set = set(legal_names) if legal_names else set(DIRECTIONAL + [CLICK_ACTION])

        coords = self._action6_coords_cache or []

        # Update no-progress counter from the most recent transition.
        last_transitions = self._last_transition()
        if last_transitions is not None:
            if last_transitions.score_delta > 0 or last_transitions.changed_cells > 0:
                self._no_progress_turns = 0
            else:
                self._no_progress_turns += 1

        # Prefer replaying productive ACTION6 coords either right away or
        # whenever we've stalled.
        if (
            CLICK_ACTION in legal_set
            and coords
            and (self._no_progress_turns >= self.NO_PROGRESS_CLICK_THRESHOLD or self._click_cursor == 0)
        ):
            x, y = coords[self._click_cursor % len(coords)]
            self._click_cursor += 1
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {
                "phase": "play",
                "kind": "replay_productive_click",
                "x": int(x),
                "y": int(y),
            }
            return action

        ranked_actions = self._rank_non_click_actions(list(legal_set))
        exploratory = [a for a in DIRECTIONAL if a in legal_set and a not in ranked_actions]
        candidates = ranked_actions + exploratory

        if candidates:
            name = candidates[self._action_cursor % len(candidates)]
            self._action_cursor += 1
            action = getattr(GameAction, name)
            return self._reasoned(action, f"play: ranked candidate {name}")

        if CLICK_ACTION in legal_set and coords:
            x, y = coords[self._click_cursor % len(coords)]
            self._click_cursor += 1
            action = GameAction.ACTION6
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {"phase": "play", "kind": "fallback_click"}
            return action

        # Last resort — random legal action.
        fallback_name = random.choice(list(legal_set) or DIRECTIONAL)
        action = getattr(GameAction, fallback_name)
        if action.is_simple():
            action.reasoning = "play: random fallback"
        elif action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"phase": "play", "kind": "random_fallback"}
        return action

    def _last_transition(self) -> Transition | None:
        for transitions in self.world.action_effects.values():
            if transitions:
                last = transitions[-1]
                if last.turn == self.world.turn - 1:
                    return last
        return None


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the offline ARC-AGI-3 controller on one game.")
    parser.add_argument("--game", required=True, help="game_id, e.g. ls20")
    parser.add_argument("--card", required=True, help="card_id from the ARC-AGI-3 API")
    parser.add_argument(
        "--root-url",
        default=os.environ.get("ARC_ROOT_URL", "https://three.arcprize.org"),
        help="ARC-AGI-3 API root URL (verify against docs.arcprize.org/create-agent)",
    )
    parser.add_argument("--record", action="store_true", help="record frames for replay")
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--out", default=None, help="write final scorecard JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    agent = OfflineControllerAgent(
        card_id=args.card,
        game_id=args.game,
        agent_name="offline-controller",
        ROOT_URL=args.root_url,
        record=args.record,
        tags=args.tags,
    )
    agent.main()

    payload: dict[str, Any] = {"game_id": args.game}
    try:
        scorecard = agent.get_scorecard()  # type: ignore[attr-defined]
        if hasattr(scorecard, "model_dump"):
            payload["scorecard"] = scorecard.model_dump()
        else:
            payload["scorecard"] = scorecard
    except Exception as exc:  # noqa: BLE001
        log.warning("get_scorecard() failed: %s", exc)

    text = json.dumps(payload, indent=2, default=str)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    _cli()
