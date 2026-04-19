"""Claude tool-use loop that plays ARC-AGI-3 games.

Per game:
  1. scripted probe seeds the world model (cheap facts Claude shouldn't pay to rediscover)
  2. Claude picks actions via tool use, informed by world_model.summary()
  3. budget guardrails fall back to random-from-probe if Claude stalls
  4. scorecard collected across all games and returned
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
    run_probe,
)
from .world_model import (
    WorldModel,
    available_actions,
    grid_of,
    is_done,
    score_of,
)

log = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-7"
DEFAULT_PER_GAME_WALL_SECONDS = 240
DEFAULT_PER_GAME_TURNS = 40
ACTION_REGEX = re.compile(r"(RESET|ACTION[1-7])", re.IGNORECASE)

SYSTEM_PROMPT = """You are playing an unknown turn-based puzzle game on a 2D grid.
You were not told the rules, the goal, or the controls. You must figure them out.

You see:
- A compact text representation of the current grid (. = empty; 1..9,a..f = colors)
- A list of legal actions for this turn
- Your own running theories about the game
- A summary of what each action has done so far

Your job each turn: pick ONE action that best tests a theory or makes progress.
When you learn something new, call update_theory. When a theory is falsified,
call invalidate_theory. Only call reset_game if the game is over or you are
certain you've reached a dead end.

Key heuristics:
- If score increased after an action, that action is good — try it again.
- If an action changed the grid, study the change; it's a clue.
- Actions with zero effect are probably invalid right now — skip them.
- ACTION6 takes (x, y) in 0..63; aim for visible objects, not empty cells.
"""

TOOLS = [
    {
        "name": "take_action",
        "description": "Take one action in the game.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"],
                },
                "x": {"type": "integer", "minimum": 0, "maximum": 63, "description": "Only for ACTION6."},
                "y": {"type": "integer", "minimum": 0, "maximum": 63, "description": "Only for ACTION6."},
                "reason": {"type": "string", "description": "One sentence: why this action tests a theory or advances play."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "update_theory",
        "description": "Add or replace a hypothesis about the game's goal or mechanics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["text"],
        },
    },
    {
        "name": "invalidate_theory",
        "description": "Mark a previous hypothesis as falsified, by index in the hypothesis list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0},
                "reason": {"type": "string"},
            },
            "required": ["index"],
        },
    },
    {
        "name": "reset_game",
        "description": "Restart the current game from scratch. Only use when game is over or stuck.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


@dataclass
class GameResult:
    game_id: str
    final_score: int
    turns_used: int
    wall_seconds: float
    hypotheses: list[str] = field(default_factory=list)
    finished: bool = False


@dataclass
class Scorecard:
    results: list[GameResult] = field(default_factory=list)
    raw: Any = None

    def to_dict(self) -> dict:
        return {
            "results": [r.__dict__ for r in self.results],
            "raw": self.raw,
        }


# -------- Anthropic client (lazy import) -----------------------------------

def _anthropic_client() -> Any:
    import anthropic  # type: ignore
    return anthropic.Anthropic()


# -------- Env step wrapper -------------------------------------------------

def _step(env: Any, action_name: str, *coords: int) -> Any:
    from arcengine import GameAction  # type: ignore
    action = getattr(GameAction, action_name)
    if coords:
        return env.step(action, *coords)
    return env.step(action)


# -------- Tool dispatch ----------------------------------------------------

def _dispatch_tool(
    tool_name: str,
    tool_input: dict,
    *,
    env: Any,
    world: WorldModel,
    state: dict,
) -> str:
    if tool_name == "take_action":
        action = str(tool_input.get("action", "")).upper()
        if action not in {*DIRECTIONAL, CLICK_ACTION, UNDO_ACTION, RESET_ACTION}:
            return f"error: unknown action {action!r}"
        coords = ()
        label = action
        if action == CLICK_ACTION:
            x = int(tool_input.get("x", 0))
            y = int(tool_input.get("y", 0))
            coords = (x, y)
            label = f"ACTION6({x},{y})"
        prev = state["last_frame"]
        try:
            if action == RESET_ACTION:
                next_frame = env.reset()
            else:
                next_frame = _step(env, action, *coords)
        except Exception as exc:  # noqa: BLE001
            return f"error: env rejected {label}: {exc}"
        t = world.update(prev, label, next_frame)
        state["last_frame"] = next_frame
        state["turns"] += 1
        if is_done(next_frame):
            state["done"] = True
        return (
            f"ok: {label}  Δscore={t.score_delta:+d}  cells_changed={t.changed_cells}  "
            f"done={is_done(next_frame)}"
        )

    if tool_name == "update_theory":
        idx = world.add_hypothesis(
            str(tool_input.get("text", "")),
            float(tool_input.get("confidence", 0.5)),
        )
        return f"ok: hypothesis #{idx} recorded"

    if tool_name == "invalidate_theory":
        ok = world.invalidate_hypothesis(int(tool_input.get("index", -1)))
        return "ok" if ok else "error: no such hypothesis"

    if tool_name == "reset_game":
        next_frame = env.reset()
        world.frame_history.append(next_frame)
        world.current_score = score_of(next_frame)
        state["last_frame"] = next_frame
        state["done"] = False
        return "ok: game reset"

    return f"error: unknown tool {tool_name!r}"


# -------- Fallback action picker -------------------------------------------

def _fallback_action(world: WorldModel, legal: list[str]) -> tuple[str, tuple[int, ...]]:
    """Pick the action with the best historical Δscore; break ties by change rate."""
    best: tuple[float, float, str] = (-1e9, -1e9, "")
    for key, trs in world.action_effects.items():
        if key == "ACTION6":
            continue
        if key not in legal:
            continue
        avg_delta = sum(t.score_delta for t in trs) / max(len(trs), 1)
        avg_change = sum(t.changed_cells for t in trs) / max(len(trs), 1)
        if (avg_delta, avg_change, key) > best:
            best = (avg_delta, avg_change, key)
    if best[2]:
        return best[2], ()
    return random.choice(legal or DIRECTIONAL), ()


def _fallback_step(env: Any, world: WorldModel, state: dict) -> None:
    legal = available_actions(state["last_frame"]) or DIRECTIONAL
    action, coords = _fallback_action(world, legal)
    try:
        next_frame = _step(env, action, *coords) if action != RESET_ACTION else env.reset()
    except Exception as exc:  # noqa: BLE001
        log.warning("fallback action %s failed: %s", action, exc)
        state["done"] = True
        return
    world.update(state["last_frame"], action, next_frame)
    state["last_frame"] = next_frame
    state["turns"] += 1
    if is_done(next_frame):
        state["done"] = True


# -------- Main per-game loop ----------------------------------------------

def play_game(
    arc: Any,
    game_id: str,
    *,
    max_turns: int = DEFAULT_PER_GAME_TURNS,
    max_seconds: int = DEFAULT_PER_GAME_WALL_SECONDS,
    model: str = CLAUDE_MODEL,
) -> GameResult:
    env = arc.make(game_id)

    # Phase 1: scripted probe seeds the world model
    world, probe_report = run_probe(env, game_id)
    log.info("probe complete: %s", probe_report.as_text().splitlines()[0])

    # Phase 2: Claude loop
    client = _anthropic_client()
    state = {"last_frame": env.reset(), "turns": 0, "done": False}
    world.current_score = score_of(state["last_frame"])
    wall_start = time.monotonic()
    messages: list[dict] = []

    static_primer = (
        "Scripted probe results (already applied to world model):\n\n"
        + probe_report.as_text()
    )

    consecutive_noops = 0

    while state["turns"] < max_turns and not state["done"]:
        if time.monotonic() - wall_start > max_seconds:
            log.info("%s: wall-clock budget exhausted", game_id)
            break

        user_turn = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": static_primer,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "CURRENT WORLD MODEL:\n\n" + world.summary(),
                },
                {
                    "type": "text",
                    "text": f"Legal actions right now: {available_actions(state['last_frame']) or 'unknown'}",
                },
            ],
        }
        messages.append(user_turn)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: Claude call failed (%s); falling back", game_id, exc)
            _fallback_step(env, world, state)
            messages.pop()  # drop the user turn we failed to answer
            continue

        assistant_blocks = response.content
        messages.append({"role": "assistant", "content": assistant_blocks})

        tool_uses = [b for b in assistant_blocks if getattr(b, "type", "") == "tool_use"]
        took_action = False
        tool_results: list[dict] = []

        if tool_uses:
            for block in tool_uses:
                result_text = _dispatch_tool(
                    block.name, dict(block.input), env=env, world=world, state=state
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
                )
                if block.name == "take_action":
                    took_action = True
            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool use — regex-extract an action from the text, per docs fallback.
            text = "".join(getattr(b, "text", "") for b in assistant_blocks)
            match = ACTION_REGEX.search(text.upper())
            if match:
                action = match.group(1)
                result = _dispatch_tool(
                    "take_action", {"action": action}, env=env, world=world, state=state
                )
                messages.append({"role": "user", "content": [{"type": "text", "text": f"(regex-parsed) {result}"}]})
                took_action = True
            else:
                log.debug("%s turn %d: no tool use and no action in text", game_id, state["turns"])

        if took_action:
            last_tr = None
            for trs in world.action_effects.values():
                if trs and (last_tr is None or trs[-1].turn > last_tr.turn):
                    last_tr = trs[-1]
            if last_tr and last_tr.score_delta == 0 and last_tr.changed_cells == 0:
                consecutive_noops += 1
            else:
                consecutive_noops = 0
        if consecutive_noops >= 5:
            log.info("%s: 5 consecutive no-ops, switching to fallback policy", game_id)
            while state["turns"] < max_turns and not state["done"] and time.monotonic() - wall_start <= max_seconds:
                _fallback_step(env, world, state)
            break

    return GameResult(
        game_id=game_id,
        final_score=world.current_score,
        turns_used=state["turns"],
        wall_seconds=time.monotonic() - wall_start,
        hypotheses=[h.text for h in world.goal_hypotheses],
        finished=state["done"],
    )


# -------- Top-level entry --------------------------------------------------

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


def run_competition(
    game_ids: list[str] | None = None,
    *,
    max_turns_per_game: int = DEFAULT_PER_GAME_TURNS,
    max_seconds_per_game: int = DEFAULT_PER_GAME_WALL_SECONDS,
    model: str = CLAUDE_MODEL,
) -> Scorecard:
    import arc_agi  # type: ignore

    arc = arc_agi.Arcade()
    if game_ids is None:
        game_ids = _list_games(arc)
    if not game_ids:
        raise RuntimeError("No game_ids provided and Arcade did not expose a game list.")

    scorecard = Scorecard()
    for gid in game_ids:
        log.info("== playing %s ==", gid)
        try:
            result = play_game(
                arc,
                gid,
                max_turns=max_turns_per_game,
                max_seconds=max_seconds_per_game,
                model=model,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("game %s crashed: %s", gid, exc)
            result = GameResult(game_id=gid, final_score=0, turns_used=0, wall_seconds=0.0)
        scorecard.results.append(result)
        log.info("%s: score=%s turns=%s", gid, result.final_score, result.turns_used)

    try:
        scorecard.raw = arc.get_scorecard()
    except Exception as exc:  # noqa: BLE001
        log.warning("get_scorecard() failed: %s", exc)
    return scorecard


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", action="append", help="game_id; pass multiple times for multiple games")
    parser.add_argument("--budget", type=int, default=DEFAULT_PER_GAME_WALL_SECONDS, help="seconds per game")
    parser.add_argument("--turns", type=int, default=DEFAULT_PER_GAME_TURNS, help="Claude turns per game")
    parser.add_argument("--model", default=CLAUDE_MODEL)
    parser.add_argument("--out", default=None, help="write scorecard JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except ImportError:
        pass

    if not os.getenv("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY not set — Claude calls will fail")

    scorecard = run_competition(
        game_ids=args.game,
        max_turns_per_game=args.turns,
        max_seconds_per_game=args.budget,
        model=args.model,
    )
    payload = scorecard.to_dict()
    print(json.dumps(payload, indent=2, default=str))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)


if __name__ == "__main__":
    _cli()
