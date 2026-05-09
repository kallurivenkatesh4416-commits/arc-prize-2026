"""Runtime-free smoke test for the offline controller (SDK-native shape).

Drives `OfflineControllerAgent.choose_action(frames, latest_frame)` directly
with synthesized `FrameData` objects so we never touch the network. The
controller subclasses `arc_agi_3.Agent`, but we never call `agent.main()`
(which would issue HTTP) — only the policy-decision method.

Run: python -m tests.test_offline_smoke

Skips gracefully when `arc_agi_3` is unavailable (e.g. Python <3.12), so
this is safe to run in any CI matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo importable when run as `python -m tests.test_offline_smoke`
# from the project root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_frame(
    *,
    state,
    score: int,
    grid,
    available,
    last_action=None,
    last_data=None,
    game_id: str = "smoke",
):
    """Build a `FrameData` mirroring what the SDK loop would hand to choose_action.

    `grid` is the 2D array; we wrap it in a 3D list-of-lists-of-lists as the
    real SDK does (`frame.frame[0]` is the active grid).
    """
    from arc_agi_3 import FrameData, ActionInput

    action_input_kwargs: dict = {}
    if last_action is not None:
        action_input_kwargs["id"] = last_action
    if last_data is not None:
        action_input_kwargs["data"] = last_data
    action_input = ActionInput(**action_input_kwargs) if action_input_kwargs else ActionInput()

    return FrameData(
        game_id=game_id,
        frame=[grid],
        state=state,
        score=score,
        action_input=action_input,
        available_actions=list(available),
    )


def _instantiate_agent():
    """Construct the OfflineControllerAgent without triggering any HTTP."""
    from agent.offline_controller import OfflineControllerAgent

    # Agent.__init__ creates a requests.Session but does not call out; safe.
    return OfflineControllerAgent(
        card_id="test-card",
        game_id="smoke",
        agent_name="offline-controller-smoke",
        ROOT_URL="http://localhost:0",
        record=False,
        tags=["smoke"],
    )


def main() -> int:
    # Graceful skip if the real SDK is unavailable.
    try:
        from arc_agi_3 import GameAction, GameState  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: arc_agi_3 unavailable ({exc!r}); "
              "offline smoke needs Python >=3.12 with the real SDK installed.")
        return 0

    from arc_agi_3 import GameState
    from arc_agi_3 import GameAction
    agent = _instantiate_agent()

    failures: list[str] = []
    GRID_BLANK = [[0, 0, 0, 0] for _ in range(4)]
    GRID_PAINTED = [[3, 0, 0, 0]] + [[0, 0, 0, 0] for _ in range(3)]
    LEGAL = [
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
        GameAction.ACTION6,
    ]

    # ------------------------------------------------------------------
    # 1) RESET on NOT_PLAYED
    # ------------------------------------------------------------------
    frame_not_played = _make_frame(
        state=GameState.NOT_PLAYED, score=0, grid=GRID_BLANK, available=LEGAL,
    )
    action = agent.choose_action([frame_not_played], frame_not_played)
    if action is not GameAction.RESET:
        failures.append(f"NOT_PLAYED expected RESET, got {action!r}")

    # ------------------------------------------------------------------
    # 2) Probe phase: at least 3 DISTINCT non-RESET actions before play
    # ------------------------------------------------------------------
    probe_actions: list = []
    frames: list = [frame_not_played]
    score = 0
    grid = [row[:] for row in GRID_BLANK]
    for turn in range(8):
        latest = _make_frame(
            state=GameState.NOT_FINISHED, score=score, grid=grid, available=LEGAL,
        )
        frames.append(latest)
        chosen = agent.choose_action(frames, latest)
        probe_actions.append(chosen)

    distinct_non_reset = {a for a in probe_actions if a is not GameAction.RESET}
    if len(distinct_non_reset) < 3:
        failures.append(
            f"probe phase issued <3 distinct actions: {[a.name for a in distinct_non_reset]}"
        )

    # ------------------------------------------------------------------
    # 3) ACTION6 returned during probing must be complex with x,y data
    # ------------------------------------------------------------------
    action6_returns = [a for a in probe_actions if a is GameAction.ACTION6]
    if action6_returns:
        a6 = action6_returns[-1]
        if not a6.is_complex():
            failures.append("ACTION6 should report is_complex() == True")
        # Complex action coordinates live on action_data (a ComplexAction
        # pydantic model with x/y fields). The enum is a singleton, so
        # action_data reflects the most-recent set_data call.
        data = getattr(a6, "action_data", None)
        if data is None:
            failures.append("ACTION6 has no action_data after choose_action")
        else:
            x = getattr(data, "x", None)
            y = getattr(data, "y", None)
            if x is None or y is None:
                # Some builds expose dict-like data
                if hasattr(data, "model_dump"):
                    dumped = data.model_dump()
                    x = dumped.get("x")
                    y = dumped.get("y")
            if x is None or y is None:
                failures.append(f"ACTION6 action_data missing x/y: {data!r}")

    # ------------------------------------------------------------------
    # 4) After ACTION1 yields score_delta>0, the policy should eventually
    #    return ACTION1 again during play.
    # ------------------------------------------------------------------
    # Push enough "productive" turns to exit the probe phase. The agent's
    # PROBE_BUDGET is 24, so we feed plenty of turns where ACTION1 is the
    # only action that moves score.
    play_actions: list = []
    for turn in range(60):
        # The frame the agent now observes reflects the OUTCOME of its last
        # chosen action. We label `last_action` based on what the agent just
        # returned, so the world_model.update inside choose_action sees a
        # consistent (frames[-2].action_input -> latest) transition.
        prev_choice = play_actions[-1] if play_actions else None
        if prev_choice is GameAction.ACTION1:
            score += 1
            grid = [row[:] for row in GRID_PAINTED]
        # else: pure no-op; grid/score unchanged.

        latest = _make_frame(
            state=GameState.NOT_FINISHED,
            score=score,
            grid=grid,
            available=LEGAL,
            last_action=prev_choice,
        )
        frames.append(latest)
        chosen = agent.choose_action(frames, latest)
        play_actions.append(chosen)

        # Once we've left probe and seen ACTION1 repeated, we're done.
        if (
            getattr(agent, "_phase", None) == "PLAY"
            and chosen is GameAction.ACTION1
            and turn > agent.PROBE_BUDGET  # well past probe
        ):
            break

    if not any(a is GameAction.ACTION1 for a in play_actions[agent.PROBE_BUDGET:]):
        failures.append(
            "after ACTION1 reward, controller never returned ACTION1 again in play phase"
        )

    # ------------------------------------------------------------------
    # 5) RESET on GAME_OVER
    # ------------------------------------------------------------------
    frame_game_over = _make_frame(
        state=GameState.GAME_OVER, score=score, grid=grid, available=LEGAL,
    )
    frames.append(frame_game_over)
    chosen = agent.choose_action(frames, frame_game_over)
    if chosen is not GameAction.RESET:
        failures.append(f"GAME_OVER expected RESET, got {chosen!r}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"probe_actions   = {[a.name for a in probe_actions]}")
    print(f"distinct_probe  = {sorted(a.name for a in distinct_non_reset)}")
    print(f"play_seen_a1    = {any(a is GameAction.ACTION1 for a in play_actions)}")
    print(f"final_phase     = {getattr(agent, '_phase', '?')}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
