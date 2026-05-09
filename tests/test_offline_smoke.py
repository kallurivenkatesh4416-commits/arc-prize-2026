"""Runtime-free smoke test for the offline controller.

Stands up a tiny mock arc_agi-style environment (no network, no real package)
and runs `play_game_offline` end-to-end to catch import errors, API drift in
WorldModel/explorer, and obvious policy bugs before a Kaggle attempt.

Run: python -m tests.test_offline_smoke
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

# Make the repo importable when run as `python -m tests.test_offline_smoke`
# from the project root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stub `arcengine.GameAction` before agent modules import it. Mirrors the
# real attribute access pattern: getattr(GameAction, "ACTION1") etc.
arcengine = types.ModuleType("arcengine")


class _GameAction:
    RESET = "RESET"
    ACTION1 = "ACTION1"
    ACTION2 = "ACTION2"
    ACTION3 = "ACTION3"
    ACTION4 = "ACTION4"
    ACTION5 = "ACTION5"
    ACTION6 = "ACTION6"
    ACTION7 = "ACTION7"


arcengine.GameAction = _GameAction
sys.modules.setdefault("arcengine", arcengine)


class FakeEnv:
    """Deterministic 4x4 game.

    - ACTION1 always paints cell (0,0) and grants +1 score.
    - ACTION6(1,1) toggles a cell and grants +2 score the first time.
    - All other actions are no-ops.
    - Done after 12 successful steps.
    """

    def __init__(self) -> None:
        self.size = 4
        self._reset_state()

    def _reset_state(self) -> None:
        self.grid = [[0] * self.size for _ in range(self.size)]
        self.score = 0
        self.done = False
        self._steps = 0
        self._reward_used = False
        self.legal = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"]

    def reset(self) -> dict:
        self._reset_state()
        return self._frame()

    def step(self, action: str, *coords: int) -> dict:
        if self.done:
            return self._frame()
        if action == "ACTION1":
            self.grid[0][0] = 3
            self.score += 1
        elif action == "ACTION6" and coords == (1, 1) and not self._reward_used:
            self.grid[1][1] = 5
            self.score += 2
            self._reward_used = True
        # other actions: pure no-op
        self._steps += 1
        if self._steps >= 12:
            self.done = True
        return self._frame()

    def _frame(self) -> dict:
        return {
            "grid": [row[:] for row in self.grid],
            "score": self.score,
            "done": self.done,
            "available_actions": list(self.legal),
        }


class FakeArcade:
    def make(self, game_id: str) -> FakeEnv:
        return FakeEnv()

    def get_scorecard(self) -> dict:
        return {"games": ["smoke"]}


def main() -> int:
    from agent.offline_controller import play_game_offline

    arc = FakeArcade()
    result = play_game_offline(arc, "smoke", max_turns=20, max_seconds=10)

    print(f"game_id      = {result.game_id}")
    print(f"final_score  = {result.final_score}")
    print(f"turns_used   = {result.turns_used}")
    print(f"finished     = {result.finished}")
    print(f"action6_seen = {result.action6_coords}")

    failures: list[str] = []
    if result.game_id != "smoke":
        failures.append(f"game_id mismatch: {result.game_id!r}")
    if result.turns_used < 3:
        failures.append(
            f"turns_used {result.turns_used} < 3; controller did not exercise multi-turn policy"
        )
    if result.final_score < 1:
        failures.append(
            f"final_score {result.final_score} < 1; "
            "policy never repeated the +1 ACTION1 reward"
        )

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
