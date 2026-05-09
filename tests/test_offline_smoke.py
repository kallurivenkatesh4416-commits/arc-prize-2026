"""Runtime-free smoke test for the Toolkit-style offline controller.

Run: python -m tests.test_offline_smoke
"""
from __future__ import annotations

import sys
import json
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeAction:
    def __init__(self, name: str) -> None:
        self.name = name

    def is_complex(self) -> bool:
        return self.name == "ACTION6"


class FakeGameAction:
    pass


for _name in ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]:
    setattr(FakeGameAction, _name, FakeAction(_name))

sys.modules.setdefault("arcengine", types.SimpleNamespace(GameAction=FakeGameAction))


@dataclass
class FakeFrame:
    game_id: str
    frame: list[list[list[int]]]
    state: str
    score: int
    levels_completed: int
    available_actions: list[Any]


class FakeEnv:
    def __init__(self) -> None:
        self.game_id = "smoke"
        self.score = 0
        self.levels_completed = 0
        self.actions: list[str] = []
        self.action_space = [
            FakeGameAction.ACTION1,
            FakeGameAction.ACTION2,
            FakeGameAction.ACTION3,
            FakeGameAction.ACTION4,
            FakeGameAction.ACTION6,
        ]
        self.grid = [[0, 0, 0, 0] for _ in range(4)]
        self.grid[1][1] = 3

    def reset(self) -> FakeFrame:
        self.score = 0
        self.levels_completed = 0
        self.actions.append("RESET")
        return self._frame("NOT_FINISHED")

    def step(self, action: FakeAction, data: dict[str, Any] | None = None, reasoning: Any = None):
        self.actions.append(action.name)
        if action.name == "ACTION1":
            self.score += 1
            self.grid[0][0] = self.score
        elif action.name == "ACTION6" and data:
            self.grid[int(data["y"]) % 4][int(data["x"]) % 4] = 7
        state = "WIN" if self.score >= 3 else "NOT_FINISHED"
        if state == "WIN":
            self.levels_completed = 1
        return self._frame(state)

    def _frame(self, state: str) -> FakeFrame:
        return FakeFrame(
            game_id=self.game_id,
            frame=[[row[:] for row in self.grid]],
            state=state,
            score=self.score,
            levels_completed=self.levels_completed,
            available_actions=self.action_space,
        )


def main() -> int:
    from agent.offline_controller import OfflineControllerAgent
    from agent.transition_log import TransitionLogger

    env = FakeEnv()
    with tempfile.TemporaryDirectory() as tmpdir:
        transitions_path = Path(tmpdir) / "transitions-test-card.jsonl"
        logger = TransitionLogger(transitions_path)
        try:
            agent = OfflineControllerAgent(
                game_id="smoke",
                env=env,
                scorecard_id="test-card",
                max_actions=16,
                transition_logger=logger,
            )
            result = agent.run()
        finally:
            logger.close()

        records = [
            json.loads(line)
            for line in transitions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    failures: list[str] = []
    non_reset = [a for a in env.actions if a != "RESET"]
    distinct_probe = set(non_reset[:5])
    if len(distinct_probe) < 3:
        failures.append(f"probe phase issued too few distinct actions: {sorted(distinct_probe)}")
    if "ACTION1" not in non_reset[5:]:
        failures.append("play phase did not replay productive ACTION1")
    if result.score < 3 or not result.won:
        failures.append(f"expected fake env win with score >=3, got score={result.score} won={result.won}")
    if not records:
        failures.append("transition log was empty")
    else:
        required = {
            "turn",
            "phase",
            "action",
            "state",
            "score",
            "score_delta",
            "changed_cells",
            "levels_completed",
            "available_actions",
            "elapsed_ms",
        }
        missing = required - set(records[0])
        if missing:
            failures.append(f"transition record missing keys: {sorted(missing)}")

    if len(agent.world.distinct_grid_hashes) < 2:
        failures.append(
            f"world model recorded <2 distinct grid states: "
            f"{len(agent.world.distinct_grid_hashes)}"
        )
    if not agent.world.action_novelty:
        failures.append("world model recorded no per-action novelty hits")

    print(f"actions        = {env.actions}")
    print(f"distinct_probe = {sorted(distinct_probe)}")
    print(f"result         = {result.to_dict()}")
    print(f"transition_n   = {len(records)}")
    print(f"distinct_grids = {len(agent.world.distinct_grid_hashes)}")
    print(f"novelty        = {dict(agent.world.action_novelty)}")

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
