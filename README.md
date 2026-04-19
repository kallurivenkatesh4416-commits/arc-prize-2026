# arc-prize-2026

Submission for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) Kaggle competition.

A Claude-driven agent that plays interactive turn-based game environments with no instructions, no rules, and no stated goal. The agent explores, builds a world model, hypothesizes the goal, and tries to win.

## Structure

```
agent/
  explorer.py      scripted probe that maps actions to effects
  world_model.py   evolving theory of the game
  llm_agent.py     Claude tool-use loop
submission.ipynb   Kaggle entry point
requirements.txt
LICENSE            MIT
```

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env     # fill in ARC_API_KEY and ANTHROPIC_API_KEY
python -m agent.llm_agent --game ls20
```

## Run on Kaggle

1. Upload this repo as a Kaggle Dataset (or clone from GitHub in cell 1).
2. Create two Kaggle Secrets: `ARC_API_KEY`, `ANTHROPIC_API_KEY`.
3. Open `submission.ipynb` and click **Save & Run All**.
4. Scorecard is written to `/kaggle/working/scorecard.json`.

## License

MIT. See [LICENSE](LICENSE).
