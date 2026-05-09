# arc-prize-2026

Submission workspace for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) Kaggle competition.

The prize-eligible path is an offline controller that uses a scripted probe, builds a world model, and follows a score-delta policy with no internet or external model calls.

The Claude-driven agent remains as an online research harness for generating traces and designing better offline policies.

## Structure

```
agent/
  explorer.py      scripted probe that maps actions to effects
  world_model.py   evolving theory of the game
  llm_agent.py     Claude tool-use loop
  offline_controller.py  prize-eligible offline controller
submission.ipynb   online research notebook (Claude-driven; not prize-safe)
submission_offline.ipynb prize-safe offline notebook
requirements.txt
LICENSE            MIT-0
```

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env     # fill in ARC_API_KEY and ANTHROPIC_API_KEY
python -m agent.llm_agent --game ls20
python -m agent.offline_controller --game ls20
```

## Run on Kaggle

1. Upload this repo as a Kaggle Dataset.
2. Open `submission_offline.ipynb` and click **Save & Run All**.
3. Scorecard is written to `/kaggle/working/scorecard.json`.

For online research runs only, use `submission.ipynb` with `ANTHROPIC_API_KEY`
and internet enabled.

## License

MIT-0. See [LICENSE](LICENSE).
