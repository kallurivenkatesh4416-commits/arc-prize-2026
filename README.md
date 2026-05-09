# arc-prize-2026

Submission workspace for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) Kaggle competition.

The prize-eligible path is an offline controller that uses a scripted probe, builds a world model, and follows a score-delta policy with no internet or external model calls.

The online research path uses the ARC-AGI-3 SDK's builtin OpenAI-based `llm`
template for generating traces and designing better offline policies.

## Structure

```
agent/
  explorer.py            pure helpers (object detection); probe loop now lives in offline_controller
  world_model.py         evolving theory of the game (FrameData accessors)
  llm_agent.py           legacy Claude loop (parked, not used; kept for reference)
  offline_controller.py  prize-eligible offline controller (arc_agi_3.Agent subclass)
submission.ipynb         research notebook (uses SDK builtin LLM template via Swarm)
submission_offline.ipynb prize-safe offline notebook
tests/test_offline_smoke.py  runtime-free smoke test
requirements.txt
LICENSE                  MIT-0
```

## Run locally

Requires Python 3.12+. The `arc-agi-3` and `arcengine` packages don't support 3.10/3.11.

```powershell
py -3.12 -m venv .venv
& .venv\Scripts\python.exe -m pip install -r requirements.txt
cp .env.example .env     # fill in ARC_API_KEY (and OPENAI_API_KEY for the research path)
& .venv\Scripts\python.exe -m agent.offline_controller --game ls20 --card <card_id>
```

## Run on Kaggle

1. Upload this repo as a Kaggle Dataset.
2. Open `submission_offline.ipynb` and click **Save & Run All**.
3. Scorecard is written to `/kaggle/working/scorecard.json`.

For online research runs only, use `submission.ipynb` with `OPENAI_API_KEY`
set. It uses the SDK's builtin `llm` template via `arc_agi_3.Swarm`.

## License

MIT-0. See [LICENSE](LICENSE).
