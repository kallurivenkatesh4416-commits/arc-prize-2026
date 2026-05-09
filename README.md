# arc-prize-2026

Submission workspace for the [ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) Kaggle competition.

The prize-eligible path is an offline controller that uses the current `arc_agi` Toolkit, a scripted probe, a world model, and a score-delta policy with no external model calls.

The online research path uses the ARC-AGI-3 SDK's builtin OpenAI-based `llm`
template for generating traces and designing better offline policies.

## Structure

```
agent/
  explorer.py            probe helpers and object detection
  world_model.py         evolving theory of the game
  llm_agent.py           legacy Claude loop (parked, not used; kept for reference)
  offline_controller.py  prize-eligible offline controller using arc_agi.Arcade
submission.ipynb         research notebook (uses SDK builtin LLM template via Swarm)
submission_offline.ipynb prize-safe offline notebook
tests/test_offline_smoke.py  runtime-free smoke test
requirements.txt
LICENSE                  MIT-0
```

## Run Locally

Requires Python 3.12+. The current `arc-agi` / `arcengine` packages do not support Python 3.10/3.11.

```powershell
py -3.12 -m venv .venv
& .venv\Scripts\python.exe -m pip install -r requirements.txt
cp .env.example .env     # fill in ARC_API_KEY (and OPENAI_API_KEY for the research path)
& .venv\Scripts\python.exe -m tests.test_offline_smoke
& .venv\Scripts\python.exe -m agent.offline_controller --game ls20 --record --out runs\ls20.json
```

The controller writes transition JSONL evidence to `runs/transitions-<scorecard_id>.jsonl` by default.

For the real API smoke from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_real_arc.ps1
```

## Run On Kaggle

1. Upload this repo as a Kaggle Dataset.
2. Open `submission_offline.ipynb` and click **Save & Run All**.
3. Scorecard is written to `/kaggle/working/scorecard.json`.

For online research runs only, use `submission.ipynb` with `OPENAI_API_KEY`
set. It uses the SDK's builtin `llm` template via `arc_agi_3.Swarm`.

## License

MIT-0. See [LICENSE](LICENSE).
