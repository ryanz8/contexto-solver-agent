# AGENTS.md

## Overview

Automated solver for the [Contexto](https://contexto.me) daily word game. Uses GloVe embeddings + a hybrid exploit/explore strategy (UCB cluster sampling, Boltzmann neighbor selection, temperature decay) to guess the secret word via the Contexto API.

## Project Structure

```
contexto_solver/          # Core package
  __init__.py             # Lazy imports (play_game_and_record, load_*_game_id)
  config.py               # All hyperparameters and file paths (single source of truth)
  embedding.py            # GloveEmbedding: load, whiten, normalize, fp16
  api.py                  # ContextoAPI + DistanceScoreMapper (isotonic regression)
  solver.py               # HybridSolver: KMeans clustering, UCB, Boltzmann sampling
  pipeline.py             # play_game_and_record(): main orchestration function
  io.py                   # File I/O: game IDs, bad words, results, timestamps
main.py                   # Minimal single-game entry point
automation_script.py      # Full orchestrator: CLI args, batch mode, RESULTS.md generation
view_results.ipynb        # Jupyter notebook for ad-hoc result exploration
```

## System information
This project is running on a Windows system.

A venv has been created for this project. Activate it with:
```
.\.venv\Scripts\activate
```

Before attempting to run any code from this project, make sure to activate this venv.

## How to Run

```bash
# Single game (next game ID auto-incremented)
python main.py

# Automation (play + regenerate RESULTS.md)
python automation_script.py

# Batch mode
python automation_script.py --start 100 --end 110

# Reset
python automation_script.py --reset-state     # clear IDs + results
python automation_script.py --reset-docs       # reset RESULTS.md only
```

**Prerequisite**: `glove.6B.300d.txt` in repo root (~822MB uncompressed). Download from https://nlp.stanford.edu/data/glove.6B.zip. Gitignored.

## Dependencies

`scikit-learn`, `scipy`, `requests`, `numpy` (see `requirements.txt`). Dev extras in `pyproject.toml`: `ipykernel`, `jinja2`, `matplotlib`, `pandas`.

## State Files

| File | Purpose |
|------|---------|
| `current_game_id.txt` | Tracks next game to play (auto-incremented) |
| `last_successful_game_id.txt` | Last game that found the answer |
| `bad_words.json` | Blacklisted words (404s, zero-score) — persists across games |
| `results/game_{id}_{timestamp}.jsonl` | One JSON object per game with full trajectory |
| `RESULTS.md` | Auto-generated report (stats + trajectory tables) |

## Pipeline Flow

1. Read `current_game_id.txt`, increment to get next game ID
2. Load GloVe embeddings -> whiten (center + remove top-3 PCs) -> normalize -> fp16
3. Filter vocab: regex `^[a-z]{2,20}$`, cap at `VOCAB_SIZE` (80k)
4. Load `bad_words.json`, exclude from candidate pool
5. **Early phase**: Probe category words (`EARLY_PROBES`) + farthest-point seeds
6. **Main loop**: `HybridSolver.propose_next()` -> query API -> `solver.update()` -> repeat until hit or vocab exhausted
7. **Stagnation**: After 5 steps without improvement, jump to farthest word from current best
8. Save JSONL result, update game IDs, persist bad words

## Key Design Decisions

- **No test suite** — validation is done via batch runs and trajectory inspection
- **No logging module** — uses print statements throughout
- **No async** — single-threaded, API latency dominates runtime
- **Git as database** — GitHub Actions commits results directly to the repo
- **Isotonic regression** for distance-to-score mapping kicks in after 40 observations; log-prior fallback before that
- **Hit detection** is multi-strategy: `distance <= 1` OR `correct`/`is_answer` flags OR `rank == 1`
- **LRU cache** on `api.query()` (8192 entries) prevents duplicate HTTP calls
- **Neighbor cache** in solver is never invalidated (acceptable approximation)

## Configuration

All tunable parameters live in `contexto_solver/config.py`. Key ones:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `VOCAB_SIZE` | 80000 | Top-N GloVe words to keep |
| `KMEANS_K` | 512 | Cluster count (capped by vocab size) |
| `NEIGHBOR_K` | 64 | Local neighbor pool size |
| `INITIAL_TEMPERATURE` | 0.9 | Boltzmann sampling temperature |
| `TEMPERATURE_DECAY` | 0.97 | Per-step multiplicative decay |
| `MIN_TEMPERATURE` | 0.25 | Floor for temperature |
| `EXPLOIT_BASE` | 0.55 | Base probability of exploiting vs exploring |
| `UCB_ALPHA` | 1.3 | UCB exploration coefficient |
| `EARLY_PROBE_TURNS` | 8 | Number of category probe words |
| `RATE_LIMIT_SLEEP` | 0.1 | Seconds between API calls |

## Pitfalls

- **Whitening order matters**: mean-center BEFORE removing top PCs
- **Distance threshold**: Some Contexto servers return 0 for the answer, others return 1 — both are treated as hits
- **Temperature decays on every `update()` call**, including during seed phase — seeds cause faster early cooling
- **`bad_words.json` grows unboundedly** across games; may need periodic pruning
- **Game ID increments immediately** after a game ends, even if subsequent file operations fail
- **`vocab.index(w)` in pipeline.py seed phase** is O(n) per probe word — fine for ~8 probes but would be slow at scale

## GitHub Actions

Workflow at `.github/workflows/daily-contexto.yml`:
- Trigger: `workflow_dispatch` (cron is commented out)
- Caches GloVe file between runs (key: `glove-6B-300d-v1`)
- Runs `python automation_script.py`, then commits + pushes results
- 60-minute timeout, single concurrency group
