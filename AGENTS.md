# AI coding tools guidelines

- **Goal:** keep all reasoning and outputs in plain text, Git-tracked artifacts. Avoid opaque binaries or formats that mask diffs.
- **Privacy:** never paste access tokens or patient-like data. If any credentials are present in the environment, do not log them.
- **Reproducibility:** prefer deterministic seeds. Each experiment takes seeds from `config/experiments.yaml`.
- **Code style:** small, composable functions; explicit parameters; light comments when control flow is non-obvious.
- **Dependencies:** add new packages through `uv add ...` so `pyproject.toml` and `uv.lock` stay in sync.
- **Checks:** keep plots and CSV outputs under `results/`. Avoid committing massive artifacts; regenerate via `uv run python -m plaintext_sims.run_all`.
