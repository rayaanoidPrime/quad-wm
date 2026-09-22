# Contributing

## Small pull requests

Keep changes focused: one infrastructure or research decision per pull request when possible. Explain the motivation, the affected recipe/evaluation rule, and how the change was verified.

## Before opening a pull request

```bash
uv run python -m compileall -q src tests
uv run pytest -q
ruff check src tests
```

GPU, Slurm, MuJoCo-rendering, and GrandTour tests should be labelled explicitly because they are not guaranteed to run in GitHub Actions.

## Research changes

If a change affects a data split, normalization, horizon, reward, metric, or checkpoint format, update the relevant config/docs and add an entry under `docs/adr/` when the decision is non-obvious.

## Data and secrets

Do not commit raw datasets, checkpoints, W&B API keys, `.env` files, or generated run directories. Use external storage roots and reference them through environment variables.
