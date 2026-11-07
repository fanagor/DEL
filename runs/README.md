# Paper Runs

Run paths group experiments by controller, task and varied parameters:

```text
runs/<model>/<task>/<run_id>/
    config.yaml
    final_model_checkpoint.pt
    evaluation.json
```

The selection contains 69 completed runs: 36 arithmetic DFST runs, 6 universal DFST runs and 27 DFST_Softmax runs.

The release removes tracking identities and disables tracking in the checkpoint configuration. Model parameters, optimizer state and random-generator state retain the values from the selected final training step. `evaluation.json` contains only the final-step result and records the sanitized checkpoint’s SHA-256 checksum. The plotting notebooks verify this checksum before reading a result.

`config.yaml` specifies resolved hidden dimensions and initial-state settings. Intentional nulls remain: `embed_dim: null` selects fixed one-hot embeddings, `data.max_steps: null` removes the execution limit, and `initialization.checkpoint: null` selects ordinary initialization. A numeric embedding dimension would change which parameters are learned.

The configurations retain the scientific training settings and disable tracking by default. To retrain, use:

```sh
del train --config runs/DFST/add2/warmup_25000_steps_250000/config.yaml run.wandb.enabled=false run.name=add2-reproduction
```

Load released weights with the same loader as training checkpoints:

```python
from differentiable_exact_learning.training import load_run
model, expert, config, saved = load_run(
    "runs/DFST/add2/warmup_25000_steps_250000", device="cpu"
)
```

The public figure selection lives in [paper/figures.yaml](../paper/figures.yaml).
Maintainers can import training outputs with `scripts/export_paper_runs.py --manifest /private/path/selection.yaml`.
The private manifest supplies source paths and tracking identities for validation.
Public artifacts contain no tracking identities; model parameters, optimizer state
and random-generator state remain available for analysis and continuation.
