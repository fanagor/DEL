# GRU Runs

Store final binary-arithmetic artifacts under `<task>/<run_id>/`, with
`config.yaml`, `final_model_checkpoint.pt` and `evaluation.json` in each run.
For loss and ULG curves, store complete histories separately under
`analyses/training_dynamics/GRU/<task>/<run_id>/` and select those paths in
[training_dynamics.ipynb](../../notebooks/training_dynamics.ipynb).
