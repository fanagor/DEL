# Training Recipes

Choose a recipe by the computation you want to learn. The names `add2` and `mult2` refer to binary addition and multiplication. The suffix `10` selects decimal operands. The `universal` task trains one controller to interpret multiple binary Turing machines encoded on the grid.

## Choose a Starting Configuration

The table pairs each starting recipe with a saved paper configuration. Recipes use relative paths, automatic device selection, and local logging. To match a particular paper run, use the saved configuration in the last column, including the original checkpoint and evaluation schedule.

| Recipe | Hidden dimensions | Dataset size | Training steps | Warmup steps | Related run |
|---|---:|---:|---:|---:|---|
| [DFST/add2.yaml](DFST/add2.yaml) | 11 | 25 | 500,000 | 50,000 | [warmup_50000_steps_500000](../runs/DFST/add2/warmup_50000_steps_500000/config.yaml) |
| [DFST/mult2.yaml](DFST/mult2.yaml) | 32 | 100 | 500,000 | 50,000 | [warmup_50000_steps_500000](../runs/DFST/mult2/warmup_50000_steps_500000/config.yaml) |
| [DFST/add10.yaml](DFST/add10.yaml) | 19 | 250 | 4,000,000 | 50,000 | [warmup_50000_steps_4000000](../runs/DFST/add10/warmup_50000_steps_4000000/config.yaml) |
| [DFST/mult10.yaml](DFST/mult10.yaml) | 108 | 10,000 | 4,000,000 | 50,000 | [warmup_50000_steps_4000000](../runs/DFST/mult10/warmup_50000_steps_4000000/config.yaml) |
| [DFST/universal.yaml](DFST/universal.yaml) | 15 | 75 | 500,000 | 0 | [data_75_steps_500000](../runs/DFST/universal/data_75_steps_500000/config.yaml) |
| [DFST_Softmax/add2.yaml](DFST_Softmax/add2.yaml) | 11 | 25 | 100,000 | 0 | [hidden_11_data_25](../runs/DFST_Softmax/add2/hidden_11_data_25/config.yaml) |
| [DFST_Softmax/mult2.yaml](DFST_Softmax/mult2.yaml) | 32 | 100 | 100,000 | 0 | [hidden_32_data_100](../runs/DFST_Softmax/mult2/hidden_32_data_100/config.yaml) |
| [DFST_Softmax/universal.yaml](DFST_Softmax/universal.yaml) | 15 | 75 | 100,000 | 0 | [hidden_15_data_75](../runs/DFST_Softmax/universal/hidden_15_data_75/config.yaml) |
| [RNN/add2.yaml](RNN/add2.yaml) | 32 | 25 | 1,000,000 | 0 | [RNN add2 history](../analyses/training_dynamics/RNN/add2/config.yaml) |
| [RNN/mult2.yaml](RNN/mult2.yaml) | 64 | 100 | 1,000,000 | 0 | [RNN mult2 history](../analyses/training_dynamics/RNN/mult2/config.yaml) |
| [GRU/add2.yaml](GRU/add2.yaml) | 32 | 25 | 1,000,000 | 100,000 | [GRU add2 history](../analyses/training_dynamics/GRU/add2/config.yaml) |
| [GRU/mult2.yaml](GRU/mult2.yaml) | 64 | 100 | 1,000,000 | 100,000 | [GRU mult2 history](../analyses/training_dynamics/GRU/mult2/config.yaml) |

The four recurrent recipes use the seeds from the [training-dynamics comparison](../notebooks/training_dynamics.ipynb), clip gradients at 0.1, and decay the learning rate to `1e-7`. Binary-addition RNN training starts at 0.01 with orthogonal recurrent weights. The other three recipes use a peak of 0.003, identity recurrence with zero input weights, and a learnable all-ones initial state. Both GRUs warm up for 100,000 updates, then hold the peak until update 900,000 for addition or 300,000 for multiplication.

All DFST and Softmax DFST recipes save checkpoints every 10,000 updates and log training measurements every 100. The following table shows when evaluation runs and which scan the evaluator uses.

| Recipe | Evaluation interval | Final evaluation | Evaluation scan |
|---|---:|---|---|
| `DFST/add2.yaml` | 250,000 | Yes | parallel |
| `DFST/mult2.yaml` | 250,000 | Yes | parallel |
| `DFST/add10.yaml` | 250,000 | Yes | parallel |
| `DFST/mult10.yaml` | 250,000 | Yes | parallel |
| `DFST/universal.yaml` | Final only | Yes | parallel |
| `DFST_Softmax/add2.yaml` | 50,000 | Yes | sequential |
| `DFST_Softmax/mult2.yaml` | 50,000 | Yes | parallel |
| `DFST_Softmax/universal.yaml` | 50,000 | Yes | sequential |

DFST learning rates decay from 0.01 for binary arithmetic or 0.001 for decimal arithmetic and the universal learning task to 1e-7. DFST_Softmax uses a constant learning rate of 0.1.

## Run a Recipe

Run commands from the repository root after installing the package with `pip install -e .`.

First inspect the resolved settings for a binary-addition DFST.

```sh
del config --config configs/DFST/add2.yaml
```

To train with those settings, change the subcommand to `train`.

```sh
del train --config configs/DFST/add2.yaml
```

By default, DEL derives the run name from the configuration and expert. Set `run.name` for a readable prefix or pass `--run-dir` to choose the directory. Training prints the output directory, which contains the saved configuration, loss history, checkpoints and evaluations. Recipe runs start from scratch and include final evaluation. Full training and length-generalization evaluation can take substantial compute.

For a short CPU check, reduce both the training budget and the evaluation ceiling as follows.

```sh
del train --config configs/DFST/add2.yaml \
  training.steps=20 training.warmup_steps=0 training.batch_size=4 \
  training.log_every=5 training.checkpoint_every=10 \
  evaluation.every=0 evaluation.max_length=3 evaluation.samples=2 \
  evaluation.snapshot.enabled=false run.device=cpu run.name=smoke
```

`python -m differentiable_exact_learning` accepts the same arguments as `del`.

## Change Settings

DEL loads package defaults, applies the recipe, and finally applies dotted command-line overrides. Some null values carry specific meanings. For example, `model.embed_dim: null` selects frozen one-hot inputs, `data.max_steps: null` allows expert execution until halt, and `initialization.checkpoint: null` starts training from scratch.

The following example previews a Softmax DFST with 22 hidden coordinates and 75 training demonstrations.

```sh
del config --config configs/DFST_Softmax/add2.yaml \
  model.hidden_size=22 data.size=75
```

To vary two settings together, use a shell loop. This example trains a universal DFST at each combination of dataset size and update budget.

```sh
for size in 75 150 225; do
  for steps in 250000 500000; do
    del train --config configs/DFST/universal.yaml \
      data.size=$size training.steps=$steps
  done
done
```

Arithmetic schedule sweeps vary `training.steps` and `training.warmup_steps`. Softmax sweeps vary `model.hidden_size` and `data.size`. The total training budget determines the learning-rate schedule, so changing the budget defines a different experiment. When shortening a run, set a warmup smaller than the selected budget. `data.seed` controls sampling, and `run.model_seed` controls model initialization.

To enable W&B, install `pip install -e '.[tracking]'`, authenticate with your W&B service, and pass `run.wandb.enabled=true run.wandb.project=<project>`. Local training writes metrics without a tracking account.

## Reproduce a Selected Run

Exact experiment configurations live alongside the weights in `runs/<model>/<task>/<run_id>/config.yaml`. Choose the saved configuration to reproduce a specific sweep cell.

```sh
del train --config runs/DFST/add2/warmup_50000_steps_500000/config.yaml \
  run.device=auto run.wandb.enabled=false run.output_dir=outputs/reproductions
```

The released universal Softmax runs encode scaling-A programs with state $m$ as the halt state. The matching recipe and all nine released run configurations set `data.scaling_a_terminal_step=false`. The checkpoint loader recovers the same setting from the original expert fingerprint. Ordinary universal DFST runs use a separate terminal state.

See [paper/figures.yaml](../paper/figures.yaml) for figure selections and the selected run directories for saved training configurations. Adding or editing a recipe does not change those selections or the released artifacts.
