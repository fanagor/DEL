# Decimal Addition by Metric Extraction

We trained a 19-dimensional DFST on 250 demonstrations for 4,000,000 updates before extraction. The [released run](../../../../../runs/DFST/add10/warmup_50000_steps_4000000) contains the original weights and training configuration. The [case configuration](config.yaml) identifies that checkpoint by checksum and records the settings for metric extraction.

## Inspect the Recovered Controller

Metric extraction groups hidden states observed during autonomous execution and merges compatible transitions to recover a deterministic controller. The [extracted table](machines/extracted.csv) lists each observed read and the corresponding write, motion, and successor across 52 source states. Compare those rules with the [expert table](machines/expert.csv). The [machine directory](machines/) also provides JSON and PDF versions and the state-numbering maps.

Inspect the [sampled inputs](data/inputs.json), [threshold attempts](data/epsilon_attempts.json), and [graph-stability measurements](data/graph_stability.csv) to trace the construction from neural trajectories to discrete states. The [figures](figures/) summarize the measurements, and the [execution checks](checks/) document the finite comparisons for this case.

To open the saved analysis, select `CASE_INDEX = 2` in the [method notebook](../../../../../notebooks/metric_FST_extraction.ipynb) and leave `RECOMPUTE = False`. To repeat the procedure from the checkpoint, select `RECOMPUTE = True`. Enable `SAVE_INTERMEDIATES` to retain intermediate artifacts as well as the final controller.

## Verify Expert Equivalence

The proof covers all pairs of nonempty decimal operands, including operands of unequal length, leading zeros, and zero-valued operands. The [mathematical argument](proof/equivalence_proof.pdf) establishes agreement of the extracted and expert controllers throughout every valid execution. The [checker](proof/verify_equivalence.py) verifies the finite obligations and performs supplementary replay checks. From the repository root, run the following command.

```sh
python analyses/metric_extraction/DFST/add10/warmup_50000_steps_4000000/proof/verify_equivalence.py
```

The [saved report](proof/equivalence_verification.json) records the verification results. The [paper](https://arxiv.org/abs/2611.21486v1) states the additional assumptions needed to extend the discrete-controller guarantee to neural execution.
