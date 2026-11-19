# Binary Multiplication by Softmax Annealing

We trained a 64-dimensional Softmax DFST on 100 demonstrations for 100,000 updates before extraction. The [released run](../../../../../runs/DFST_Softmax/mult2/hidden_64_data_100) contains the original weights and training configuration. The [case configuration](config.yaml) identifies that checkpoint by checksum and records the settings for softmax annealing.

## Inspect the Recovered Controller

After annealing, column-wise argmax selects a successor state, write symbol, and motion for each state and read symbol. The [extracted table](machines/extracted.csv) lists writes, motions, and successors for 52 states with exported transitions. Compare those rules with the [expert table](machines/expert.csv). The [machine directory](machines/) also provides JSON and PDF versions and the state-numbering maps.

Follow the [sharpness measurements](data/sharpness.csv) and [training log](data/training.jsonl) to see how the probability columns concentrate. The [probability snapshots](data/probabilities/) drive the notebook slider, and the [annealed checkpoint](checkpoints/annealed_model_checkpoint.pt) preserves the final weights. The [figures](figures/) summarize the measurements, and the [execution checks](checks/) document the finite comparisons for this case.

To open the saved analysis, select `CASE_INDEX = 1` in the [method notebook](../../../../../notebooks/anneal_dfst_softmax_to_fst.ipynb) and leave `RECOMPUTE = False`. To repeat the procedure from the checkpoint, select `RECOMPUTE = True`. Enable `SAVE_INTERMEDIATES` to retain intermediate artifacts as well as the final controller.

## Verify Expert Equivalence

The proof covers all pairs of nonempty binary operands, including operands of unequal length, leading zeros, and zero-valued operands. The [mathematical argument](proof/equivalence_proof.pdf) establishes agreement of the extracted and expert controllers throughout every valid execution. The [checker](proof/verify_equivalence.py) verifies the finite obligations and performs supplementary replay checks. From the repository root, run the following command.

```sh
python analyses/annealing/DFST_Softmax/mult2/hidden_64_data_100/proof/verify_equivalence.py
```

The [saved report](proof/equivalence_verification.json) records the verification results. The [paper](https://arxiv.org/abs/2611.21486v1) states the additional assumptions needed to extend the discrete-controller guarantee to neural execution.
