# Universal Teleporting Agent: annealing

Notebook case index: `2`. Source run: `runs/DFST_Softmax/universal/hidden_30_data_150`.

Proof domain: Correctly encoded binary Turing machines with arbitrary finite binary inputs, including empty input.

`config.yaml` records the source checkpoint checksum and method settings.
`machines/` contains the extracted and expert transition maps in JSON, readable
CSV and PDF, together with the state maps. `figures/` contains the main plots;
`data/` contains the measurements and inputs needed for inspection and plotting.
Sampled PTO results, counterexamples and coverage checks live in `checks/`.
Annealing cases also contain `data/probabilities/` for the slider and
`checkpoints/annealed_model_checkpoint.pt` for the final annealed weights.

Use `CASE_INDEX = 2` with `RECOMPUTE = False` in the corresponding
method notebook to display this case. Set `RECOMPUTE = True` to compute a new
analysis. `SAVE_INTERMEDIATES = True` additionally saves debugging artifacts
under `intermediates/`; the default produces the compact layout.

The [proof](proof/equivalence_proof.pdf) concerns the extracted discrete controller.
The self-contained `proof/` bundle includes the checker and all fingerprinted
dependencies. From the repository root, run:

```sh
python analyses/annealing/DFST_Softmax/universal/hidden_30_data_150/proof/verify_equivalence.py
```

The checker verifies finite obligations and supplementary executions. The
mathematical argument establishes protocol coverage and the unbounded claim.
Equivalence of the continuous neural model requires additional assumptions.
