"""Model invariants: scan order, gradient flow, streaming, and stochasticity."""

import copy
import unittest

import torch

from differentiable_exact_learning.models import MODEL_NAMES, build_model, matrix_prefix_scan
from differentiable_exact_learning.objectives import action_loss


def make_model(name, **overrides):
    config = {
        "name": name,
        "hidden_size": 4 if name in {"DFST", "DFST_Softmax"} else 5,
        "num_layers": 2 if name in {"RNN", "GRU"} else 1,
        "init": {"transition": "random", "symbol": "random", "motion": "random", "head": "default"},
    }
    config.update(overrides)
    return build_model(config, input_size=3, symbol_size=4, motion_size=2, expert_states=4).double()


def all_chunks(model, tokens, **kwargs):
    chunks = list(model.iter_outputs(tokens, **kwargs))
    return tuple(torch.cat([chunk[head] for chunk in chunks], dim=1) for head in (0, 1))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.tokens = torch.tensor([[0, 1, 2, 0, 2, 1, 0], [2, 1, 0, 1, 0, 2, 1]])

    def test_prefix_scan_matches_ordered_matrix_products(self):
        for length in (0, 1, 2, 3, 7, 8, 9):
            with self.subTest(length=length):
                matrices = torch.randn(2, length, 3, 3, dtype=torch.float64) / 2
                actual = matrix_prefix_scan(matrices)
                if length == 0:
                    self.assertEqual(actual.shape, matrices.shape)
                    continue
                running = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
                reference = []
                for t in range(length):
                    running = matrices[:, t] @ running
                    reference.append(running)
                torch.testing.assert_close(actual, torch.stack(reference, dim=1))

    def test_prefix_scan_gradients_match_sequential_products(self):
        matrices = (torch.randn(2, 7, 3, 3, dtype=torch.float64) / 2).requires_grad_()
        reference_input = matrices.detach().clone().requires_grad_()
        weights = torch.randn_like(matrices)
        (matrix_prefix_scan(matrices) * weights).sum().backward()
        running = torch.eye(3, dtype=torch.float64).expand(2, -1, -1)
        reference = []
        for t in range(reference_input.shape[1]):
            running = reference_input[:, t] @ running
            reference.append(running)
        (torch.stack(reference, dim=1) * weights).sum().backward()
        torch.testing.assert_close(matrices.grad, reference_input.grad)

    def test_parallel_and_sequential_dfst_outputs_and_gradients(self):
        for name in ("DFST", "DFST_Softmax"):
            for embed_dim in (None, 5):
                with self.subTest(name=name, embed_dim=embed_dim):
                    parallel = make_model(name, embed_dim=embed_dim)
                    sequential = copy.deepcopy(parallel)
                    sequential.scan = "sequential"
                    p_outputs = parallel(self.tokens)
                    s_outputs = sequential(self.tokens)
                    for left, right in zip(p_outputs, s_outputs):
                        torch.testing.assert_close(left, right)
                    sum(item.square().sum() for item in p_outputs).backward()
                    sum(item.square().sum() for item in s_outputs).backward()
                    for p, s in zip(parallel.parameters(), sequential.parameters()):
                        if p.requires_grad:
                            torch.testing.assert_close(p.grad, s.grad)

    def test_chunked_predictions_match_full_sequence_for_every_family(self):
        for name in MODEL_NAMES:
            for embed_dim in (None, 5):
                with self.subTest(name=name, embed_dim=embed_dim):
                    model = make_model(name, embed_dim=embed_dim).eval()
                    symbols, motions, state = model(self.tokens)
                    for chunk_size in (1, 2, 4, 20):
                        cs, cm = all_chunks(model, self.tokens, chunk_size=chunk_size)
                        torch.testing.assert_close(cs, symbols)
                        torch.testing.assert_close(cm, motions)
                    if name in {"RNN", "GRU"}:
                        self.assertEqual(state.shape, (2, 2, 5))

    def test_chunking_preserves_gradients_to_learned_initial_state(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                config_init = {
                    "transition": "random",
                    "symbol": "random",
                    "motion": "random",
                    "head": "default",
                    "state": "randn",
                    "learnable_state": True,
                }
                full = make_model(name, embed_dim=4, init=config_init)
                chunked = copy.deepcopy(full)
                ys, ym, _ = full(self.tokens)
                cs, cm = all_chunks(chunked, self.tokens, chunk_size=2)
                (ys.square().sum() + ym.square().sum()).backward()
                (cs.square().sum() + cm.square().sum()).backward()
                self.assertIsNotNone(full.h0.grad)
                self.assertGreater(full.h0.grad.abs().sum().item(), 0)
                for left, right in zip(full.parameters(), chunked.parameters()):
                    if left.requires_grad:
                        torch.testing.assert_close(left.grad, right.grad)

    def test_masked_batches_match_individual_traces_and_gradients(self):
        lengths = [2, self.tokens.shape[1]]
        mask = torch.arange(self.tokens.shape[1])[None, :] < torch.tensor(lengths)[:, None]
        for name in MODEL_NAMES:
            for embed_dim in (None, 5):
                scans = ("parallel", "sequential") if name.startswith("DFST") else ("parallel",)
                for scan in scans:
                    with self.subTest(name=name, embed_dim=embed_dim, scan=scan):
                        model = make_model(name, embed_dim=embed_dim, scan=scan)
                        reference = copy.deepcopy(model)
                        symbols, motions, states = model(self.tokens, mask=mask)
                        batch_loss = (
                            symbols.square().sum() + motions.square().sum() + states.square().sum()
                        )
                        reference_loss = 0
                        for row, length in enumerate(lengths):
                            ys, ym, state = reference(self.tokens[row : row + 1, :length])
                            torch.testing.assert_close(symbols[row : row + 1, :length], ys)
                            torch.testing.assert_close(motions[row : row + 1, :length], ym)
                            state_row = (
                                states[:, row : row + 1]
                                if name in {"RNN", "GRU"}
                                else states[row : row + 1]
                            )
                            torch.testing.assert_close(state_row, state)
                            reference_loss = (
                                reference_loss
                                + ys.square().sum()
                                + ym.square().sum()
                                + state.square().sum()
                            )
                        self.assertEqual(symbols[~mask].abs().sum().item(), 0)
                        self.assertEqual(motions[~mask].abs().sum().item(), 0)
                        batch_loss.backward()
                        reference_loss.backward()
                        for left, right in zip(model.parameters(), reference.parameters()):
                            if left.requires_grad:
                                torch.testing.assert_close(left.grad, right.grad)

    def test_mask_prevents_padded_dfst_overflow_and_nan_gradients(self):
        for scan in ("parallel", "sequential"):
            with self.subTest(scan=scan):
                model = build_model({"name": "DFST", "hidden_size": 1, "scan": scan}, 2, 2, 2, 1)
                with torch.no_grad():
                    model.A[:, 0, 0] = torch.tensor([10.0, 1.0])
                    model.B[:, 0, 0] = 1
                    model.C[:, 0, 0] = 1
                reference = copy.deepcopy(model)
                tokens = torch.ones(2, 100, dtype=torch.long)
                tokens[0] = 0
                mask = torch.ones_like(tokens, dtype=torch.bool)
                mask[0, 1:] = False
                targets = torch.ones_like(tokens)
                symbols, motions, _ = model(tokens, mask=mask)
                self.assertTrue(
                    bool(torch.isfinite(symbols).all() and torch.isfinite(motions).all())
                )
                loss = action_loss(symbols, motions, targets, targets, mask)
                loss.backward()
                reference_loss = 0
                for row, length in enumerate((1, 100)):
                    ys, ym, _ = reference(tokens[row : row + 1, :length])
                    target = targets[row : row + 1, :length]
                    valid = mask[row : row + 1, :length]
                    reference_loss = (
                        reference_loss + action_loss(ys, ym, target, target, valid) * length / 101
                    )
                reference_loss.backward()
                for left, right in zip(model.parameters(), reference.parameters()):
                    if left.requires_grad:
                        self.assertTrue(bool(torch.isfinite(left.grad).all()))
                        torch.testing.assert_close(left.grad, right.grad)

    def test_mask_validation(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = make_model(name)
                for mask in (
                    torch.ones_like(self.tokens),
                    torch.ones(2, 1, dtype=torch.bool),
                    torch.zeros_like(self.tokens, dtype=torch.bool),
                    torch.tensor([[True, False, True, True, True, True, True]]).expand(2, -1),
                ):
                    with self.assertRaises(ValueError):
                        model(self.tokens, mask=mask)
                result = model(self.tokens[:, :0], mask=torch.empty(2, 0, dtype=torch.bool))
                self.assertEqual(result[0].shape, (2, 0, 4))

    def test_stabilization_does_not_change_nonhomogeneous_models(self):
        for name in ("DFST_Softmax", "RNN", "GRU"):
            with self.subTest(name=name):
                model = make_model(name)
                ordinary = all_chunks(model, self.tokens, chunk_size=2)
                stabilized = all_chunks(model, self.tokens, chunk_size=2, stabilize=True)
                for left, right in zip(ordinary, stabilized):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_dfst_stabilization_preserves_argmax(self):
        model = make_model("DFST")
        ordinary = all_chunks(model, self.tokens, chunk_size=2)
        stabilized = all_chunks(model, self.tokens, chunk_size=2, stabilize=True)
        for left, right in zip(ordinary, stabilized):
            torch.testing.assert_close(left.argmax(-1), right.argmax(-1))

    def test_stabilized_dfst_survives_growth_inside_a_chunk(self):
        model = build_model({"name": "DFST", "hidden_size": 2}, 1, 2, 2, 2)
        with torch.no_grad():
            model.A.copy_(10 * torch.eye(2).unsqueeze(0))
            model.B.copy_(torch.eye(2).unsqueeze(0))
            model.C.copy_(torch.eye(2).flip(0).unsqueeze(0))
            model.h0.copy_(torch.tensor([1.0, 2.0]))
            tokens = torch.zeros((2, 1000), dtype=torch.long)
            symbols, motions = all_chunks(model, tokens, chunk_size=256, stabilize=True)
        self.assertTrue(bool(torch.isfinite(symbols).all() and torch.isfinite(motions).all()))
        torch.testing.assert_close(symbols.argmax(-1), torch.ones_like(tokens))
        torch.testing.assert_close(motions.argmax(-1), torch.zeros_like(tokens))

    def test_stabilized_dfst_avoids_unvisited_prefix_overflow(self):
        model = build_model({"name": "DFST", "hidden_size": 2}, 1, 2, 2, 2)
        with torch.no_grad():
            model.A.copy_(torch.diag(torch.tensor([1.0, 10.0])).unsqueeze(0))
            model.B.copy_(torch.eye(2).unsqueeze(0))
            model.C.copy_(torch.eye(2).unsqueeze(0))
            model.h0.copy_(torch.tensor([1.0, 0.0]))
            tokens = torch.zeros((1, 1000), dtype=torch.long)
            symbols, motions = all_chunks(model, tokens, chunk_size=256, stabilize=True)
        for output in (symbols, motions):
            self.assertTrue(bool(torch.isfinite(output).all()))
            torch.testing.assert_close(output.argmax(-1), torch.zeros_like(tokens))

    def test_stabilized_dfst_uses_learned_embeddings(self):
        model = make_model("DFST", embed_dim=5)
        normal = all_chunks(model, self.tokens, chunk_size=2)
        stabilized = all_chunks(model, self.tokens, chunk_size=2, stabilize=True)
        for left, right in zip(normal, stabilized):
            torch.testing.assert_close(left.argmax(-1), right.argmax(-1))

    def test_softmax_outputs_and_effective_tensors_are_stochastic(self):
        for embed_dim in (None, 5):
            with self.subTest(embed_dim=embed_dim):
                model = make_model("DFST_Softmax", embed_dim=embed_dim)
                for tensor in model.effective_tensors()[:3]:
                    self.assertTrue(bool((tensor >= 0).all()))
                    torch.testing.assert_close(tensor.sum(-2), torch.ones_like(tensor.sum(-2)))
                symbols, motions, state = model(self.tokens)
                for tensor in (symbols, motions, state):
                    self.assertTrue(bool((tensor >= 0).all()))
                    torch.testing.assert_close(tensor.sum(-1), torch.ones_like(tensor.sum(-1)))

    def test_effective_tensors_reconstruct_token_dynamics(self):
        for name in ("DFST", "DFST_Softmax"):
            for embed_dim in (None, 5):
                with self.subTest(name=name, embed_dim=embed_dim):
                    model = make_model(name, embed_dim=embed_dim)
                    A, B, C, h0 = model.effective_tensors()
                    h = h0.expand(self.tokens.shape[0], -1)
                    symbols, motions = [], []
                    for t in range(self.tokens.shape[1]):
                        x = self.tokens[:, t]
                        symbols.append((B[x] @ h.unsqueeze(-1)).squeeze(-1))
                        motions.append((C[x] @ h.unsqueeze(-1)).squeeze(-1))
                        h = (A[x] @ h.unsqueeze(-1)).squeeze(-1)
                    reference = (torch.stack(symbols, 1), torch.stack(motions, 1), h)
                    for left, right in zip(model(self.tokens), reference):
                        torch.testing.assert_close(left, right)

    def test_paper_dfst_defaults(self):
        model = build_model({"name": "DFST"}, 3, 4, 2, 5)
        torch.testing.assert_close(model.A, torch.eye(5).expand(3, -1, -1))
        torch.testing.assert_close(model.B, torch.zeros(3, 4, 5))
        torch.testing.assert_close(model.C, torch.zeros(3, 2, 5))
        torch.testing.assert_close(model.h0, torch.ones(5))
        self.assertTrue(model.h0.requires_grad)
        self.assertEqual(model.output_kind, "logits")

    def test_dfst_random_initial_state_matches_original_scaling(self):
        for name in ("DFST", "DFST_Softmax"):
            for scheme in ("random", "randn"):
                with self.subTest(name=name, scheme=scheme):
                    torch.manual_seed(71)
                    expected = torch.randn(9) / 3
                    torch.manual_seed(71)
                    # Fixed one-hot embeddings and identity/zero matrix
                    # initialization consume no random values before h0.
                    model = build_model(
                        {"name": name, "hidden_size": 9, "init": {"state": scheme}},
                        input_size=3,
                        symbol_size=4,
                        motion_size=2,
                        expert_states=9,
                    )
                    torch.testing.assert_close(model.h0, expected, rtol=0, atol=0)

    def test_null_initial_settings_use_family_defaults(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = make_model(name, init={"state": None, "learnable_state": None})
                dfst = name.startswith("DFST")
                expected = torch.ones_like(model.h0) if dfst else torch.zeros_like(model.h0)
                torch.testing.assert_close(model.h0, expected)
                self.assertEqual(model.h0.requires_grad, dfst)

    def test_only_softmax_dfst_has_regularization(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = make_model(name, reg_lambda=0.2)
                loss = model.regularization_loss()
                torch.testing.assert_close(loss, model.reg_loss())
                self.assertEqual(loss.ndim, 0)
                if name == "DFST_Softmax":
                    self.assertGreater(loss.item(), 0)
                    loss.backward()
                    self.assertIsNotNone(model.A.grad)
                else:
                    self.assertEqual(loss.item(), 0)

    def test_gru_gate_bias_and_identity_input_initialization(self):
        model = make_model("GRU", init={"recurrent": "identity", "gate_bias": 2.0})
        H = model.hidden_size
        for layer in range(model.num_layers):
            bias = getattr(model.core, f"bias_ih_l{layer}")
            torch.testing.assert_close(bias[H : 2 * H], torch.full_like(bias[H : 2 * H], 2.0))
            self.assertGreater(getattr(model.core, f"weight_ih_l{layer}").abs().sum().item(), 0)
            recurrent = getattr(model.core, f"weight_hh_l{layer}")
            torch.testing.assert_close(recurrent[2 * H :], torch.eye(H, dtype=recurrent.dtype))
            torch.testing.assert_close(recurrent[: 2 * H], torch.zeros_like(recurrent[: 2 * H]))

    def test_identity_zero_input_initialization(self):
        for name, gates in (("RNN", 1), ("GRU", 3)):
            model = make_model(name, init={"recurrent": "identity_zero_input"})
            H = model.hidden_size
            for layer in range(model.num_layers):
                core = {
                    key: getattr(model.core, f"{key}_l{layer}")
                    for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
                }
                for key in ("weight_ih", "bias_ih", "bias_hh"):
                    torch.testing.assert_close(core[key], torch.zeros_like(core[key]))
                recurrent = core["weight_hh"]
                candidate = (gates - 1) * H
                torch.testing.assert_close(
                    recurrent[candidate:], torch.eye(H, dtype=recurrent.dtype)
                )
                torch.testing.assert_close(
                    recurrent[:candidate], torch.zeros_like(recurrent[:candidate])
                )

    def test_initial_state_variants(self):
        for name in MODEL_NAMES:
            variants = ("zeros", "ones", "randn", "one_hot")
            for scheme in variants:
                with self.subTest(name=name, scheme=scheme):
                    model = make_model(name, init={"state": scheme, "learnable_state": True})
                    ys, ym, state = model(self.tokens)
                    self.assertTrue(
                        bool(
                            torch.isfinite(ys).all()
                            and torch.isfinite(ym).all()
                            and torch.isfinite(state).all()
                        )
                    )
                    self.assertIn("h0", dict(model.named_parameters()))

    def test_empty_sequences_and_invalid_inputs(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = make_model(name)
                symbols, motions, _ = model(self.tokens[:, :0])
                self.assertEqual(symbols.shape, (2, 0, 4))
                self.assertEqual(motions.shape, (2, 0, 2))
                self.assertEqual(list(model.iter_outputs(self.tokens[:, :0])), [])
                with self.assertRaises(ValueError):
                    list(model.iter_outputs(self.tokens, chunk_size=0))
                with self.assertRaises(ValueError):
                    model(self.tokens[0])
        with self.assertRaises(ValueError):
            make_model("unknown")
        with self.assertRaises(ValueError):
            make_model("DFST", scan="unknown")


if __name__ == "__main__":
    unittest.main()
