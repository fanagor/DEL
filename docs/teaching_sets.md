# Teaching Sets

Teaching cores provide deterministic examples that exercise reachable expert
transitions. Enable a core with `data.teaching_set: true`.

## Configuration

The teaching core depends on the selected expert and uses static datasets
(`data.source: static`). Setting `teaching_set: true`
with `data.source: online`, `data.max_length` below the core's maximum length, or
`data.size < len(teaching_set)` fails immediately at run startup.

`data.variant` selects `default` or `new` grid controllers. The `tm` and
`teleporting` environments each provide one implementation and ignore this field.

## Teaching Set Sizes

| Environment | Task | Variant | Teaching examples (base 10 / base 2) | Max length | Observed reachable-rule coverage (base 10 / base 2) |
|---|---|---|---:|---:|---:|
| `grid` | `addition` | `default` | `120` / `8` | `2` | `201` / `33` (100%) |
| `grid` | `addition` | `new` | `181` / `21` | `3` | `124` / `36` (100%) |
| `tm` | `addition` | Not applicable | `253` / `19` | `3` | `385` / `33` (100%) |
| `grid` | `multiplication` | `default` | `1,645` / `33` | `5` | `1,035` / `79` (100%) |
| `grid` | `multiplication` | `new` | `700` / `28` | `5` | `423` / `63` (100%) |
| `tm` | `multiplication` | Not applicable | `481` / `21` | `3` | `4,357` / `209` (100%) |
| `teleporting` | `universal` | Not applicable | `75` | `3` | `67` (100%) |

## Arithmetic Agents

All arithmetic teaching cores include both operand orders and contain no duplicate pairs:

- **Addition (`grid` `default`):** Exact `R² + 2R` core (`≤ 2` digits per
  operand): all `R²` single-digit sums `a + b` (`0 ≤ a, b < R`), plus `dh + 1`
  and `1 + dh` (`0 ≤ d < R`, `h = R - 1`).
- **Addition (`grid` `new`, and `tm` `default`):** Preserves leading zeros (`≤ 3` digits per operand). `grid` `new`
  includes all `R²` single-digit sums, 1- and 2-step ripple-carry stops and
  blank overflows (`dh + 1`, `dhh + 1`), unequal widths and zero-digit `b`
  bypass witnesses (`1 + 000`, `0 + 001`, `d + d0d`, `dh + (R-d)0d`), and
  `hhh + hhh`. `tm` augments those anchors with a deterministic symmetric cover
  over 2- and 3-digit witnesses to cover every dotted-digit (`⋅d`) tape scan
  transition.
- **Multiplication (`grid` `default`):** Explicit `R³ + 6R² + R(R-1)/2` core
  (`≤ 5` digits per operand), consisting of repeated digits, adjacent digits,
  carry boundaries, partial-product carry witnesses, and four-digit palindromic
  squares covering mixed later-erasure contexts.
- **Multiplication (`grid` `new`):** Explicit `7R²` transition core (`≤ 5`
  digits per operand), sharing the first three families with the default variant.
  The controller leaves the untrimmed `(m+n)`-digit product on row `1` at
  `[-n, m-1]`, halting at `(-n, 1)`.
- **Multiplication (`tm` `default`):** Trims leading zeros (`≤ 3` digits per
  operand), retaining `0^m × 0^n` (`1 ≤ m, n ≤ 3`), zero vs. repdigit
  (`000 × ddd`), identity (`1 × ddd`), carry/overflow anchors (`hh × hh`,
  `hhh × hhh`, `101 × 101`, `101 × hhh`), and a deterministic greedy cover over
  `≤ 3`-digit candidates with reverse pruning.

## Controller Sizes

Controller size below counts **all states, including halt**. Complete tables
include fallback halt rules for unreachable state/symbol pairs and the halt
self-loops; reachable rules count only rules actually executed on valid inputs.
Each arithmetic alphabet has `R+2` symbols (digits, blank, and operator).

| Controller family | States Q(R) | Complete table entries | Reachable rules | Teaching examples |
|---|---:|---:|---:|---:|
| `add_grid` | `R+9` | `(R+9)(R+2)` | `R²+9R+11` | `R²+2R` |
| `add_grid_new` | `R+13` | `(R+13)(R+2)` | `11R+14` | `R²+8R+1` |
| `mult_grid` | `R(R+1)/2+3R+23` | `Q(R)(R+2)` | `(R³+6R²+43R+40)/2` | `R³+6R²+R(R-1)/2` |
| `mult_grid_new` | `2R+22` | `(2R+22)(R+2)` | `2R²+21R+13` | `7R²` |
| `universal_tp` (binary tape) | `15` | `135` | `67` | `75` |

| Controller | States (including halt) | Complete table entries | Reachable rules | Teaching examples |
|---|---:|---:|---:|---:|
| `add2` | 11 | 44 | 33 | 8 |
| `add2_new` | 15 | 60 | 36 | 21 |
| `add10` | 19 | 228 | 201 | 120 |
| `add10_new` | 23 | 276 | 124 | 181 |
| `mult2` | 32 | 128 | 79 | 33 |
| `mult2_new` | 26 | 104 | 63 | 28 |
| `mult10` | 108 | 1,296 | 1,035 | 1,645 |
| `mult10_new` | 42 | 504 | 423 | 700 |
| `universal_tp` | 15 | 135 | 67 | 75 |

### Default Grid Multiplication

The `mult_grid` controller computes products through four phases:

1. Copy successively shorter input prefixes one row down and one column left.
2. Replace each row with a digit-times-operand partial product, including a
   leading carry digit even when zero.
3. Add neighboring rows right to left, storing carry in the controller.
4. When no lower row remains, preserve the blank, move up to `finish` and emit
   the final symbol-preserving `S` action.

For operand lengths `m,n`, the final product contains exactly `m+n` digits,
including leading zeros, at columns `[-n, m-1]` of row `n-1`. The head stops
on the adjacent left blank at `(-n-1, n-1)`. Decimal `2x3` leaves `06`, and
`00x000` leaves `00000`.

The controller uses `R(R+1)/2+3R+23` states, including finish and halt, with
contiguous state IDs. State complexity scales as `Theta(R²)` and complete
transition-table complexity scales as `Theta(R³)`.

Dataset cache keys include the expert-table fingerprint. Teaching datasets
also include a digest of the exact ordered teaching examples. Checkpoint
loading verifies the saved expert fingerprint against the selected controller.

### Explicit Multiplication Transition Cores

The [README construction](../README.md#explicit-grid-multiplication-teaching-sets)
defines the algebraic families and explains the coverage roles. Both
variants use symmetric, deduplicated sets, with operands of at most five digits
and all single-digit products included. The formulas determine every grid teaching example without random selection
or greedy pruning.

For digits `a,b in {0,...,R-1}`, let `h=R-1`; concatenation denotes digit words,
`a^k` denotes repetition, and `Sym` adds reversed operand pairs. The shared core contains:

- `(a^k,b^k)` for `k=1,...,5`: `5R²` ordered examples.
- `Sym{(ab,11): a != b}`: `2R(R-1)` examples.
- `Sym{(hbhhh,11)}`: `2R` examples.

These disjoint families give exactly `7R²` examples for `mult_grid_new`.
For `mult_grid`, also include `Sym{(as,f)}` for every digit `a`, factor
`1 <= f < R`, and carry `0 <= c < f`, where `s=ceil(R*c/f)`. Reading `s`
establishes carry `c` before reading `a`. This disjoint family contributes
`R²(R-1)` examples.

The default core also includes `{(abba,abba): 0 <= a < b < R}`. Each
four-digit palindromic square exercises both mixed later-erasure contexts:
remembering factor `a` while reading `b`, and remembering factor `b` while
reading `a`, after at least one prefix erasure. Equal operands preserve
symmetry; strings preserve leading zeros. The `R(R-1)/2` examples add no
duplicates to the three shared families or the carry-boundary family, giving `R³+6R²+R(R-1)/2` examples.
This family covers contexts that a one-step expert-transition coverage check
cannot distinguish when extraction splits an expert state into several states.

| Radix R | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mult_grid` | 33 | 84 | 166 | 285 | 447 | 658 | 924 | 1,251 | 1,645 |
| `mult_grid_new` | 28 | 63 | 112 | 175 | 252 | 343 | 448 | 567 | 700 |

The cores cover every reachable transition, including self-loops, for each
supported radix `2..10`. Coverage requires at least one execution of each
reachable transition; coverage does not require multiple consecutive
repetitions of every repeatable self-loop.

Both grid variants require `data.max_length >= 5` and `data.size` at least the
corresponding core size. Teaching-example digests distinguish datasets with
different ordered teaching inputs.

## Universal Teleporting Agent

For the universal learning task, use `data.environment: teleporting`. With
`data.teaching_set: true`, static dataset generation injects `75` `(tm, word)`
examples: all 15 binary words of lengths `0..3` (including the empty word)
for each of these 5 TMs, all with identical input-word sets:

- `w = 1` state trit: `parity` (lengths `0..3`, 15 words)
- `w = 2` state trits: `shift_left` (`0..3`, 15), `shift_right` (`0..3`, 15), `scaling_A_m5` (`0..3`, 15)
- `w = 3` state trits: `scaling_B_m5` (`0..3`, 15)

The core contains `5 × (1+2+4+8) = 75` examples and covers all 67 reachable
transitions of the universal teleporting agent. Teaching sets require `data.source=static`,
`data.max_length >= 3`, and `data.size >= 75`. Setting `data.size=75` gives
exactly this core. Any remaining `data.size - 75` examples are sampled from
`data.train_tasks`. The core excludes `scaling_A_m10` and `scaling_B_m10`.
The ordinary sampler can draw these programs when `data.train_tasks` includes
the corresponding names and `data.size > 75`.

## Inspect Teaching Examples

Run these commands from the repository root in the DEL environment.
`del teaching-set` prints a JSON manifest containing the ordered examples,
example count, covered-transition count and controller size. Inspection
simulates expert executions without training a model. Redirect the output to a
file when reviewing a large core, for example with `> /tmp/mult10_teaching.json`.

```bash
# Default decimal addition: 120 examples, operands up to 2 digits.
del teaching-set --config configs/DFST/add10.yaml \
  data.teaching_set=true data.max_length=2 data.size=120

# Default decimal multiplication: 1,645 examples, operands up to 5 digits.
del teaching-set --config configs/DFST/mult10.yaml \
  data.teaching_set=true data.max_length=5 data.size=1645

# Alternative grid addition: 181 examples, operands up to 3 digits.
del teaching-set --config configs/DFST/add10.yaml \
  data.variant=new data.teaching_set=true data.max_length=3 data.size=181

# Alternative grid multiplication: 700 examples, operands up to 5 digits.
del teaching-set --config configs/DFST/mult10.yaml \
  data.variant=new data.teaching_set=true data.max_length=5 data.size=700

# Universal learning task: 75 program/input pairs, input lengths 0–3.
del teaching-set --config configs/DFST/universal.yaml \
  data.teaching_set=true data.max_length=3 data.size=75
```

## Train with a Teaching Core

`data.size` counts the complete static dataset, including teaching examples.
Setting `data.size` to the core size trains on the teaching core alone. A larger
value adds ordinary samples up to the requested total. The examples below set
`data.source=static` explicitly and assign distinct experiment names.

```bash
# Decimal addition: 120 teaching examples plus 130 ordinary samples.
del train --config configs/DFST/add10.yaml \
  data.source=static data.teaching_set=true data.max_length=3 data.size=250 \
  run.name=add10-teaching-250

# Decimal multiplication: the complete 1,645-example teaching core.
del train --config configs/DFST/mult10.yaml \
  data.source=static data.teaching_set=true data.max_length=5 data.size=1645 \
  run.name=mult10-teaching-1645

# Universal learning task: the complete 75-example teaching core.
del train --config configs/DFST/universal.yaml \
  data.source=static data.teaching_set=true data.max_length=3 data.size=75 \
  run.name=universal-teaching-75

# Universal learning task: 75 teaching examples plus 925 ordinary samples.
del train --config configs/DFST/universal.yaml \
  data.source=static data.teaching_set=true data.max_length=5 data.size=1000 \
  run.name=universal-teaching-1000
```

For the universal learning task, ordinary samples use the programs listed in
`data.train_tasks` and binary input lengths up to `data.max_length`. The teaching
core uses the fixed five programs and all binary words of lengths 0–3.
Training commands use the optimizer and training budget from the selected recipe;
see the [recipe guide](../configs/README.md) for those settings.
