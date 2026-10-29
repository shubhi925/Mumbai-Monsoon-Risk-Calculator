# Mumbai Monsoon Risk Calculator — Streamlit edition

Monte Carlo cost and schedule contingency for construction projects exposed to the
south-west monsoon. Takes three-point estimates (optimistic, most likely, pessimistic)
for each activity's monsoon-driven extra delay and extra cost, runs a correlated
simulation over the project network, and reports P50, P80 and P90 with histograms and
cumulative curves.

Replaces a flat contingency percentage with a distribution you can defend.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. No API keys, no network access needed.

## Files

| File | What it is |
|---|---|
| `app.py` | The Streamlit interface: eight tabs, all charts and explanations |
| `engine.py` | Simulation engine — sampler, CPM, PERT moments, validation, self-tests. No Streamlit imports, so it is testable and reusable |
| `test_engine.py` | pytest suite (14 tests). Run with `pytest -q` |
| `sample_activities.csv` | The G+4 Mumbai benchmark case, in the tool's import format |
| `requirements.txt`, `.streamlit/config.toml` | Dependencies and theme |

## The tabs

**Inputs** — editable activity table, CSV import and export, live network check
(critical path, float per activity, monsoon overlap from the planned start date), and
input validation that catches cycles, unknown predecessors, O > M > P violations and
activities carrying monsoon risk while sitting outside June–September.

**Results** — percentile metrics, a plain-words summary, four charts each with a "how
to read this" note, a contingency table by confidence level, an interactive
"test your own contingency" box, and a drivers table showing which activities actually
move the answer.

**Scenarios** — mild, typical and severe side by side on the same seed, plus a
confidence-level curve for all three.

**Validation** — the hand PERT check, the project-level comparison, merge-event
diagnostics, self-tests and convergence/seed stability. See below.

**Rainfall** — upload daily June–September rainfall and derive the mild and severe
multipliers empirically instead of assuming them.

**Input guide / Why rain delays / Model and trust** — how to set O, M and P defensibly;
the physical mechanisms behind rain delay; and an explicit list of what the model
cannot do.

## Two sampling engines

| Engine | Generator | Beta quantile | Use it for |
|---|---|---|---|
| `legacy` (default) | mulberry32, vectorised | 2000-bin discretised inverse | Reproducing the original HTML calculator **bit for bit**, so results match the validation report |
| `precise` | numpy PCG64 | exact, via scipy | Production runs. Quantile error ~1e-16 against ~1e-8 |

The legacy path is vectorised because mulberry32's internal counter advances by a
constant, so output *k* depends only on `seed + k·C` and the whole stream can be
generated at once. A 10,000-iteration run over twelve activities takes about 0.06 s
instead of the original's per-iteration JavaScript loop.

## What the validation tab tells you

Three things worth understanding before you quote any number from this tool.

**1. Activity-level agreement is not project-level agreement.** Every activity's
sampled mean should match `(O + 4M + P)/6` within three standard errors, and it does.
That proves the *sampler* is correct. It does not license a hand critical-path sum as a
project estimate.

**2. Merge-event bias.** Where two branches reach a merge with similar finish times,
neither is reliably critical, and because `E[max(X,Y)] ≥ max(E[X], E[Y])` the expected
completion is later than any single path's expectation. On the bundled benchmark case
the two branches into A11 finish 0.007 days apart, each governs about half the
iterations, and the resulting gap is 0.46 days — which accounts for the entire
difference between the simulated mean and the hand calculation. The tab measures this
directly. Where the branches are far apart it vanishes and the hand sum is fine.

**3. The textbook standard deviation is wrong, twice.** `(P−O)/6` understates the true
beta SD by 5–12%, and root-sum-of-squares assumes independence when ρ > 0. On the
benchmark case the textbook figure is 7.27 days against 10.92 simulated. Correcting
both — exact beta SD plus `2ρ²Σσᵢσⱼ` cross terms — gives 10.90, within 0.17%. The tool
was right; the hand formula was not.

## Units and conventions

- Durations and delays are **calendar days**. The monsoon-overlap check adds durations
  to the planned start date, so working days would misplace the monsoon. Rain days are
  calendar-day events anyway.
- Costs are **₹ lakh**.
- `delay_O/M/P` and `cost_O/M/P` are monsoon-driven **increments** on top of the
  baseline duration and cost, not totals.
- Predecessors are separated by `;` in CSV, `;` or `,` in the editor.
- Anything accruing **per idle day** (labour standing by, plant on standby, overheads,
  interest) goes in the sidebar's daily-cost panel, *not* in an activity's
  `cost_O/M/P`. Double counting idle labour in both places is the commonest error.

## Known limits

Stated in full on the "Model and trust" tab. The three that matter most:

- **ρ is assumed, not measured.** Moving it from 0 to 0.9 roughly doubles the project
  standard deviation — more than any individual estimate does. Always report the value
  used.
- **Slip into the monsoon is not modelled.** Each activity's spread is fixed, so if an
  early stage overruns and pushes a later one into June, the model does not widen that
  later activity's distribution in response.
- **The daily rate is constant and certain.** In reality it probably rises with delay
  length as penalties and financing compound.

Nothing the tool produces is a forecast. It is the consequence of your assumptions,
expressed as probabilities.
