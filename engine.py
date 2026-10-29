"""
Monte Carlo engine for monsoon cost and schedule contingency.

Two sampling back-ends:

  "legacy"   reproduces the original HTML calculator bit for bit - the same
             mulberry32 generator, Box-Muller pairing, Abramowitz-Stegun normal
             CDF and 2000-bin discretised beta inverse. Vectorised here because
             mulberry32's internal counter advances by a constant, so the k-th
             output depends only on seed + k*C and the whole stream can be
             generated at once.

  "precise"  numpy PCG64 plus exact beta quantiles from scipy. Better numerics
             (quantile error ~1e-16 against ~1e-8) but not stream-compatible
             with the HTML tool.

Keep "legacy" for anything that has to agree with the validation report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

try:
    from scipy.stats import beta as _beta
    from scipy.special import ndtr as _ndtr
    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

U32 = np.uint64(0xFFFFFFFF)
MULB_C = np.uint64(0x6D2B79F5)
TABLE_BINS = 2000


# ----------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------
@dataclass
class Activity:
    id: str
    name: str = ""
    dur: float = 0.0
    cost: float = 0.0
    pre: list = field(default_factory=list)
    dO: float = 0.0
    dM: float = 0.0
    dP: float = 0.0
    cO: float = 0.0
    cM: float = 0.0
    cP: float = 0.0
    pi: list = field(default_factory=list)
    exposure: float | None = None


@dataclass
class Config:
    start: str = "2027-02-01"
    budget: float = 900.0
    buf: float = 30.0
    rate: float = 0.6561
    rho: float = 0.5
    n: int = 10000
    seed: int = 42
    m_mild: float = 0.6
    m_typ: float = 1.0
    m_sev: float = 1.5
    engine: str = "legacy"
    base: float = 0.0
    head: float = 0.0


# ----------------------------------------------------------------------
# legacy random stream (bit-exact with the HTML tool)
# ----------------------------------------------------------------------
def mulberry32_stream(seed: int, count: int) -> np.ndarray:
    """Vectorised mulberry32. Output k (1-indexed) depends only on seed + k*C."""
    k = np.arange(1, count + 1, dtype=np.uint64)
    s = (np.uint64(int(seed) & 0xFFFFFFFF) + k * MULB_C) & U32
    t = ((s ^ (s >> np.uint64(15))) * (np.uint64(1) | s)) & U32
    t2 = ((t + (((t ^ (t >> np.uint64(7))) * (np.uint64(61) | t)) & U32)) ^ t) & U32
    return (((t2 ^ (t2 >> np.uint64(14))) & U32).astype(np.float64)) / 4294967296.0


def box_muller_stream(seed: int, count: int) -> np.ndarray:
    """Normals in the exact order the HTML tool's mkNorm() yields them."""
    pairs = (count + 1) // 2
    u = mulberry32_stream(seed, 2 * pairs)
    a = u[0::2].copy()
    b = u[1::2]
    a[a == 0.0] = np.nextafter(0.0, 1.0)          # mirrors `while(!u) u=r()`
    m = np.sqrt(-2.0 * np.log(a))
    out = np.empty(2 * pairs)
    out[0::2] = m * np.cos(2 * np.pi * b)          # returned first
    out[1::2] = m * np.sin(2 * np.pi * b)          # held as the spare
    return out[:count]


def phi_as(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26 normal CDF, as used by the HTML tool."""
    s = np.where(x < 0, -1.0, 1.0)
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    p = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
          - 0.284496736) * t + 0.254829592) * t
    return 0.5 * (1.0 + s * (1.0 - p * np.exp(-z * z)))


def _shapes(O: float, M: float, P: float) -> tuple[float, float]:
    span = P - O
    return 1.0 + 4.0 * (M - O) / span, 1.0 + 4.0 * (P - M) / span


def beta_quantile_table(O: float, M: float, P: float, bins: int = TABLE_BINS):
    """The HTML tool's discretised inverse CDF, as a vectorised callable."""
    if not (P > O):
        return lambda u: np.full_like(np.asarray(u, dtype=float), O)
    a, b = _shapes(O, M, P)
    x = (np.arange(bins) + 0.5) / bins
    dens = x ** (a - 1.0) * (1.0 - x) ** (b - 1.0)
    c = np.concatenate([[0.0], np.cumsum(dens)])
    total = c[-1]

    def q(u):
        target = np.asarray(u, dtype=float) * total
        hi = np.clip(np.searchsorted(c, target, side="left"), 1, bins)
        lo = hi - 1
        denom = c[hi] - c[lo]
        frac = np.clip(np.divide(target - c[lo], denom,
                                 out=np.zeros_like(target), where=denom > 0), 0.0, 1.0)
        return O + (P - O) * ((lo + frac) / bins)
    return q


def beta_quantile_exact(O: float, M: float, P: float):
    if not (P > O):
        return lambda u: np.full_like(np.asarray(u, dtype=float), O)
    a, b = _shapes(O, M, P)
    return lambda u: O + (P - O) * _beta.ppf(np.asarray(u, dtype=float), a, b)


# ----------------------------------------------------------------------
# PERT moments
# ----------------------------------------------------------------------
def pert_mean(O, M, P):
    return (O + 4.0 * M + P) / 6.0


def pert_sd_classical(O, M, P):
    return (P - O) / 6.0


def pert_sd_beta(O, M, P):
    """True SD of the sampled PERT-beta. a+b=6 always, so the denominator is 252."""
    if P <= O:
        return 0.0
    a, b = _shapes(O, M, P)
    return (P - O) * math.sqrt(a * b / 252.0)


# ----------------------------------------------------------------------
# network
# ----------------------------------------------------------------------
def topological_order(acts: list[Activity]) -> list[int]:
    """Kahn's algorithm, matching the HTML tool's queue order."""
    deg = [len(a.pi) for a in acts]
    queue = [i for i, d in enumerate(deg) if d == 0]
    order: list[int] = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for j, a in enumerate(acts):
            if i in a.pi:
                deg[j] -= 1
                if deg[j] == 0:
                    queue.append(j)
    return order


def cpm_baseline(acts, order) -> float:
    f = [0.0] * len(acts)
    out = 0.0
    for i in order:
        start = max([f[p] for p in acts[i].pi], default=0.0)
        f[i] = start + acts[i].dur
        out = max(out, f[i])
    return out


def early_starts(acts, order) -> list[float]:
    es = [0.0] * len(acts)
    for i in order:
        es[i] = max([es[p] + acts[p].dur for p in acts[i].pi], default=0.0)
    return es


def floats_and_critical_path(acts, order):
    """Total float per activity and the zero-float chain."""
    base = cpm_baseline(acts, order)
    ef = [0.0] * len(acts)
    for i in order:
        ef[i] = max([ef[p] for p in acts[i].pi], default=0.0) + acts[i].dur
    succ = {i: [] for i in range(len(acts))}
    for j, a in enumerate(acts):
        for p in a.pi:
            succ[p].append(j)
    lf = [base] * len(acts)
    for i in reversed(order):
        if succ[i]:
            lf[i] = min(lf[j] - acts[j].dur for j in succ[i])
    fl = {acts[i].id: lf[i] - ef[i] for i in range(len(acts))}
    cp = [acts[i].id for i in order if abs(fl[acts[i].id]) < 1e-9]
    return fl, cp, base


def monsoon_exposure(acts, order, start_iso: str):
    """Fraction of each activity's planned span falling inside 1 Jun - 30 Sep."""
    try:
        d0 = date.fromisoformat(start_iso)
    except Exception:
        for a in acts:
            a.exposure = None
        return False
    es = early_starts(acts, order)
    for i, a in enumerate(acts):
        s = d0 + timedelta(days=float(es[i]))
        e = s + timedelta(days=float(a.dur))
        span = (e - s).days or 1
        ov = 0
        for y in range(d0.year, d0.year + 3):
            lo, hi = max(s, date(y, 6, 1)), min(e, date(y, 10, 1))
            ov += max(0, (hi - lo).days)
        a.exposure = ov / span
    return True


# ----------------------------------------------------------------------
# simulation
# ----------------------------------------------------------------------
@dataclass
class Result:
    acts: list
    order: list
    cfg: Config
    n: int
    base: float
    delay: np.ndarray
    extra: np.ndarray
    act_delay: np.ndarray        # (k, n)
    act_cost: np.ndarray         # (k, n)
    finishes: np.ndarray         # (k, n) simulated finish times
    dm: float
    cm: float
    label: str = ""


def simulate(acts, order, cfg: Config, n=None, seed=None,
             dm=1.0, cm=1.0, engine=None, label="") -> Result:
    n = int(n or cfg.n)
    seed = int(cfg.seed if seed is None else seed)
    engine = engine or cfg.engine
    k = len(acts)
    rho = float(cfg.rho)
    s = math.sqrt(max(0.0, 1.0 - rho * rho))

    if engine == "legacy":
        # The HTML tool draws, per iteration: z, then (ld, lc) for each activity.
        need = n * (1 + 2 * k)
        nz = box_muller_stream(seed, need).reshape(n, 1 + 2 * k)
        z = nz[:, 0]
        eps = nz[:, 1::2]            # (n, k) -> the ld noise
        eta = nz[:, 2::2]            # (n, k) -> the lc noise
        cdf = phi_as
        qd = [beta_quantile_table(a.dO, a.dM, a.dP) for a in acts]
        qc = [beta_quantile_table(a.cO, a.cM, a.cP) for a in acts]
    else:
        if not HAVE_SCIPY:
            raise RuntimeError("precise engine needs scipy")
        g = np.random.default_rng(seed)
        z = g.standard_normal(n)
        eps = g.standard_normal((n, k))
        eta = g.standard_normal((n, k))
        cdf = _ndtr
        qd = [beta_quantile_exact(a.dO, a.dM, a.dP) for a in acts]
        qc = [beta_quantile_exact(a.cO, a.cM, a.cP) for a in acts]

    ld = rho * z[:, None] + s * eps
    lc = 0.85 * ld + 0.5268 * eta

    act_delay = np.empty((k, n))
    act_cost = np.empty((k, n))
    for i in range(k):
        act_delay[i] = qd[i](cdf(ld[:, i])) * dm
        act_cost[i] = qc[i](cdf(lc[:, i])) * cm

    finishes = np.zeros((k, n))
    for i in order:
        if acts[i].pi:
            start = np.max(finishes[acts[i].pi, :], axis=0)
        else:
            start = np.zeros(n)
        finishes[i] = start + acts[i].dur + act_delay[i]

    base = cpm_baseline(acts, order)
    delay = finishes.max(axis=0) - base
    extra = act_cost.sum(axis=0) + delay * cfg.rate

    return Result(acts=acts, order=order, cfg=cfg, n=n, base=base,
                  delay=delay, extra=extra, act_delay=act_delay,
                  act_cost=act_cost, finishes=finishes, dm=dm, cm=cm, label=label)


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def percentile(sorted_arr: np.ndarray, p: float) -> float:
    """Linear interpolation on the sorted sample, matching the HTML tool's pc()."""
    x = (len(sorted_arr) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return float(sorted_arr[lo] + (sorted_arr[hi] - sorted_arr[lo]) * (x - lo))


def summarise(r: Result) -> dict:
    e, d = np.sort(r.extra), np.sort(r.delay)
    head = r.cfg.head
    return dict(
        e=e, d=d,
        mean_cost=float(r.extra.mean()), mean_delay=float(r.delay.mean()),
        sd_cost=float(r.extra.std(ddof=1)), sd_delay=float(r.delay.std(ddof=1)),
        se_cost=float(r.extra.std(ddof=1) / math.sqrt(r.n)),
        se_delay=float(r.delay.std(ddof=1) / math.sqrt(r.n)),
        cost_p={p: percentile(e, p) for p in (.5, .7, .8, .9, .95)},
        delay_p={p: percentile(d, p) for p in (.5, .7, .8, .9, .95)},
        p_over_budget=1.0 if head < 0 else float(np.mean(r.extra > head)),
        p_late=float(np.mean(r.delay > r.cfg.buf)),
    )


def merge_diagnostics(r: Result) -> dict | None:
    """
    Quantify E[max] - max[E] at the merge feeding the project finish.

    A hand critical-path sum cannot see this, so it is the main reason a
    manual project mean can differ legitimately from the simulated one.
    """
    sinks = [i for i in range(len(r.acts))
             if not any(i in a.pi for a in r.acts)]
    merges = [(i, a) for i, a in enumerate(r.acts) if len(a.pi) > 1]
    if not merges:
        return None
    # the merge whose predecessors' mean finishes are closest together
    best = None
    for i, a in merges:
        means = [float(r.finishes[p].mean()) for p in a.pi]
        gap = max(means) - min(means)
        if best is None or gap < best[0]:
            best = (gap, i, a, means)
    gap, i, a, means = best
    pre_f = r.finishes[a.pi, :]
    e_max = float(pre_f.max(axis=0).mean())
    max_e = max(means)
    governs = [float(np.mean(pre_f[j] == pre_f.max(axis=0))) for j in range(len(a.pi))]
    return dict(
        merge_id=a.id,
        pred_ids=[r.acts[p].id for p in a.pi],
        pred_mean_finish=means,
        spread=gap,
        e_max=e_max, max_e=max_e, jensen=e_max - max_e,
        governs=governs,
        sinks=[r.acts[s].id for s in sinks],
    )


def hand_pert_project(acts, cp_ids, rho: float, rate: float):
    """Hand calculation for the project, including the corrected SD formula."""
    idx = {a.id: a for a in acts}
    cp = [idx[x] for x in cp_ids]
    mean_delay = sum(pert_mean(a.dO, a.dM, a.dP) for a in cp)
    sd_class = math.sqrt(sum(pert_sd_classical(a.dO, a.dM, a.dP) ** 2 for a in cp))
    sig = [pert_sd_beta(a.dO, a.dM, a.dP) for a in cp]
    sd_beta_rss = math.sqrt(sum(v * v for v in sig))
    cross = sum(sig[i] * sig[j] for i in range(len(sig)) for j in range(i + 1, len(sig)))
    sd_corr = math.sqrt(sum(v * v for v in sig) + 2 * rho ** 2 * cross)
    cost_sum = sum(pert_mean(a.cO, a.cM, a.cP) for a in acts)
    return dict(mean_delay=mean_delay, sd_class=sd_class, sd_beta=sd_beta_rss,
                sd_corr=sd_corr, cost_sum=cost_sum,
                mean_cost=cost_sum + mean_delay * rate)


# ----------------------------------------------------------------------
# input validation
# ----------------------------------------------------------------------
def validate(acts: list[Activity], cfg: Config):
    errors, warnings = [], []
    if not acts:
        errors.append("Add at least one activity.")
        return errors, warnings, []

    index = {}
    for i, a in enumerate(acts):
        if not a.id:
            errors.append(f"Row {i+1}: ID is empty")
        elif a.id in index:
            errors.append(f"Duplicate ID {a.id}")
        index[a.id] = i

    for a in acts:
        a.pi = []
        for p in a.pre:
            if p not in index:
                errors.append(f"{a.id}: unknown predecessor {p}")
            elif p == a.id:
                errors.append(f"{a.id} cannot precede itself")
            else:
                a.pi.append(index[p])
        if a.dur <= 0:
            errors.append(f"{a.id}: duration must be above 0")
        if a.cost < 0:
            errors.append(f"{a.id}: cost cannot be negative")
        for label, (O, M, P) in (("delay", (a.dO, a.dM, a.dP)),
                                 ("extra cost", (a.cO, a.cM, a.cP))):
            if O < 0:
                errors.append(f"{a.id}: {label} optimistic is negative")
            if not (O <= M <= P):
                errors.append(f"{a.id}: {label} must satisfy O <= M <= P")

    if cfg.budget <= 0:
        errors.append("Budget must be above 0")
    if not (0 <= cfg.rho <= 0.9):
        errors.append("Correlation must be between 0 and 0.9")
    if not (1000 <= cfg.n <= 200000):
        errors.append("Iterations must be between 1,000 and 200,000")
    for nm, v in (("mild", cfg.m_mild), ("typical", cfg.m_typ), ("severe", cfg.m_sev)):
        if v < 0:
            errors.append(f"Scenario multiplier ({nm}) must be 0 or more")

    order: list[int] = []
    if not errors:
        order = topological_order(acts)
        if len(order) < len(acts):
            errors.append("Dependency loop detected: check the predecessors.")

    cfg.base = sum(a.cost for a in acts)
    cfg.head = cfg.budget - cfg.base
    if not errors and cfg.head < 0:
        warnings.append(
            f"Activity costs (Rs {cfg.base:,.1f} L) exceed the budget, "
            "so the baseline already overruns before any monsoon risk."
        )

    if not errors:
        ok = monsoon_exposure(acts, order, cfg.start)
        if not ok:
            warnings.append("Enter a valid planned start date to check monsoon overlap.")
        else:
            outside, inside_zero = [], []
            for a in acts:
                risky = a.dP > 0 or a.cP > 0
                # only flag material risk, so small judgement allowances for
                # unseasonal showers do not bury the real warnings
                material = a.dP > 0.1 * a.dur or a.cP > 0.1 * max(a.cost, 1e-9)
                if a.exposure == 0 and risky and material:
                    outside.append(a.id)
                if a.exposure and a.exposure > 0.5 and not risky:
                    inside_zero.append(a.id)
            if outside:
                warnings.append(
                    "Outside June-September but carrying material monsoon risk: "
                    + ", ".join(outside)
                    + ". Justify it (pre-monsoon showers, cyclone, or the risk of the "
                      "activity slipping into the monsoon) or reduce the pessimistic values.")
            if inside_zero:
                warnings.append(
                    "Mostly inside the monsoon but with zero delay and cost risk entered: "
                    + ", ".join(inside_zero) + ".")
    return errors, warnings, order


# ----------------------------------------------------------------------
# rainfall -> scenario multipliers
# ----------------------------------------------------------------------
def rainfall_multipliers(df, threshold: float = 64.5):
    """
    Derive mild and severe multipliers from daily Jun-Sep rainfall.

    Assumption, not an observed relationship: delay and extra cost scale in
    proportion to the seasonal total relative to the median year.
    """
    import pandas as pd
    d = df.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    dcol = d.columns[0]
    vcol = d.columns[1]
    d[dcol] = pd.to_datetime(d[dcol], errors="coerce", dayfirst=True)
    d[vcol] = pd.to_numeric(d[vcol], errors="coerce")
    d = d.dropna(subset=[dcol, vcol])
    d = d[(d[vcol] >= 0) & (d[dcol].dt.month.between(6, 9))]
    if d.empty:
        return None, "No valid June-September rows found. Check the file format."
    g = d.groupby(d[dcol].dt.year).agg(
        total_mm=(vcol, "sum"), heavy_days=(vcol, lambda s: int((s >= threshold).sum())),
        days_with_data=(vcol, "size")).reset_index().rename(columns={dcol: "year"})
    g.columns = ["year", "total_mm", "heavy_days", "days_with_data"]
    if len(g) < 5:
        return g, (f"Only {len(g)} year(s) of June-September data found. "
                   "At least 5 are needed before percentiles mean anything.")
    tot = np.sort(g["total_mm"].to_numpy(dtype=float))
    med = percentile(tot, .5)
    if med <= 0:
        return g, "Median seasonal total is zero; cannot derive multipliers."
    return g, dict(median_mm=med,
                   mild=percentile(tot, .2) / med,
                   severe=percentile(tot, .9) / med,
                   years=len(g))


# ----------------------------------------------------------------------
# self-tests (mirrors the HTML tool's, plus checks it did not have)
# ----------------------------------------------------------------------
def _fixed(spec, rate=0.5):
    """Build activities whose O=M=P, so every draw is deterministic."""
    acts = []
    for i, (dur, pre, d, c) in enumerate(spec):
        acts.append(Activity(id=f"X{i}", dur=dur, pi=list(pre),
                             dO=d, dM=d, dP=d, cO=c, cM=c, cP=c))
    cfg = Config(rho=0.0, rate=rate, n=50, seed=1)
    return acts, cfg


def self_tests(acts=None, order=None, cfg=None) -> list[dict]:
    out = []

    def check(name, ok, detail=""):
        out.append(dict(name=name, ok=bool(ok), detail=detail))

    def fixed_case(spec, order_, exp_d, exp_c):
        a, c = _fixed(spec)
        r = simulate(a, order_, c, n=50, seed=1)
        return (np.allclose(r.delay, exp_d) and np.allclose(r.extra, exp_c),
                f"delay {r.delay[0]:.3f} (expected {exp_d}), "
                f"cost {r.extra[0]:.3f} (expected {exp_c})")

    ok, det = fixed_case([(10, [], 0, 0), (10, [0], 0, 0)], [0, 1], 0, 0)
    check("Zero disruption gives zero extra cost and delay", ok, det)

    ok, det = fixed_case([(10, [], 3, 1), (10, [0], 4, 2)], [0, 1], 7, 6.5)
    check("Series chain: delays 3 + 4 give 7 days and 1 + 2 + 7x0.5 = 6.5 lakh", ok, det)

    ok, det = fixed_case([(10, [], 0, 0), (5, [], 3, 0), (5, [0, 1], 0, 0)], [0, 1, 2], 0, 0)
    check("Float absorbs a delay: branch +3 days against 5 days of float gives 0", ok, det)

    ok, det = fixed_case([(10, [], 0, 0), (5, [], 8, 0), (5, [0, 1], 0, 0)], [0, 1, 2], 3, 1.5)
    check("Delay beyond float propagates: branch +8 against 5 of float gives 3", ok, det)

    # PERT identity: the sampled mean must equal (O+4M+P)/6
    a = [Activity(id="T", dur=10, cost=1, dO=0, dM=2, dP=7, cO=0, cM=2, cP=8)]
    r = simulate(a, [0], Config(rho=0, n=40000, seed=5))
    got, want = float(r.act_delay[0].mean()), pert_mean(0, 2, 7)
    se = float(r.act_delay[0].std(ddof=1) / math.sqrt(40000))
    check("Sampled activity mean matches the PERT mean within 3 standard errors",
          abs(got - want) <= 3 * se,
          f"{got:.4f} vs {want:.4f}, standard error {se:.4f}")

    # Exact beta SD, not the (P-O)/6 approximation, should match the sample SD
    got_sd, want_sd = float(r.act_delay[0].std(ddof=1)), pert_sd_beta(0, 2, 7)
    check("Sampled SD matches the exact beta SD, not the (P-O)/6 approximation",
          abs(got_sd - want_sd) / want_sd < 0.02,
          f"sample {got_sd:.4f}, exact beta {want_sd:.4f}, "
          f"classical {pert_sd_classical(0,2,7):.4f}")

    # Correlation structure: pairwise latent correlation should be rho^2
    a2 = [Activity(id="P", dur=10, dO=0, dM=2, dP=7),
          Activity(id="Q", dur=10, dO=0, dM=2, dP=7)]
    r2 = simulate(a2, [0, 1], Config(rho=0.5, n=40000, seed=11))
    obs = float(np.corrcoef(r2.act_delay[0], r2.act_delay[1])[0, 1])
    check("Between-activity correlation is rho squared, not rho",
          abs(obs - 0.25) < 0.03, f"observed {obs:.3f}, expected 0.250 at rho = 0.5")

    # Monotonicity on the user's own model
    if acts and order and cfg and len(acts) > 1:
        means = []
        for m in (0.6, 1.0, 1.5):
            s = summarise(simulate(acts, order, cfg, n=5000, seed=cfg.seed, dm=m, cm=m))
            means.append((s["mean_cost"], s["mean_delay"]))
        mono = all(means[i][0] <= means[i + 1][0] and means[i][1] <= means[i + 1][1]
                   for i in range(2))
        check("Severity is monotonic on your model: mild <= typical <= severe", mono,
              "cost " + " -> ".join(f"{c:.1f}" for c, _ in means) +
              " | delay " + " -> ".join(f"{d:.1f}" for _, d in means))
    return out
