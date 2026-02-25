"""
Analyze the SymFD sensitivity sweep results.

Groups runs by (n_collapse_pairs, buffer_size) and computes
mean ± std of moving_avg_return across seeds.

Usage:
    python analyze_sym_fd_sweep.py                         # uses default SWEEP_ID
    python analyze_sym_fd_sweep.py <entity/project/sweep>  # explicit sweep ID
"""
import sys
import wandb
import pandas as pd
import numpy as np

# ── Sweep ID ─────────────────────────────────────────────────────────────────
DEFAULT_SWEEP_ID = "YOUR_ENTITY/Deep-CVI-Experiments/SWEEP_ID_HERE"
SWEEP_ID = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SWEEP_ID

# ── Fetch runs ────────────────────────────────────────────────────────────────
api = wandb.Api()
sweep = api.sweep(SWEEP_ID)
print(f"Fetching runs from sweep: {SWEEP_ID}")

rows = []
for run in sweep.runs:
    if run.state != "finished":
        continue
    rows.append({
        "seed":              run.config.get("seed"),
        "n_collapse_pairs":  run.config.get("n_collapse_pairs"),
        "buffer_size":       run.config.get("buffer_size"),
        "moving_avg":        run.summary.get("charts/moving_avg_return"),
        "q_values":          run.summary.get("losses/q_values"),
        "target_q":          run.summary.get("diagnostics/target_q_values"),
        "q_slope_spread":    run.summary.get("diagnostics/q_slope_spread"),
        "cf_phase_max_pair": run.summary.get("diagnostics/cf_phase_max_pair"),
        "q_action_gap":      run.summary.get("diagnostics/q_action_gap"),
        "run_id":            run.id,
    })

df = pd.DataFrame(rows)
print(f"Total finished runs: {len(df)}")

# ── Aggregate over seeds ──────────────────────────────────────────────────────
group_cols = ["n_collapse_pairs", "buffer_size"]
agg = df.groupby(group_cols).agg(
    n_seeds      = ("seed", "count"),
    avg_return    = ("moving_avg", "mean"),
    std_return    = ("moving_avg", "std"),
    min_return    = ("moving_avg", "min"),
    max_return    = ("moving_avg", "max"),
    avg_q         = ("q_values", "mean"),
    avg_spread    = ("q_slope_spread", "mean"),
    avg_phase_max = ("cf_phase_max_pair", "mean"),
).reset_index().sort_values("avg_return", ascending=False)

# ── Display results ───────────────────────────────────────────────────────────
pd.set_option("display.float_format", "{:.1f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)

print("\n" + "=" * 120)
print("SENSITIVITY RESULTS — grouped by (n_collapse_pairs, buffer_size), aggregated over seeds")
print("=" * 120)
print(agg.to_string(index=False))

# ── Best config ───────────────────────────────────────────────────────────────
best = agg.iloc[0]
print("\n" + "=" * 120)
print("BEST CONFIG")
print("=" * 120)
print(f"  n_collapse_pairs = {int(best['n_collapse_pairs'])}")
print(f"  buffer_size      = {int(best['buffer_size'])}")
print(f"  moving_avg       = {best['avg_return']:.1f} ± {best['std_return']:.1f}")
print(f"  range            = [{best['min_return']:.1f}, {best['max_return']:.1f}]")
print(f"  avg Q            = {best['avg_q']:.1f}")
print(f"  avg slope spread = {best['avg_spread']:.1f}")
print(f"  n_seeds          = {int(best['n_seeds'])}")

# ── Per-seed breakdown for the best config ────────────────────────────────────
best_runs = df[
    (df["n_collapse_pairs"] == best["n_collapse_pairs"]) &
    (df["buffer_size"] == best["buffer_size"])
].sort_values("seed")

print("\n  Per-seed breakdown:")
for _, r in best_runs.iterrows():
    print(f"    seed {int(r['seed']):3d}: moving_avg={r['moving_avg']:.1f}  Q={r['q_values']:.1f}  slope_spread={r['q_slope_spread']:.1f}")

# ── Thesis-ready: is moving_avg ≥ 460 for ALL seeds? ─────────────────────────
target = 460
all_pass = (best_runs["moving_avg"] >= target).all()
print(f"\n  All seeds ≥ {target}?  {'✓ YES — ready for harder envs' if all_pass else '✗ NO — needs more tuning'}")
