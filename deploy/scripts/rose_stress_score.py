#!/usr/bin/env python3
"""Score RoSE flight stress-test cells from their ground-truth trajectory CSVs.

Reads every *.csv in a results dir (as produced by rose_stress_harness.py), computes the
metrics from docs/ROSE_FLIGHT_STRESS_PLAN.md section 5.1 over the steady window (excluding
the takeoff transient), and classifies each cell PASS / DEGRADED / FAIL against the section
5.2 level-1 thresholds. Emits a markdown table to stdout and writes scores.md + scores.json.

Stdlib only (csv + math) so it runs under any python3 — no numpy needed.

CSV columns (written by crazyflie_mpc_env.py):
    t,x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz, f0..f3, u0..u3, tx,ty,tz

Usage:
    deploy/scripts/rose_stress_score.py <results-dir> [--steady-start 3.0]
"""
import argparse
import csv
import glob
import json
import math
import os

# level-1 PASS bounds (section 5.2). DEGRADED = exceeds one bound by <2x; FAIL = >2x, crash,
# divergence, NaN, or a truncated run.
BOUNDS = {
    "alt_rms_err_m": 0.03,
    "alt_ripple_m": 0.05,
    "vel_rms_ms": 0.10,
    "max_tilt_deg": 10.0,
}
CRASH_Z = 0.05        # below this = crashed
DIVERGE_M = 5.0       # horizontal/pos norm beyond this = diverged
MIN_STEADY_S = 7.0    # need at least this much steady flight to be scoreable


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(path):
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        header = next(r, None)
        for line in r:
            if len(line) < 25:
                continue
            rows.append([_f(v) for v in line])
    return rows


def score_cell(path, steady_start):
    rows = load(path)
    m = {"cell": os.path.splitext(os.path.basename(path))[0], "n": len(rows)}
    if not rows:
        m["verdict"] = "FAIL"; m["note"] = "empty"; return m

    t_end = rows[-1][0]
    m["duration_s"] = round(t_end, 2)

    # --- survival scan over the WHOLE run ---
    crashed = diverged = nan = False
    survive_t = t_end
    for row in rows:
        t, x, y, z = row[0], row[1], row[2], row[3]
        if any(math.isnan(v) for v in (x, y, z)):
            nan = True; survive_t = t; break
        if z < CRASH_Z:
            crashed = True; survive_t = t; break
        if math.sqrt(x * x + y * y + z * z) > DIVERGE_M + 1.0:  # +1 since hover z~1
            diverged = True; survive_t = t; break
    m["survival_s"] = round(survive_t, 2)

    # --- steady-window metrics ---
    steady = [r for r in rows if r[0] >= steady_start and r[0] <= survive_t]
    if len(steady) < 10:
        m["verdict"] = "FAIL"
        m["note"] = "crash/diverge/NaN" if (crashed or diverged or nan) else "no steady window"
        return m

    zs = [r[3] for r in steady]
    tzs = [r[24] for r in steady]
    m["alt_rms_err_m"] = round(math.sqrt(sum((z - tz) ** 2 for z, tz in zip(zs, tzs)) / len(zs)), 4)
    m["alt_ripple_m"] = round(max(zs) - min(zs), 4)

    vmag = [math.sqrt(r[8] ** 2 + r[9] ** 2 + r[10] ** 2) for r in steady]
    m["vel_rms_ms"] = round(math.sqrt(sum(v * v for v in vmag) / len(vmag)), 4)

    # tilt from level = angle of body-z axis vs world-z = acos(1 - 2(qx^2+qy^2))
    tilts = []
    for r in steady:
        qx, qy = r[5], r[6]
        c = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
        tilts.append(math.degrees(math.acos(c)))
    m["max_tilt_deg"] = round(max(tilts), 2)

    # horizontal drift rate: ||(x,y)|| growth per second across the steady window
    def hnorm(r):
        return math.sqrt(r[1] ** 2 + r[2] ** 2)
    dt = steady[-1][0] - steady[0][0]
    m["horiz_drift_ms"] = round((hnorm(steady[-1]) - hnorm(steady[0])) / dt, 5) if dt > 0 else 0.0

    # control effort + saturation (u limit unknown from CSV alone -> report magnitude only)
    us = [max(abs(r[18]), abs(r[19]), abs(r[20]), abs(r[21])) for r in steady]
    m["max_abs_u"] = round(max(us), 3)
    m["ctrl_rms_u"] = round(math.sqrt(sum(u * u for u in us) / len(us)), 3)

    # --- verdict ---
    truncated = survive_t < steady_start + MIN_STEADY_S
    if crashed or diverged or nan or truncated:
        m["verdict"] = "FAIL"
        m["note"] = ("crash" if crashed else "diverge" if diverged else
                     "NaN" if nan else "truncated")
        return m
    worst = 1.0
    breach = []
    for k, bound in BOUNDS.items():
        ratio = m[k] / bound
        worst = max(worst, ratio)
        if ratio > 1.0:
            breach.append("%s=%.3f>%.3f" % (k, m[k], bound))
    if worst <= 1.0:
        m["verdict"] = "PASS"
    elif worst <= 2.0:
        m["verdict"] = "DEGRADED"; m["note"] = ", ".join(breach)
    else:
        m["verdict"] = "FAIL"; m["note"] = "exceeds 2x: " + ", ".join(breach)
    return m


COLS = ["cell", "verdict", "duration_s", "survival_s", "alt_rms_err_m", "alt_ripple_m",
        "vel_rms_ms", "max_tilt_deg", "horiz_drift_ms", "max_abs_u", "note"]


def render_md(scored):
    head = "| " + " | ".join(COLS) + " |"
    sep = "| " + " | ".join("---" for _ in COLS) + " |"
    lines = [head, sep]
    for m in scored:
        lines.append("| " + " | ".join(str(m.get(c, "")) for c in COLS) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--steady-start", type=float, default=3.0,
                    help="seconds of takeoff transient to exclude (default 3.0)")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.results_dir, "*.csv")))
    if not csvs:
        raise SystemExit("no CSVs in %s" % args.results_dir)
    scored = [score_cell(p, args.steady_start) for p in csvs]

    md = render_md(scored)
    print(md)
    counts = {}
    for m in scored:
        counts[m["verdict"]] = counts.get(m["verdict"], 0) + 1
    summary = "  ".join("%s=%d" % (k, counts[k]) for k in ("PASS", "DEGRADED", "FAIL") if k in counts)
    print("\n" + summary)

    with open(os.path.join(args.results_dir, "scores.md"), "w") as f:
        f.write(md + "\n\n" + summary + "\n")
    with open(os.path.join(args.results_dir, "scores.json"), "w") as f:
        json.dump(scored, f, indent=2)


if __name__ == "__main__":
    main()
