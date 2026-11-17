#!/usr/bin/env python3
"""Forecast a launch day's Twitch category viewers, hours 2-24, from hour 1.

Model: pool the release-day curves of the case's comparable launches.
Each comparable contributes, per hour h, the log ratio ln(viewers_h /
viewers_1) of its own release day, weighted by how close its hour-1
audience is to this case's hour-1 print (Gaussian kernel on log scale).
Only comparables whose hour 1 was already a real audience (at least
5% of their day peak, i.e. rolling-unlock launches like this one)
shape the pool. The forecast for an hour is the case's hour-1 print
times the exponentiated average, in log space, of the weighted
empirical quantiles of those log ratios and the quantiles of a
weighted normal fit to them: the blend keeps the empirical shape
while smoothing the noisy 5% and 95% tails of a small pool.

Deterministic, standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from statistics import NormalDist

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZS = [-1.6448536269514722, -0.6744897501960817, 0.0,
      0.6744897501960817, 1.6448536269514722]
MIN_H1_SHARE = 0.05   # comparable's hour-1 viewers / its day peak
KERNEL_BW = 1.0       # log-scale bandwidth for the similarity kernel
MIN_POOL = 8          # below this the share filter is dropped


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_curve(comp):
    """Release-day hourly viewers of one comparable, keyed by hour 0-23."""
    try:
        start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    day = {}
    for point in comp.get("hourly") or []:
        try:
            at = parse_ts(point["from"])
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers > 0 and start <= at < start + timedelta(hours=24):
            day[int((at - start).total_seconds() // 3600)] = viewers
    return day


def build_pool(comparables):
    """Usable pool entries: (hour-1 viewers, {hour index: viewers})."""
    pool = []
    for comp in comparables or []:
        day = day_curve(comp)
        if 0 not in day or len(day) < 20:
            continue
        pool.append((day[0], day, day[0] / max(day.values())))
    return pool


def weighted_quantile(pairs, tau):
    """First value whose cumulative weight reaches tau of the total."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= tau * total:
            return value
    return pairs[-1][0]


def weighted_mean_sd(pairs):
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0], 0.3
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    return mean, math.sqrt(var) if var > 0 else 0.3


_NORMAL = NormalDist()


def z_score(tau):
    """Standard normal quantile at tau, from the standard library."""
    return _NORMAL.inv_cdf(tau)


def hour1_print(case):
    known = case.get("known") or []
    if isinstance(known, dict):
        known = [known]
    for row in known:
        try:
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers > 0:
            return viewers
    for row in reversed(case.get("history_hourly") or []):
        try:
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers > 0:
            return viewers
    return 1.0


def baseline_row(p, hour_index, levels):
    return [p * math.exp(z * 0.15 * math.sqrt(hour_index))
            for z in ZS[:len(levels)]]


def forecast_case(case):
    p = max(1.0, hour1_print(case))
    levels = case.get("levels") or DEFAULT_LEVELS
    hours = [row["hour"] for row in case.get("forecast") or []
             ] or list(range(2, 25))
    pool = build_pool(case.get("comparables"))
    shaped = [entry for entry in pool if entry[2] >= MIN_H1_SHARE]
    if len(shaped) >= MIN_POOL:
        pool = shaped
    log_p = math.log(p)
    rows = []
    for hour in hours:
        hi = hour - 1
        pairs = []
        for h1, day, _share in pool:
            if hi in day:
                weight = math.exp(-0.5 * ((math.log(h1) - log_p)
                                          / KERNEL_BW) ** 2)
                pairs.append((math.log(day[hi] / h1), weight))
        if len(pairs) >= 4:
            pairs.sort()
            mean, sd = weighted_mean_sd(pairs)
            quantiles = []
            for tau in levels:
                emp = weighted_quantile(pairs, tau)
                nrm = mean + z_score(tau) * sd
                quantiles.append(p * math.exp(0.5 * (emp + nrm)))
        else:
            quantiles = baseline_row(p, hi, levels)
        quantiles = sorted(quantiles)
        quantiles = [min(1e9, max(0.01, round(q, 2))) for q in quantiles]
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        names = []
    for name in names:
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = str(case.get("id") or name[:-3]) + ".out"
        with open(os.path.join(out_dir, out_name), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
