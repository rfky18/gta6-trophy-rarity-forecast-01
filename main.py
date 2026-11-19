"""gta6-trophy-rarity-forecast-01: earners/players two-curve forecaster.

Forecasts every GTA VI PS5 trophy's Exophase earn rate at a case's target from
the case's history of page reads. Python 3.13, standard library only, no
network, deterministic (same bytes on every run), fast (milliseconds a case).

Model, in one paragraph: a trophy's displayed rate is earners / players_tracked,
and both curves move. Exophase refreshes a game's counts on its own schedule,
hours apart, so the page a read shows is the state as of the last refresh before
it. The program therefore forecasts TWO curves in log time (hours since the
19 Nov 00:00 UTC release, via ln(1 + hours), the transform the house curve
baseline uses): the tracked-player count N(t), fitted on the reads after the
first refresh, and each trophy's earner count E_i(t), fitted the same way;
the median rate is E_i/N at the target, less the expected refresh lag. The
quantile spread combines each fit's residual with a random-walk term that
widens with the horizon, so the band is honest about how far the target is.

The one case the history cannot speak about is a page that has never
refreshed: every read identical, no refresh seen yet. Then the program falls
back to the launch-refresh prior the contest's own rules disclose (RULES.txt:
the page sat unchanged at 638 tracked from 23:48Z and refreshed between
04:00Z and 04:30Z to 3,868 tracked and 4,276 trophies earned): it expects
that first refresh before the target, the tracked count multiplying about
sixfold, each new player bringing about one trophy (the opening activity),
the rest of the new trophies shared among the trophies players already hold.
That prior is used ONLY when the history is flat; any refresh visible in
the history switches the program to the fitted curves above.

How the grader runs it: `python3 -P -s /bot/main.py` in python:3.13-slim.
It reads every <id>.in in the input directory (one case per run) and writes
<id>.out to the output directory: {"trophies": [{"id": n, "q": [q05, q25, q50,
q75, q95]}, ...]}, five quantiles per trophy, 0-100, non-decreasing.
MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and /output
outside the grader's container.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The game's release: 2026-11-19T00:00:00Z. Log time is measured from here.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Launch-refresh prior (used ONLY when the case history is flat) --------
# From RULES.txt: the launch-night page was unchanged at 638 players tracked
# and 943 trophies earned from 23:48Z, then refreshed between 04:00Z and
# 04:30Z to 3,868 tracked and 4,276 trophies earned. New players 3,230,
# new trophies 3,333: about one trophy per new player (the opening activity
# every new player does first), the rest spread over the trophies already held.
PRIOR_TRACKED_MULT = 3868.0 / 638.0        # ~6.06x at the first refresh
PRIOR_TROPHIES_PER_NEW_PLAYER = 3333.0 / 3230.0  # ~1.032
PRIOR_SPREAD = 0.20  # logit spread while the refresh is only a prior

# Random-walk spread scale (logit per sqrt day), as the house baseline uses.
RW_SIGMA = 0.5
MIN_SPREAD = 0.15
MAX_SPREAD = 1.50
MAX_REFRESH_LAG_HOURS = 6.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(when: datetime) -> float:
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def log_time(hours: float) -> float:
    return math.log1p(hours)


def least_squares(xs, ys):
    """Return (intercept, slope, residual std) of y ~ a + b*x, or None."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return (a, b, resid)


def first_refresh_index(reads):
    """Index of the first read where the page differs from the first read.

    A refresh shows as a new players_tracked count (or any rate moving).
    None when the page never changed: no refresh has been seen yet.
    """
    if not reads:
        return None
    n0 = reads[0]["players_tracked"]
    r0 = reads[0]["rates"]
    for i, read in enumerate(reads):
        if read["players_tracked"] != n0:
            return i
        if read["rates"] != r0:
            return i
    return None


def refresh_lag_hours(reads, first_idx):
    """Half the median gap between successive distinct tracked counts.

    The displayed value at the target is the state as of the last refresh
    before it, so the fits are read that far before the target. Zero when
    fewer than two distinct counts were seen.
    """
    seen = []
    for read in reads[first_idx:]:
        n = read["players_tracked"]
        if not seen or seen[-1][0] != n:
            seen.append((n, parse_time(read["at"])))
    if len(seen) < 3:
        return 0.0
    gaps = [(seen[i + 1][1] - seen[i][1]).total_seconds() / 3600.0
            for i in range(len(seen) - 1)]
    gaps.sort()
    med = gaps[len(gaps) // 2]
    return min(MAX_REFRESH_LAG_HOURS, max(0.0, med / 2.0))


def forecast_flat_history(case, last, horizon_days):
    """Prior-path forecast: the page never refreshed in the history.

    Expects the first refresh before the target (RULES.txt prior): tracked
    multiplies ~6x, each new player brings ~1 trophy, the opening activity
    first. A trophy every tracked player already holds stays held.
    """
    n_last = float(last["players_tracked"])
    n_star = n_last * PRIOR_TRACKED_MULT
    new_players = n_star - n_last
    total_earned = float(last["trophies_earned"])
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate_last = last["rates"][i]
        if rate_last is None:
            rate_last = 0.0
        earners_last = rate_last / 100.0 * n_last
        if rate_last >= 99.99:
            # Held by every tracked player: the new ones hold it too.
            earners_star = n_star
        else:
            others = max(total_earned - earners_last, 0.0)
            # The new trophies beyond one per new player, shared pro rata
            # to what each trophy already holds.
            extra_pool = max(new_players * (PRIOR_TROPHIES_PER_NEW_PLAYER - 1.0), 0.0)
            base = max(total_earned - n_last, 0.0)  # trophies beyond the opener
            share = earners_last / base if base > 0 else 0.0
            earners_star = earners_last + extra_pool * share
        rate_star = 100.0 * earners_star / n_star if n_star > 0 else rate_last
        rows.append((trophy["id"], rate_star, PRIOR_SPREAD))
    return rows


def forecast_fitted(case, reads, first_idx, horizon_days):
    """Two-curve forecast from a history that contains at least one refresh."""
    lag = refresh_lag_hours(reads, first_idx)
    target = parse_time(case["target"])
    x_target = log_time(hours_since_release(target) - lag)

    # --- Tracked players N(t): log N ~ a + b * ln(1 + hours since release).
    # One point per hour bin (median N in the bin) so the 10-minute cadence
    # does not overweight the flat stretches between refreshes.
    bins = {}
    for read in reads[first_idx:]:
        h = int(hours_since_release(parse_time(read["at"])))
        bins.setdefault(h, []).append(float(read["players_tracked"]))
    xs, ys = [], []
    for h in sorted(bins):
        vals = sorted(bins[h])
        n_med = vals[len(vals) // 2]
        if n_med > 0:
            xs.append(log_time(float(h)))
            ys.append(math.log(n_med))
    fit_n = least_squares(xs, ys)
    if fit_n is None:
        n_star = float(reads[-1]["players_tracked"])
        resid_n = 0.5
    else:
        a_n, b_n, resid_n = fit_n
        n_star = math.exp(a_n + b_n * x_target)
    n_star = max(n_star, 1.0)

    # --- Earners E_i(t) per trophy: log E_i ~ a_i + b_i * ln(1 + hours).
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        xs_i, ys_i = [], []
        for read in reads[first_idx:]:
            rate = read["rates"][i]
            if rate is None:
                continue
            earners = rate / 100.0 * float(read["players_tracked"])
            if earners < 1.0:
                continue
            xs_i.append(log_time(hours_since_release(parse_time(read["at"]))))
            ys_i.append(math.log(earners))
        fit_i = least_squares(xs_i, ys_i)
        rate_last = reads[-1]["rates"][i]
        if rate_last is None:
            rate_last = 0.0
        if fit_i is None:
            # Too little movement to fit: hold the last rate, wide band.
            rows.append((trophy["id"], rate_last, 1.0))
            continue
        a_i, b_i, resid_i = fit_i
        earners_star = math.exp(a_i + b_i * x_target)
        earners_star = min(max(earners_star, 0.0), n_star)
        rate_star = 100.0 * earners_star / n_star
        spread = math.sqrt(resid_i ** 2 + resid_n ** 2
                           + (RW_SIGMA * math.sqrt(horizon_days)) ** 2)
        spread = min(MAX_SPREAD, max(MIN_SPREAD, spread))
        rows.append((trophy["id"], rate_star, spread))
    return rows


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    last = case["history"][-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    first_idx = first_refresh_index(case["history"])
    if first_idx is None:
        medians = forecast_flat_history(case, last, horizon_days)
    else:
        medians = forecast_fitted(case, case["history"], first_idx, horizon_days)
    rows = []
    for tid, rate_star, spread in medians:
        lg = logit(rate_star)
        qs = [round4(clamp(inv_logit(lg + z * spread))) for z in Z]
        # Non-decreasing, after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
