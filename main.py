"""gta6-trophy-rarity-forecast-01: earners/players two-curve forecaster.

Forecasts every GTA VI PS5 trophy's earn rate on Exophase at the case's
target as five quantiles (levels 0.05/0.25/0.5/0.75/0.95) on the logit
scale. Python 3.13 standard library only, deterministic, no network.

Model
-----
Two curves are fitted in log-time x = ln(1 + hours since release):

  * N(t): the tracked-player count, fitted as log N ~ a + b*x on hourly
    bins (median N per bin) over the reads since the first refresh.
  * E_i(t): each trophy's earner count, fitted as log E_i ~ a_i + b_i*x
    over the reads since the first refresh where the trophy's earners
    are at least 1.

The forecast rate is 100 * E_i(target) / N(target), with the target
shifted back by half the median Exophase refresh lag (the page shows
rates as of its last refresh, not as of the read). The spread combines
both fits' residual std with a random-walk term over the horizon.

When the history has never refreshed (every read identical), the fits
above are flat and useless, so a launch-refresh prior is used instead
(see flat_forecast): the first launch-night refresh disclosed in
RULES.txt, scaled to the case's own player count.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves the entry directory off sys.path: put it back
# first, so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# GTA VI release: 2026-11-19T00:00:00Z.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Launch-refresh prior, from RULES.txt: on launch night Exophase's page
# sat unchanged at 638 players tracked and 943 trophies earned from
# 23:48Z until the 04:00Z read, then refreshed by the 04:30Z read to
# 3,868 players tracked and 4,276 trophies earned. So the refresh added
# 3,868 - 638 = 3,230 new players and 4,276 - 943 = 3,333 new trophies:
# about one trophy per new player (the opening activity every new
# player does), with the remaining ~103 trophies spread over the
# trophies the old players already held, in proportion to how many
# held each one. A trophy every tracked player already held (100%)
# stays held by everyone. This prior applies ONLY to a history that
# has never refreshed; a history with a refresh is fitted instead.
FLAT_N_RATIO = 3868.0 / 638.0
FLAT_TROPHY_RATIO = 3333.0 / 3230.0
FLAT_SPREAD = 0.20


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(lg: float) -> float:
    # Numerically stable: 100 / (1 + exp(-lg)) without overflow.
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt: datetime) -> float:
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


def log_time(hours: float) -> float:
    return math.log1p(max(0.0, hours))


def median(values) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def least_squares(xs, ys):
    """Ordinary least squares y ~ intercept + slope * x.

    Returns (intercept, slope, residual std) or None when there are
    fewer than 3 points or x has no variance.
    """
    n = len(xs)
    if n < 3:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    resid = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    return (intercept, slope, resid)


def first_refresh_index(reads):
    """Index of the first read that differs from the first read, or None.

    A read differs when its players_tracked or its rates differ from
    the first read's.
    """
    first = reads[0]
    for i, read in enumerate(reads):
        if (read["players_tracked"] != first["players_tracked"]
                or read["rates"] != first["rates"]):
            return i
    return None


def refresh_lag_hours(reads, first_idx: int) -> float:
    """Half the median gap between Exophase refreshes, in hours.

    Collects (N, at) each time the tracked-player count changes from
    the first refreshed read on; with fewer than 3 distinct counts
    there is no lag to estimate, so 0.0. Otherwise the median gap
    between changes, halved (the page lags its refresh by half a
    refresh interval on average), bounded to [0, 6].
    """
    changes = []
    prev = None
    for read in reads[first_idx:]:
        n = read["players_tracked"]
        if n != prev:
            changes.append((n, parse_time(read["at"])))
            prev = n
    if len({n for n, _ in changes}) < 3:
        return 0.0
    gaps = [
        (changes[k + 1][1] - changes[k][1]).total_seconds() / 3600.0
        for k in range(len(changes) - 1)
    ]
    lag = median(gaps) / 2.0
    return min(6.0, max(0.0, lag))


def flat_forecast(case: dict, reads):
    """Launch-refresh prior for a history that has never refreshed.

    Returns [(rate_star, spread)] per trophy, in trophy order.
    See the FLAT_* constants above for where the numbers come from.
    """
    last = reads[-1]
    n_last = float(last["players_tracked"])
    n_star = n_last * FLAT_N_RATIO
    new_players = n_star - n_last
    base = max(float(last["trophies_earned"]) - n_last, 0.0)
    extra_pool = new_players * (FLAT_TROPHY_RATIO - 1.0)
    out = []
    for i, _trophy in enumerate(case["trophies"]):
        rate_last = last["rates"][i]
        if rate_last is None:
            rate_last = 0.0
        earners_last = rate_last / 100.0 * n_last
        if rate_last >= 99.99:
            earners_star = n_star
        else:
            share = earners_last / base if base > 0 else 0.0
            earners_star = earners_last + extra_pool * share
        rate_star = 100.0 * earners_star / n_star
        out.append((rate_star, FLAT_SPREAD))
    return out


def fitted_forecast(case: dict, reads, first_idx: int):
    """Log-time fits of N(t) and of each trophy's earner count.

    Returns [(rate_star, spread)] per trophy, in trophy order.
    """
    lag = refresh_lag_hours(reads, first_idx)
    target = parse_time(case["target"])
    x_target = log_time(hours_since_release(target) - lag)
    horizon_days = case["horizon_minutes"] / 1440.0

    # (a) Fit N(t): median N per whole-hour bin since the refresh,
    # log N ~ a + b * x.
    bins = {}
    for read in reads[first_idx:]:
        hour = int(hours_since_release(parse_time(read["at"])))
        bins.setdefault(hour, []).append(read["players_tracked"])
    xs = [log_time(hour) for hour in sorted(bins)]
    ys = [math.log(median(bins[hour])) for hour in sorted(bins)]
    fit_n = least_squares(xs, ys)
    if fit_n is None:
        n_star = float(reads[-1]["players_tracked"])
        resid_n = 0.5
    else:
        a_n, b_n, resid_n = fit_n
        n_star = max(1.0, math.exp(a_n + b_n * x_target))

    # (b) Fit each trophy's earner count the same way.
    out = []
    for i, _trophy in enumerate(case["trophies"]):
        xs_i = []
        ys_i = []
        for read in reads[first_idx:]:
            rate = read["rates"][i]
            if rate is None:
                continue
            earners = rate / 100.0 * read["players_tracked"]
            if earners < 1.0:
                continue
            xs_i.append(log_time(hours_since_release(parse_time(read["at"]))))
            ys_i.append(math.log(earners))
        fit = least_squares(xs_i, ys_i)
        if fit is None:
            last_rate = reads[-1]["rates"][i]
            rate_star = last_rate if last_rate is not None else 0.0
            spread = 1.0
        else:
            a_i, b_i, resid_i = fit
            earners_star = min(max(math.exp(a_i + b_i * x_target), 0.0), n_star)
            rate_star = 100.0 * earners_star / n_star
            spread = math.sqrt(
                resid_i ** 2 + resid_n ** 2
                + (0.5 * math.sqrt(horizon_days)) ** 2
            )
            spread = min(1.50, max(0.15, spread))
        out.append((rate_star, spread))
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    reads = case["history"]
    if not reads:
        raise ValueError(f"{name} has no history")
    first_idx = first_refresh_index(reads)
    if first_idx is None:
        rates = flat_forecast(case, reads)
    else:
        rates = fitted_forecast(case, reads, first_idx)
    rows = []
    for trophy, (rate_star, spread) in zip(case["trophies"], rates):
        lg = logit(rate_star)
        qs = [round4(clamp(inv_logit(lg + z * spread))) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
