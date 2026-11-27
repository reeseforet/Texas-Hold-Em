"""Official-engine matches for the ablation and the overnight run.

Stacks persist, side pots are the engine's, and the button moves. The deck
seed is separate from each bot's Monte Carlo seed. The same seed is used for
every policy so the comparison is paired until a player busts.
"""

import math
import time

from match import AdaptiveBot, RangeThresholdBot, ThresholdBot, load_params, play_match


STATION = {"raise_at": 0.9, "call_margin": -0.08, "raise_fraction": 0.25, "preflop_floor": 0.0}
ROCK = {"raise_at": 0.72, "call_margin": 0.08, "raise_fraction": 0.5, "preflop_floor": 0.28}


def style_params():
    return {
        "aggressor": load_params("aggressor"),
        "evolved": load_params("evolved"),
        "station": STATION,
        "rock": ROCK,
    }


def tables():
    """Five- and six-handed mixes, plus a table of each single style."""
    return {
        "mixed-5": ["aggressor", "evolved", "station", "rock"],
        "mixed-6": ["aggressor", "evolved", "station", "rock", "aggressor"],
        "vs-aggressor": ["aggressor"] * 4,
        "vs-evolved": ["evolved"] * 4,
        "vs-station": ["station"] * 4,
        "vs-rock": ["rock"] * 4,
    }


def _rotate(seats, shift):
    shift %= len(seats)
    return seats[shift:] + seats[:shift]


def build_match(policy, opponents, trials, shift):
    """Return (player_id, factory) pairs. Hero sits in a rotated seat."""
    params = style_params()
    specs = [("Hero", policy)]
    seen = {}
    for style in opponents:
        seen[style] = seen.get(style, 0) + 1
        specs.append((f"{style}-{seen[style]}", style))
    specs = _rotate(specs, shift)

    def factory(kind, player_id):
        if kind == "adaptive":
            return lambda: AdaptiveBot(player_id, trials=trials["adaptive"], use_ranges=True)
        if kind == "ev":
            return lambda: AdaptiveBot(player_id, trials=trials["ev"], use_ranges=False)
        if kind == "aggressor_bayes":
            return lambda: RangeThresholdBot(player_id, params["aggressor"], trials=trials["bayes"])
        if kind in params:
            return lambda: ThresholdBot(player_id, params[kind], trials=trials["threshold"])
        raise KeyError(kind)

    return [(player_id, factory(kind, player_id)) for player_id, kind in specs]


def play_one(policy, table, seed, hands, trials, shift=0):
    started = time.perf_counter()
    builders = build_match(policy, tables()[table], trials, shift)
    holders = []
    wrapped = []
    for player_id, factory in builders:
        def make(factory=factory, holders=holders):
            bot = factory()
            holders.append(bot)
            return bot
        wrapped.append((player_id, make))
    deltas = play_match(wrapped, hands=hands, seed=seed)
    hero = next(bot for bot in holders if getattr(bot, "player_id", None) == "Hero")
    latencies = getattr(hero, "latencies", []) or []
    others = [value for player_id, value in deltas.items() if player_id != "Hero"]
    return {
        "policy": policy,
        "table": table,
        "seed": seed,
        "hands": hands,
        "hero_chips": deltas["Hero"],
        "won": deltas["Hero"] > max(others) if others else deltas["Hero"] > 0,
        "survived": deltas["Hero"] > -1000,
        "latency_ms": round(1000 * sum(latencies) / len(latencies), 2) if latencies else None,
        "decisions": len(getattr(hero, "decisions", []) or []),
        "elapsed": round(time.perf_counter() - started, 3),
        "deltas": deltas,
    }


def mean_se(values):
    count = len(values)
    if count == 0:
        return {"n": 0, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0}
    mean = sum(values) / count
    if count == 1:
        return {"n": 1, "mean": mean, "se": 0.0, "low": mean, "high": mean}
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    se = math.sqrt(variance / count)
    return {"n": count, "mean": mean, "se": se, "low": mean - 1.96 * se, "high": mean + 1.96 * se}


def bb_per_100(rows):
    hands = sum(row["hands"] for row in rows)
    chips = sum(row["hero_chips"] for row in rows)
    if hands == 0:
        return 0.0
    return (chips / hands) / 20 * 100


def summarize(rows):
    """Group by policy and table. Intervals are 95% standard-error intervals."""
    groups = {}
    for row in rows:
        groups.setdefault((row["policy"], row["table"]), []).append(row)
    report = []
    for (policy, table), group in sorted(groups.items()):
        chips = mean_se([row["hero_chips"] for row in group])
        wins = sum(1 for row in group if row["won"]) / len(group)
        survived = sum(1 for row in group if row["survived"]) / len(group)
        latencies = [row["latency_ms"] for row in group if row["latency_ms"] is not None]
        report.append({
            "policy": policy,
            "table": table,
            "matches": len(group),
            "chips": chips,
            "bb_per_100": round(bb_per_100(group), 2),
            "win_rate": round(wins, 3),
            "survival": round(survived, 3),
            "latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        })
    return report


def paired_gain(rows, candidate, baseline, table):
    """Mean chip difference on seeds both policies played at this table."""
    by_seed = {}
    for row in rows:
        if row["table"] != table or row["policy"] not in (candidate, baseline):
            continue
        by_seed.setdefault(row["seed"], {})[row["policy"]] = row["hero_chips"]
    diffs = []
    for pair in by_seed.values():
        if candidate in pair and baseline in pair:
            diffs.append(pair[candidate] - pair[baseline])
    return mean_se(diffs)


def convincing(rows, candidate="adaptive"):
    """True only when the paired gain over both frozen baselines clears noise."""
    if not rows:
        return False
    tables_played = sorted({row["table"] for row in rows if row["policy"] == candidate})
    if not tables_played:
        return False
    for table in tables_played:
        for baseline in ("aggressor", "evolved"):
            gain = paired_gain(rows, candidate, baseline, table)
            if gain["n"] < 8 or gain["low"] <= 0:
                return False
    return True
