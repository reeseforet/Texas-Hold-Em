"""One-step expected value for the legal actions at this decision.

Folding is worth 0: the chips already in the pot are sunk. Checking or
calling uses the share of the pot we get if the called players show down.
A raise wins the current pot only when every opponent who can fold does
fold. Otherwise the pot is split into side pots by how much each caller
can actually put in.

Called ranges get tighter as the bet grows. Unknown opponents use the
population fold rate, not a forced call and not a frequent fold.
"""

import time

from bot import parse_card
from equity import board_percentiles, sample_ranks
from opponent_model import FOLD_PRIOR


def _me(state, hero_id):
    return next(player for player in state["players"] if player["id"] == hero_id)


def _live(state, hero_id):
    return [player for player in state["players"] if player["id"] != hero_id and not player.get("folded")]


def raise_targets(state, me):
    """Total street bets that satisfy the server's minimum and stack cap."""
    current = int(state.get("current_bet", 0))
    minimum = int(state.get("minimum_raise", state.get("big_blind", 20)))
    pot = int(state.get("pot", 0))
    ceiling = int(me["bet"] + me["stack"])
    if ceiling <= current:
        return []
    targets = set()
    minimum_target = current + minimum
    if minimum_target < ceiling:
        targets.add(minimum_target)
    for fraction in (0.5, 1.0):
        extra = max(minimum, int(pot * fraction))
        target = current + extra
        if minimum_target <= target < ceiling:
            targets.add(target)
    if ceiling > current:
        targets.add(ceiling)
    legal = []
    for target in sorted(targets):
        extra = target - current
        all_in = target == ceiling
        if extra >= minimum or all_in:
            legal.append(target)
    return legal


def sized_fold_probability(base, size):
    """Fold chance for a bet of `size` pots. Stays near the measured rate.

    A bigger bet folds a few more hands. It cannot turn a 27% folder into
    someone who folds 80% of the time.
    """
    tilt = 0.65 + 0.55 * min(max(float(size), 0.0), 1.5)
    return min(0.62, max(0.05, base * tilt))


def fold_product(model, opponents, size):
    """Chance every listed opponent folds. Missing history uses the population prior."""
    if not opponents:
        return 0.0
    product = 1.0
    for player_id in opponents:
        base = FOLD_PRIOR if model is None else model.fold_to_bet(player_id)
        product *= sized_fold_probability(base, size)
    return product


def _layer_payout(contributions, ranks):
    """Award each contribution layer to the best eligible hand. Chips are floats."""
    payout = {player_id: 0.0 for player_id in contributions}
    previous = 0.0
    for level in sorted(set(contributions.values())):
        if level <= previous:
            continue
        contributors = [player_id for player_id, amount in contributions.items() if amount >= level - 1e-9]
        layer = (level - previous) * len(contributors)
        best = max(ranks[player_id] for player_id in contributors)
        winners = [player_id for player_id in contributors if ranks[player_id] == best]
        share = layer / len(winners)
        for player_id in winners:
            payout[player_id] += share
        previous = level
    return payout


def showdown_change(pot, hero_id, committed, finals, ranks, hero_paid):
    """Chips added to hero's stack at showdown.

    `committed` is the street bets already in the pot for players who stay in.
    `finals` is each stayer's street bet after this action, with the uncalled
    part of our raise already removed. `hero_paid` is only the new chips that
    are not returned.
    """
    if hero_id not in finals:
        return -hero_paid
    dead = max(0.0, pot - committed)
    side = _layer_payout(finals, ranks)[hero_id]
    best = max(ranks[player_id] for player_id in finals)
    winners = [player_id for player_id in finals if ranks[player_id] == best]
    main = dead / len(winners) if hero_id in winners else 0.0
    return main + side - hero_paid


def _expectation(samples, pot, hero_id, players, fold_ps, finals_if_call, hero_bet, hero_cap):
    """Average stack change over independent fold/call subsets and sampled hands."""
    if not samples or not players and hero_cap == hero_bet:
        return 0.0
    total = 0.0
    count = len(players)
    for mask in range(1 << count):
        probability = 1.0
        stayers = []
        for index, player in enumerate(players):
            if mask >> index & 1:
                probability *= 1.0 - fold_ps[index]
                stayers.append(player)
            else:
                probability *= fold_ps[index]
        if probability <= 0.0:
            continue
        committed = float(hero_bet)
        finals = {}
        for player in stayers:
            finals[player["id"]] = finals_if_call[player["id"]]
            committed += int(player["bet"])
        others = list(finals.values())
        hero_final = min(hero_cap, max(others)) if others else hero_bet
        finals[hero_id] = hero_final
        paid = max(0.0, hero_final - hero_bet)
        change = 0.0
        for ranks in samples:
            mapped = {hero_id: ranks["hero"]}
            for player in stayers:
                mapped[player["id"]] = ranks[player["id"]]
            change += showdown_change(pot, hero_id, committed, finals, mapped, paid)
        total += probability * (change / len(samples))
    return total


def choose(state, model, hero_id, trials=300, rng=None, time_limit=1.5):
    started = time.perf_counter()
    me = _me(state, hero_id)
    hero = [parse_card(card) for card in me["hole_cards"]]
    board = []
    for card in state.get("community_cards") or state.get("board") or []:
        if card.get("label") not in (None, "??"):
            board.append(parse_card(card))
    players = _live(state, hero_id)
    opponent_ids = [player["id"] for player in players]
    pot = int(state.get("pot", 0))
    current = int(state.get("current_bet", 0))
    hero_bet = int(me["bet"])
    to_call = min(max(0, current - hero_bet), int(me["stack"]))
    percentiles = board_percentiles(hero, board)

    def remaining():
        return max(0.05, time_limit - (time.perf_counter() - started))

    equity_samples = sample_ranks(
        hero, board, opponent_ids, model, trials=trials, time_limit=remaining(),
        rng=rng, size=None, percentiles=percentiles,
    )
    equity = 0.0
    if equity_samples:
        for ranks in equity_samples:
            mine = ranks["hero"]
            others = [rank for player_id, rank in ranks.items() if player_id != "hero"]
            best = max([mine] + others) if others else mine
            if mine == best:
                equity += 1 / (1 + sum(rank == mine for rank in others))
        equity /= len(equity_samples)

    fold_rates = {}
    for player in players:
        if int(player.get("stack", 0)) <= 0 or player.get("all_in"):
            fold_rates[player["id"]] = 0.0
        elif model is None:
            fold_rates[player["id"]] = FOLD_PRIOR
        else:
            fold_rates[player["id"]] = model.fold_to_bet(player["id"])

    options = []
    if to_call == 0:
        options.append(("check", 0, equity * pot))
    else:
        options.append(("fold", 0, 0.0))
        call_finals = {}
        for player in players:
            call_finals[player["id"]] = min(current, int(player["bet"]) + int(player["stack"]))
        call_folds = []
        for player in players:
            added = max(0, current - int(player["bet"]))
            size = added / max(pot, 1)
            if int(player.get("stack", 0)) <= 0 or player.get("all_in") or added <= 0:
                call_folds.append(0.0)
            else:
                call_folds.append(sized_fold_probability(fold_rates[player["id"]], size))
        options.append((
            "call", 0,
            _expectation(equity_samples, pot, hero_id, players, call_folds, call_finals, hero_bet, current),
        ))

    can_raise = any(int(player.get("stack", 0)) > 0 and not player.get("all_in") for player in players)
    targets = raise_targets(state, me) if can_raise and to_call < int(me["stack"]) else []
    sample_cache = {}
    for target in targets:
        if time.perf_counter() - started > time_limit and sample_cache:
            break
        risk = max(0, target - hero_bet)
        size = risk / max(pot, 1)
        key = round(size, 1)
        if key not in sample_cache:
            sample_cache[key] = sample_ranks(
                hero, board, opponent_ids, model, trials=trials, time_limit=remaining(),
                rng=rng, size=size, percentiles=percentiles,
            )
        samples = sample_cache[key] or equity_samples
        finals = {}
        folds = []
        for player in players:
            room = int(player["bet"]) + int(player["stack"])
            finals[player["id"]] = min(target, room)
            added = max(0, finals[player["id"]] - int(player["bet"]))
            if int(player.get("stack", 0)) <= 0 or player.get("all_in"):
                folds.append(0.0)
            else:
                folds.append(sized_fold_probability(fold_rates[player["id"]], added / max(pot, 1)))
        value = _expectation(samples, pot, hero_id, players, folds, finals, hero_bet, target)
        options.append(("raise", int(target), value))

    action, amount, value = max(options, key=lambda item: (item[2], -item[1]))
    return action, amount, {
        "equity": round(equity, 3),
        "trials": len(equity_samples),
        "ev": round(value, 1),
        "options": [(name, chips, round(ev, 1)) for name, chips, ev in options],
        "fold_rates": {player_id: round(rate, 3) for player_id, rate in fold_rates.items()},
    }
