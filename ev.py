"""Compare legal poker actions by their expected chip gain.

The baseline estimates a showdown without simulating future betting.
"""

import random
import time

from bot import FULL_DECK, best_hand, parse_card
from opponent_model import FOLD_PRIOR, class_strength, hand_class


# Each possible pair of hole cards and its starting-hand class.
COMBOS = []
for index, first in enumerate(FULL_DECK):
    for second in FULL_DECK[index + 1:]:
        COMBOS.append((first, second, hand_class(first, second)))


def _tightness(strength, size, fold_rate):
    """1 for the current range. Callers are stronger when the bet is larger.

    Hands below the threshold are almost never in the called sample. There
    are many more weak combos than strong ones, so a mild discount would
    leave the calling range unchanged. Players who rarely fold keep a wider
    calling range.
    """
    if size is None:
        return 1.0
    strength = max(0.0, min(1.0, strength))
    threshold = min(0.92, 0.38 + 0.24 * min(float(size), 2.5))
    threshold *= min(1.2, 0.55 + fold_rate)
    threshold = min(0.95, threshold)
    if strength < threshold:
        return 0.002
    span = max(1.0 - threshold, 0.05)
    return 0.35 + 0.65 * ((strength - threshold) / span)


def board_percentiles(hero, board):
    """Each remaining combo's rank on this board, from 0 (worst) to 1 (best).

    Preflop there is no board, so the caller uses preflop class strength.
    """
    if len(board) < 3:
        return None
    dead = set(hero + board)
    combos = []
    scores = []
    for first, second, _hand in COMBOS:
        if first in dead or second in dead:
            continue
        combos.append((first, second))
        scores.append(best_hand([first, second] + board))
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    last = max(len(order) - 1, 1)
    return {combos[index]: rank / last for rank, index in enumerate(order)}


def _combo_table(hero, board, opponent_ids, model, size, percentiles):
    dead = set(hero + board)
    combos = [(first, second, hand) for first, second, hand in COMBOS if first not in dead and second not in dead]
    tables = []
    for player_id in opponent_ids:
        weights = []
        for first, second, hand in combos:
            weight = 1.0 if model is None else model.combo_weight(player_id, first, second)
            if size is not None:
                if percentiles is None:
                    strength = class_strength(hand)
                else:
                    strength = percentiles.get((first, second), class_strength(hand))
                fold_rate = FOLD_PRIOR if model is None else model.fold_to_bet(player_id)
                weight *= _tightness(strength, size, fold_rate)
            weights.append(max(weight, 1e-12))
        tables.append(weights)
    return combos, tables


def sample_ranks(hero, board, opponent_ids, model=None, trials=250, time_limit=2.0, rng=None, size=None, percentiles=None):
    """One showdown ranking per completed trial. Hero key is 'hero'."""
    rng = rng or random.Random()
    opponents = [player_id for player_id in opponent_ids if player_id is not None]
    missing = 5 - len(board)
    dead = set(hero + board)
    pool = [card for card in FULL_DECK if card not in dead]
    while len(opponents) > 1 and len(opponents) * 2 + missing > len(pool):
        opponents.pop()
    if percentiles is None and size is not None:
        percentiles = board_percentiles(hero, board)
    combos, tables = _combo_table(hero, board, opponents, model, size, percentiles)
    started = time.perf_counter()
    results = []
    for _ in range(trials):
        if results and time.perf_counter() - started > time_limit:
            break
        available = set(pool)
        hands = {}
        failed = False
        for player_id, weights in zip(opponents, tables):
            picked = None
            for _attempt in range(30):
                first, second, _hand = rng.choices(combos, weights=weights, k=1)[0]
                if first in available and second in available:
                    picked = (first, second)
                    break
            if picked is None:
                failed = True
                break
            available.remove(picked[0])
            available.remove(picked[1])
            hands[player_id] = picked
        if failed:
            continue
        rest = [card for card in pool if card in available]
        rng.shuffle(rest)
        full_board = list(board) + rest[:missing]
        ranks = {"hero": best_hand(list(hero) + full_board)}
        for player_id, hand in hands.items():
            ranks[player_id] = best_hand([hand[0], hand[1]] + full_board)
        results.append(ranks)
    return results


def pot_share(state_cards, opponent_ids, model=None, trials=250, time_limit=2.0, rng=None, action=None, size=None):
    """Our share of the pot, in [0, 1], using weighted opponent ranges."""
    hero, board = state_cards
    if action == "call" and size is None:
        size = 1.0
    ranks = sample_ranks(hero, board, opponent_ids, model, trials, time_limit, rng, size)
    if not ranks:
        return 0.0, 0
    total = 0.0
    for row in ranks:
        mine = row["hero"]
        others = [rank for player_id, rank in row.items() if player_id != "hero"]
        best = max([mine] + others) if others else mine
        if mine == best:
            winners = 1 + sum(rank == mine for rank in others)
            total += 1 / winners
    return total / len(ranks), len(ranks)


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


def sized_fold_probability(base, size, observed=False):
    """Fold chance for a bet of `size` pots. Stays near the measured rate.

    A bigger bet folds a few more hands. It cannot turn a 27% folder into
    someone who folds 80% of the time.
    """
    tilt = 0.65 + 0.55 * min(max(float(size), 0.0), 1.5)
    return min(0.62, max(0 if observed else 0.05, base * tilt))


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


def preflop_change(contributions, ranks, hero, paid, order):
    """Net chips from engine-style contribution layers, including folded money."""
    if len(ranks) == 1:
        return sum(contributions.values()) - paid
    payout = previous = 0
    for level in sorted(set(contributions.values())):
        eligible = [pid for pid in ranks if contributions[pid] >= level]
        if hero in eligible:
            best = max(ranks[pid] for pid in eligible)
            winners = [pid for pid in order if pid in eligible and ranks[pid] == best]
            if hero in winners:
                layer = (level-previous)*sum(v >= level for v in contributions.values())
                share, odd = divmod(layer,len(winners))
                payout += share + (winners.index(hero)<odd)
        previous = level
    return payout-paid


def _expectation(samples, pot, hero_id, players, fold_ps, finals_if_call, hero_bet, hero_cap, preflop=None):
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
            if preflop is None:
                change += showdown_change(pot, hero_id, committed, finals, mapped, paid)
            else:
                original, order = preflop
                totals = {**original, **finals}
                totals[hero_id] = max(hero_bet, hero_cap)
                # Refund chips that nobody matched. Folded bets still count.
                top, second = sorted(totals, key=totals.get, reverse=True)[:2]
                excess = totals[top] - totals[second]
                totals[top] -= excess
                change += preflop_change(totals, mapped, hero_id,
                                          totals[hero_id]-hero_bet, order)
        total += probability * (change / len(samples))
    return total


def choose(state, model, hero_id, trials=300, rng=None, time_limit=1.5, *, experiment=None):
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

    exact = None
    if experiment == "preflop-payout" and state["phase"] == "PREFLOP":
        original = {p["id"]:p["bet"] for p in state["players"]}
        if sum(original.values()) == pot and len(original) >= 2:
            seats = list(original)
            dealer = state.get("dealer")
            split = seats.index(dealer)+1 if dealer in seats else 0
            exact = original,seats[split:]+seats[:split]
    options = []
    responses = {}
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
                call_folds.append(0.0 if experiment == "always-call" else
                                  sized_fold_probability(fold_rates[player["id"]], size, experiment == "observed-folds"))
        responses["call:0"] = dict(zip(opponent_ids, call_folds))
        options.append((
            "call", 0,
            _expectation(equity_samples, pot, hero_id, players, call_folds, call_finals, hero_bet, hero_bet+to_call if exact else current, exact),
        ))

    can_raise = any(int(player.get("stack", 0)) > 0 and not player.get("all_in") for player in players)
    targets = raise_targets(state, me) if can_raise and to_call < int(me["stack"]) else []
    if experiment == "deep-shove" and state["phase"] == "PREFLOP":
        bb = state.get("big_blind", 20)
        if me["stack"] >= 30*bb and me["stack"] > 4*pot:
            targets = [t for t in targets if t != me["bet"]+me["stack"]]
    sample_cache = {}
    for target in targets:
        if time.perf_counter() - started > time_limit and sample_cache:
            break
        risk = max(0, target - hero_bet)
        size = risk / max(pot, 1)
        key = round(size, 1)
        if experiment == "always-call":
            sample_cache[key] = equity_samples
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
                folds.append(0.0 if experiment == "always-call" else
                             sized_fold_probability(fold_rates[player["id"]], added / max(pot, 1),
                                                    experiment == "observed-folds"))
        responses[f"raise:{target}"] = dict(zip(opponent_ids, folds))
        value = _expectation(samples, pot, hero_id, players, folds, finals, hero_bet, target, exact)
        options.append(("raise", int(target), value))

    def score(item):
        action, target, value = item
        additional = min(me["stack"],max(0,target-hero_bet)) if action=="raise" else (
            to_call if action=="call" else 0)
        if experiment == "realization" and state["phase"] in ("PREFLOP","FLOP","TURN"):
            effective = max((p["stack"] for p in players),default=0)
            if min(me["stack"],effective)>additional:
                value -= .12*max(0,value+additional) if action in ("call","raise") else 0
        if experiment == "position" and state["phase"] == "PREFLOP" and action in ("call","raise"):
            seats = [p["id"] for p in state["players"] if not p.get("folded")]
            dealer = state.get("dealer")
            if dealer in seats:
                distance = (seats.index(hero_id)-seats.index(dealer)-1)%len(seats)
                value -= .15*additional*(1-distance/max(1,len(seats)-1))
        if experiment == "survival":
            value -= .25*additional**2/max(1,me["stack"])
        return value
    action, amount, value = max(options, key=lambda item: (score(item), -item[1]))
    return action, amount, {
        "equity": round(equity, 3),
        "trials": len(equity_samples),
        "ev": round(value, 1),
        "options": [(name, chips, round(ev, 1)) for name, chips, ev in options],
        "scores": [(row[0],row[1],round(score(row),2)) for row in options],
        "responses": responses,
        "fold_rates": {player_id: round(rate, 3) for player_id, rate in fold_rates.items()},
    }
