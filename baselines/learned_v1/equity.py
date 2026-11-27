"""Range-weighted Monte Carlo equity.

Opponent hands are drawn from RangeModel combo weights. Cards already used
by us, the board, or another opponent are removed. Ties split the pot.

Weights are built once per decision. A larger bet tightens the calling
range toward hands that are strong on this board, so a shove is not priced
as if the same hands call a small bet.
"""

import random
import time

from bot import FULL_DECK, best_hand
from opponent_model import FOLD_PRIOR, class_strength, hand_class

# (card, card, class) for every unordered two-card combination.
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
