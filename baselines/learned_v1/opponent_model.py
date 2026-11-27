"""Bayesian hand ranges and smoothed action rates.

Each opponent has a weight for every preflop hand class. An observed action
updates those weights by Bayes' rule with a fixed likelihood:

    P(class | action) ∝ P(action | class, aggression) P(class)

Fold-to-bet and raise rates start from a population prior measured on the
frozen baseline styles: about 27% of faced bets are folds, and about 30% of
actions are raises. Each prior is worth 8 observations, so a short match can
move the rate without letting one action pin it at 0 or 1.

Hand weights start higher for stronger preflop classes. That prior is used
immediately. Observed actions multiply it and gradually take over. Fold-to-bet
counts a fold, a call, or a raise, and only when that player faced a bet.
"""

# 8 pseudo-observations. Means: fold 0.27, raise 0.30.
PRIOR_OBS = 8.0
FOLD_PRIOR = 0.27
RAISE_PRIOR = 0.30
FOLD_A = FOLD_PRIOR * PRIOR_OBS
FOLD_B = (1.0 - FOLD_PRIOR) * PRIOR_OBS
RAISE_A = RAISE_PRIOR * PRIOR_OBS
RAISE_B = (1.0 - RAISE_PRIOR) * PRIOR_OBS


def hand_class(card_a, card_b):
    high, low = sorted((card_a[0], card_b[0]), reverse=True)
    suited = card_a[1] == card_b[1] and high != low
    return (high, low, suited)


def all_classes():
    classes = []
    for high in range(14, 1, -1):
        for low in range(high, 1, -1):
            if high == low:
                classes.append((high, low, False))
            else:
                classes.append((high, low, True))
                classes.append((high, low, False))
    return classes


CLASSES = all_classes()
PRIOR_TOTAL = None


def prior_weight(hand):
    """Conservative starting range: stronger classes are more likely, not certain."""
    return 0.40 + 0.60 * class_strength(hand)


def class_strength(hand):
    """0–1 score used only inside the action likelihood. Not win equity."""
    high, low, suited = hand
    if high == low:
        return 0.50 + high / 28
    score = (high + low) / 36
    if suited:
        score += 0.08
    if high - low == 1:
        score += 0.04
    if high == 14:
        score += 0.05
    return min(0.97, score)


def likelihood(action, strength, aggression):
    """P(action | hand strength, how often this player raises).

    aggression near 0 makes a raise strong evidence of a good hand.
    aggression near 1 keeps a raise compatible with weak hands.
    """
    aggression = min(0.9, max(0.1, aggression))
    if action == "raise":
        return (0.12 + 0.88 * strength) ** (1.6 - aggression)
    if action == "call":
        return 0.35 + 0.65 * (1 - abs(strength - 0.42))
    if action == "fold":
        return 0.15 + 0.85 * (1 - strength)
    return 0.55 + 0.45 * (1 - abs(strength - 0.35))


class RangeModel:
    def __init__(self):
        self.players = {}
        self.last_marker = None
        self.last_showdown = None
        self.seen_bet = {}
        self.showdown_scores = []

    def _player(self, player_id):
        record = self.players.get(player_id)
        if record is None:
            record = {
                "raise_a": RAISE_A,
                "raise_b": RAISE_B,
                "fold_a": FOLD_A,
                "fold_b": FOLD_B,
                "weights": {hand: prior_weight(hand) for hand in CLASSES},
                "shown": {hand: 0.0 for hand in CLASSES},
                "actions": 0,
            }
            self.players[player_id] = record
        return record

    def aggression(self, player_id):
        record = self._player(player_id)
        return record["raise_a"] / (record["raise_a"] + record["raise_b"])

    def fold_to_bet(self, player_id):
        """Smoothed fold rate. The denominator includes calls and raises that faced a bet."""
        record = self._player(player_id)
        return record["fold_a"] / (record["fold_a"] + record["fold_b"])

    def class_probabilities(self, player_id):
        record = self._player(player_id)
        total = sum(record["weights"].values())
        if total <= 0:
            share = 1 / len(record["weights"])
            return {hand: share for hand in record["weights"]}
        return {hand: weight / total for hand, weight in record["weights"].items()}

    def ready(self, player_id):
        return self._player(player_id)["actions"] >= 8

    def update_action(self, player_id, action, faced_bet):
        """A raise that faces a bet counts as not folding.

        P(fold | facing a bet) = (2 + folds) / (4 + folds + calls + raises).
        An opening bet, when nobody has bet yet, is not part of that rate.
        """
        record = self._player(player_id)
        if action == "raise":
            record["raise_a"] += 1
            if faced_bet:
                record["fold_b"] += 1
            kind = "raise"
        elif action == "fold" and faced_bet:
            record["fold_a"] += 1
            record["raise_b"] += 1
            kind = "fold"
        elif action == "call" and faced_bet:
            record["fold_b"] += 1
            record["raise_b"] += 1
            kind = "call"
        elif action == "check":
            record["raise_b"] += 1
            kind = "check"
        else:
            return
        record["actions"] += 1
        aggression = record["raise_a"] / (record["raise_a"] + record["raise_b"])
        # Temper the likelihood. A full multiply made one raise wipe out most
        # classes, and those ranges predicted showdowns worse than uniform.
        for hand, weight in list(record["weights"].items()):
            softened = likelihood(kind, class_strength(hand), aggression) ** 0.35
            updated = weight * softened
            record["weights"][hand] = max(updated, 0.25 * prior_weight(hand))
        peak = max(record["weights"].values())
        if peak > 0:
            for hand in record["weights"]:
                record["weights"][hand] /= peak

    def reveal(self, player_id, cards):
        if len(cards) != 2:
            return
        shown = hand_class(cards[0], cards[1])
        record = self._player(player_id)
        record["shown"][shown] = record["shown"].get(shown, 0) + 1

    def combo_weight(self, player_id, card_a, card_b):
        """Unnormalized P(these two cards), including the population prior."""
        record = self._player(player_id)
        hand = hand_class(card_a, card_b)
        return record["weights"][hand] * (1 + record["shown"].get(hand, 0))

    def note_showdown(self, player_id, cards):
        """Record how likely this shown hand was before we update on the reveal."""
        global PRIOR_TOTAL
        if len(cards) != 2:
            return
        if PRIOR_TOTAL is None:
            PRIOR_TOTAL = sum(prior_weight(hand) for hand in CLASSES)
        hand = hand_class(cards[0], cards[1])
        adaptive = self.class_probabilities(player_id).get(hand, 0.0)
        self.showdown_scores.append({
            "adaptive": adaptive,
            "prior": prior_weight(hand) / PRIOR_TOTAL,
            "uniform": 1 / len(CLASSES),
        })

    def observe_state(self, state, hero_id):
        """Update from one public state. Repeated broadcasts are ignored."""
        hand_number = state.get("hand_number")
        if state.get("phase") == "SHOWDOWN" and self.last_showdown != hand_number:
            self.last_showdown = hand_number
            for player in state.get("players", []):
                if player.get("id") == hero_id or player.get("folded"):
                    continue
                cards = []
                for card in player.get("hole_cards") or []:
                    label = card.get("label", "??")
                    if label in ("??", "", None) or card.get("rank") in ("?", None):
                        cards = []
                        break
                    from bot import parse_card
                    cards.append(parse_card(card))
                if len(cards) == 2:
                    self.note_showdown(player["id"], cards)
                    self.reveal(player["id"], cards)

        last = state.get("last_action") or {}
        player = last.get("player")
        kind = last.get("type")
        marker = (hand_number, player, kind, last.get("amount"), last.get("target"))
        previous_bet = self.seen_bet.get(hand_number, 0)
        self.seen_bet[hand_number] = int(state.get("current_bet") or 0)
        if marker == self.last_marker or not player or player == hero_id:
            return
        self.last_marker = marker
        # Fold and call are only legal facing a bet. A raise faces a bet only
        # when the previous state already had a wager, including the big blind.
        faced_bet = kind in ("fold", "call") or (kind == "raise" and previous_bet > 0)
        if kind == "check":
            self.update_action(player, "check", False)
        elif kind in ("fold", "call", "raise"):
            self.update_action(player, kind, faced_bet)


MODEL = RangeModel()
