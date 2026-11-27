"""Play persistent-stack matches on the official poker engine.

The deck and the Monte Carlo sampler use different random streams, so
changing the trial count does not change the cards.
"""

import random
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "ai-poker-server"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app.engine import Card, PokerEngine, Suit

from bot import MY_ID, parse_card, win_chance
from equity import pot_share
from ev import choose as ev_choose
from opponent_model import RangeModel


def load_params(name):
    import json
    path = Path(__file__).resolve().parent / "baselines" / f"{name}.json"
    return json.loads(path.read_text())


def _fallback(state, hero_id):
    me = next(player for player in state["players"] if player["id"] == hero_id)
    to_call = max(0, int(state["current_bet"]) - int(me["bet"]))
    if to_call <= 0:
        return "check", 0
    if to_call >= me["stack"]:
        return "call", 0
    return "call", 0


def _rng_of(actor):
    rng = getattr(actor, "rng", None)
    if rng is None:
        rng = random.Random()
        actor.rng = rng
    return rng


class ThresholdBot:
    """Fixed raise/call thresholds. Uniform opponent hands."""

    def __init__(self, player_id, params, trials=40):
        self.player_id = player_id
        self.params = params
        self.trials = trials
        self.rng = None
        self.latencies = []

    def observe(self, state):
        return None

    def equity(self, state):
        previous = MY_ID
        try:
            # win_chance reads the global id. Matches are single-threaded.
            import bot
            bot.MY_ID = self.player_id
            return win_chance(state, trials=self.trials, rng=_rng_of(self))
        finally:
            import bot
            bot.MY_ID = previous

    def __call__(self, state):
        started = time.perf_counter()
        equity = self.equity(state)
        me = next(player for player in state["players"] if player["id"] == self.player_id)
        to_call = max(0, int(state["current_bet"]) - int(me["bet"]))
        pot = int(state.get("pot", 0))
        action, amount = "fold", 0
        if state.get("phase") == "PREFLOP" and to_call > 0 and equity < self.params.get("preflop_floor", 0):
            action, amount = "fold", 0
        elif equity >= self.params["raise_at"] and me["stack"] > to_call:
            extra = max(int(state["minimum_raise"]), int(pot * self.params["raise_fraction"]))
            total = min(int(state["current_bet"]) + extra, int(me["bet"]) + int(me["stack"]))
            if total > int(state["current_bet"]):
                action, amount = "raise", int(total)
            elif to_call <= 0:
                action, amount = "check", 0
            else:
                action, amount = self._call_or_fold(equity, to_call, pot)
        elif to_call <= 0:
            action, amount = "check", 0
        else:
            action, amount = self._call_or_fold(equity, to_call, pot)
        self.latencies.append(time.perf_counter() - started)
        return action, amount

    def _call_or_fold(self, equity, to_call, pot):
        price = to_call / (pot + to_call) if pot + to_call else 1
        if equity >= price + self.params["call_margin"]:
            return "call", 0
        return "fold", 0


class RangeThresholdBot(ThresholdBot):
    """Aggressor thresholds, but equity is weighted by the Bayesian range."""

    def __init__(self, player_id, params, trials=80):
        super().__init__(player_id, params, trials=trials)
        self.model = RangeModel()

    def observe(self, state):
        self.model.observe_state(state, self.player_id)

    def equity(self, state):
        self.observe(state)
        me = next(player for player in state["players"] if player["id"] == self.player_id)
        hero = [parse_card(card) for card in me["hole_cards"]]
        board = []
        for card in state.get("community_cards") or []:
            if card.get("label") not in (None, "??"):
                board.append(parse_card(card))
        opponents = [player["id"] for player in state["players"] if player["id"] != self.player_id and not player.get("folded")]
        share, _used = pot_share((hero, board), opponents, self.model, trials=self.trials, rng=_rng_of(self))
        return share


class AdaptiveBot:
    def __init__(self, player_id, trials=300, use_ranges=True):
        self.player_id = player_id
        self.trials = trials
        self.model = RangeModel() if use_ranges else None
        self.rng = None
        self.latencies = []
        self.decisions = []

    def observe(self, state):
        if self.model is not None:
            self.model.observe_state(state, self.player_id)

    def __call__(self, state):
        self.observe(state)
        if not state.get("players"):
            return "check", 0
        me = next(player for player in state["players"] if player["id"] == self.player_id)
        if len(me.get("hole_cards") or []) < 2:
            return _fallback(state, self.player_id)
        started = time.perf_counter()
        action, amount, info = ev_choose(
            state, self.model, self.player_id, trials=self.trials, rng=_rng_of(self),
        )
        elapsed = time.perf_counter() - started
        self.latencies.append(elapsed)
        to_call = max(0, int(state["current_bet"]) - int(me["bet"]))
        self.decisions.append({
            "hand": state.get("hand_number"),
            "phase": state.get("phase"),
            "cards": [card.get("label") for card in me.get("hole_cards") or []],
            "board": [card.get("label") for card in state.get("community_cards") or []],
            "pot": state.get("pot"),
            "to_call": to_call,
            "stack": me.get("stack"),
            "action": action,
            "amount": amount,
            "equity": info["equity"],
            "ev": info["ev"],
            "options": info["options"],
            "fold_rates": info["fold_rates"],
            "latency": round(elapsed, 4),
            "all_in": action == "raise" and amount >= int(me["bet"]) + int(me["stack"]),
        })
        return action, amount


def play_hand(engine, actors):
    engine.start_hand()
    steps = 0
    while engine.phase not in ("WAITING", "SHOWDOWN") and steps < 400:
        steps += 1
        actor_id = engine.current_turn
        if actor_id is None:
            break
        for player_id, actor in actors.items():
            observe = getattr(actor, "observe", None)
            if observe is not None:
                observe(engine.snapshot(player_id))
        state = engine.snapshot(actor_id)
        try:
            action, amount = actors[actor_id](state)
            result = engine.place_action(actor_id, action, int(amount))
        except Exception:
            result = {"ok": False}
        if not result.get("ok"):
            me = next(player for player in state["players"] if player["id"] == actor_id)
            to_call = int(state["current_bet"]) - int(me["bet"])
            fallback = ("call", 0) if to_call > 0 else ("check", 0)
            engine.place_action(actor_id, fallback[0], fallback[1])
    if engine.phase == "SHOWDOWN":
        for player_id, actor in actors.items():
            observe = getattr(actor, "observe", None)
            if observe is not None:
                observe(engine.snapshot(player_id))
        engine.finalize_showdown()
    return steps


def _build_deck(rng):
    deck = [Card(rank, suit) for suit in Suit for rank in range(2, 15)]
    rng.shuffle(deck)
    return deck


def play_match(actor_builders, hands=20, seed=None):
    """actor_builders: list of (player_id, zero-arg factory). Stacks persist.

    `seed` deals the cards. Each seat gets its own sampler stream, so two
    policies with the same seed see the same deck until a player busts.
    """
    deck_rng = random.Random(0 if seed is None else seed)
    engine = PokerEngine()
    engine._build_deck = lambda: _build_deck(deck_rng)
    actors = {}
    for index, (player_id, factory) in enumerate(actor_builders):
        engine.add_player(player_id, player_id)
        actor = factory()
        actor.rng = random.Random((0 if seed is None else seed) + 10007 * (index + 1))
        actors[player_id] = actor
    for _ in range(hands):
        if not engine.can_start_hand():
            break
        play_hand(engine, actors)
    starting = engine.starting_stack
    deltas = {player_id: engine.players[player_id].stack - starting for player_id in actors}
    return deltas
