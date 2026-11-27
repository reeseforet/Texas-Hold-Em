"""Load Learned V1 and update opponent ranges from public actions."""
from collections import Counter, defaultdict
from itertools import combinations
import json
import hashlib
import math
from pathlib import Path

from bot import FULL_DECK, best_hand, parse_card
from opponent_model import FOLD_PRIOR, class_strength, hand_class

ACTIONS = ("fold", "check", "call", "bet", "raise")
COMBOS = tuple(combinations(FULL_DECK, 2))
PREFLOP = {hand: class_strength(hand_class(*hand)) for hand in COMBOS}


class LearnedV1:
    def __init__(self):
        path = Path(__file__).with_name("baselines") / "learned_v1" / "model.json"
        artifact = path.read_bytes()
        self.model_digest = hashlib.sha256(artifact).hexdigest()
        model = json.loads(artifact)["logistic"]
        self.coefficients = dict(zip(model["classes"], zip(model["coef"], model["intercept"])))
        self.seen = defaultdict(lambda: [2, 8])
        self.weights = {}
        self.acted = {}
        self.previous = None
        self.last_choice = 0

    def combo_weight(self, player, first, second):
        weights = self.weights.get(player, {})
        return weights.get((first, second), weights.get((second, first), 1.0))

    def fold_to_bet(self, player):
        return FOLD_PRIOR

    def can_raise(self, state, hero):
        wager = state["current_bet"]
        faced = self.acted.get((state["phase"], hero), 0)
        return not faced or wager - faced >= state["minimum_raise"]

    def observe(self, state, hero_id):
        prior = self.previous
        self.previous = state
        if prior is None or prior["hand_number"] != state["hand_number"]:
            self.weights.clear()
            self.acted.clear()
            self.last_choice = 0
            return
        action = state.get("last_action") or {}
        player = prior.get("current_turn")
        name = action.get("type")
        if player is None or action.get("player") != player or name not in ACTIONS:
            return
        if (prior["current_turn"] == state["current_turn"]
                and prior["phase"] == state["phase"]
                and prior.get("last_action") == action):
            return
        current = prior["current_bet"]
        tag = "bet" if name == "raise" and current == 0 else name
        choice = ACTIONS.index(tag)
        if player != hero_id:
            self._update(prior, player, choice)
        self.acted[prior["phase"],player] = action.get("target", current)
        self.seen[player][0] += int(tag in ("bet","raise"))
        self.seen[player][1] += 1
        self.last_choice = choice + 1

    def _update(self, state, player_id, choice):
        players = [p for p in state["players"] if not p["folded"]]
        player = next(p for p in players if p["id"] == player_id)
        ids = [p["id"] for p in players]
        dealer = state["dealer"]
        position = ((ids.index(player_id)-ids.index(dealer)-1)%len(ids)
                    if dealer in ids else 0)
        board = [parse_card(c) for c in state["community_cards"]]
        blocked = set(board)
        pot, bb, current = state["pot"], state["big_blind"], state["current_bet"]
        cost = min(player["stack"], max(0,current-player["bet"]))
        others = [p for p in players if p["id"] != player_id]
        effective = min(player["stack"],max((p["stack"] for p in others),default=0))
        suits = Counter(s for _,s in board)
        raises, observed = self.seen[player_id]
        ctx = (len(board)/5, position/max(1,len(ids)-1),
               float(len({r for r,_ in board})!=len(board)),max(suits.values(),default=0)/5,
               min(2,cost/max(1,pot)), math.log1p(effective/bb), math.log1p(pot/bb),
               (len(ids)-1)/5,self.last_choice/5,raises/observed)
        raise_ok = (self.can_raise(state,player_id) and player["bet"]+player["stack"]>current
                    and any(p["stack"] and p["connected"] for p in others))
        mask = (True,not cost,bool(cost),raise_ok and current==0,raise_ok and current>0)
        if not mask[choice]:
            return
        allowed = [i for i,ok in enumerate(mask) if ok]
        old = self.weights.get(player_id,{})
        updated = {}
        for hand in COMBOS:
            if not blocked.isdisjoint(hand):
                continue
            if len(board)<3:
                strength = PREFLOP[hand]
            else:
                rank = best_hand([*hand,*board])
                strength = (rank[0]+rank[1]/15)/9
            x = (strength,strength*strength,strength*ctx[4],strength*ctx[0],*ctx)
            scores = {i:b+sum(a*v for a,v in zip(w,x))
                      for i in allowed for w,b in (self.coefficients[i],)}
            top = max(scores.values())
            likelihood = math.exp(scores[choice]-top)/sum(math.exp(z-top) for z in scores.values())
            updated[hand] = max(1e-9,old.get(hand,1.0)*likelihood)
        self.weights[player_id] = updated
