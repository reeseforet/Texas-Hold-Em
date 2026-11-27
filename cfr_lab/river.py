"""Bounded heads-up river game and optional learned-EV policy adapter."""
from itertools import combinations
import random
import time

from bot import FULL_DECK, best_hand, parse_card
from cfr_lab.core import Solver, uniform


class River:
    """Unopened river, equal effective stacks, half/pot opens and one min-raise.

    Deal probability is proportional to w0(hand0)*w1(hand1) on disjoint
    combinations. Opponents observe only their own strength bucket and history.
    Showdown uses exact cards. Utilities center the dead pot at pot/2.
    """
    def __init__(self, board, ranges, pot, stack, minimum, buckets=12):
        self.board = tuple(board)
        if len(self.board) != 5 or len(set(self.board)) != 5 or not set(board)<=set(FULL_DECK):
            raise ValueError("Expected five distinct board cards")
        if min(pot,stack,minimum,buckets) <= 0:
            raise ValueError("Positive pot, stack, minimum and bucket count required")
        self.pot,self.stack,self.minimum = pot,stack,minimum
        self.buckets = buckets
        self.hands = []
        self.weights = []
        for rows in ranges:
            legal = {}
            for hand, weight in rows.items():
                if (len(hand)==2 and len(set(hand))==2 and set(hand)<=set(FULL_DECK)
                        and set(board).isdisjoint(hand) and weight>0):
                    legal[tuple(sorted(hand))] = weight
            if not legal:
                raise ValueError("Empty compatible range")
            self.hands.append(tuple(legal))
            self.weights.append(tuple(legal.values()))
        ranks = {h:best_hand(list(h)+list(board)) for h in set(self.hands[0]+self.hands[1])}
        ordered = sorted(set(ranks.values()))
        ordinal = {rank:i*buckets//len(ordered) for i,rank in enumerate(ordered)}
        self.rank = ranks
        self.bucket = {h:ordinal[r] for h,r in ranks.items()}
        self.openers = tuple(sorted({min(stack,max(minimum,pot//2)),min(stack,max(minimum,pot))}))

    @property
    def deals(self):
        if len(self.hands[0])*len(self.hands[1])>100_000:
            raise ValueError("Exact evaluation requires a limited support fixture")
        entries = [((a,b),wa*wb) for a,wa in zip(self.hands[0],self.weights[0])
                   for b,wb in zip(self.hands[1],self.weights[1]) if set(a).isdisjoint(b)]
        total = sum(w for _,w in entries)
        if not total:
            raise ValueError("No disjoint deals")
        return [(d,w/total) for d,w in entries]

    def sample(self,rng):
        for _ in range(100):
            a,b = (rng.choices(h,w)[0] for h,w in zip(self.hands,self.weights))
            if set(a).isdisjoint(b):
                return a,b
        raise ValueError("Unable to sample disjoint private hands")

    def info(self,d,h,p):
        return p,self.bucket[d[p]],h

    def node(self,d,h):
        p = len(h)%2
        if not h or h==("check",):
            return p,("check",*self.openers),None
        sizes = [a for a in h if isinstance(a,int)]
        if h[-1]=="fold":
            previous = sizes[-2] if len(sizes)>1 else 0
            return None,(),(self.pot/2+previous)*(1 if p==0 else -1)
        if h[-1] in ("call","check"):
            r0,r1 = (self.rank[hand] for hand in d)
            return None,(),((r0>r1)-(r0<r1))*(self.pot/2+(sizes[-1] if sizes else 0))
        target = sizes[-1]
        new_target = min(self.stack, target+max(target,self.minimum))
        raises = (new_target,) if len(sizes)==1 and new_target>target else ()
        return p,("fold","call",*raises),None


def experiment_action(state, hero_id, ranges, rng, fallback, *, mode="hybrid",
                      seconds=.25, iterations=1000):
    """Optional offline policy for first-to-act unopened heads-up rivers.

    The chance model receives both public-inference ranges. Hero cards are
    read only after training to select the relevant root information set.
    """
    if mode not in ("cfr","hybrid","ev","uncertain"):
        raise ValueError("Unknown CFR experiment mode")
    if mode=="ev":
        return fallback()
    start = time.monotonic()
    players = [p for p in state["players"] if not p["folded"]]
    if (state["phase"]!="RIVER" or len(players)!=2 or state["current_bet"]
            or any(p["stack"]<=0 or p["bet"] for p in players)):
        return fallback()
    hero = next(p for p in players if p["id"]==hero_id)
    villain = next(p for p in players if p["id"]!=hero_id)
    # First actor is the seat clockwise from the button heads-up.
    if state.get("dealer") != villain["id"]:
        return fallback()
    if mode=="uncertain":
        # Low evidence in the current hand is an experimental proxy.
        if ranges.weights.get(villain["id"]):
            return fallback()
    elif mode=="hybrid" and rng.random()<.5:
        return fallback()
    try:
        board = [parse_card(c) for c in state["community_cards"]]
        combos = [h for h in combinations(FULL_DECK,2) if set(board).isdisjoint(h)]
        beliefs = [{h:ranges.combo_weight(p["id"],*h) for h in combos} for p in (hero,villain)]
        game = River(board,beliefs,state["pot"], min(hero["stack"],villain["stack"]),state["minimum_raise"])
        solver = Solver(rng.randrange(2**32))
        # Never submit a short raise unless it is the actual hero all-in.
        if game.stack < game.minimum and hero["stack"] != game.stack:
            return fallback()
        while solver.iterations < iterations and time.monotonic()-start < seconds*.8:
            solver.train(game, min(16,iterations-solver.iterations),external=True)
        hole = tuple(sorted(parse_card(c) for c in hero["hole_cards"]))
        key = (0,game.bucket[hole],())
        visits = sum(solver.total.get(key,()))
        if solver.iterations < 64 or visits < 8 or time.monotonic()-start >= seconds:
            return fallback()
        actions = ("check",*game.openers)
        action = rng.choices(actions,solver.average().get(key,uniform(len(actions))))[0]
        return ("check",0) if action=="check" else ("raise",action)
    except (KeyError,ValueError):
        return fallback()
