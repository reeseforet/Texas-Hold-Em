"""Two-player zero-sum tabular CFR and external-sampling MCCFR."""
from collections import defaultdict
from itertools import permutations
import random


def uniform(n):
    return [1/n]*n


def matching(regrets):
    positive = [max(0, r) for r in regrets]
    total = sum(positive)
    return [r/total for r in positive] if total else uniform(len(regrets))


class Solver:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.regret = {}
        self.total = {}
        self.iterations = 0

    def train(self, game, steps, external=False):
        for _ in range(steps):
            # Freeze this iteration's policy until both traversals finish.
            cache = {}
            updates = {}

            def policy(key, n):
                if key not in cache:
                    cache[key] = matching(self.regret.get(key, [0]*n))
                return cache[key]

            def add(table, key, values):
                row = table.setdefault(key, [0.0]*len(values))
                for i, value in enumerate(values):
                    row[i] += value

            def visit(deal, h, traverser, reach, chance):
                player, actions, utility = game.node(deal, h)
                if player is None:
                    return utility if traverser == 0 else -utility
                key = game.info(deal, h, player)
                p = policy(key, len(actions))
                if external and player != traverser:
                    # In two-player external sampling, the other player's
                    # own reach is represented by its sampled actions.
                    add(self.total, key, p)
                    choice = self.rng.choices(actions, p)[0]
                    return visit(deal, h+(choice,), traverser, reach, chance)
                if not external and player == traverser:
                    add(self.total, key, [reach[player]*x for x in p])
                values = []
                for action, prob in zip(actions, p):
                    next_reach = list(reach)
                    next_reach[player] *= prob
                    values.append(visit(deal, h+(action,), traverser, next_reach, chance))
                expected = sum(q*v for q,v in zip(p,values))
                if player == traverser:
                    weight = 1 if external else chance*reach[1-player]
                    add(updates, key, [weight*(v-expected) for v in values])
                return expected

            for traverser in (0,1):
                if external:
                    visit(game.sample(self.rng), (), traverser, (1,1), 1)
                else:
                    for deal, probability in game.deals:
                        visit(deal, (), traverser, (1,1), probability)
            for key, values in updates.items():
                add(self.regret, key, values)
            self.iterations += 1

    def average(self):
        return {k:[x/sum(v) for x in v] if sum(v) else uniform(len(v))
                for k,v in self.total.items()}


def evaluate(game, profile):
    """Exact profile value and infoset-constrained best responses."""
    def probs(deal,h,player,actions):
        return profile.get(game.info(deal,h,player),uniform(len(actions)))

    def value(deal,h,br=None,choices=None):
        player, actions, utility = game.node(deal,h)
        if player is None:
            return utility
        if player == br:
            return value(deal,h+(choices[game.info(deal,h,player)],),br,choices)
        return sum(q*value(deal,h+(a,),br,choices)
                   for q,a in zip(probs(deal,h,player,actions),actions) if q)

    ev = sum(p*value(d,()) for d,p in game.deals)
    improvements = []
    for br in (0,1):
        grouped = defaultdict(list)

        def gather(deal,h,weight):
            player, actions, _ = game.node(deal,h)
            if player is None:
                return
            if player == br:
                grouped[game.info(deal,h,player)].append((deal,h,weight,actions))
                p = [1]*len(actions)
            else:
                p = probs(deal,h,player,actions)
            for a,q in zip(actions,p):
                if q:
                    gather(deal,h+(a,),weight*q)

        for d,p in game.deals:
            gather(d,(),p)
        choices = {}
        sign = 1 if br == 0 else -1
        for key, nodes in sorted(grouped.items(),key=lambda row:len(row[1][0][1]),reverse=True):
            choices[key] = max(nodes[0][3],key=lambda a:sum(
                sign*w*value(d,h+(a,),br,choices) for d,h,w,_ in nodes))
        best = sum(sign*p*value(d,(),br,choices) for d,p in game.deals)
        improvements.append(best-sign*ev)
    return {"value_p0":ev,"nashconv":sum(improvements),"exploitability":sum(improvements)/2}


class Kuhn:
    deals = [(d,1/6) for d in permutations(range(3),2)]

    def sample(self,rng):
        return rng.choice(self.deals)[0]

    def info(self,d,h,p):
        return p,d[p],h

    def node(self,d,h):
        if h in ((),("check",)):
            return len(h),("check","bet"),None
        if h in (("bet",),("check","bet")):
            return len(h)%2,("fold","call"),None
        if h[-1] == "fold":
            return None,(),1 if len(h)==2 else -1
        return None,(),(1 if d[0]>d[1] else -1)*(2 if h[-1]=="call" else 1)


class Leduc:
    # Two physical copies per rank. The third card is predealt and remains hidden
    # until street two. One ante each, fixed bets of 2 and 4, two raises per street.
    deals = [(tuple(x//2 for x in d),1/120) for d in permutations(range(6),3)]

    def sample(self,rng):
        return tuple(x//2 for x in rng.sample(range(6),3))

    @staticmethod
    def state(h):
        paid = [1,1]
        street = player = raises = checks = 0
        bets = [0,0]
        folded = None
        for a in h:
            if a == "fold":
                folded = player
                break
            if a == "raise":
                target = max(bets)+(2 if street==0 else 4)
                paid[player] += target-bets[player]
                bets[player] = target
                raises += 1
            elif a == "call":
                paid[player] += max(bets)-bets[player]
            else:
                checks += 1
            if a == "call" or checks == 2:
                street += 1
                player = raises = checks = 0
                bets = [0,0]
            else:
                player = 1-player
        return paid,street,player,raises,bets,folded

    def info(self,d,h,p):
        street = self.state(h)[1]
        return p,d[p],d[2] if street else None,h

    def node(self,d,h):
        paid,street,p,raises,bets,folded = self.state(h)
        if folded is not None:
            return None,(),-paid[0] if folded==0 else paid[1]
        if street == 2:
            strengths = [(card==d[2],card) for card in d[:2]]
            return None,(),paid[1] if strengths[0]>strengths[1] else (
                -paid[0] if strengths[0]<strengths[1] else (paid[1]-paid[0])/2)
        facing = bets[p]<max(bets)
        actions = ("fold","call") if facing else ("check",)
        return p,actions+(("raise",) if raises<3 else ()),None
