"""Train opponent models and compare poker policies."""
import argparse
from bisect import bisect_left
from itertools import accumulate
from collections import Counter, defaultdict
from itertools import combinations
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from benchmark import style_params
from bot import FULL_DECK, best_hand, parse_card
from match import AdaptiveBot, PokerEngine, ThresholdBot, _build_deck
from opponent_model import FOLD_PRIOR, class_strength, hand_class, likelihood

ACTIONS = ("fold", "check", "call", "bet", "raise")
TRAIN = ("aggressor", "evolved", "station", "rock")
VALID = ("balanced",)
FINAL = ("pressure",)
POLICIES = {
    **style_params(),
    "balanced": dict(raise_at=.48, call_margin=.03, raise_fraction=.5, preflop_floor=.16),
    "pressure": dict(raise_at=.28, call_margin=.04, raise_fraction=1, preflop_floor=.14),
}
COMBOS = list(combinations(FULL_DECK, 2))
PRE = {hand: class_strength(hand_class(*hand)) for hand in COMBOS}


def legal(state, player_id, acted_at):
    player = next(p for p in state["players"] if p["id"] == player_id)
    current = state["current_bet"]
    facing = player["bet"] < current
    reopen = not acted_at or current - acted_at >= state["minimum_raise"]
    raise_ok = reopen and player["bet"] + player["stack"] > current and any(
        p["id"] != player_id and p["stack"] and not p["folded"] and p["connected"]
        for p in state["players"]
    )
    return [True, not facing, facing, raise_ok and current == 0, raise_ok and current > 0]


def context(state, player_id, aggression, previous):
    others = [p for p in state["players"] if not p["folded"]]
    player = next(p for p in others if p["id"] == player_id)
    ids = [p["id"] for p in others]
    dealer = state["dealer"]
    position = ((ids.index(player_id) - ids.index(dealer) - 1) % len(ids)
                if dealer in ids else 0)
    board = tuple(parse_card(c) for c in state["community_cards"])
    bb = state["big_blind"]
    pot = state["pot"]
    call = min(player["stack"], max(0, state["current_bet"] - player["bet"]))
    effective = min(player["stack"], max((p["stack"] for p in others if p != player), default=0))
    suits = Counter(s for _, s in board)
    return board, (
        len(board) / 5, position / max(1, len(ids) - 1),
        float(len({r for r, _ in board}) != len(board)),
        max(suits.values(), default=0) / 5,
        min(2, call / max(1, pot)), math.log1p(effective / bb),
        math.log1p(pot / bb), (len(ids) - 1) / 5, previous / 5, aggression,
    )


def strength(hand, board):
    if len(board) < 3:
        return PRE.get(hand, PRE.get(hand[::-1], 0))
    rank = best_hand(list(hand + board))
    return (rank[0] + rank[1] / 15) / 9


def features(strengths, ctx):
    return np.array([[s, s*s, s*ctx[4], s*ctx[0], *ctx] for s in strengths])


def bucket(s, ctx, mask):
    return (round(ctx[0]*5), min(4, int(s*5)), min(3, int(ctx[4]*4)), tuple(mask))


class Predictor:
    def __init__(self, rows, kind):
        self.kind = kind
        self.counts = defaultdict(lambda: np.zeros(len(ACTIONS)))
        self.prior = np.ones(len(ACTIONS))
        self.linear = None
        self.classes = None
        for row in rows:
            self.counts[bucket(row["strength"], row["context"], row["legal"])][row["action"]] += 1
            self.prior[row["action"]] += 1
        self.prior /= self.prior.sum()
        if kind == "logistic":
            x = np.concatenate([features([r["strength"]], r["context"]) for r in rows])
            y = [r["action"] for r in rows]
            if len(set(y)) >= 2:
                model = LogisticRegression(max_iter=300).fit(x, y)
                self.linear = model
                self.classes = model.classes_

    def predict(self, strengths, ctx, mask):
        strengths = list(strengths)
        if self.linear is not None:
            p = np.full((len(strengths), len(ACTIONS)), 1e-6)
            p[:, self.classes] = self.linear.predict_proba(features(strengths, ctx))
        elif self.kind == "table":
            p = np.array([self.counts.get(bucket(s, ctx, mask), 0) + 12*self.prior for s in strengths])
        elif self.kind == "heuristic":
            p = np.array([[likelihood("raise" if a == "bet" else a, s, ctx[-1])
                           for a in ACTIONS] for s in strengths])
        else:
            p = np.tile(self.prior, (len(strengths), 1))
        p = p * np.asarray(mask)
        return p / p.sum(axis=1, keepdims=True)

    def artifact(self):
        return dict(kind=self.kind, prior=self.prior.tolist(),
                    table=[[*key[:3], list(key[3]), value.tolist()] for key, value in self.counts.items()]
                    if self.kind == "table" else [],
                    logistic=None if self.linear is None else dict(
                        classes=self.classes.tolist(), coef=self.linear.coef_.tolist(),
                        intercept=self.linear.intercept_.tolist()))


class HandRanges:
    """Weights for the current deal. Exact private cards are absent from updates."""
    def __init__(self, predictor):
        self.predictor = predictor
        self.weights = {}

    def reset(self):
        self.weights.clear()

    def update(self, pid, board, ctx, mask, action):
        available = [h for h in COMBOS if not set(h).intersection(board)]
        if hasattr(self.predictor, "predict_hands"):
            p = self.predictor.predict_hands(available, board, ctx, mask)[:, action]
        else:
            p = self.predictor.predict([strength(h, board) for h in available], ctx, mask)[:, action]
        old = self.weights.get(pid, {})
        self.weights[pid] = {h: max(1e-9, old.get(h, 1) * prob)
                             for h, prob in zip(available, p)}

    def combo_weight(self, pid, first, second):
        weights = self.weights.get(pid, {})
        return weights.get((first, second), weights.get((second, first), 1))

    def fold_to_bet(self, pid):
        return FOLD_PRIOR


def payout(contributions, ranks, hero):
    """Expected fractional share of exact whole-hand contribution layers."""
    result = previous = 0
    for level in sorted(set(contributions.values())):
        if level <= previous:
            continue
        eligible = [pid for pid in ranks if contributions[pid] >= level]
        if hero in eligible:
            top = max(ranks[pid] for pid in eligible)
            if ranks[hero] == top:
                layer = (level-previous)*sum(v >= level for v in contributions.values())
                result += layer/sum(ranks[pid] == top for pid in eligible)
        previous = level
    return result


def response_action(state, hero, model, ranges, public, rng, risk=0, trials=80):
    """Score the same dealt scenarios for every candidate. Offline experiment."""
    from ev import raise_targets
    players = {p["id"]: p for p in state["players"]}
    me = players[hero]
    board, _ = context(state, hero, public[hero][0], public[hero][1])
    mine = tuple(parse_card(c) for c in me["hole_cards"])
    live = [pid for pid,p in players.items() if pid != hero and not p["folded"]]
    current = state["current_bet"]
    options = [("fold",0),("call",0)] if current > me["bet"] else [("check",0)]
    if public[hero][2][3] or public[hero][2][4]:
        options.extend(("raise", n) for n in raise_targets(state, me))
    if not live:
        return options[-1]
    dead = set(mine + board)
    combos = [h for h in COMBOS if dead.isdisjoint(h)]
    cumulative = {pid:list(accumulate(ranges.combo_weight(pid,*h) for h in combos)) for pid in live}
    strength_cache = {}
    scenarios = []
    end = time.monotonic()+1.0
    for _ in range(trials):
        used = set(dead)
        hidden = {}
        order = rng.sample(live,len(live))
        for pid in order:
            weights = cumulative[pid]
            for _attempt in range(60):
                h = combos[bisect_left(weights,rng.random()*weights[-1])]
                if used.isdisjoint(h):
                    break
            else:
                h = tuple(rng.sample([c for c in FULL_DECK if c not in used],2))
            hidden[pid] = h
            used.update(h)
        full = board + tuple(rng.sample([c for c in FULL_DECK if c not in used],5-len(board)))
        ranks = {pid:best_hand(list(h+full)) for pid,h in {**hidden,hero:mine}.items()}
        for h in hidden.values():
            if h not in strength_cache:
                strength_cache[h] = strength(h,board)
        scenarios.append((ranks,hidden,{pid:rng.random() for pid in live}))
        if time.monotonic() >= end:
            break
    starting = {pid:entry[3] for pid,entry in public.items()}
    scores = []
    for action,target in options:
        if action == "fold":
            scores.append(0)
            continue
        target = target if action == "raise" else current
        cost = min(me["stack"],max(0,target-me["bet"]))
        responses = {}
        for pid in live:
            p = players[pid]
            price = min(p["stack"],max(0,target-p["bet"]))
            if not price:
                continue
            _, ctx = context(state,pid,public[pid][0],public[pid][1])
            ctx = (*ctx[:4],min(2,price/max(1,state["pot"]+cost)),*ctx[5:])
            probs = model.predict([strength_cache[h[pid]] for _,h,_ in scenarios],
                                  ctx,[True,False,True,False,p["stack"]>price])
            responses[pid] = (price,probs)
        gains = []
        for i,(ranks,_,draws) in enumerate(scenarios):
            contributions = starting.copy()
            contributions[hero] += cost
            ranks = ranks.copy()
            for pid,(price,probs) in responses.items():
                if draws[pid] < probs[i,0]:
                    del ranks[pid]
                else:
                    # Collapse reraises into calls for this one-step estimate.
                    contributions[pid] += price
            change = payout(contributions,ranks,hero)-cost
            gains.append(change-risk*max(0,-change)**2/max(1,me["stack"]))
        scores.append(sum(gains)/len(gains))
    return options[max(range(len(options)),key=scores.__getitem__)]


def play(seed, styles, hands, policy=None, model=None, collect=False, *,
         guard_raises=False, style_overrides=None, range_factory=HandRanges,
         hero_policy=None):
    engine = PokerEngine()
    decks = random.Random(seed)
    engine._build_deck = lambda: _build_deck(decks)
    button_shift = seed % len(styles)
    styles = styles[button_shift:] + styles[:button_shift]
    actors = {}
    hero = next((str(i) for i, style in enumerate(styles) if style == "hero"), None)
    ranges = range_factory(model) if policy in ("learned", "response", "risk") else None
    response_rng = random.Random(seed+701)
    for i, style in enumerate(styles):
        pid = str(i)
        engine.add_player(pid, pid)
        if style == "hero" and policy in {"adaptive", "uniform", "learned", "response", "risk"}:
            actor = AdaptiveBot(pid, trials=80, use_ranges=policy == "adaptive")
            if policy == "learned":
                from ev import choose
                actor = lambda state, pid=pid, rng=random.Random(seed+701): choose(
                    state, ranges, pid, trials=80, rng=rng)[0:2]
        elif style == "demo":
            actor = lambda state, pid=pid: ("call" if next(p["bet"] for p in state["players"] if p["id"] == pid) < state["current_bet"] else "check", 0)
        else:
            kind = policy if style == "hero" else style
            params = (style_overrides or {}).get((kind, i), POLICIES.get(kind))
            actor = ThresholdBot(pid, params, trials=30)
        if isinstance(actor, (ThresholdBot, AdaptiveBot)):
            actor.rng = random.Random(seed + 10007*(i+1))
        actors[pid] = actor
    seen = defaultdict(lambda: [2, 8])
    rows, latencies = [], []
    errors = timeouts = dealt = hero_hands = hero_actions = guarded = 0
    rejection_reasons = Counter()
    early_shoves = 0
    first_hand_bust = False
    for _ in range(hands):
        if not engine.can_start_hand():
            break
        engine.start_hand()
        dealt += 1
        hero_hands += int(hero is not None and bool(engine.players[hero].hole_cards))
        if ranges:
            ranges.reset()
        previous = 0
        hand_start = len(rows)
        while engine.phase not in ("SHOWDOWN", "WAITING"):
            pid = engine.current_turn
            state = engine.snapshot(pid)
            mask = legal(state, pid, engine.acted_at.get(pid))
            aggression = seen[pid][0]/seen[pid][1]
            for who, actor in actors.items():
                if isinstance(actor, AdaptiveBot):
                    actor.observe(engine.snapshot(who))
            started = time.monotonic()
            if pid == hero and policy in ("response","risk"):
                public = {
                    who:(seen[who][0]/seen[who][1],previous,
                         legal(state,who,engine.acted_at.get(who)),p.hand_contribution)
                    for who,p in engine.players.items()
                }
                action, amount = response_action(state,pid,model,ranges,public,response_rng,
                                                 risk=.2 if policy == "risk" else 0)
            elif pid == hero and hero_policy is not None:
                action, amount = hero_policy(state, pid, ranges, response_rng, lambda: actors[pid](state))
            else:
                action, amount = actors[pid](state)
            elapsed = time.monotonic()-started
            if pid == hero:
                latencies.append(elapsed)
                hero_actions += 1
            if elapsed >= 15:
                timeouts += int(pid == hero)
                action, amount = ("check" if mask[1] else "fold"), 0
            current = state["current_bet"]
            if guard_raises and action == "raise" and not (mask[3] or mask[4]):
                guarded += int(pid == hero)
                action, amount = ("check" if mask[1] else "call"), 0
            tag = ("bet" if current == 0 else "raise") if action == "raise" else action
            result = engine.place_action(pid, action, amount)
            if not result["ok"]:
                errors += int(pid == hero)
                if pid == hero:
                    rejection_reasons[result["error"]] += 1
                continue_action = "check" if mask[1] else "fold"
                result = engine.place_action(pid, continue_action, 0)
                if not result["ok"]:
                    raise RuntimeError(result)
                tag = continue_action
            if pid == hero and tag == "raise" and dealt <= 3 and state["phase"] == "PREFLOP":
                me = next(p for p in state["players"] if p["id"] == pid)
                early_shoves += int(amount == me["bet"] + me["stack"])
            choice = ACTIONS.index(tag)
            if not mask[choice]:
                raise RuntimeError(("action mask disagrees with engine", tag, state))
            track_range = ranges is not None and (pid != hero or hero_policy is not None)
            if collect or track_range:
                board, ctx = context(state, pid, aggression, previous)
                if track_range:
                    ranges.update(pid, board, ctx, mask, choice)
                    observe_response = getattr(ranges, "observe_response", None)
                    if observe_response is not None and mask[2]:
                        observe_response(pid, choice == ACTIONS.index("fold"))
                if collect:
                    hole = tuple(parse_card(c) for c in next(p for p in state["players"] if p["id"] == pid)["hole_cards"])
                    rows.append(dict(seed=seed, hand=dealt, player=pid, style=styles[int(pid)],
                                     board=board, hole=hole, strength=strength(hole, board),
                                     context=ctx, legal=mask, action=choice))
            seen[pid][0] += int(tag in ("bet", "raise"))
            seen[pid][1] += 1
            previous = choice + 1
        if engine.phase == "SHOWDOWN":
            for pid, actor in actors.items():
                if isinstance(actor, AdaptiveBot):
                    actor.observe(engine.snapshot(pid))
            if collect:
                for row in rows[hand_start:]:
                    row["revealed"] = not engine.players[row["player"]].folded
            engine.finalize_showdown()
        elif collect:
            for row in rows[hand_start:]:
                row["revealed"] = False
        if dealt == 1 and hero is not None:
            first_hand_bust = engine.players[hero].stack == 0
        if sum(p.stack for p in engine.players.values()) != len(styles)*engine.starting_stack:
            raise RuntimeError("chips were not conserved")
    if collect:
        return rows
    stack = engine.players[hero].stack
    rank = 1 + sum(p.stack > stack for p in engine.players.values())
    return dict(policy=policy, seed=seed, chips=stack-engine.starting_stack, hands=hero_hands,
                dealt=dealt, placement=rank, win=rank == 1, eliminated=stack == 0,
                early_shoves=early_shoves, first_hand_bust=first_hand_bust, invalid=errors, guarded=guarded, rejection_reasons=dict(rejection_reasons),
                timeouts=timeouts, decisions=hero_actions,
                latency_ms=[round(t*1000, 2) for t in latencies])


def metrics(model, rows):
    probs = np.array([model.predict([r["strength"]], r["context"], r["legal"])[0] for r in rows])
    y = np.array([r["action"] for r in rows])
    actual = probs[np.arange(len(y)), y]
    chosen = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    calibration = []
    for low in np.arange(0, 1, .1):
        selected = (confidence >= low) & (confidence < low+.100001)
        if selected.any():
            calibration.append(dict(n=int(selected.sum()), predicted=float(confidence[selected].mean()),
                                    observed=float((chosen[selected] == y[selected]).mean())))
    return dict(n=len(y), accuracy=float((chosen == y).mean()),
                log_loss=float(-np.log(np.maximum(1e-12, actual)).mean()), calibration=calibration)


def range_metrics(model, rows):
    ranges = HandRanges(model)
    current = None
    scores = defaultdict(list)
    for row in rows:
        marker = row["seed"], row["hand"]
        if marker != current:
            ranges.reset()
            current = marker
        hole = tuple(tuple(c) for c in row["hole"])
        board = tuple(tuple(c) for c in row["board"])
        ranges.update(row["player"], board, row["context"], row["legal"], row["action"])
        valid = [h for h in COMBOS if not set(h).intersection(board)]
        mass = sum(ranges.combo_weight(row["player"], *h) for h in valid)
        score = -math.log(ranges.combo_weight(row["player"], *hole)/mass)
        for label in ("all", "revealed" if row["revealed"] else "unrevealed"):
            scores[label].append((score, math.log(len(valid))))
    return {k: dict(n=len(v), combo_log_loss=float(np.mean(v, axis=0)[0]),
                    uniform_log_loss=float(np.mean(v, axis=0)[1])) for k,v in scores.items()}


def atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)


def collect(start, count, styles, hands, deadline):
    rows = []
    for seed in range(start, start+count):
        if time.monotonic() >= deadline:
            break
        rows.extend(play(seed, list(styles), hands, collect=True))
    return rows


def summary(rows):
    result = {}
    indexed = {(r["table"], r["seed"], r["policy"]): r["chips"] for r in rows}
    for table, policy in sorted({(r["table"], r["policy"]) for r in rows}):
        group = [r for r in rows if (r["table"], r["policy"]) == (table, policy)]
        latencies = [t for r in group for t in r["latency_ms"]]
        deltas = [r["chips"]-indexed[(table,r["seed"],"adaptive")] for r in group
                  if (table,r["seed"],"adaptive") in indexed]
        ranks = [r["placement"] for r in group]
        mean = float(np.mean(deltas)) if deltas else None
        ci = None
        if len(deltas) >= 2:
            half = 1.96*float(np.std(deltas,ddof=1))/math.sqrt(len(deltas))
            ci = [mean-half,mean+half]
        result[f"{table}/{policy}"] = dict(
            matches=len(group), chips=float(np.mean([r["chips"] for r in group])),
            paired_vs_adaptive=mean, paired_ci95=ci, pairs=len(deltas),
            win=float(np.mean([r["win"] for r in group])),
            eliminated=float(np.mean([r["eliminated"] for r in group])),
            placement=float(np.mean(ranks)),
            bb100=5*sum(r["chips"] for r in group)/max(1,sum(r["hands"] for r in group)),
            invalid=sum(r["invalid"] for r in group), timeouts=sum(r["timeouts"] for r in group),
            latency_p95=float(np.percentile(latencies,95)) if latencies else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error("--hours must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + (120 if args.smoke else args.hours*3600)
    train = collect(10000, 3 if args.smoke else 600,
                    TRAIN + ("aggressor","station"), 3 if args.smoke else 16,
                    min(deadline, start + max(15, (deadline-start)*.22)))
    other = TRAIN + VALID + ("balanced",)
    valid = collect(20000, 2 if args.smoke else 50, other, 3 if args.smoke else 16, deadline)
    final = collect(30000, 2 if args.smoke else 50, TRAIN+FINAL+("pressure",),
                    3 if args.smoke else 16, deadline)
    with (args.output/"decisions.jsonl").open("x") as f:
        for split, data in (("train",train),("validation",valid),("test",final)):
            for row in data:
                f.write(json.dumps(dict(split=split, **row))+"\n")
    models = {kind: Predictor(train,kind) for kind in ("table","logistic","prior","heuristic")}
    validation = {k: metrics(m, [r for r in valid if r["style"] in VALID]) for k,m in models.items()}
    selected = min(("table","logistic"), key=lambda k: validation[k]["log_loss"])
    test = {k: metrics(m, [r for r in final if r["style"] in FINAL]) for k,m in models.items()}
    hand_ranges = {k: range_metrics(m, final) for k,m in models.items()}
    atomic(args.output/"model.json", models[selected].artifact())
    atomic(args.output/"model_metrics.json", dict(selected=selected, validation=validation,
           test=test, ranges=hand_ranges, samples=dict(train=len(train), validation=len(valid), test=len(final))))
    print("Selected", selected, "validation log loss", validation[selected]["log_loss"], flush=True)
    rows = []
    tables = {"mixed-six": ["hero", "aggressor", "evolved", "station", "rock", "pressure"],
              "mixed-five": ["hero", "aggressor", "station", "rock", "pressure"]}
    block = 0
    policies = ("adaptive","aggressor","evolved","uniform","learned","response","risk")
    with (args.output/"matches.jsonl").open("x", buffering=1) as f:
        while time.monotonic() < deadline and (not args.smoke or block < 2):
            table = list(tables)[block % len(tables)]
            seed = 1000000+block
            for policy in policies:
                if time.monotonic() >= deadline:
                    break
                row = play(seed, tables[table], 2 if args.smoke else 16, policy, models[selected])
                row["table"] = table
                rows.append(row)
                f.write(json.dumps(row)+"\n")
            atomic(args.output/"summary.json", summary(rows))
            block += 1
            if block % 10 == 0:
                print(len(rows), "matches", flush=True)
    print("Results in", args.output)


if __name__ == "__main__":
    main()
