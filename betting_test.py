"""Log betting decisions and compare strategies on the same deals."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

from bot import FULL_DECK, parse_card
from ev import choose


def describe(state, hero, model, info, action, amount):
    me = next(p for p in state["players"] if p["id"] == hero)
    board = state.get("community_cards", [])
    blockers = {parse_card(c) for c in me["hole_cards"] + board}
    combos = tuple(combinations([c for c in FULL_DECK if c not in blockers],2))
    ranges = {}
    for player in state["players"]:
        pid = player["id"]
        if pid == hero or player.get("folded"):
            continue
        # Read only the model's public-inference weights.
        weights = [(model.combo_weight(pid,*h) or 1,h) for h in combos]
        mass = sum(w for w,_ in weights)
        ranges[pid] = dict(
            effective_stack=min(me["stack"],player["stack"]),
            inferred_effective_combos=round(mass**2/sum(w*w for w,_ in weights),1),
            top_combos=[(h,round(w/mass,6)) for w,h in sorted(weights,reverse=True)[:3]],
        )
    seats = [p["id"] for p in state["players"]]
    dealer = state.get("dealer")
    position = (seats.index(hero)-seats.index(dealer))%len(seats) if dealer in seats else None
    return dict(hand=state["hand_number"], street=state["phase"],
                cards=me["hole_cards"], board=board, seats_from_button=position,
                pot=state["pot"], stack=me["stack"], street_bet=me["bet"],
                to_call=min(me["stack"],max(0,state["current_bet"]-me["bet"])),
                opponent_ranges=ranges,
                equity=info["equity"], trials=info["trials"],
                candidates=[dict(action=a, amount=n, ev=v, score=score,
                                 max_new_commitment=(n-me["bet"] if a=="raise" else
                                     min(me["stack"],max(0,state["current_bet"]-me["bet"])) if a=="call" else 0),
                                 responses={p:dict(fold=x,continue_=1-x) for p,x in
                                            info["responses"].get(f"{a}:{n}",{}).items()})
                            for (a,n,v),(_,_,score) in zip(info["options"],info["scores"])],
                selected=[action,amount], reason=("highest estimated score" if
                    [action,amount]==list(max(info["scores"],key=lambda row:(row[2],-row[1]))[:2])
                    else "legality guard or tie among rounded candidate scores"))


def observed_ranges():
    from train_model import HandRanges

    class ObservedRanges(HandRanges):
        def __init__(self, predictor):
            super().__init__(predictor)
            self.responses = defaultdict(lambda: [0,0])

        def observe_response(self, player, folded):
            counts = self.responses[player]
            counts[0] += int(folded)
            counts[1] += 1

        def fold_to_bet(self, player):
            folds, total = self.responses[player]
            return (1.08 + folds)/(4 + total)

    return ObservedRanges


def run(out, pairs, hands):
    from train_model import atomic, play
    from learned_v2 import Control
    out.mkdir(parents=True)
    model = Control()
    rows = []
    variants = ("control","always-call","observed-folds","position","deep-shove","realization","survival","preflop-payout")
    with (out/"matches.jsonl").open("x",buffering=1) as log, (
        out/"early-decisions.jsonl").open("x",buffering=1) as decisions:
        for table,opponents in (("demo",["demo"]*5),
                                ("mixed",["aggressor","station","rock","evolved","pressure"])):
            for seed in range(62000,62000+pairs):
                for variant in variants:
                    if variant=="always-call" and table!="demo":
                        continue
                    def decide(state,hero,ranges,rng,fallback):
                        action,amount,info = choose(state,ranges,hero,trials=80,rng=rng,
                            experiment=None if variant=="control" else variant)
                        if state["hand_number"]<=3 and state["phase"]=="PREFLOP":
                            decisions.write(json.dumps(dict(seed=seed,table=table,variant=variant,
                                **describe(state,hero,ranges,info,action,amount)))+"\n")
                        return action,amount
                    row = play(seed,["hero",*opponents],hands,"learned",model,
                               guard_raises=True,hero_policy=decide,
                               **({"range_factory": observed_ranges()} if variant == "observed-folds" else {}))
                    row.update(table=table, policy=variant)
                    rows.append(row)
                    log.write(json.dumps(row)+"\n")
    report = {}
    for table in ("demo","mixed"):
        control = {r["seed"]:r["chips"] for r in rows if r["table"]==table and r["policy"]=="control"}
        for variant in variants:
            group = [r for r in rows if r["policy"]==variant and r["table"]==table]
            if not group:
                continue
            n = len(group)
            deltas = [r["chips"]-control[r["seed"]] for r in group]
            mean = statistics.mean(deltas)
            half = 1.96*statistics.stdev(deltas)/math.sqrt(n) if n>1 else None
            latency = sorted(t for r in group for t in r["latency_ms"])
            with_shove = [r for r in group if r["early_shoves"]]
            report[f"{table}/{variant}"] = dict(matches=n, chips=statistics.mean(r["chips"] for r in group),
                paired_gain=mean, ci95=[mean-half,mean+half] if half is not None else None,
                first=statistics.mean(r["win"] for r in group),
                eliminated=statistics.mean(r["eliminated"] for r in group),
                early_shove_matches=len(with_shove), early_shoves=sum(r["early_shoves"] for r in group),
                first_hand_busts=sum(r["first_hand_bust"] for r in group),
                eliminated_with_early_shove=sum(r["eliminated"] for r in with_shove),
                invalid=sum(r["invalid"] for r in group), timeouts=sum(r["timeouts"] for r in group),
                p95_ms=latency[math.ceil(.95*len(latency))-1] if latency else None)
    atomic(out/"summary.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--pairs",type=int,default=12)
    parser.add_argument("--hands",type=int,default=8)
    args = parser.parse_args()
    run(args.output,args.pairs,args.hands)
