"""Train local CFR river checkpoints and compare experimental policies."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
from pathlib import Path
import random
import statistics
import time

import joblib

from bot import FULL_DECK
from cfr_lab.core import Kuhn, Leduc, Solver, evaluate, uniform
from cfr_lab.river import River, experiment_action
from train_model import atomic, play
from learned_v2 import Control, SmoothedRanges, report


def river_fixture(seed):
    rng = random.Random(seed)
    board = rng.sample(FULL_DECK,5)
    hands = [h for h in combinations(FULL_DECK,2) if set(board).isdisjoint(h)]
    # Select private hand supports independently of any actual deal.
    ranges = [{h:rng.uniform(.5,1.5) for h in rng.sample(hands,20)} for _ in range(2)]
    return River(board,ranges,pot=(80,160,240)[seed%3],stack=(160,320)[seed%2],minimum=20)


def checkpoint(path,game,solver,diagnostics):
    profile = solver.average()
    rows = []
    for key, regret in solver.regret.items():
        p,b,h = key
        rows.append(dict(player=p,bucket=b,history=h,regret=regret,
                         strategy_sum=solver.total.get(key),average=profile.get(key)))
    atomic(path,dict(iterations=solver.iterations,board=game.board,
                ranges=[list(zip(h,w)) for h,w in zip(game.hands,game.weights)],
                pot=game.pot,stack=game.stack,minimum=game.minimum,
                buckets=game.buckets,diagnostics=diagnostics,infosets=rows))


def fixture_value(game, profile, seed, runs=500):
    """Paired draws against two fixed opponents, averaging both seats."""
    rng = random.Random(seed)
    gains = {}
    for opponent in ("uniform","passive"):
        changes = []
        for _ in range(runs):
            deal = game.sample(rng)
            uniforms = [rng.random() for _ in range(8)]

            def trial(seat,use_cfr):
                h = ()
                while True:
                    p,actions,utility = game.node(deal,h)
                    if p is None:
                        return utility if seat==0 else -utility
                    if p==seat and use_cfr:
                        probs = profile.get(game.info(deal,h,p),uniform(len(actions)))
                    elif p!=seat and opponent=="passive":
                        probs = [float(a==("call" if "call" in actions else "check")) for a in actions]
                    else:
                        probs = uniform(len(actions))
                    u = uniforms[len(h)]
                    for action,prob in zip(actions,probs):
                        u -= prob
                        if u<0:
                            break
                    h += (action,)
            changes.append(sum(trial(seat,True)-trial(seat,False) for seat in (0,1))/2)
        mean = statistics.mean(changes)
        half = 1.96*statistics.stdev(changes)/runs**.5
        gains[opponent] = dict(mean=mean,ci95=[mean-half,mean+half],pairs=runs)
    return gains


def benchmarks(out, deadline, smoke):
    definitions = (("kuhn-cfr",Kuhn(),False,4000),("kuhn-external",Kuhn(),True,20000),
                   ("leduc-external",Leduc(),True,60000))
    results = {}
    for name,game,external,count in definitions:
        solver = Solver(13)
        results[name] = [dict(iterations=0,**evaluate(game,{}))]
        for target in (count//10,count):
            solver.train(game,target-solver.iterations,external)
            results[name].append(dict(iterations=target,**evaluate(game,solver.average())))
        atomic(out/"convergence.json",results)
        print(name,results[name][-1],flush=True)
    river_stop = min(deadline,time.monotonic()+(10 if smoke else (deadline-time.monotonic())*.25))
    for seed in range(2 if smoke else 18):
        if time.monotonic()>=river_stop and not smoke:
            break
        game = river_fixture(seed)
        solver = Solver(seed)
        diagnostics = [dict(iterations=0,**evaluate(game,{}))]
        cap = 1500 if smoke else 200000
        while solver.iterations<cap and (smoke or time.monotonic()<river_stop):
            solver.train(game,
                         min(1000 if smoke else 5000,cap-solver.iterations),external=True)
        diagnostics.append(dict(iterations=solver.iterations,**evaluate(game,solver.average())))
        diagnostics[-1]["against_fixed"] = fixture_value(game,solver.average(),303000+seed)
        checkpoint(out/f"river-{seed:02}.json",game,solver,diagnostics)
        results[f"river-{seed:02}"] = diagnostics
        atomic(out/"convergence.json",results)


FIELDS = (("aggressor","evolved","station","rock","pressure"),
          ("balanced","pressure","station","evolved","rock"),
          ("pressure","pressure","balanced","aggressor","rock"))


def match(task):
    seed,seats,hands,policy,v2 = task
    opponents = list(FIELDS[seed%len(FIELDS)])
    random.Random(seed).shuffle(opponents)
    styles = ["hero",*opponents[:seats-1]]
    model = Control()
    mode = "ev"
    counts = {"eligible":0,"used":0}
    range_factory = None
    if v2 and policy=="v2":
        model = joblib.load(v2[0])
        if v2[1]:
            range_factory = SmoothedRanges
    if policy in {"cfr","hybrid","uncertain"}:
        mode = policy
    def choose(state, hero, ranges, rng, fallback):
        if (state["phase"]=="RIVER" and state.get("dealer")!=hero
                and len([p for p in state["players"] if not p["folded"]])==2):
            counts["eligible"] += 1
        fell_back = False
        def tracked_fallback():
            nonlocal fell_back
            fell_back = True
            return fallback()
        action = experiment_action(state,hero,ranges,rng,tracked_fallback,mode=mode)
        counts["used"] += int(not fell_back)
        return action
    kind = policy if policy in {"adaptive","aggressor","evolved"} else "learned"
    options = dict(range_factory=range_factory) if range_factory else {}
    row = play(seed,styles,hands,kind,model,guard_raises=True,**options,
               hero_policy=choose if kind=="learned" else None)
    row.update(policy=policy,table=f"{seats}-player",cfr=counts,opponents=styles[1:])
    return row


def run(args):
    out = args.output
    out.mkdir(parents=True,exist_ok=False)
    deadline = time.monotonic()+args.hours*3600
    benchmarks(out,deadline,args.smoke)
    v2 = None
    if args.v2_run:
        selected = json.loads((args.v2_run/"selection.json").read_text())["finalist"]
        if selected!="control":
            artifact = args.v2_run/(selected.removesuffix("-smooth")+".joblib")
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            v2 = str(artifact.resolve()),selected.endswith("-smooth")
    policies = ["control","cfr","hybrid","uncertain","adaptive","aggressor","evolved"]
    if v2:
        policies.append("v2")
    atomic(out/"config.json",dict(seed=args.seed,hours=args.hours,workers=args.workers,
                                  smoke=args.smoke,v2=v2,policies=policies))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (out/"matches.jsonl").open("x",buffering=1) as log:
        block = 0
        while (block<4 if args.smoke else time.monotonic()<deadline):
            seats = 2 if block%2==0 else 6
            seeds = [(args.seed+block*6+i,seats,6 if args.smoke else 16,policy,v2)
                     for i in range(1 if args.smoke else 6) for policy in policies]
            for row in pool.map(match,seeds):
                rows.append(row)
                log.write(json.dumps(row)+"\n")
            result = report(rows)
            for key,group in result.items():
                table,policy = key.split("/")
                subset = [r for r in rows if r["table"]==table and r["policy"]==policy]
                group["cfr_used"] = sum(r["cfr"]["used"] for r in subset)
                group["cfr_eligible_proxy"] = sum(r["cfr"]["eligible"] for r in subset)
            atomic(out/"summary.json",result)
            block += 1
            if block%10==0:
                print("paired blocks",block,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours",type=float,default=8)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seed",type=int,default=460001)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--v2-run",type=Path,help="Existing local Learned V2 run with a selected model")
    args = parser.parse_args()
    if args.hours<=0 or args.workers<1:
        parser.error("hours and workers must be positive")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted. Completed checkpoints are in",args.output)
