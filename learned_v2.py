"""Train and validate six-player learned-range challengers locally."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import random
import statistics
import math
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier

from train_model import (
    ACTIONS, COMBOS, HandRanges, POLICIES, atomic, features, play, strength, summary,
)
from opponent_model import class_strength, hand_class

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "baselines/learned_v1/model.json"
FAMILIES = {
    "train": ("aggressor", "evolved", "station", "rock", "balanced", "pressure"),
    "validation": ("late_attack", "loose_flat", "short_value", "small_ball", "nit"),
    "final": ("wide_press", "trap", "patient", "overbet", "loose_value"),
}
EXTRA = {
    "late_attack": (.39, .01, .70, .14), "loose_flat": (.77, -.10, .52, .07),
    "short_value": (.43, .07, .45, .18), "small_ball": (.41, .01, .28, .11),
    "nit": (.75, .12, .55, .29), "wide_press": (.25, .01, .82, .09),
    "trap": (.78, -.04, .55, .15), "patient": (.57, .10, .40, .21),
    "overbet": (.38, .02, 1.7, .17), "loose_value": (.52, -.08, .90, .09),
}
POLICIES.update({name:dict(zip(("raise_at","call_margin","raise_fraction","preflop_floor"), vals))
                 for name,vals in EXTRA.items()})


@lru_cache(maxsize=24)
def hand_matrix(hands, board):
    result = []
    board_suits = [s for _,s in board]
    board_ranks = {r for r,_ in board}
    for hand in hands:
        pre = class_strength(hand_class(*hand))
        made = strength(hand, board)
        ranks = board_ranks | {r for r,_ in hand}
        if 14 in ranks:
            ranks.add(1)
        straight = max(len(ranks.intersection(range(low,low+5))) for low in range(1,11))
        flush = max(board_suits.count(s)+sum(t==s for _,t in hand) for s in "cdhs")
        pair = sum(r in board_ranks for r,_ in hand)
        result.append((made, pre, straight/5, flush/7, pair/2,
                       float(hand[0][1]==hand[1][1]), max(r for r,_ in hand)/14))
    return np.asarray(result)


def expanded(hands, board, ctx):
    h = hand_matrix(tuple(hands), tuple(board))
    s = h[:,0]
    return np.column_stack((h, s*s, s*ctx[4], s*ctx[0],
                            h[:,1]*ctx[1], h[:,0]*ctx[9], h[:,2]*ctx[0],
                            np.broadcast_to(ctx,(len(h),len(ctx)))))


def normalise(p, mask, temperature=1):
    p = np.maximum(p,1e-10)**(1/temperature) * mask
    return p/p.sum(axis=-1,keepdims=True)


class Control:
    def __init__(self):
        self.weights = json.loads(CONTROL.read_text())["logistic"]

    def raw(self, x):
        z = x @ np.asarray(self.weights["coef"]).T + self.weights["intercept"]
        p = np.zeros((len(x),len(ACTIONS)))
        p[:,self.weights["classes"]] = np.exp(z-z.max(axis=1,keepdims=True))
        return p/p.sum(axis=1,keepdims=True)

    def predict(self, strengths, ctx, mask):
        return normalise(self.raw(features(strengths,ctx)), mask)


class Challenger:
    def __init__(self, estimator, temperature=1):
        self.estimator = estimator
        self.temperature = temperature

    def raw(self, x):
        p = np.zeros((len(x),len(ACTIONS)))
        p[:,self.estimator.classes_] = self.estimator.predict_proba(x)
        return p

    def predict_hands(self, hands, board, ctx, mask):
        return normalise(self.raw(expanded(hands,board,ctx)),mask,self.temperature)


class SmoothedRanges(HandRanges):
    def update(self,pid,board,ctx,mask,action):
        valid = [h for h in COMBOS if set(board).isdisjoint(h)]
        probs = self.predictor.predict_hands(valid,board,ctx,mask)[:,action]
        previous = self.weights.get(pid,{})
        weights = np.array([previous.get(h,1.0) for h in valid],dtype=float)
        weights *= np.maximum(probs,1e-8)**.65
        weights = .97*weights/weights.sum()+.03/len(valid)
        self.weights[pid] = dict(zip(valid,weights))


def prepare(rows):
    x = np.concatenate([expanded([r["hole"]], r["board"], r["context"]) for r in rows])
    base = np.concatenate([features([r["strength"]],r["context"]) for r in rows])
    return x,base,np.asarray([r["legal"] for r in rows]),np.asarray([r["action"] for r in rows])


def prediction_metrics(p,y):
    pred = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = pred==y
    precision,recall,f1,n = precision_recall_fscore_support(y,pred,labels=range(5),zero_division=0)
    bins = []
    for lo in np.arange(0,1,.1):
        mask = (confidence >= lo) & (confidence < lo+.100001)
        if mask.any():
            bins.append([int(mask.sum()),float(confidence[mask].mean()),float(correct[mask].mean())])
    return dict(n=len(y),accuracy=float(correct.mean()),
                log_loss=float(-np.log(p[np.arange(len(y)),y]).mean()),
                brier=float(np.mean(np.square(p-np.eye(5)[y]).sum(axis=1))),
                ece=sum(n*abs(a-b) for n,a,b in bins)/len(y),calibration=bins,
                classes={name:dict(precision=float(precision[i]),recall=float(recall[i]),
                                   f1=float(f1[i]),n=int(n[i])) for i,name in enumerate(ACTIONS)})


def range_score(model,rows,smooth=False):
    ranges = SmoothedRanges(model) if smooth else HandRanges(model)
    marker = None
    scores = {"all":[], "revealed":[], "unrevealed":[]}
    for row in rows:
        key = row["seed"],row["hand"]
        if key != marker:
            ranges.reset()
            marker = key
        pid,board = row["player"],row["board"]
        ranges.update(pid,board,row["context"],row["legal"],row["action"])
        w = ranges.weights[pid]
        hole = row["hole"]
        score = -np.log(w.get(hole,w.get(hole[::-1]))/sum(w.values()))
        entry = (score,np.log(len(w)))
        scores["all"].append(entry)
        scores["revealed" if row["revealed"] else "unrevealed"].append(entry)
    return {k:dict(n=len(v),log_loss=float(np.mean(v,axis=0)[0]),
                   uniform=float(np.mean(v,axis=0)[1])) for k,v in scores.items() if v}


def game(params):
    seed, split, hands, policy, model, guard = params
    rng = random.Random(seed*113)
    styles = list(FAMILIES[split])
    rng.shuffle(styles)
    if policy:
        styles = ["hero",*styles[:5]]
    while len(styles)<6:
        styles.append(rng.choice(FAMILIES[split]))
    overrides = {}
    for i in range(6):
        for style in FAMILIES[split]:
            overrides[style,i] = {k:max(-.15 if k=="call_margin" else 0,
                                        v+rng.uniform(-.13,.13)*(1 if k=="raise_fraction" else .25))
                                  for k,v in POLICIES[style].items() if isinstance(v,(int,float))}
    kind = "learned" if model is not None else policy
    result = play(seed,styles,hands,kind,model,collect=policy is None,
                  guard_raises=guard,style_overrides=overrides,
                  range_factory=SmoothedRanges if policy and policy.endswith("-smooth") else HandRanges)
    if policy:
        result.update(policy=policy,table=split,opponents=styles[1:])
    return result


def paired_batch(pool,seed,split,hands,policies,models,guard=True):
    return list(pool.map(game,[(seed,split,hands,name,models.get(name),guard) for name in policies]))


def report(rows):
    result = summary(rows)
    control = {(r["table"],r["seed"]):r["chips"] for r in rows if r["policy"]=="control"}
    groups = {}
    for row in rows:
        groups.setdefault((row["table"],row["policy"]),[]).append(row)
    for (table,policy),group in groups.items():
        deltas = [r["chips"]-control[table,r["seed"]] for r in group if (table,r["seed"]) in control]
        half = 1.96*statistics.stdev(deltas)/math.sqrt(len(deltas)) if len(deltas)>1 else None
        average = statistics.mean(deltas) if deltas else None
        result[f"{table}/{policy}"].update(paired_vs_control=average,
            control_ci95=[average-half,average+half] if half is not None else None,
            control_pairs=len(deltas),guarded=sum(r["guarded"] for r in group))
    return result


def dump_model(path,model):
    tmp = path.with_suffix(".tmp")
    joblib.dump(model,tmp,compress=3)
    tmp.replace(path)


def run(args):
    out = args.output
    out.mkdir(parents=True)
    deadline = time.monotonic()+args.hours*3600
    started = time.monotonic()
    hands = 3 if args.smoke else 16
    models = {"control":Control()}
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (
        out/"matches.jsonl").open("x",buffering=1) as log, (
        out/"decisions.jsonl").open("x",buffering=1) as data_log:
        data = {}
        for split,start,count in (("train",40000,5 if args.smoke else 6000),
                                  ("validation",60000,3 if args.smoke else 48)):
            data[split] = []
            for offset in range(0,count,args.workers):
                limit = started + args.hours*3600*.28
                if not args.smoke and split=="train" and data[split] and (time.monotonic()>=limit or len(data[split])>=args.max_rows):
                    break
                batch = [(seed,split,hands,None,None,True)
                         for seed in range(start+offset,start+min(count,offset+args.workers))]
                for block in pool.map(game,batch):
                    data[split].extend(block)
                    for row in block:
                        data_log.write(json.dumps(dict(split=split,**row))+"\n")
        train,valid = data["train"],data["validation"]
        tx,_,_,ty = prepare(train)
        vx,vb,vm,vy = prepare(valid)
        print("Training decisions",len(train),"validation decisions",len(valid),flush=True)
        candidates = {
            "logistic-c03":LogisticRegression(C=.3,max_iter=250),
            "logistic-c3":LogisticRegression(C=3,max_iter=250),
            "forest":RandomForestClassifier(n_estimators=40 if args.smoke else 100,
                         min_samples_leaf=10,max_depth=14,n_jobs=args.workers,random_state=71),
            "mlp":MLPClassifier(hidden_layer_sizes=(32,),max_iter=25 if args.smoke else 70,random_state=71),
        }
        try:
            from xgboost import XGBClassifier
            candidates["xgboost"] = XGBClassifier(n_estimators=24 if args.smoke else 120,
                max_depth=5,learning_rate=.10,n_jobs=args.workers,random_state=71)
        except Exception as error:
            if not isinstance(error,(ImportError,OSError)) and type(error).__module__ != "xgboost.core":
                raise
            print("XGBoost unavailable:",str(error).strip().splitlines()[0],flush=True)
            candidates["hist-gradient"] = HistGradientBoostingClassifier(
                max_iter=30 if args.smoke else 120,max_leaf_nodes=24,random_state=71)
        scores = {"control":prediction_metrics(normalise(models["control"].raw(vb),vm),vy)}
        sizes = [len(ty)] if args.smoke else sorted({min(len(ty),n) for n in (100000,250000,500000,args.max_rows)})
        for name,estimator in candidates.items():
            for size in sizes if name.startswith("logistic") else [sizes[-1]]:
                if not args.smoke and time.monotonic()>started+args.hours*3600*.53:
                    break
                if len(set(ty[:size])) != len(ACTIONS) and name=="xgboost":
                    continue
                estimator.fit(tx[:size],ty[:size])
                if hasattr(estimator,"n_jobs"):
                    estimator.set_params(n_jobs=1)
                model = Challenger(estimator)
                raw = model.raw(vx)
                grid = {t:prediction_metrics(normalise(raw,vm,t),vy) for t in (.65,.8,1,1.25,1.6)}
                model.temperature = min(grid,key=lambda t:grid[t]["log_loss"])
                key = f"{name}-{size}"
                model_path = out/f"{key}.joblib"
                dump_model(model_path,model)
                models[key] = joblib.load(model_path)
                scores[key] = dict(**grid[model.temperature],temperature=model.temperature,
                                   uncalibrated_log_loss=grid[1]["log_loss"])
                atomic(out/"validation.json",scores)
        # Evaluate whole matched prefixes so range evidence never crosses splits.
        sample_keys = sorted({r["seed"] for r in valid})[:(1 if args.smoke else 5)]
        sample = [r for r in valid if r["seed"] in sample_keys]
        for key,model in models.items():
            scores[key]["ranges"] = range_score(model,sample)
        atomic(out/"validation.json",scores)
        shortlist = sorted((k for k in models if k!="control"),
                           key=lambda k:scores[k]["ranges"]["all"]["log_loss"])[:3]
        if shortlist:
            models[shortlist[0]+"-smooth"] = models[shortlist[0]]
            shortlist.append(shortlist[0]+"-smooth")
        policies = ["control",*shortlist]
        batches = 2 if args.smoke else 80
        for i in range(batches):
            if not args.smoke and time.monotonic() > started+args.hours*3600*.72:
                break
            block = paired_batch(pool,80000+i,"validation",hands,policies,models)
            rows.extend(block)
            for row in block:
                log.write(json.dumps(row)+"\n")
            atomic(out/"summary.json",report(rows))
        leaders = {k:np.mean([r["chips"] for r in rows if r["policy"]==k])
                   for k in shortlist if any(r["policy"]==k for r in rows)}
        finalist = max(leaders,key=leaders.get) if leaders else "control"
        atomic(out/"selection.json",dict(finalist=finalist,validation_chip_means=leaders,
                                      validation_shortlist=shortlist))
        final_data = list(pool.map(game,[(70000+i,"final",hands,None,None,True)
                                        for i in range(2 if args.smoke else 8)]))
        holdout = [row for match in final_data for row in match]
        for row in holdout:
            data_log.write(json.dumps(dict(split="final",**row))+"\n")
        fx,fb,fm,fy = prepare(holdout)
        reports = {}
        for key in set(("control",finalist)):
            model = models[key]
            predictions = normalise(model.raw(fb),fm) if key=="control" else normalise(
                model.raw(fx),fm,model.temperature)
            reports[key] = dict(actions=prediction_metrics(predictions,fy),
                                ranges=range_score(model,holdout,key.endswith("-smooth")))
        atomic(out/"final_predictions.json",reports)
        final_policies = list(dict.fromkeys(["adaptive","aggressor","evolved","uniform","control",finalist]))
        i = 0
        while time.monotonic()<deadline and (not args.smoke or i<2):
            block = paired_batch(pool,90000+i,"final",hands,final_policies,models)
            rows.extend(block)
            for row in block:
                log.write(json.dumps(row)+"\n")
            atomic(out/"summary.json",report(rows))
            i += 1
            if i%20==0:
                print("Final paired blocks",i,flush=True)
    print("Finalist",finalist,"final paired blocks",i,"results",out,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours",type=float,default=8)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--max-rows",type=int,default=250000)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.hours<=0 or args.workers<=0 or args.max_rows<=0:
        parser.error("hours, workers and max rows must be positive")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted. Completed match logs and checkpoints are in",args.output)


if __name__ == "__main__":
    from learned_v2 import main
    main()
