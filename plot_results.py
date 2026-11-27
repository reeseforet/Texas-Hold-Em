"""Create charts from saved training results."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(run):
    metrics = json.loads((run/"model_metrics.json").read_text())
    summary = json.loads((run/"summary.json").read_text())
    matches = [json.loads(line) for line in (run/"matches.jsonl").read_text().splitlines()]
    figures = run/"figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(figures/f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    for table in sorted({row["table"] for row in matches}):
        totals = defaultdict(list)
        for row in matches:
            if row["table"] == table:
                totals[row["policy"]].append(row["chips"])
        fig, ax = plt.subplots(figsize=(8,5))
        for policy, chips in sorted(totals.items()):
            ax.plot(np.arange(1,len(chips)+1), np.cumsum(chips), label=policy)
        ax.set(xlabel="Match", ylabel="Cumulative chips", title=table)
        ax.legend()
        save(fig, f"{table}-cumulative")
        names = sorted(totals)
        rows = [summary[f"{table}/{name}"] for name in names]
        fig, axes = plt.subplots(1,2,figsize=(12,5),layout="constrained")
        for i,row in enumerate(rows):
            gain, ci = row["paired_vs_adaptive"], row["paired_ci95"]
            if gain is not None:
                axes[0].errorbar(gain,i,xerr=None if ci is None else (ci[1]-ci[0])/2,fmt="o")
        axes[0].axvline(0,color="grey")
        axes[0].set(yticks=range(len(names)),yticklabels=names,xlabel="Paired chip gain vs adaptive")
        y = np.arange(len(names))
        axes[1].barh(y-.18,[row["win"] for row in rows],.36,label="Tied or outright first")
        axes[1].barh(y+.18,[row["eliminated"] for row in rows],.36,label="Eliminated")
        axes[1].set(yticks=y,yticklabels=names,xlabel="Match fraction")
        axes[1].legend()
        save(fig,f"{table}-ablation")

    fig, axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    names = list(metrics["test"])
    axes[0].bar(names,[metrics["test"][name]["log_loss"] for name in names])
    axes[0].set(ylabel="Action log loss",title="Held-out opponent style")
    axes[1].plot([0,1],[0,1],"--",color="grey")
    for name in names:
        buckets = metrics["test"][name]["calibration"]
        axes[1].plot([b["predicted"] for b in buckets],[b["observed"] for b in buckets],"o-",label=name)
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel="Predicted confidence",ylabel="Observed accuracy")
    axes[1].legend()
    save(fig,"prediction-calibration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    render(parser.parse_args().run)
