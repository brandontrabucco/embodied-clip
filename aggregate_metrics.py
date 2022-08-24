import json
import glob

import numpy as np
import argparse


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="results_bc/transformer-0.pt")
    args = parser.parse_args()

    metrics = []

    for file in glob.glob(args.ckpt + ".*.val.json"):
        with open(file, "r") as file:
            metrics.extend(json.load(file))

    print("Prop Fixed Strict:", len(metrics), np.array([
        x["unshuffle/prop_fixed_strict"] for x in metrics]).mean())
