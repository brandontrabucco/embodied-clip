import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    with open("experiment_output/metrics/OnePhaseRGBClipResNet50Dagger_40proc/2022-08-15_00-58-05/metrics__test_2022-08-15_00-58-05.json", "r") as f:

        metrics_list = json.load(f)

    metrics = dict(train=[], val=[], test=[])

    metrics["test"] = [x["unshuffle/prop_fixed_strict"] 
                       for x in metrics_list[0]["tasks"] if x["task_info"]["stage"] == "test"]

    metrics["train"] = [x["unshuffle/prop_fixed_strict"] 
                        for x in metrics_list[0]["tasks"] if x["task_info"]["stage"] == "train"]

    metrics["val"] = [x["unshuffle/prop_fixed_strict"] 
                      for x in metrics_list[0]["tasks"] if x["task_info"]["stage"] == "val"]

    for key in metrics.keys():
        print(key, np.mean(metrics[key]))
