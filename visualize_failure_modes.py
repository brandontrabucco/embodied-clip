import pickle as pkl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":

    with open("interactions.pkl", "rb") as f:
        interactions = pkl.load(f)

    results = []
    room_types = [
        x for x in [
        "bathroom",
        "bedroom",
        "living room",
        "kitchen"] for i in range(250)
    ]

    for task_id, task in enumerate(interactions):

        for value in task.values():

            results.append(dict(
                stage="correct_nav0",
                scene=room_types[task_id],
                value=(
                    value["correct_nav0"]
                ),
            ))

            results.append(dict(
                stage="correct_pick",
                scene=room_types[task_id],
                value=(
                    value["correct_nav0"] and 
                    value["correct_pick"]
                ),
            ))

            results.append(dict(
                stage="correct_nav1",
                scene=room_types[task_id],
                value=(
                    value["correct_nav0"] and 
                    value["correct_pick"] and 
                    value["correct_nav1"]
                ),
            ))

            results.append(dict(
                stage="correct_place",
                scene=room_types[task_id],
                value=(
                    value["correct_nav0"] and 
                    value["correct_pick"] and 
                    value["correct_nav1"] and 
                    value["correct_place"]
                ),
            ))

    df = pd.DataFrame.from_records(results)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=df, x="stage", y="value", hue="scene")

    plt.xlabel("Rearrangement Stage")
    plt.ylabel("Success")

    plt.tight_layout()
    plt.savefig("failure_modes.png")

    plt.clf()

    results = []

    for task_id, task in enumerate(interactions):

        for value in task.values():

            results.append(dict(
                stage="failed_pick",
                scene=room_types[task_id],
                value=(
                    value["correct_nav0"] and 
                    not value["correct_pick"]
                ),
            ))

            results.append(dict(
                stage="wrong_pick",
                scene=room_types[task_id],
                value=(
                    value["wrong_pick"]
                ),
            ))

    df = pd.DataFrame.from_records(results)

    plt.figure(figsize=(3, 4))
    sns.barplot(data=df, x="stage", y="value", hue="scene")

    plt.xlabel("Pickup Failed Reason")
    plt.ylabel("Failed")

    plt.tight_layout()
    plt.savefig("failure_modes_pick.png")