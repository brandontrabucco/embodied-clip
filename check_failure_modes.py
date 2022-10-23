from baseline_configs.one_phase.one_phase_rgb_expert_dagger import (
    OnePhaseRGBExpertDaggerExperimentConfig
)
from allenact.utils.inference import InferenceAgent

import torch
import numpy as np
from collections import defaultdict, namedtuple

import stringcase
import pickle as pkl


if __name__ == "__main__":

    exp_config = OnePhaseRGBExpertDaggerExperimentConfig()

    task_sampler_args = exp_config.stagewise_task_sampler_args(
        stage="val", devices=[0], process_ind=0, total_processes=1)

    task_sampler = exp_config.make_sampler_fn(
        **task_sampler_args, force_cache_reset=False, epochs=1)

    object_interactions_list = []

    for task_id in range(task_sampler.length):

        agent = InferenceAgent.from_experiment_config(
            exp_config=exp_config, 
            device=torch.device("cpu"),
            checkpoint_path="/home/ubuntu/cvpr2023_expert/checkpoints/"
            "OnePhaseRGBExpertDagger/2022-10-12_23-17-43/"
            "exp_OnePhaseRGBExpertDagger__stage_00__steps_000072161550.pt"
        )

        agent.reset()

        task = task_sampler.next_task()
        observations = task.get_observations()
        
        object_type = None
        object_name = None

        object_interactions = defaultdict(lambda: dict(
            correct_nav0=False,
            correct_pick=False,
            wrong_pick=False,
            correct_nav1=False,
            correct_place=False
        ))

        while not task.is_done():

            action = agent.act(observations)
            observations = task.step(action).observation
            start_poses, goal_poses, current_poses = task.env.poses

            goal_poses = {x["name"]: x for x in goal_poses}
            current_poses = {x["name"]: x for x in current_poses}

            if "drop_held_object_with_snap" == task.action_names()[action]:

                correct_place = task.env.are_poses_equal(
                    goal_poses[object_name], 
                    current_poses[object_name],
                    treat_broken_as_unequal=True
                )

                object_interactions[
                    object_name]["correct_place"] = correct_place

            if task.greedy_expert._last_to_interact_object_pose is not None:
                object_type = task.greedy_expert._last_to_interact_object_pose["type"]
                object_name = task.greedy_expert._last_to_interact_object_pose["name"]

            if "pickup" in task.action_names()[action]:

                correct_pick = (
                    task.action_names()[action] == 
                    f"pickup_{stringcase.snakecase(object_type)}"
                    and task.env.held_object is not None 
                    and task.env.held_object["name"] == object_name
                )

                object_interactions[
                    object_name]["correct_pick"] = correct_pick

                object_interactions[object_name][
                    "wrong_pick"] = not correct_pick and (
                        task.env.held_object is not None and
                        task.env.held_object["name"] != object_name
                    )

            if task.env.held_object is not None:
                object_name = task.env.held_object["name"]
                object_type = current_poses[object_name]["type"]

            is_unshuffle_navigation = (
                task.greedy_expert._last_to_interact_object_pose is not None
            )

            if not goal_poses[object_name]["pickupable"]: continue

            navigation_success = (
                np.linalg.norm(observations["map"]["goal_position"][:2]) < 1.5
            )

            if is_unshuffle_navigation:
                object_interactions[object_name][
                    "correct_nav0"] = navigation_success

            else:  # moving to goal
                object_interactions[object_name][
                    "correct_nav1"] = navigation_success

        prop_fixed_strict = task.metrics()["unshuffle/prop_fixed_strict"]

        for object_name, data in object_interactions.items():
            print(f"Task={task_id:04d} %FixedStrict={prop_fixed_strict:.3f}", 
                  object_name, data)

        object_interactions_list.append(dict(object_interactions))
        
        with open("interactions.pkl", "wb") as f:
            pkl.dump(object_interactions_list, f)