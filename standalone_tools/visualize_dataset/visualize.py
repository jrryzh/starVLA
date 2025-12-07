import numpy as np
import matplotlib.pyplot as plt

from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
import numpy as np

import matplotlib.pyplot as plt

cfg = OmegaConf.load("./results/1204_agibot_random_pick_place_ckpt_25ksteps/config.yaml")
vla_dataset_cfg = cfg.datasets.vla_data
vla_dataset_cfg.data_root_dir = "/home/zhangjinyu/code_repo/starVLA/data"

# dataset
vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg, use_instruction_segments=True)

num_samples = 50
max_steps = 150

left_arm_state_joints_across_time = []
right_arm_state_joints_across_time = []
left_arm_gt_action_joints_across_time = []
right_arm_gt_action_joints_across_time = []

# Sample 50 trajectories

# Get total number of trajectories in the dataset
total_trajectories = len(vla_dataset.datasets[0])
print(f"Total trajectories in dataset: {total_trajectories}")

# Sample 50 trajectories (or all if less than 50)
import random
sampled_traj_ids = sorted(random.sample(range(total_trajectories), min(num_samples, total_trajectories)))
print(f"Sampled trajectory IDs: {sampled_traj_ids}")

# Load episode metadata to show task information
import json
episodes_meta = []
try:
    with open("/home/zhangjinyu/code_repo/starVLA/data/task_3124/meta/episodes.jsonl", "r") as f:
        for line in f:
            episodes_meta.append(json.loads(line.strip()))
    print("\nSampled trajectories details:")
    for traj_id in sampled_traj_ids:
        if traj_id < len(episodes_meta):
            episode = episodes_meta[traj_id]
            print(f"  Traj {traj_id}: {episode['tasks'][0]}, length={episode['length']}")
except Exception as e:
    print(f"Could not load episode metadata: {e}")

# Save sampled trajectory IDs to a file for reference
with open("./results/1204_agibot_random_pick_place_ckpt_25ksteps/visualize_dataset/sampled_trajectories.txt", "w") as f:
    f.write(f"Sampled {len(sampled_traj_ids)} trajectories from {total_trajectories} total trajectories\n")
    f.write(f"Sampled IDs: {sampled_traj_ids}\n\n")
    if episodes_meta:
        f.write("Trajectory details:\n")
        for traj_id in sampled_traj_ids:
            if traj_id < len(episodes_meta):
                episode = episodes_meta[traj_id]
                f.write(f"  Traj {traj_id}: {episode['tasks'][0]}, length={episode['length']}\n")
print(f"\nSaved trajectory sampling information to: ./results/1204_agibot_random_pick_place_ckpt_25ksteps/visualize_dataset/sampled_trajectories.txt")

for traj_idx, traj_id in enumerate(sampled_traj_ids):
    print(f"Processing trajectory {traj_idx + 1}/{len(sampled_traj_ids)}: traj_id = {traj_id}")

    left_arm_state_joints_across_time = []
    right_arm_state_joints_across_time = []
    left_arm_gt_action_joints_across_time = []
    right_arm_gt_action_joints_across_time = []

    for step_count in range(max_steps):
        try:
            data_point = vla_dataset.datasets[0].get_step_data(traj_id, step_count)
            left_arm_state_joints = data_point["state.left_arm"][0]
            right_arm_state_joints = data_point["state.right_arm"][0]
            left_arm_gt_action_joints = data_point["action.left_arm"][0]
            right_arm_gt_action_joints = data_point["action.right_arm"][0]

            left_arm_state_joints_across_time.append(left_arm_state_joints)
            right_arm_state_joints_across_time.append(right_arm_state_joints)
            left_arm_gt_action_joints_across_time.append(left_arm_gt_action_joints)
            right_arm_gt_action_joints_across_time.append(right_arm_gt_action_joints)
        except:
            # If trajectory is shorter than max_steps, break
            break

    if len(left_arm_state_joints_across_time) == 0:
        continue

    # Size is (max_steps, num_joints == 7)
    left_arm_state_joints_across_time = np.array(left_arm_state_joints_across_time)
    right_arm_state_joints_across_time = np.array(right_arm_state_joints_across_time)
    left_arm_gt_action_joints_across_time = np.array(left_arm_gt_action_joints_across_time)
    right_arm_gt_action_joints_across_time = np.array(right_arm_gt_action_joints_across_time)

    # Plot the joint angles across time
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 2*7))

    for i, ax in enumerate(axes):
        ax.plot(left_arm_state_joints_across_time[:, i], label="left arm state joints")
        ax.plot(right_arm_state_joints_across_time[:, i], label="right arm state joints")
        ax.plot(left_arm_gt_action_joints_across_time[:, i], label="left arm gt action joints")
        ax.plot(right_arm_gt_action_joints_across_time[:, i], label="right arm gt action joints")
        ax.set_title(f"Joint {i}")
        ax.legend()

    plt.tight_layout()
    # plt.show()
    os.makedirs(f"./results/1204_agibot_random_pick_place_ckpt_25ksteps/visualize_dataset", exist_ok=True)
    plt.savefig(f"./results/1204_agibot_random_pick_place_ckpt_25ksteps/visualize_dataset/joint_angles_traj_{traj_id}.png")
    plt.close()  # Close the figure to save memory

    print(f"Saved plot for trajectory {traj_id}")