from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment_agibot.model_server import ModelServer
from deployment_agibot.interface import M1Inference
from omegaconf import OmegaConf
import numpy as np

import matplotlib.pyplot as plt


if __name__ == "__main__":

    cfg = OmegaConf.load("./results/1204_agibot_random_pick_place_ckpt_25ksteps/config.yaml")
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.data_root_dir = "/home/zhangjinyu/code_repo/starVLA/data"

    # dataset
    vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # model server
    # Server Configuration
    model_name_or_path = (
        # "/home/zhangjinyu/code_repo/starVLA/results/1121_agibot_random_pick_place_all/final_model/pytorch_model.pt"
        # "/home/zhangjinyu/code_repo/starVLA/results/1124_agibot_random_pick_place_ckpt_50ksteps/checkpoint/model.pt"
        "/home/zhangjinyu/code_repo/starVLA/results/1204_agibot_random_pick_place_ckpt_25ksteps/checkpoint/model_bf16.pt"
    )
    use_joint_fast_tokenizer=False
    post_process_gripper = False  # Whether to steady the gripper action
    use_recorder = True  # Whether to save debug records
    model_action_mask = [1] * 8 + [1] * 8  # Whether to save debug records
    output_action_horizon = 30  # Whether to save debug records

    host: str = "0.0.0.0"  # Host IP Address
    port: int = 38887  # Host Port

    server = ModelServer(
        model_name_or_path=model_name_or_path,
        post_process_gripper=post_process_gripper,
        use_recorder=use_recorder,
        model_action_mask=model_action_mask,
        output_action_horizon=output_action_horizon,
        use_official_fast_tokenizer=False,
        is_libero=False,
        use_joint_fast_tokenizer=use_joint_fast_tokenizer,
    )

    DRAW_MEAN_STD = False

    # Storage for multiple samples
    all_gt = []
    all_pred = []
    max_samples = 10  # 可调，采多少样本（避免整套数据太大）

    for idx, sample in enumerate(vla_dataset):
        print(sample)

        normalized_gt_action = sample["action"] # (8, 16)
        action_norm_stats = M1Inference.get_action_stats(None, policy_ckpt_path=model_name_or_path)
        gt_action = M1Inference.unnormalize_actions(normalized_actions=normalized_gt_action, action_norm_stats=action_norm_stats)

        # format into payload
        payload = {
            "observation.images.head": np.array(sample["image"][0]),
            "observation.images.hand_left": np.array(sample["image"][1]),
            "observation.images.hand_right": np.array(sample["image"][2]),
            "observation.states.arm_joint_states": np.zeros((14)), # placeholder
            "observation.states.gripper_states": np.zeros((2)), # placeholder
            "sub_task_instruction": sample["lang"],
        }

        pred_action = server.predict_action(payload) # (8, 16)
        print(f"pred_action:\n {pred_action}")
        print(f"gt_action:\n {gt_action}")

        # ---- Save for later plotting ----
        all_gt.append(gt_action)
        all_pred.append(pred_action)

        if idx + 1 >= max_samples:
            break

        # horizon = pred_action.shape[0]

        # # 7 left, 7 right
        # joint_names = [f"L{i}" for i in range(7)] + [f"R{i}" for i in range(7)]
        # gripper_names = ["Gripper_L", "Gripper_R"]

        # # split joints and grippers
        # gt_joints = gt_action[:, :14]          # (8, 14)
        # gt_grippers = gt_action[:, 14:]        # (8, 2)

        # pred_joints = pred_action[:, :14]      # (8, 14)
        # pred_grippers = pred_action[:, 14:]    # (8, 2)

        # # -------------------------------
        # # Plot joints
        # # -------------------------------
        # plt.figure(figsize=(16, 10))
        # for j in range(14):
        #     plt.subplot(4, 4, j+1)
        #     plt.plot(range(horizon), gt_joints[:, j], label="GT", linewidth=2)
        #     plt.plot(range(horizon), pred_joints[:, j], label="Pred", linestyle="--")
        #     plt.title(joint_names[j])
        #     plt.tight_layout()

        # plt.legend()
        # plt.suptitle("Open-loop Joint Prediction (GT vs Pred)", fontsize=18)
        # plt.subplots_adjust(top=0.92)
        # plt.savefig(f"./results/1204_agibot_random_pick_place_ckpt_25ksteps/openloop_joint_prediction.png")

        # # -------------------------------
        # # Plot grippers
        # # -------------------------------
        # plt.figure(figsize=(8, 4))
        # for g in range(2):
        #     plt.subplot(1, 2, g+1)
        #     plt.plot(range(horizon), gt_grippers[:, g], label="GT", linewidth=2)
        #     plt.plot(range(horizon), pred_grippers[:, g], label="Pred", linestyle="--")
        #     plt.title(gripper_names[g])
        #     plt.tight_layout()

        # plt.legend()
        # plt.suptitle("Gripper Prediction (GT vs Pred)", fontsize=16)
        # plt.subplots_adjust(top=0.8)
        # plt.show()

        # break

    if DRAW_MEAN_STD:

        # Convert to numpy arrays: shape (N, 8, 16)
        all_gt = np.array(all_gt)
        all_pred = np.array(all_pred)

        horizon = all_gt.shape[1]

        joint_names = [f"L{i}" for i in range(7)] + [f"R{i}" for i in range(7)]
        gripper_names = ["Gripper_L", "Gripper_R"]

        # Split joint & gripper dims
        gt_joints = all_gt[:, :, :14]       # (N, 8, 14)
        gt_grippers = all_gt[:, :, 14:]     # (N, 8, 2)

        pred_joints = all_pred[:, :, :14]   # (N, 8, 14)
        pred_grippers = all_pred[:, :, 14:] # (N, 8, 2)

        # ========================================
        # Plot mean ± std for joints
        # ========================================
        plt.figure(figsize=(16, 12))

        for j in range(14):
            plt.subplot(4, 4, j + 1)

            gt_mean = gt_joints[:, :, j].mean(axis=0)
            gt_std  = gt_joints[:, :, j].std(axis=0)

            pred_mean = pred_joints[:, :, j].mean(axis=0)
            pred_std  = pred_joints[:, :, j].std(axis=0)

            # GT
            plt.plot(gt_mean, label="GT mean", linewidth=2, color="blue")
            plt.fill_between(range(horizon), gt_mean-gt_std, gt_mean+gt_std, alpha=0.2, color="blue")

            # Pred
            plt.plot(pred_mean, label="Pred mean", linewidth=2, linestyle="--", color="orange")
            plt.fill_between(range(horizon), pred_mean-pred_std, pred_mean+pred_std, alpha=0.2, color="orange")

            plt.title(joint_names[j])

        plt.legend()
        plt.suptitle("Multi-Sample Open-loop Prediction vs GT (Joints)", fontsize=18)
        plt.subplots_adjust(top=0.92)
        # plt.show()
        plt.savefig(f"./results/1204_agibot_random_pick_place_ckpt_25ksteps/openloop_joint_prediction_std_mean.png")
    else:
        # to np array: (N, horizon, 16)
        all_gt = np.array(all_gt)
        all_pred = np.array(all_pred)

        horizon = all_gt.shape[1]
        num_samples = all_gt.shape[0]

        joint_names = [f"L{i}" for i in range(7)] + [f"R{i}" for i in range(7)]
        gripper_names = ["Gripper_L", "Gripper_R"]

        # split
        gt_joints = all_gt[:, :, :14]        # (N, 8, 14)
        gt_grippers = all_gt[:, :, 14:]      # (N, 8, 2)

        pred_joints = all_pred[:, :, :14]    # (N, 8, 14)
        pred_grippers = all_pred[:, :, 14:]  # (N, 8, 2)

        # ----------------------------------
        # Plot each sample directly (no mean)
        # ----------------------------------
        plt.figure(figsize=(16, 12))

        for j in range(14):
            plt.subplot(4, 4, j + 1)

            # plot all samples
            for s in range(num_samples):
                # GT = solid
                plt.plot(
                    gt_joints[s, :, j],
                    color="blue",
                    alpha=0.3,
                    linewidth=1
                )

                # Pred = dashed
                plt.plot(
                    pred_joints[s, :, j],
                    color="orange",
                    alpha=0.3,
                    linestyle="--",
                    linewidth=1
                )

            plt.title(joint_names[j])
            plt.tight_layout()

        # create a manual legend (only once)
        import matplotlib.lines as mlines
        gt_line = mlines.Line2D([], [], color='blue', label='GT')
        pred_line = mlines.Line2D([], [], color='orange', linestyle='--', label='Pred')
        plt.legend(handles=[gt_line, pred_line], bbox_to_anchor=(1.0, 0.5))

        plt.suptitle("Open-loop Prediction vs GT (Multiple Samples, No Averaging)", fontsize=18)
        plt.subplots_adjust(top=0.92, right=0.88)
        plt.savefig(f"./results/1204_agibot_random_pick_place_ckpt_25ksteps/openloop_joint_prediction_all.png")

