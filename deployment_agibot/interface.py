from collections import deque
from typing import Optional, Sequence
from typing import Dict, Any
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from examples.SimplerEnv.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import numpy as np
from pathlib import Path

from starVLA.model.tools import read_mode_config
from starVLA.model.framework.base_framework import baseframework


class M1Inference:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = True,
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
    ) -> None:
        
        # we don't need client here, just baseframework
        self.vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
            policy_ckpt_path,
        ).to("cuda").eval()
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None


    def step(
        self, 
        payload: Dict[str, Any],
        step: int = 0,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param payload: Dict[str, Any]
        :return: (raw action, processed action)
        """
        # payload keys
        # dict_keys(['observation.timestamp', 'observation.images.head', 'observation.images.hand_left', 'observation.images.hand_right', 'observation.states.head_joint_states', 'observation.states.arm_joint_states', 'observation.states.waist_joint_states', 'observation.states.gripper_states', 'task_instruction', 'sub_task_instruction'])
        images = [payload["observation.images.head"], payload["observation.images.hand_left"], payload["observation.images.hand_right"]]
        task_description = payload["sub_task_instruction"]
        # Convert state to numpy array with shape [1, state_dim] to match training format
        arm_states = np.array(payload["observation.states.arm_joint_states"])
        gripper_states = np.array(payload["observation.states.gripper_states"])
        state = np.concatenate([arm_states, gripper_states]).reshape(1, 1, -1).astype(np.float32)

        state = self.normalize_state(state, self.state_norm_stats)

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        vla_input = {
            "batch_images": [images],
            "instructions": [self.task_description],
            "state": None, # state, # HACK: state not used for now
            # Note(jinyu): other keys are reserved
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        
        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            response = self.vla.predict_action(**vla_input)
            # unnormalize the action
            normalized_actions = response["normalized_actions"] # B, chunk, D        
            normalized_actions = normalized_actions[0]    
            self.raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        
        # if I do the trick
        # raw_actions = self.raw_actions[step % action_chunk_size][None]    
        raw_actions = self.raw_actions[None]

        # raw_action = {
        #     "world_vector": np.array(raw_actions[0, :3]),
        #     "rotation_delta": np.array(raw_actions[0, 3:6]),
        #     # TODO: check later, two grippers here
        #     "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        # }
        return {"raw_action": raw_actions[0]}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def normalize_state(state: np.ndarray, state_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Normalize the state
        """
        return(state - np.array(state_norm_stats["mean"])) / (np.array(state_norm_stats["std"]) + 1e-6)

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = M1Inference._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]
    
    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = M1Inference._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config['framework']['action_model']['future_action_window_size'] + 1

    # TODO: not used for now
    def _resize_image(self, image: np.ndarray) -> Image.Image:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        # Convert numpy array to PIL Image
        return Image.fromarray(image)

    # TODO: not used for now
    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key