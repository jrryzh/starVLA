import copy
import glob
import logging
import os
import site
import sys
from collections.abc import Sequence

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"project_root: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 或 ":16:8"，对 cuBLAS 确定性必需
import json
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Union

# from deployment_agibot.a2d_description import A2dJoint2EefIK

# pip install --user json_numpy uvicorn fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
# Add user site packages to Python path
site.addsitedir(site.getusersitepackages())
import json_numpy
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # 禁掉动态算法搜索
torch.use_deterministic_algorithms(True)  # ≥1.8；早期版本用 torch.set_deterministic(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from transformers.utils import logging

from deployment_agibot.interface import M1Inference

torch.set_printoptions(sci_mode=False, precision=5)
logger = logging.get_logger("ModelServer")
np.set_printoptions(linewidth=400, precision=5, suppress=True)
torch.set_printoptions(linewidth=400, precision=5, sci_mode=False)
json_numpy.patch()


def denorm(action, norm_dict, space):
    action_denorm = action * np.array(norm_dict[space]["std"]) + np.array(norm_dict[space]["mean"])
    return action_denorm


def steady_gripper(gripper_values, threshold=0.03, steady_value=0.01):
    gripper_values[gripper_values < threshold] = steady_value
    return gripper_values


def get_action_setting(data_args):
    ## get action chunk encode method

    if hasattr(data_args, "dataset_processors") and data_args.dataset_processors is not None:
        action_chunk_encode_method = (
            "abs"
            if not data_args.dataset_processors[0]["action_use_delta"]
            else data_args.dataset_processors[0]["delta_type"]
        )
    else:
        # 处理LeRobot配置文件没有dataset_processors的情况
        action_chunk_encode_method = "abs"

    assert action_chunk_encode_method in ["abs", "chunk", "frame"]

    # Get used camera
    used_camera_list = None
    if hasattr(data_args, "runtime_processors") and data_args.runtime_processors is not None:
        for item in data_args.runtime_processors:
            if "RuntimeImagePreprocessLoad" in item["type"] or "RuntimeImagePreprocessLoadAgilex" in item["type"]:
                camera_list = item.get("camera_list", None)
                if camera_list is not None:
                    used_camera_list = []
                    camera_list = list(camera_list.values())
                    if "head_color" in camera_list:
                        used_camera_list.append("cam_tensor_head_top")
                    if "hand_right_color" in camera_list:
                        used_camera_list.append("cam_tensor_wrist_right")
                    if "hand_left_color" in camera_list:
                        used_camera_list.append("cam_tensor_wrist_left")

    ## get norm statistics from config
    norm_action_state = False
    state_norm_dict = None
    action_norm_dict = None
    a_min = None
    a_max = None
    if hasattr(data_args, "runtime_processors") and data_args.runtime_processors is not None:
        for item in data_args.runtime_processors:
            if item["type"] == "RuntimeStateActionNormAgilex":
                norm_action_state = True
                state_norm_dict = item["state_norm_dict"]
                action_norm_dict = item["action_norm_dict"]
                a_min = -3
                a_max = 3

    return (
        action_chunk_encode_method,
        norm_action_state,
        state_norm_dict,
        action_norm_dict,
        a_min,
        a_max,
        used_camera_list,
    )


class ModelInfer:
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        model_action_mask: Union[list, np.array] = None,
        use_official_fast_tokenizer: bool = True,
        use_joint_fast_tokenizer: bool = False,
        is_libero: bool = True,
    ) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.model_name_or_path = model_name_or_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.bfloat16

        # Load Model using HF AutoClasses
        self.vla_model = M1Inference(
            policy_ckpt_path=model_name_or_path,
            image_size=[224, 224],
        )
        # convert joints to eef
        # self.conventor = A2dJoint2EefIK()
        # self.conventor.reset()

    def model_infer(
        self, input_ids, attention_mask, position_ids, pixel_values, image_grid_thw, state, ctrl_freqs, action_gts=None
    ):
        # Run model Inference
        model_start_time = time.time()
        with torch.no_grad():
            vla_outputs = self.vla_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                state=state,
                ctrl_freqs=ctrl_freqs,
                action_gts=action_gts,
            )

        model_end_time = time.time()  # record the end time
        elapsed_time = (model_end_time - model_start_time) * 1000  # calculate the runtime and convert to ms unit
        act_outputs = vla_outputs[1][0].detach().float().cpu().numpy()  # action outputs
        print(f"vla_model executed in {elapsed_time:.3f} ms, get action output shape:{act_outputs.shape}")

        return act_outputs

    @staticmethod
    def joint2eef(conventor, left_joint, right_joint, waist_pitch, waist_lift, head_joint):
        all_left_eef = []
        all_right_eef = []
        for f_idx in range(left_joint.shape[0]):
            left_eef, right_eef = conventor.get_eef_pos(
                waist_pitch=waist_pitch,
                waist_lift=waist_lift,
                left_joints=left_joint[f_idx],
                right_joints=right_joint[f_idx],
                head_joints=head_joint,
            )
            all_left_eef.append(left_eef)
            all_right_eef.append(right_eef)
        return np.concatenate([all_left_eef]), np.concatenate([all_right_eef])

    # TODO: implement later
    # return eef + gripper actions
    def predict_action(self, payload: Dict[str, Any]) -> str:
        return self.vla_model.step(payload)

# === Server Interface ===
class ModelServer:
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        post_process_gripper: bool = False,
        use_recorder: bool = False,
        model_action_mask=None,
        output_action_horizon=30,
        use_norm=False,
        dataset_stats_path=None,
        use_official_fast_tokenizer=True,
        is_libero=True,
        use_joint_fast_tokenizer=False,
    ) -> Path:
        """
        A simple server for Alpha1 models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        if use_norm or dataset_stats_path is not None:
            # load lerobot stats json
            dataset_stats_path = (
                os.path.join(os.path.dirname(model_name_or_path), "dataset_stats.json")
                if dataset_stats_path is None
                else dataset_stats_path
            )
            with open(dataset_stats_path, "r") as f:
                self.dataset_stats = json.load(f)
        else:
            self.dataset_stats = None

        self.model = ModelInfer(
            model_name_or_path=model_name_or_path,
        )

        self.post_process_gripper = post_process_gripper

        # Recorder for debugging purposes
        self.model_action_mask = model_action_mask
        self.use_recorder = use_recorder
        self.output_action_horizon = output_action_horizon
        self.recorder_idx = 0
        model_id = "-".join(model_name_or_path.split("/")[-2:])  # Get the last part of the path as model name
        self.recorder_saving_dir = os.path.join(
            os.getcwd(), "debug_records", f"{model_id}_{time.strftime('%Y%m%dT%H%M%S', time.localtime())}"
        )
        os.makedirs(self.recorder_saving_dir, exist_ok=True)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        if self.use_recorder:
            recorder_payload = copy.deepcopy(payload)

        # payload["image_resize"] = (224, 224)
        # payload["image_resize"] = (224, 224)
        # payload["chat_sources"] = [
        #     [
        #         {
        #             "from": "human",
        #             "value": f"<image>\n<image>\n<image>\nWhat action should the robot take to {payload['sub_task_instruction']}?",
        #         },
        #         {"from": "gpt", "value": ""},
        #     ]
        # ]
        response = self.model.predict_action(payload)
        response = response['raw_action']
        # joint MC 部署
        # _, response = self.model.predict_action(payload)
        # response = np.concatenate([response[:, :7], response[:, 8:15], response[:, 7:8], response[:, 15:16]], axis=-1)

        if self.use_recorder:
            self.recorder_idx += 1
            save_path = os.path.join(self.recorder_saving_dir, f"{self.recorder_idx:05d}.pkl")
            with open(save_path, "wb") as fid:
                pickle.dump({"payload": recorder_payload, "response": response}, fid)
                print(f"Saved debug record to {save_path}")

        return response

    def post_process(self, response):
        assert len(response.shape) == 2, f"Expect response shape to be 2D, got {response.shape}"
        if self.output_action_horizon < 30:
            response[self.output_action_horizon :] = response[self.output_action_horizon]
            print(f"Actually execute action horizon: {response.shape}")

        if self.post_process_gripper:
            if response.shape[-1] == 16:
                response[:, 7:8] = steady_gripper(response[:, 7:8])
                response[:, 15:16] = steady_gripper(response[:, 15:16])
            else:
                raise ValueError(f"Only support joint control for now")
        else:
            pass
        return response

    def model_infer(self, **kwargs) -> str:
        response = self.model.model_infer(**kwargs)
        return response

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        import uvicorn
        from fastapi import FastAPI, Request

        self.app = FastAPI()

        @self.app.post("/act")
        async def act_endpoint(request: Request):
            # payload = await request.json()
            payload_bytes = await request.body()

            # print(f"payload_bytes: {payload_bytes}")

            payload = pickle.loads(payload_bytes)

            print(f"payload: {payload}")

            outputs = self.predict_action(payload)

            print(f"outputs: {outputs}")

            return outputs.tolist() if hasattr(outputs, "tolist") else outputs

        uvicorn.run(self.app, host=host, port=port)


def deploy(
    model_name_or_path=None,
    host=None,
    port=None,
    debug=False,
    post_process_gripper=False,
    use_recorder=False,
    model_action_mask=None,
    output_action_horizon=30,
    use_official_fast_tokenizer=False,
    is_libero=False,
    use_joint_fast_tokenizer=False,
) -> None:
    server = ModelServer(
        model_name_or_path=model_name_or_path,
        post_process_gripper=post_process_gripper,
        use_recorder=use_recorder,
        model_action_mask=model_action_mask,
        output_action_horizon=output_action_horizon,
        use_official_fast_tokenizer=use_official_fast_tokenizer,
        is_libero=is_libero,
        use_joint_fast_tokenizer=use_joint_fast_tokenizer,
    )
    if not debug:
        server.run(host, port=port)
    else:
        # debug_pickle = "/home/zhangjinyu/code_repo/starVLA/deployment_agibot/gm_00000.pkl"
        debug_pickle = "/home/zhangjinyu/code_repo/starVLA/debug_records/checkpoint-model.pt_20251127T100349/00003.pkl"
        payload= pickle.load(open(debug_pickle, "rb"))['payload']
        actions = server.predict_action(payload)
        print(f"actions:\n {actions}")

        # print(raw_target["action_target"])
        # raw_target["action_target"]["right_end_effector_6d_pose"]
        # raw_target["action_target"]["left_end_effector_6d_pose"]


if __name__ == "__main__":
    # Server Configuration
    model_name_or_path = (
        # "/home/zhangjinyu/code_repo/starVLA/results/1121_agibot_random_pick_place_all/final_model/pytorch_model.pt"
        "/home/zhangjinyu/code_repo/starVLA/results/1124_agibot_random_pick_place_ckpt_50ksteps/checkpoint/model.pt"
    )
    use_joint_fast_tokenizer=False
    post_process_gripper = False  # Whether to steady the gripper action
    use_recorder = True  # Whether to save debug records
    model_action_mask = [1] * 8 + [1] * 8  # Whether to save debug records
    output_action_horizon = 30  # Whether to save debug records

    host: str = "0.0.0.0"  # Host IP Address
    port: int = 38887  # Host Port
    debug: bool = True


    deploy(
        model_name_or_path,
        host,
        port,
        debug=debug,
        post_process_gripper=post_process_gripper,
        use_recorder=use_recorder,
        model_action_mask=model_action_mask,
        output_action_horizon=output_action_horizon,
        use_joint_fast_tokenizer=use_joint_fast_tokenizer,
    )
