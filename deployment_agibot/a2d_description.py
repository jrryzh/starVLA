
from pathlib import Path

import numpy as np
import pip
from scipy.spatial.transform import Rotation as R

try:
    import ik_solver
except ImportError:
    pip.main(["install", "/mnt/public/xiongziyu/dependency/ik_solver-0.3.1-cp312-cp312-linux_x86_64.whl"])
    import ik_solver  # 再次尝试导入


class A2dJoint2EefIK:
    def __init__(self, use_relaxed_ik=False, use_elbow=False):
        urdf_path = str(Path(__file__).parent / "urdf/A2D_ik.urdf")
        config_path = str(Path(__file__).parent / "urdf/solver.yaml")
        self.solver = ik_solver.Solver(
            urdf_path=urdf_path,
            config_path=config_path,
            use_relaxed_ik=use_relaxed_ik,
            use_elbow=use_elbow,
        )
        self.initialize = False

    def reset(self):
        self.initialize = False

    def get_eef_pos(self, waist_pitch, waist_lift, left_joints, right_joints, head_joints):
        if not self.initialize:
            self.solver.initialize_states(
                left_arm_init=np.array(left_joints, dtype=np.float32),
                right_arm_init=np.array(right_joints, dtype=np.float32),
                head_init=np.array(head_joints, dtype=np.float32),
            )
            self.initialize = True

        # T_arm_left_ee = self.solver.compute_part_fk(
        #    q_part=np.array(left_joints, dtype=np.float32),
        #    part=ik_solver.RobotPart.LEFT_ARM,
        #    from_base=False,
        # )
        # T_arm_right_ee = self.solver.compute_part_fk(
        #    q_part=np.array(right_joints, dtype=np.float32),
        #    part=ik_solver.RobotPart.RIGHT_ARM,
        #    from_base=False,
        # )

        Q_full = np.concatenate(
            [
                np.array([waist_lift, waist_pitch]),
                head_joints,
                left_joints,
                right_joints,
            ]
        )
        T_arm_left_ee = self.solver.compute_fk(q=Q_full, start_link="arm_base_link", end_link="arm_left_link7")
        T_arm_right_ee = self.solver.compute_fk(q=Q_full, start_link="arm_base_link", end_link="arm_right_link7")

        left_position = T_arm_left_ee[:3, 3]
        left_rotation = T_arm_left_ee[:3, :3]

        right_position = T_arm_right_ee[:3, 3]
        right_rotation = T_arm_right_ee[:3, :3]

        left_euler = R.from_matrix(left_rotation).as_euler("xyz", degrees=False)
        right_euler = R.from_matrix(right_rotation).as_euler("xyz", degrees=False)

        left_pose = np.array(
            [
                left_position[0],
                left_position[1],
                left_position[2],
                left_euler[0],
                left_euler[1],
                left_euler[2],
            ]
        )
        right_pose = np.array(
            [
                right_position[0],
                right_position[1],
                right_position[2],
                right_euler[0],
                right_euler[1],
                right_euler[2],
            ]
        )
        return left_pose, right_pose
