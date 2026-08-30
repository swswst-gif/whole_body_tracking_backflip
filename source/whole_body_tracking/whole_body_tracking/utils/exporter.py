# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        onnx_path = os.path.join(path, filename)
        torch.onnx.export(
            self,
            obs,
            onnx_path,
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def get_default_joint_pos(robot_data) -> list[float]:
    """Return the nominal default joint positions across IsaacLab versions.

    Older or alternate task setups may attach a ``default_joint_pos_nominal`` override to the articulation data,
    while standard IsaacLab exposes the nominal values via ``default_joint_pos[0]``. Use whichever is available and
    gracefully fall back to the built-in default if neither is explicitly set.
    """
    if hasattr(robot_data, "default_joint_pos_nominal"):
        pos = robot_data.default_joint_pos_nominal
    elif hasattr(robot_data, "default_joint_pos"):
        pos = robot_data.default_joint_pos[0]
    else:
        raise AttributeError("ArticulationData does not expose default joint positions")

    if hasattr(pos, "cpu"):
        pos = pos.cpu()
    return pos.reshape(-1).tolist()


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    robot_data = env.scene["robot"].data
    metadata = {
        "run_path": run_path,
        "joint_names": robot_data.joint_names,
        "joint_stiffness": robot_data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": robot_data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": get_default_joint_pos(robot_data),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

        # self-contained single file (weights embedded); do NOT use save_as_external_data,
        # otherwise the .onnx alone is unusable without its companion .data file
        onnx.save(model, onnx_path)
