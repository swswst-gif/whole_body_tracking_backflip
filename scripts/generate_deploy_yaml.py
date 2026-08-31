#!/usr/bin/env python3
"""从导出的 policy.onnx 自动生成 params/deploy.yaml。

前提：policy.onnx 必须带有 utils/exporter.py 中 attach_onnx_metadata()
写入的 metadata（play.py 导出时自动完成）。

用法：
    # 1) 先导出 ONNX（play.py 里已有 --export_only 分支）
    python scripts/rsl_rl/play.py --task Tracking-Flat-G1-Wo-State-Estimation-v0 \
        --load_run <run_name> --checkpoint model_xxx.pt --export_only

    # 2) 从 ONNX metadata 生成 deploy.yaml
    python scripts/generate_deploy_yaml.py \
        --onnx logs/rsl_rl/<run_name>/exported/policy.onnx

    # Low-Freq 变体（step_dt 应为 0.04）：
    python scripts/generate_deploy_yaml.py --onnx <policy.onnx> --step-dt 0.04

    # 默认姿态取 motion 第一帧（default_joint_pos 与 actions offset 同步替换）：
    python scripts/generate_deploy_yaml.py --onnx <policy.onnx> --motion artifacts/motion:v3/motion.npz

依赖：onnx、pyyaml、numpy（训练环境里通常都有）。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import onnx
import yaml

# ---------------------------------------------------------------------------
# G1 29-DoF 常量（与真机 SDK / DDS 低层协议一致）
# ---------------------------------------------------------------------------

# 真机电机顺序（Unitree SDK motor order），也就是 deploy.yaml 中
# stiffness / damping 采用的顺序。
SDK_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# 各关节力矩上限（与 robots/g1.py 的 effort_limit_sim 一致），
# 用于由 stiffness 反推 action scale = 0.25 * effort / stiffness。
EFFORT_LIMITS = [
    (r".*_hip_yaw_joint", 88.0),
    (r".*_hip_roll_joint", 139.0),
    (r".*_hip_pitch_joint", 88.0),
    (r".*_knee_joint", 139.0),
    (r".*_ankle_pitch_joint", 50.0),
    (r".*_ankle_roll_joint", 50.0),
    ("waist_yaw_joint", 88.0),
    ("waist_roll_joint", 50.0),
    ("waist_pitch_joint", 50.0),
    (r".*_shoulder_pitch_joint", 25.0),
    (r".*_shoulder_roll_joint", 25.0),
    (r".*_shoulder_yaw_joint", 25.0),
    (r".*_elbow_joint", 25.0),
    (r".*_wrist_roll_joint", 25.0),
    (r".*_wrist_pitch_joint", 5.0),
    (r".*_wrist_yaw_joint", 5.0),
]

# Isaac Lab 策略观测名 -> (deploy.yaml 观测名, 维度规则)。
# deploy 观测名必须与 sim2real 部署 C++ 注册的名字一致
# （include/isaaclab/envs/mdp/observations/observations.h、State_Mimic.cpp）。
OBS_TERMS = {
    "command": ("motion_command", lambda n_joints: 2 * n_joints),
    "motion_anchor_pos_b": ("motion_anchor_pos_b", lambda n_joints: 3),
    "motion_anchor_ori_b": ("motion_anchor_ori_b", lambda n_joints: 6),
    "base_lin_vel": ("base_lin_vel", lambda n_joints: 3),
    "base_ang_vel": ("base_ang_vel", lambda n_joints: 3),
    "joint_pos": ("joint_pos_rel", lambda n_joints: n_joints),
    "joint_vel": ("joint_vel_rel", lambda n_joints: n_joints),
    "actions": ("last_action", lambda n_joints: n_joints),
}


def parse_metadata(model: onnx.ModelProto) -> dict[str, str]:
    """把 ONNX metadata_props 读成 dict[str, str]。"""
    return {prop.key: prop.value for prop in model.metadata_props}


def parse_csv_list(value: str) -> list:
    """解析 exporter 写入的逗号分隔字符串，尽量转成 float，否则保留字符串。"""
    if not value:
        return []
    out = []
    for item in value.split(","):
        item = item.strip()
        try:
            out.append(float(item))
        except ValueError:
            out.append(item)
    return out


def effort_of(joint_name: str) -> float:
    for pattern, effort in EFFORT_LIMITS:
        if re.fullmatch(pattern, joint_name):
            return effort
    raise ValueError(f"找不到关节 '{joint_name}' 的力矩上限，请检查 EFFORT_LIMITS")


def build_deploy_yaml(meta: dict[str, str], step_dt: float) -> dict:
    """由 ONNX metadata 构造 deploy.yaml 对应的 Python 字典。"""

    required = [
        "joint_names",
        "joint_stiffness",
        "joint_damping",
        "default_joint_pos",
        "observation_names",
        "observation_history_lengths",
    ]
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(
            f"ONNX 缺少 metadata: {missing}。请确认这个 policy.onnx 是 play.py 通过 "
            "export_motion_policy_as_onnx + attach_onnx_metadata 导出的"
        )

    joint_names = [str(x) for x in parse_csv_list(meta["joint_names"])]
    n_joints = len(joint_names)
    if n_joints != len(SDK_JOINT_ORDER):
        raise ValueError(f"期望 {len(SDK_JOINT_ORDER)} 个关节，ONNX metadata 里是 {n_joints} 个")

    stiffness_policy = [float(x) for x in parse_csv_list(meta["joint_stiffness"])]
    damping_policy = [float(x) for x in parse_csv_list(meta["joint_damping"])]
    default_pos_policy = [float(x) for x in parse_csv_list(meta["default_joint_pos"])]
    if not (len(stiffness_policy) == len(damping_policy) == len(default_pos_policy) == n_joints):
        raise ValueError("joint_stiffness / joint_damping / default_joint_pos 长度与 joint_names 不一致")

    # action scale = 0.25 * effort / stiffness（与 robots/g1.py 的 G1_ACTION_SCALE 一致）
    scale_policy = [0.25 * effort_of(name) / k for name, k in zip(joint_names, stiffness_policy)]

    # policy 关节顺序 -> 真机 SDK 电机顺序
    try:
        joint_ids_map = [SDK_JOINT_ORDER.index(name) for name in joint_names]
    except ValueError as e:
        raise ValueError(f"ONNX metadata 里的关节名不在 SDK_JOINT_ORDER 中: {e}")

    # 按名字组装 SDK 顺序的 stiffness / damping
    name_to_stiffness = dict(zip(joint_names, stiffness_policy))
    name_to_damping = dict(zip(joint_names, damping_policy))
    stiffness_sdk = [name_to_stiffness[name] for name in SDK_JOINT_ORDER]
    damping_sdk = [name_to_damping[name] for name in SDK_JOINT_ORDER]

    # 观测：顺序必须与策略输入一致（metadata 里就是 policy 组顺序）
    obs_names = [str(x) for x in parse_csv_list(meta["observation_names"])]
    hist_raw = [float(x) for x in parse_csv_list(meta["observation_history_lengths"])]
    if len(hist_raw) == 1 and len(obs_names) > 1:
        hist_raw = hist_raw * len(obs_names)
    if len(hist_raw) != len(obs_names):
        raise ValueError("observation_names 与 observation_history_lengths 长度不一致")

    observations = {}
    total_obs_dim = 0
    for name, hist in zip(obs_names, hist_raw):
        if name not in OBS_TERMS:
            raise ValueError(
                f"策略观测 '{name}' 不在部署代码支持列表 {sorted(OBS_TERMS)} 中，"
                "请先确认 sim2real 部署 C++ 是否实现了该观测项"
            )
        deploy_name, dim_rule = OBS_TERMS[name]
        dim = dim_rule(n_joints)
        total_obs_dim += dim
        params = {"command_name": "motion"} if deploy_name.startswith("motion_") else {}
        observations[deploy_name] = {
            "params": params,
            "clip": None,
            "scale": [1.0] * dim,
            "history_length": int(round(hist)) if hist else 1,
        }

    deploy = {
        "joint_ids_map": joint_ids_map,
        "step_dt": step_dt,
        "stiffness": stiffness_sdk,       # SDK 电机顺序
        "damping": damping_sdk,           # SDK 电机顺序
        "default_joint_pos": default_pos_policy,  # policy 顺序
        "commands": {},
        "actions": {
            "JointPositionAction": {
                "clip": None,
                "joint_names": [".*"],
                "scale": scale_policy,          # policy 顺序
                "offset": default_pos_policy,   # policy 顺序
                "joint_ids": None,
            }
        },
        "observations": observations,
    }
    deploy["_total_obs_dim"] = total_obs_dim  # 仅用于打印，不会写入 yaml
    return deploy


# ---------------------------------------------------------------------------
# YAML 输出格式：数字用 3 位有效数字（和手写版风格一致），列表用流式方括号
# ---------------------------------------------------------------------------


class DeployDumper(yaml.SafeDumper):
    pass


def _float_repr(dumper: DeployDumper, value: float):
    text = f"{value:.1f}" if float(value).is_integer() else f"{value:.3g}"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


def _list_repr(dumper: DeployDumper, value: list):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


yaml.add_representer(float, _float_repr, Dumper=DeployDumper)
yaml.add_representer(list, _list_repr, Dumper=DeployDumper)


def apply_first_frame_default(deploy: dict, meta: dict[str, str], motion_file: Path) -> None:
    """把 default_joint_pos 和 actions offset 替换为 motion 第一帧的关节角（policy 关节顺序）。

    注意：训练时 joint_pos_rel 观测与动作 offset 参考的都是 sim 名义默认姿态（URDF init），
    替换后这两个参考点会整体平移（本 motion 最大 0.37 rad），策略存在分布偏移风险。
    """
    n_joints = len(deploy["joint_ids_map"])
    joint_pos = np.load(str(motion_file))["joint_pos"]
    if joint_pos.ndim != 2 or joint_pos.shape[1] != n_joints:
        raise ValueError(f"motion joint_pos 形状 {joint_pos.shape} 与 deploy 关节数 {n_joints} 不一致")

    first_frame = [float(v) for v in joint_pos[0]]
    old_default = [float(x) for x in parse_csv_list(meta["default_joint_pos"])]
    deploy["default_joint_pos"] = first_frame
    deploy["actions"]["JointPositionAction"]["offset"] = list(first_frame)

    max_diff = max(abs(a - b) for a, b in zip(first_frame, old_default))
    print(f"[INFO] default_joint_pos 已替换为 motion 第一帧，与 sim 默认姿态最大差异 {max_diff:.3f} rad")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx", type=Path, required=True,
        help="导出的 policy.onnx 路径（必须带 exporter 写入的 metadata）",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="输出 yaml 路径（默认：仓库根目录/params/deploy.yaml）",
    )
    parser.add_argument(
        "--step-dt", type=float, default=0.02,
        help="策略控制周期（秒），默认 0.02 = 50 Hz；Low-Freq 变体用 0.04",
    )
    parser.add_argument(
        "--motion", type=Path, default=None,
        help="训练用的 motion.npz 路径；提供时 default_joint_pos/actions offset 取 motion 第一帧",
    )
    args = parser.parse_args()

    if not args.onnx.is_file():
        parser.error(f"ONNX 文件不存在: {args.onnx}")

    output = args.output or (Path(__file__).resolve().parents[1] / "params" / "deploy.yaml")

    model = onnx.load(str(args.onnx))
    meta = parse_metadata(model)
    deploy = build_deploy_yaml(meta, args.step_dt)
    total_obs_dim = deploy.pop("_total_obs_dim")
    if args.motion is not None:
        if not args.motion.is_file():
            parser.error(f"motion 文件不存在: {args.motion}")
        apply_first_frame_default(deploy, meta, args.motion)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(deploy, f, Dumper=DeployDumper, sort_keys=False, default_flow_style=False, width=80)

    print(f"[INFO] 已生成: {output}")
    print(f"[INFO] 关节数: {len(deploy['joint_ids_map'])}, 观测总维度: {total_obs_dim}, step_dt: {args.step_dt}")
    print("[INFO] 检查观测顺序是否与策略输入一致：", ", ".join(deploy["observations"].keys()))


if __name__ == "__main__":
    main()
