"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz

Camera controls:
    Press 1-4 to switch to a preset view (follow, side, front, top).
    Press 5 for the free camera, then use WASD / QE to move it (it always looks at the robot).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
import omni

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply_yaw

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class CameraMode:
    """Camera view presets (plain namespace for key binding)."""

    FOLLOW = "follow"
    SIDE = "side"
    FRONT = "front"
    TOP = "top"
    FREE = "free"


class CameraController:
    """Keyboard-controlled camera for replaying motions.

    Preset views (1-4) track the robot base and rotate with its heading; the free camera
    (5) is moved with WASD/QE and always looks at the robot.

    Key bindings (matching the carb string names used by ``event.input.name``):
        1-4: follow / side / front / top view
        5:   free camera
        WASD/QE: move the free camera (forward/back/left/right/down/up)
    """

    # preset views as (offset, look-at offset) relative to the robot base.
    # forward is the robot's yaw direction; up is +Z.
    PRESETS = {
        CameraMode.FOLLOW: (np.asarray([-3.0, 0.0, 1.5]), np.asarray([0.0, 0.0, 0.6])),
        CameraMode.SIDE: (np.asarray([0.0, 3.0, 1.2]), np.asarray([0.0, 0.0, 0.6])),
        CameraMode.FRONT: (np.asarray([3.0, 0.0, 1.2]), np.asarray([0.0, 0.0, 0.6])),
        CameraMode.TOP: (np.asarray([0.0, 0.0, 6.0]), np.asarray([0.0, 0.0, 0.0])),
    }
    # number keys that select the presets (FOLLOW, SIDE, FRONT, TOP) and the free camera
    _KEY_MODE_BINDING = {
        "KEY_1": CameraMode.FOLLOW,
        "KEY_2": CameraMode.SIDE,
        "KEY_3": CameraMode.FRONT,
        "KEY_4": CameraMode.TOP,
        "KEY_5": CameraMode.FREE,
    }
    # movement keys for the free camera (camera-local frame, applied each frame)
    _KEY_MOVE_BINDING = {
        "W": np.asarray([0.6, 0.0, 0.0]),
        "S": np.asarray([-0.6, 0.0, 0.0]),
        "A": np.asarray([0.0, -0.6, 0.0]),
        "D": np.asarray([0.0, 0.6, 0.0]),
        "Q": np.asarray([0.0, 0.0, -0.6]),
        "E": np.asarray([0.0, 0.0, 0.6]),
    }

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args: self._on_keyboard_event(event, *args),
        )
        self._mode = CameraMode.FOLLOW
        self._move = np.zeros(3)
        # free-camera position; lazily initialized on first free-frame
        self._eye = None

    def __del__(self):
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        msg = "Camera Controller:\n"
        msg += "\t1-4: follow / side / front / top view\n"
        msg += "\t5: free camera (WASD/QE to move, always looks at the robot)\n"
        return msg

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._KEY_MODE_BINDING:
                self._mode = self._KEY_MODE_BINDING[event.input.name]
            elif event.input.name in self._KEY_MOVE_BINDING:
                self._move += self._KEY_MOVE_BINDING[event.input.name]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._KEY_MOVE_BINDING:
                self._move -= self._KEY_MOVE_BINDING[event.input.name]
        return True

    def update(self, sim, root_states):
        """Set the camera view based on the current mode and robot pose.

        Args:
            sim: The simulation context (used to set the camera view).
            root_states: Robot root state tensor of shape (num_envs, 13).
        """
        # robot base position and yaw (env 0)
        base_pos = root_states[0, :3].cpu().numpy()
        root_quat = root_states[0, 3:7].unsqueeze(0).cpu()
        yaw_dir = quat_apply_yaw(root_quat, torch.tensor([[1.0, 0.0, 0.0]])).squeeze(0).numpy()
        yaw_angle = np.arctan2(yaw_dir[1], yaw_dir[0])

        if self._mode == CameraMode.FREE:
            # lazily start next to the robot so the free camera is not far away
            if self._eye is None:
                self._eye = base_pos + np.asarray([-3.0, 0.0, 1.5])
            # move the camera in its local frame: forward points from eye to the robot
            forward = base_pos - self._eye
            forward[2] = 0.0
            fwd_norm = np.linalg.norm(forward)
            if fwd_norm > 1e-6:
                forward /= fwd_norm
            right = np.asarray([forward[1], -forward[0], 0.0])
            self._eye += forward * self._move[0] + right * self._move[1] + np.asarray([0.0, 0.0, self._move[2]])
            target = base_pos
        else:
            offset, look_offset = self.PRESETS[self._mode]
            # rotate offset by robot yaw
            c, s = np.cos(yaw_angle), np.sin(yaw_angle)
            rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            offset = rot @ offset
            look_offset = rot @ look_offset
            eye = base_pos + offset
            target = base_pos + look_offset

        sim.set_camera_view(eye, target)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    registry_name = args_cli.registry_name
    if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
        registry_name += ":latest"
    import pathlib

    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # camera controller for switching views with the keyboard
    camera_controller = CameraController()
    print(camera_controller)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        camera_controller.update(sim, root_states)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
