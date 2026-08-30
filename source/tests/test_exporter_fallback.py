import torch
from types import SimpleNamespace

from whole_body_tracking.utils.exporter import get_default_joint_pos


def test_get_default_joint_pos_falls_back_to_default_joint_pos():
    robot = SimpleNamespace()
    robot.data = SimpleNamespace(default_joint_pos=torch.tensor([[0.1, 0.2, 0.3]]))

    assert get_default_joint_pos(robot) == [0.1, 0.2, 0.3]
