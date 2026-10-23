"""A partial reset must not overwrite another environment's sampled command."""
from types import SimpleNamespace
import unittest

import torch

from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommand


class VelocityCommandResetTests(unittest.TestCase):
  def make_command(self):
    command=UniformVelocityCommand.__new__(UniformVelocityCommand)
    command.cfg=SimpleNamespace(heading_command=True,heading_control_stiffness=.5,
      ranges=SimpleNamespace(ang_vel_z=(-1.,1.)))
    command.robot=SimpleNamespace(data=SimpleNamespace(heading_w=torch.zeros(3)))
    command.heading_target=torch.tensor([1.,1.,1.])
    command.heading_error=torch.zeros(3)
    command.vel_command_b=torch.ones(3,3)
    command.is_heading_env=torch.ones(3,dtype=torch.bool)
    command.is_standing_env=torch.tensor([False,False,True])
    return command

  def test_partial_reset_preserves_other_commands(self):
    command=self.make_command()
    before=command.vel_command_b.clone()
    command._update_command(torch.tensor([1]))
    torch.testing.assert_close(command.vel_command_b[[0,2]],before[[0,2]])
    self.assertAlmostEqual(command.vel_command_b[1,2].item(),.5,places=6)

  def test_regular_update_applies_heading_and_standing(self):
    command=self.make_command()
    command._update_command(None)
    torch.testing.assert_close(command.vel_command_b[:2,2],torch.full((2,),.5))
    torch.testing.assert_close(command.vel_command_b[2],torch.zeros(3))


if __name__=='__main__':
  unittest.main()
