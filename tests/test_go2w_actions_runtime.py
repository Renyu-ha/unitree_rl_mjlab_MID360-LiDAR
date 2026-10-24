"""Compare ordered actions to compiled MuJoCo transmissions and native mjlab."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.tasks.registry import load_env_cfg

import src.tasks  # noqa: F401
from src.assets.robots.unitree_go2w import go2w_constants as robot
from src.tasks.velocity import mdp


class Go2WActionsRuntimeTests(unittest.TestCase):
  def test_permuted_and_subset_actions_match_native_controls_and_physical_torques(self):
    legs = tuple(robot.LEG_JOINTS[i] for i in (6, 1, 10, 0, 5, 9, 2, 8, 3, 11, 4, 7))
    wheels = ("RR_wheel_joint", "FL_wheel_joint", "RL_wheel_joint", "FR_wheel_joint")
    cases = (
      ("default_pose", legs, wheels, True, False),
      ("named_affine_clip", legs, wheels, False, True),
      ("subset", ("RR_calf_joint", "FL_hip_joint", "FR_thigh_joint"),
       ("RR_wheel_joint", "FL_wheel_joint"), True, True),
    )
    for xml in (robot.GO2W_CYLINDER_XML, robot.GO2W_MESH_XML):
      for preserve in (True, False):
        for label, leg_query, wheel_query, default_offset, named in cases:
          with self.subTest(model=xml.name, preserve=preserve, case=label), patch.object(robot, "GO2W_XML", xml):
            cfg = load_env_cfg("Unitree-Go2W-Flat")
            cfg.seed, cfg.scene.num_envs = 7, 2
            cfg.curriculum, cfg.events = {}, {}
            defaults = {name: (-.1 if name[:2] in ("FL", "RL") else .1)
                        if "_hip_" in name else (.9 if "_thigh_" in name else -1.8)
                        for name in robot.LEG_JOINTS}
            # Distinct defaults and per-env bias expose target/offset misalignment.
            defaults = {name: value + .001 * index for index, (name, value) in enumerate(defaults.items())}
            defaults.update({name: 0. for name in robot.WHEEL_JOINTS})
            cfg.scene.entities["robot"].init_state.joint_pos = defaults
            position = dict(entity_name="robot", actuator_names=leg_query, preserve_order=preserve,
                            use_default_offset=default_offset, scale=robot.LEG_ACTION_SCALE, offset=.012)
            velocity = dict(entity_name="robot", actuator_names=wheel_query, preserve_order=preserve,
                            use_default_offset=False, scale=robot.WHEEL_ACTION_SCALE, offset=0.)
            if named:
              position.update(scale={r".*_hip_joint": .2, r".*_thigh_joint": .3, r".*_calf_joint": .4},
                              offset={r".*_hip_joint": .03, r".*_thigh_joint": .05, r".*_calf_joint": -.07},
                              clip={leg_query[0]: (-.04, .04)})
              velocity.update(scale={name: float(robot.WHEEL_JOINTS.index(name) + 2) for name in wheel_query},
                              offset={name: -.1 * (robot.WHEEL_JOINTS.index(name) + 1) for name in wheel_query},
                              clip={wheel_query[0]: (-.07, .09)})
            cfg.actions = {"joint_pos": mdp.OrderedJointPositionActionCfg(**position),
                           "wheel_vel": mdp.OrderedJointVelocityActionCfg(**velocity)}
            env = ManagerBasedRlEnv(cfg, device="cpu")
            try:
              env.reset()
              self._check_controls(env, defaults, position, velocity, named)
            finally:
              env.close()

  def _check_controls(self, env, defaults, position, velocity, named):
    model, asset = env.sim.mj_model, env.scene["robot"]
    # The oracle uses compiled joint transmissions, independently of the action's
    # names/IDs and Entity.find_* helpers.
    actuator_by_joint = {model.joint(int(model.actuator_trnid[i, 0])).name.rsplit("/", 1)[-1]: i
                         for i in range(model.nu)}
    joint_by_name = {model.joint(i).name.rsplit("/", 1)[-1]: i for i in range(model.njnt)}
    queries = [position["actuator_names"], velocity["actuator_names"]]
    natural = [sorted(query, key=joint_by_name.__getitem__) for query in queries]
    expected_names = [list(query) if position["preserve_order"] else order
                      for query, order in zip(queries, natural, strict=True)]
    for term_name, expected in zip(("joint_pos", "wheel_vel"), expected_names, strict=True):
      self.assertEqual(list(env.action_manager.get_term(term_name).target_names), expected)
    if position["preserve_order"]:
      self.assertTrue(all(order != expected for order, expected in zip(natural, expected_names, strict=True)))
    biases = {}
    for index, name in enumerate(asset.joint_names):
      biases[name] = torch.tensor([.001 * (index + 1), -.0015 * (index + 1)])
      asset.data.encoder_bias[:, index] = biases[name]
    count = sum(map(len, queries))
    first = torch.linspace(-.7, .9, count)
    values = torch.stack((first, -.5 * first + .13))
    before = env.sim.data.ctrl.clone()
    env.action_manager.process_action(values)
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    controls = env.sim.data.ctrl.clone()
    value_by_name = {}
    column = 0
    for group, names in enumerate(expected_names):
      for name in names:
        raw = values[:, column]
        value_by_name[name] = raw
        if group == 0:
          kind = "hip" if "_hip_" in name else ("thigh" if "_thigh_" in name else "calf")
          scale = {"hip": .2, "thigh": .3, "calf": .4}[kind] if named else robot.LEG_ACTION_SCALE
          offset = defaults[name] if position["use_default_offset"] else {"hip": .03, "thigh": .05, "calf": -.07}[kind]
          target = raw * scale + offset
          if named and name == queries[0][0]:
            target = target.clamp(-.04, .04)
          target = target - biases[name]
        else:
          index = robot.WHEEL_JOINTS.index(name)
          target = raw * (index + 2 if named else robot.WHEEL_ACTION_SCALE) + (-.1 * (index + 1) if named else 0.)
          if named and name == queries[1][0]:
            target = target.clamp(-.07, .09)
        torch.testing.assert_close(controls[:, actuator_by_joint[name]], target)
        column += 1
    unselected = sorted(set(range(model.nu)) - {actuator_by_joint[name] for name in value_by_name})
    if unselected:
      torch.testing.assert_close(controls[:, unselected], before[:, unselected])
    env.sim.forward()
    forces = env.sim.data.actuator_force.clone()
    for name in value_by_name:
      actuator_id, joint_id = actuator_by_joint[name], joint_by_name[name]
      q = env.sim.data.qpos[:, int(model.jnt_qposadr[joint_id])]
      v = env.sim.data.qvel[:, int(model.jnt_dofadr[joint_id])]
      gain, bias = model.actuator_gainprm[actuator_id], model.actuator_biasprm[actuator_id]
      expected_force = gain[0] * controls[:, actuator_id] + bias[0] + bias[1] * q + bias[2] * v
      if model.actuator_forcelimited[actuator_id]:
        expected_force = expected_force.clamp(*model.actuator_forcerange[actuator_id])
      torch.testing.assert_close(forces[:, actuator_id], expected_force, atol=2e-5, rtol=2e-5)
    # The unmodified mjlab actions receive equivalent commands in natural order.
    # They must produce exactly the same controls and actuator forces.
    for native_type, params, names in zip((JointPositionActionCfg, JointVelocityActionCfg),
                                          (position, velocity), natural, strict=True):
      native_params = deepcopy(params)
      native_params["preserve_order"] = False
      action = native_type(**native_params).build(env)
      self.assertEqual(list(action.target_names), names)
      action.process_actions(torch.stack([value_by_name[name] for name in names], dim=-1))
      action.apply_actions()
    env.scene.write_data_to_sim()
    torch.testing.assert_close(env.sim.data.ctrl[:], controls)
    env.sim.forward()
    torch.testing.assert_close(env.sim.data.actuator_force[:], forces, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
  unittest.main()
