"""Interactive policy play near the terrain ring, without boundary resets."""

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.viewer import ViserPlayViewer

from src.utils.border_inspection import find_terrain_border


class PlayVelocityCommand(UniformVelocityCommand):
  def create_gui(self, name, server, get_env_idx):
    """Keep zero-range axes fixed; Viser rejects the upstream zero-max slider."""
    self._joystick_sliders = []
    self._joystick_get_env_idx = get_env_idx
    with server.gui.add_folder(name.capitalize()):
      self._joystick_enabled = server.gui.add_checkbox("Enable", initial_value=True)
      for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        limits = getattr(self.cfg.ranges, axis)
        bound = max(abs(limits[0]), abs(limits[1]))
        slider = server.gui.add_slider(axis, min=-max(bound, 0.05), max=max(bound, 0.05),
          step=0.01, initial_value=0.0, disabled=bound == 0)
        self._joystick_sliders.append(slider)
      zero = server.gui.add_button("Zero")

      @zero.on_click
      def _(_):
        for slider in self._joystick_sliders:
          slider.value = 0.0


@dataclass(kw_only=True)
class PlayVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env):
    return PlayVelocityCommand(self, env)


def configure_play_commands(cfg):
  command = cfg.commands.get("twist")
  if isinstance(command, UniformVelocityCommandCfg):
    cfg.commands["twist"] = PlayVelocityCommandCfg(
      entity_name=command.entity_name,
      resampling_time_range=command.resampling_time_range,
      debug_vis=command.debug_vis,
      heading_command=command.heading_command,
      heading_control_stiffness=command.heading_control_stiffness,
      rel_standing_envs=command.rel_standing_envs,
      rel_heading_envs=command.rel_heading_envs,
      init_velocity_prob=command.init_velocity_prob,
      ranges=command.ranges,
      viz=command.viz,
    )


def reset_play_pose(env, env_ids, position, yaw):
  """Use a world position, independent of the randomly assigned spawn tile."""
  robot = env.scene["robot"]
  state = robot.data.default_root_state[env_ids].clone()
  state[:, :3] = torch.tensor(position, device=env.device)
  state[:, 3:7] = torch.tensor([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], device=env.device)
  state[:, 7:] = 0
  robot.write_root_state_to_sim(state, env_ids)


def configure_border_play(cfg, training_cfg, position, yaw):
  from copy import deepcopy

  cfg.scene.num_envs = 1
  cfg.seed = 42
  cfg.scene.terrain = deepcopy(training_cfg.scene.terrain)
  # Keep the curriculum generator's geometry, while disabling curriculum
  # updates and physical randomization for this controlled contact experiment.
  cfg.curriculum = {}
  cfg.terminations = {}
  cfg.events = {
    "reset_base": EventTermCfg(func=reset_play_pose, mode="reset",
      params={"position": position, "yaw": yaw}),
    "reset_robot_joints": deepcopy(cfg.events["reset_robot_joints"]),
  }
  cfg.observations["actor"].nan_policy = "error"
  cfg.observations["actor"].nan_check_per_term = True
  cfg.observations["critic"].nan_policy = "error"
  cfg.observations["critic"].nan_check_per_term = True
  cfg.viewer.elevation = -25.0


def verify_reference_geometry(model, reference_path):
  """Reject a regenerated scene that differs from the supplied failure model."""
  reference = mujoco.MjModel.from_binary_path(str(reference_path))
  if model.ngeom != reference.ngeom:
    raise ValueError("Regenerated geometry count differs from the reference MJB")
  for index in range(model.ngeom):
    if model.geom(index).name != reference.geom(index).name:
      raise ValueError(f"Reference geom name mismatch at {index}")
  # Compare terrain mesh/heightfield contents as well as geometry transforms.
  for field in ("geom_bodyid", "geom_type", "geom_pos", "geom_quat", "geom_size",
                "hfield_data", "hfield_size", "hfield_nrow", "hfield_ncol"):
    current, saved = getattr(model, field), getattr(reference, field)
    if current.shape != saved.shape or not np.array_equal(current, saved):
      raise ValueError(f"Regenerated geometry differs from the reference MJB: {field}")
  # Removing unused mesh assets can renumber mesh IDs without changing any
  # geometry. Compare the actual meshes referenced by each corresponding geom.
  checked = set()
  for index in range(model.ngeom):
    current_id, saved_id = int(model.geom_dataid[index]), int(reference.geom_dataid[index])
    if model.geom_type[index] != mujoco.mjtGeom.mjGEOM_MESH:
      if current_id != saved_id:
        raise ValueError("Regenerated geometry differs from the reference MJB: geom_dataid")
      continue
    if (current_id, saved_id) in checked:
      continue
    checked.add((current_id, saved_id))
    for field in ("mesh_pos", "mesh_quat", "mesh_scale"):
      if not np.array_equal(getattr(model, field)[current_id], getattr(reference, field)[saved_id]):
        raise ValueError(f"Regenerated geometry differs from the reference MJB: {field}")
    for kind in ("vert", "face"):
      current_start = getattr(model, f"mesh_{kind}adr")[current_id]
      current_count = getattr(model, f"mesh_{kind}num")[current_id]
      saved_start = getattr(reference, f"mesh_{kind}adr")[saved_id]
      saved_count = getattr(reference, f"mesh_{kind}num")[saved_id]
      current = getattr(model, f"mesh_{kind}")[current_start:current_start + current_count]
      saved = getattr(reference, f"mesh_{kind}")[saved_start:saved_start + saved_count]
      if current.shape != saved.shape or not np.array_equal(current, saved):
        raise ValueError(f"Regenerated geometry differs from the reference MJB: mesh_{kind}")
  print("[INFO] Geometry matches the supplied MJB (including terrain mesh/heightfield data)")


class BorderPlayViewer(ViserPlayViewer):
  def setup(self):
    super().setup()
    env = self.env.unwrapped
    model = env.sim.mj_model
    self._border_ids = find_terrain_border(model)
    if not self._border_ids:
      raise ValueError("No standard terrain border found for interactive inspection")
    self._contact_data = mujoco.MjData(model)
    self._report_time = float("-inf")
    self._wheel_ids = tuple(model.body(f"robot/{leg}_wheel_link").id for leg in ("FL", "FR", "RL", "RR"))
    self._wheel_geom_ids = {i for i in range(model.ngeom) if "wheel_link_collision" in model.geom(i).name}
    robot_geoms = [i for i in range(model.ngeom) if model.geom(i).name.startswith("robot/")
      and "collision" in model.geom(i).name]
    self._robot_masks = sorted({(int(model.geom_contype[i]), int(model.geom_conaffinity[i])) for i in robot_geoms})
    self._scene.debug_visualization_enabled = False
    command = env.command_manager.get_term("twist")
    # These handles belong to this repository's UniformVelocityCommand GUI.
    command._joystick_enabled.value = True
    self._drive_sliders = command._joystick_sliders
    self.set_command(0.0, 0.0)
    with self._server.gui.add_folder("边界检查 / Border inspection"):
      self._server.gui.add_markdown(
        "**已加载训练策略。** 点击 Play 开始；下方按钮设置速度，"
        "Commands → Twist 中的滑块可连续调速。\n\n"
        "橙线为地形与 border 的接缝。Slower 可慢放，Step 可单步。"
        "此检查关闭自动重置、扰动和物理随机化。"
      )
      for label, velocity, yaw in (("前进 0.3 m/s", 0.3, 0.0), ("后退 0.2 m/s", -0.2, 0.0),
                                   ("左转", 0.0, 0.2), ("右转", 0.0, -0.2), ("停车", 0.0, 0.0)):
        button = self._server.gui.add_button(label)

        @button.on_click
        def _(_, vx=velocity, wz=yaw):
          self.set_command(vx, wz)

      self._border_report = self._server.gui.add_markdown("")
    for geom_id in self._border_ids:
      self._server.scene.add_box(f"/fixed_bodies/border_inspection/{geom_id}",
        color=(255, 140, 0), dimensions=2 * model.geom_size[geom_id],
        position=model.geom_pos[geom_id], wireframe=True)
    top, bottom, left, right = self._border_ids
    half_x = model.geom_pos[right, 0] - model.geom_size[right, 0]
    half_y = model.geom_pos[top, 1] - model.geom_size[top, 1]
    self._half_extents = half_x, half_y
    points = np.array([[[-half_x, -half_y, .004], [half_x, -half_y, .004]],
                       [[-half_x, half_y, .004], [half_x, half_y, .004]],
                       [[-half_x, -half_y, .004], [-half_x, half_y, .004]],
                       [[half_x, -half_y, .004], [half_x, half_y, .004]]])
    self._server.scene.add_line_segments("/fixed_bodies/border_seams", points=points,
      colors=(255, 140, 0), line_width=4)
    self.sync_env_to_viewer()

  def set_command(self, vx, wz):
    self._drive_sliders[0].value = vx
    self._drive_sliders[1].value = 0.0
    self._drive_sliders[2].value = wz

  def reset_environment(self):
    self.set_command(0.0, 0.0)
    super().reset_environment()

  def sync_env_to_viewer(self):
    super().sync_env_to_viewer()
    if not hasattr(self, "_border_report") or time.monotonic() - self._report_time < 0.1:
      return
    self._report_time = time.monotonic()
    env = self.env.unwrapped
    model, data = env.sim.mj_model, self._contact_data
    data.qpos[:] = env.sim.data.qpos[0].detach().cpu().numpy()
    mujoco.mj_fwdPosition(model, data)
    contacts, wheel_distances = [], []
    for contact in data.contact:
      a, b = int(contact.geom1), int(contact.geom2)
      if a in self._wheel_geom_ids or b in self._wheel_geom_ids:
        wheel_distances.append(float(contact.dist))
      if a in self._border_ids or b in self._border_ids:
        contacts.append(f"- {model.geom(a).name} / {model.geom(b).name}: {contact.dist * 1000:.2f} mm")
    heights = ", ".join(f"{leg}: {data.xpos[body, 2]:.4f}" for leg, body in zip(("FL", "FR", "RL", "RR"), self._wheel_ids))
    penetration = max(0.0, -min(wheel_distances, default=0.0)) * 1000
    half_x, half_y = self._half_extents
    self._border_report.content = (
      f"Root XYZ: **{data.qpos[0]:.4f}, {data.qpos[1]:.4f}, {data.qpos[2]:.4f} m**\n\n"
      f"Grid: X ±{half_x:g}, Y ±{half_y:g} m\n\n"
      f"机器人碰撞掩码 (contype / conaffinity): **{self._robot_masks}**。1/1 开启自碰撞。\n\n"
      f"轮心高度 / wheel hub Z (m): **{heights}**\n\n"
      f"最大轮子穿入深度: **{penetration:.2f} mm**\n\n"
      "Border 接触（负距离表示穿入）:\n\n" + ("\n".join(contacts) or "None")
      + f"\n\nBackend: **MJWarp / {env.device}**. 距离由当前姿态在 CPU MuJoCo 中重建；不是 GPU 接触力。"
    )
