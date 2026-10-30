"""Export the G1-style deployment YAML from the resolved Go2W environment."""

import json
from pathlib import Path

import numpy as np

SDK_IDS = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 13, 12, 15, 14]
OBSERVATIONS = {
  "base_ang_vel": ("base_ang_vel", 3, {}),
  "projected_gravity": ("projected_gravity", 3, {}),
  "command": ("command", 3, {"command_name": "base_velocity"}),
  "joint_pos": ("joint_pos_rel", 12, {"asset_cfg": {"joint_ids": list(range(12))}}),
  "joint_vel": ("joint_vel_rel", 12, {"asset_cfg": {"joint_ids": list(range(12))}}),
  "actions": ("actions", 16, {}),
  "height_scan": ("height_scan", 187, {"size": 187, "miss_value": 5.0}),
  "wheel_vel": ("wheel_vel", 4, {}),
  "phase": ("phase", 2, {"command_name": "base_velocity"}),
}


def deployment_config(env, metadata):
  """Use resolved scales, history, action order and actuator parameters."""
  from src.assets.robots.unitree_go2w.go2w_constants import LEG_JOINTS, WHEEL_JOINTS

  names = metadata["action_joint_names"]
  if names != list(LEG_JOINTS + WHEEL_JOINTS):
    raise ValueError("Unsupported Go2W joint order")
  actor = env.cfg.observations["actor"]
  observations = {}
  observation_dim = 0
  for name in env.observation_manager.active_terms["actor"]:
    if name not in OBSERVATIONS:
      raise ValueError(f"Unsupported deployment observation {name!r}")
    cpp_name, dimension, params = OBSERVATIONS[name]
    term = env.observation_manager.get_term_cfg("actor", name)
    if name == "phase":
      params = {**params, "period": float(term.params["period"])}
    if not term.flatten_history_dim:
      raise ValueError("Deployment requires flattened term histories")
    scale = term.scale
    if hasattr(scale, "detach"):
      scale = scale.detach().cpu().numpy()
    scale = np.asarray(1.0 if scale is None else scale).reshape(-1).tolist()
    if len(scale) not in (1, dimension):
      raise ValueError(f"Unexpected {name} scale dimensions")
    history = max(1, term.history_length)
    observations[cpp_name] = dict(params=params, clip=term.clip, scale=scale, history_length=history)
    observation_dim += dimension * history

  index = {name: i for i, name in enumerate(metadata["joint_names"])}
  kp, kd, default = [0.0] * 16, [0.0] * 16, []
  for i, name in enumerate(names):
    natural = index[name]
    default.append(float(metadata["default_joint_pos"][natural]))
    # A MuJoCo velocity actuator stores its gain in gainprm[0], NOT biasprm[2].
    kp[SDK_IDS[i]] = float(metadata["joint_stiffness"][natural]) if i < 12 else 0.0
    kd[SDK_IDS[i]] = float(metadata["joint_damping"][natural]) if i < 12 else float(metadata["joint_stiffness"][natural])
  actions = {}
  for cpp_name, start, stop in (("JointPositionAction", 0, 12), ("JointVelocityAction", 12, 16)):
    clip = metadata["action_clip"]
    actions[cpp_name] = dict(
      joint_ids=list(range(start, stop)),
      scale=metadata["action_scale"][start:stop],
      offset=metadata["action_offset"][start:stop],
      clip=None if clip is None else [[-clip, clip]] * (stop - start),
    )
  scan_term = actor.terms.get("height_scan")
  memory = scan_term is not None and "mid360_sensor_name" in scan_term.params
  scan_params = scan_term.params if memory else {}
  scan = dict(
    size_x_m=1.6, size_y_m=1.0, resolution_m=0.1,
    x_offset_m=float(scan_params.get("x_offset_m", 0.0)),
    y_offset_m=float(scan_params.get("y_offset_m", 0.0)),
    miss_value_m=5.0, memory_max_age_s=float(scan_params.get("memory_max_age_s", 0.0)),
    mask_visible=memory,
    visible_x_min_m=float(scan_params.get("visible_x_min_m", -10.0)),
    visible_x_max_m=float(scan_params.get("visible_x_max_m", 10.0)),
    visible_y_abs_m=float(scan_params.get("visible_y_abs_m", 10.0)),
  )
  # Persist the exact training-device mask. Float32 values on the visibility
  # boundary can otherwise select a different column in C++/other devices.
  import torch
  x = torch.arange(17, device=env.device, dtype=torch.float32)*0.1 - 0.8 + scan["x_offset_m"]
  y = torch.arange(11, device=env.device, dtype=torch.float32)*0.1 - 0.5 + scan["y_offset_m"]
  gx, gy = torch.meshgrid(x, y, indexing="xy")
  visible = (gx >= scan["visible_x_min_m"]) & (gx <= scan["visible_x_max_m"]) & (gy.abs() <= scan["visible_y_abs_m"])
  scan["visible_mask"] = (visible if memory else torch.ones_like(visible)).flatten().int().cpu().tolist()
  return dict(
    go2w_deploy_schema=1, joint_ids_map=SDK_IDS, action_joint_names=names,
    step_dt=float(env.step_dt), observation_dim=observation_dim,
    stiffness=kp, damping=kd, default_joint_pos=default,
    commands={"base_velocity": {"ranges": {"lin_vel_x": [-0.2, 0.3], "lin_vel_y": [-0.15, 0.15], "ang_vel_z": [-0.3, 0.3]}}},
    actions=actions, observations=observations,
    perception_required=scan_term is not None, height_scan=scan,
  )


def write_deploy_yaml(onnx_path):
  import onnx
  import yaml

  path = Path(onnx_path)
  model = onnx.load(str(path), load_external_data=False)
  metadata = {item.key: item.value for item in model.metadata_props}
  if "go2w_deploy_contract" not in metadata:
    raise ValueError("ONNX lacks the deployment contract; re-export the checkpoint with scripts/prepare_go2w_deploy.py")
  config = json.loads(metadata["go2w_deploy_contract"])
  validate_contract(model, config, metadata)
  config["onnx_metadata"] = metadata
  params = path.parent / "params"
  params.mkdir(exist_ok=True, parents=True)
  (params / "deploy.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
  return config


def validate_contract(model, config, metadata):
  if config.get("go2w_deploy_schema") != 1:
    raise ValueError("Unsupported Go2W deployment contract")
  if len(model.graph.input) != 1 or len(model.graph.output) != 1:
    raise ValueError("Go2W requires one observation input and one action output")
  if any(value.type.tensor_type.elem_type != 1 for value in [*model.graph.input, *model.graph.output]):
    raise ValueError("Go2W deployment requires float32 input and output tensors")
  shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
  if shape(model.graph.input[0]) != [1, config["observation_dim"]] or shape(model.graph.output[0]) != [1, 16]:
    raise ValueError("ONNX tensor dimensions do not match the deployment contract")
  if model.graph.input[0].name != "obs":
    raise ValueError("Shared G1 ObservationManager requires the ONNX input name 'obs'")
  if metadata.get("observation_layout") != "term_major_history":
    raise ValueError("Unsupported observation history layout")
  if metadata.get("action_types", "").split(",") != ["position"]*12 + ["velocity"]*4:
    raise ValueError("Go2W requires 12 position actions followed by 4 wheel velocity actions")
  cpp_names = {value[0]: key for key, value in OBSERVATIONS.items()}
  if [cpp_names[name] for name in config["observations"]] != metadata["observation_names"].split(","):
    raise ValueError("Observation term order does not match the ONNX")
  if not np.isclose(float(metadata["control_dt"]), config["step_dt"]):
    raise ValueError("Control period does not match the ONNX")
  if metadata["action_joint_names"].split(",") != config["action_joint_names"]:
    raise ValueError("Action joint names do not match the ONNX")
  for key, values in (("action_scale", [x for term in config["actions"].values() for x in term["scale"]]),
                      ("action_offset", [x for term in config["actions"].values() for x in term["offset"]])):
    if not np.allclose(np.fromstring(metadata[key], sep=","), values, atol=5e-4):
      raise ValueError(f"{key} does not match the ONNX")
