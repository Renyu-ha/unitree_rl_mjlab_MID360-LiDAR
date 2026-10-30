"""Generate a portable Go2W SDK scene using only the Python standard library."""
import argparse
from copy import deepcopy
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/assets/robots/unitree_go2w/xmls"
LEGS = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf"))
WHEELS = tuple(f"{leg}_wheel_joint" for leg in ("FL", "FR", "RL", "RR"))


def flatten(path):
  root = ET.parse(path).getroot()
  def expand(parent, directory):
    for child in list(parent):
      if child.tag == "include":
        included = flatten(directory / child.attrib["file"])
        index = list(parent).index(child)
        parent.remove(child)
        for element in list(included):
          parent.insert(index, deepcopy(element))
          index += 1
      else:
        expand(child, directory)
  expand(root, path.parent)
  return root


def make_scene(output: Path, *, stairs=False, wheel_collision="cylinder"):
  output = output.resolve()
  source = ASSETS / ("go2w_mesh.xml" if wheel_collision == "mesh" else "go2w.xml")
  root = flatten(source)
  compiler = root.find("compiler")
  if compiler is not None:
    compiler.attrib.pop("meshdir", None)
  for mesh in root.findall("./asset/mesh"):
    try:
      relative = Path(os.path.relpath(ASSETS / "assets" / mesh.attrib["file"], output.parent)).as_posix()
    except ValueError:  # Windows temporary directory on another volume.
      shutil.copytree(ASSETS / "assets", output.parent / "assets", dirs_exist_ok=True)
      relative = "assets/" + mesh.attrib["file"]
    mesh.set("file", relative)
  ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
  world = root.find("worldbody")
  ET.SubElement(world, "geom", name="ground", type="plane", size="20 20 0.1", friction="0.8 0.02 0.01", group="0")
  if stairs:
    for i in range(4):
      height = (i+1)*0.08
      ET.SubElement(world, "geom", name=f"step_{i}", type="box", pos=f"{1.2+i*0.4} 0 {height/2}", size=f"0.2 1 {height/2}", group="0")
  for joint in root.findall(".//default//joint"):
    joint.set("damping", "0")
    joint.set("frictionloss", "0")
  for joint in world.iter("joint"):
    name = joint.get("name", "")
    if name not in LEGS + WHEELS:
      continue
    joint.set("damping", "0")
    joint.set("frictionloss", "0")
    joint.set("armature", "0" if name in WHEELS else ("0.02" if "calf" in name else "0.01"))
  actuator = ET.SubElement(root, "actuator")
  for i, name in enumerate(LEGS + WHEELS):
    limit = 45 if name in WHEELS or "calf" in name else 23.5
    ET.SubElement(actuator, "motor", name=f"sdk_{i}", joint=name, gear="1", ctrllimited="true", ctrlrange=f"{-limit} {limit}")
  keyframe = ET.SubElement(root, "keyframe")
  qpos = [0, 0, 0.4, 1, 0, 0, 0]
  for joint in world.iter("joint"):
    name = joint.get("name", "")
    if name in WHEELS: qpos.append(0)
    elif name in LEGS:
      qpos.append((-0.1 if name.startswith(("FL", "RL")) else 0.1) if "hip" in name else (0.9 if "thigh" in name else -1.8))
  ET.SubElement(keyframe, "key", name="home", qpos=" ".join(map(str, qpos)))
  output.parent.mkdir(parents=True, exist_ok=True)
  ET.indent(root)
  ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
  try:
    import mujoco
  except ImportError:
    return None
  model = mujoco.MjModel.from_xml_path(str(output))
  if model.nu != 16 or model.nq != 23:
    raise ValueError("Generated scene does not have the expected Go2W joints")
  return model


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=ROOT / "simulate/robots/go2w/scene.xml")
  parser.add_argument("--stairs", action="store_true")
  parser.add_argument("--wheel-collision", choices=["cylinder", "mesh"], default="cylinder")
  args = parser.parse_args()
  make_scene(args.output, stairs=args.stairs, wheel_collision=args.wheel_collision)
  print(args.output.resolve())


if __name__ == "__main__":
  main()
