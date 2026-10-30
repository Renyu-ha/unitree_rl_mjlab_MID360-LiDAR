"""Restore Git symlink placeholders after copying a Windows checkout to Linux."""
from pathlib import Path
import os

ROOT=Path(__file__).resolve().parents[1]


def repair(root=ROOT):
  directories=[root/"unitree_sdk2/thirdparty/lib",root/"simulate/mujoco/lib",root/"deploy/thirdparty"]
  repaired=[]
  for directory in directories:
    for path in directory.rglob("*.so*"):
      if path.is_symlink() or not path.is_file() or path.stat().st_size>256:
        continue
      target=path.read_text(encoding="utf-8").strip()
      if not target or Path(target).name!=target or target==path.name:
        continue
      source=path.parent/target
      if not source.is_file():
        raise RuntimeError(f"Missing symlink target: {source}")
      if os.name!="posix":
        raise RuntimeError("Run link repair on the destination Linux machine")
      path.unlink()
      path.symlink_to(target)
      repaired.append(str(path.relative_to(root)))
  return repaired


if __name__=="__main__":
  print("Restored library links:",repair())
