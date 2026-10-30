"""Install the versioned Go2W extension alongside the SDK's G1/Go2 wrappers."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def apply(sdk_root=None):
  sdk = Path(sdk_root) if sdk_root else ROOT / "unitree_sdk2"
  if not (sdk / "include/unitree/idl/go2/LowCmd_.hpp").is_file():
    raise RuntimeError(f"SDK2 is missing: {sdk}; copy the submodule contents as well")
  source = ROOT / "patches/unitree_sdk2/include"
  for path in source.rglob("*.h"):
    target = sdk / "include" / path.relative_to(source)
    if target.exists() and target.read_bytes() != path.read_bytes():
      if b"Go2W SDK extension" not in target.read_bytes()[:200]:
        raise RuntimeError(f"Refusing to overwrite an unrelated SDK file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
  return sdk


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sdk-root", type=Path)
  args = parser.parse_args()
  print(apply(args.sdk_root))
